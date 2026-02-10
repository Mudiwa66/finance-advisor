#!/usr/bin/env python3
"""Flask webhook for answering spending questions via WhatsApp (Twilio)."""

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests as http_requests
from flask import Flask, request
from markupsafe import escape
from twilio.twiml.messaging_response import MessagingResponse

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"

# ---------------------------------------------------------------------------
# Load transaction data once at startup
# ---------------------------------------------------------------------------

DATA_FILE = Path(__file__).parent / "data" / "transactions.json"

with open(DATA_FILE) as f:
    TRANSACTIONS: list[dict] = json.load(f)

# ---------------------------------------------------------------------------
# Ollama LLM integration
# ---------------------------------------------------------------------------


def build_spending_summary() -> str:
    """Pre-compute a text summary of all transactions for the LLM system prompt."""
    total_debits = sum(t["amount"] for t in TRANSACTIONS if t["amount"] < 0)
    total_credits = sum(t["amount"] for t in TRANSACTIONS if t["amount"] > 0)
    debit_count = sum(1 for t in TRANSACTIONS if t["amount"] < 0)
    credit_count = sum(1 for t in TRANSACTIONS if t["amount"] > 0)
    closing_balance = TRANSACTIONS[-1]["balance"]
    first_date = TRANSACTIONS[0]["date"]
    last_date = TRANSACTIONS[-1]["date"]

    # Top 15 merchants
    spending: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for t in TRANSACTIONS:
        if t["amount"] < 0:
            m = extract_merchant(t["description"])
            spending[m] += abs(t["amount"])
            counts[m] += 1

    top_merchants = "\n".join(
        f"  - {m}: R{amt:,.2f} ({counts[m]} transactions)" for m, amt in spending.most_common(15)
    )

    # Monthly breakdown
    monthly: dict[str, dict] = {}
    for t in TRANSACTIONS:
        month_key = t["date"][:7]  # "2024-02"
        if month_key not in monthly:
            monthly[month_key] = {"debits": 0.0, "credits": 0.0, "count": 0}
        monthly[month_key]["count"] += 1
        if t["amount"] < 0:
            monthly[month_key]["debits"] += t["amount"]
        else:
            monthly[month_key]["credits"] += t["amount"]

    month_lines = "\n".join(
        f"  - {k}: spent R{abs(v['debits']):,.2f}, income R{v['credits']:,.2f} ({v['count']} txns)"
        for k, v in sorted(monthly.items())
    )

    return (
        f"You are a helpful financial assistant for a South African FNB bank account.\n"
        f"Statement period: {first_date} to {last_date}\n"
        f"Currency: South African Rand (ZAR), displayed as R.\n\n"
        f"ACCOUNT SUMMARY:\n"
        f"  Total spending (debits): R{abs(total_debits):,.2f} ({debit_count} transactions)\n"
        f"  Total income (credits): R{total_credits:,.2f} ({credit_count} transactions)\n"
        f"  Closing balance: R{closing_balance:,.2f}\n\n"
        f"TOP MERCHANTS:\n{top_merchants}\n\n"
        f"MONTHLY BREAKDOWN:\n{month_lines}\n\n"
        f"Answer concisely. Use the data above to answer spending questions. "
        f"If you don't have enough info, say so. Keep replies under 300 words."
    )


def ask_ollama(question: str) -> dict:
    """Send a question to llama3.2 via Ollama. Returns dict with content, timing, tokens."""
    start = time.monotonic()
    try:
        resp = http_requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SPENDING_SUMMARY},
                    {"role": "user", "content": question},
                ],
                "stream": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.monotonic() - start
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        return {
            "content": data["message"]["content"],
            "llm_time": round(elapsed, 3),
            "tokens": tokens,
            "error": False,
        }
    except Exception:
        elapsed = time.monotonic() - start
        return {
            "content": (
                "Sorry, I couldn't process that right now. "
                "Try a keyword like *total*, *uber*, *march*, or type *help*."
            ),
            "llm_time": round(elapsed, 3),
            "tokens": 0,
            "error": True,
        }


# ---------------------------------------------------------------------------
# Merchant extraction
# ---------------------------------------------------------------------------

DESCRIPTION_PREFIXES = [
    "Card Purchase With Cashback",
    "Chq Card ATM Local Cash Advanc Cash",
    "Refund Chq Card Purchase Cr Vc",
    "Rtc Express Credit",
    "Rtc Express Pmt To",
    "Paypal Withdrawal",
    "Electricity Prepaid",
    "Internet Airtime",
    "Airtime Topup Airtime",
    "Payment 1Day Cr",
    "Payshap Credit",
    "Fuel Purchase",
    "Card Cashback Cashb",
    "Card Purchase",
    "Internet Pmt To",
    "POS Purchase",
    "Magtape Credit",
    "Magtape Debit",
    "Send Money App Dr Send",
    "Send Money Dr Send",
    "FNB App Transfer From",
    "FNB App Payment To",
    "FNB App Payment From",
    "FNB App Rtc Pmt To",
    "FNB OB Pmt",
    "Payment To",
    "Rtc Credit",
    "Byc Debit",
    "ATM Cash",
]

CARD_RE = re.compile(r"\d{6}\*\d{4}")
AMOUNT_PREFIX_RE = re.compile(r"^[\d,]+\.\d{2}\s+")
TRAILING_DATE_RE = re.compile(r"\s+\d{2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$")
TRAILING_NUMBERS_RE = re.compile(r"\s+\d{8,}$")


def extract_merchant(description: str) -> str:
    """Extract a human-readable merchant/payee name from a transaction description."""
    if not description.strip():
        return "Bank Fees"

    text = description
    matched_prefix = None
    for prefix in DESCRIPTION_PREFIXES:
        if text.startswith(prefix):
            matched_prefix = prefix
            text = text[len(prefix) :].strip()
            break

    text = CARD_RE.split(text)[0].strip()
    text = AMOUNT_PREFIX_RE.sub("", text)
    text = TRAILING_DATE_RE.sub("", text)
    text = TRAILING_NUMBERS_RE.sub("", text).strip()

    if text:
        return text
    if matched_prefix:
        return matched_prefix
    return description[:40]


# Pre-compute the summary now that extract_merchant is defined
SPENDING_SUMMARY = build_spending_summary()

# ---------------------------------------------------------------------------
# Month helpers
# ---------------------------------------------------------------------------

MONTH_KEYWORDS = {
    "january": "01",
    "jan": "01",
    "february": "02",
    "feb": "02",
    "march": "03",
    "mar": "03",
    "april": "04",
    "apr": "04",
    "may": "05",
    "june": "06",
    "jun": "06",
    "july": "07",
    "jul": "07",
    "august": "08",
    "aug": "08",
    "september": "09",
    "sep": "09",
    "october": "10",
    "oct": "10",
    "november": "11",
    "nov": "11",
    "december": "12",
    "dec": "12",
}

MONTH_NAMES = {
    "01": "January",
    "02": "February",
    "03": "March",
    "04": "April",
    "05": "May",
    "06": "June",
    "07": "July",
    "08": "August",
    "09": "September",
    "10": "October",
    "11": "November",
    "12": "December",
}

# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_help() -> str:
    return (
        "Welcome! Here are the commands I understand:\n\n"
        "- *total* - Total spending (all debits)\n"
        "- *income* - Total income (all credits)\n"
        "- *balance* - Closing account balance\n"
        "- *top* - Top 10 merchants by spend\n"
        "- *february* / *march* / *april* / *may* - Monthly spending\n"
        "- *uber* / *bolt* / any merchant - Spending at that merchant\n"
        "- Or just ask a question naturally!\n"
        "- *help* - Show this message"
    )


def cmd_total_spending() -> str:
    total = sum(t["amount"] for t in TRANSACTIONS if t["amount"] < 0)
    count = sum(1 for t in TRANSACTIONS if t["amount"] < 0)
    return f"Total spending: R{abs(total):,.2f}\n({count} transactions, Feb-May 2024)"


def cmd_income() -> str:
    total = sum(t["amount"] for t in TRANSACTIONS if t["amount"] > 0)
    count = sum(1 for t in TRANSACTIONS if t["amount"] > 0)
    return f"Total income: R{total:,.2f}\n({count} transactions, Feb-May 2024)"


def cmd_balance() -> str:
    last = TRANSACTIONS[-1]
    return f"Closing balance: R{last['balance']:,.2f}\n(as of {last['date']})"


def cmd_top_merchants() -> str:
    spending: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for t in TRANSACTIONS:
        if t["amount"] < 0:
            merchant = extract_merchant(t["description"])
            spending[merchant] += abs(t["amount"])
            counts[merchant] += 1

    lines = ["Top 10 merchants by spending:\n"]
    for i, (merchant, total) in enumerate(spending.most_common(10), 1):
        lines.append(f"{i}. {merchant}: R{total:,.2f} ({counts[merchant]}x)")
    return "\n".join(lines)


def cmd_month_spending(month_num: str) -> str:
    prefix = f"2024-{month_num}"
    month_txns = [t for t in TRANSACTIONS if t["date"].startswith(prefix)]

    if not month_txns:
        return f"No transactions found for {MONTH_NAMES[month_num]} 2024."

    debits = sum(t["amount"] for t in month_txns if t["amount"] < 0)
    credits = sum(t["amount"] for t in month_txns if t["amount"] > 0)
    count = len(month_txns)

    spending: Counter[str] = Counter()
    for t in month_txns:
        if t["amount"] < 0:
            spending[extract_merchant(t["description"])] += abs(t["amount"])

    lines = [
        f"{MONTH_NAMES[month_num]} 2024 summary:",
        f"  Spending: R{abs(debits):,.2f}",
        f"  Income: R{credits:,.2f}",
        f"  Transactions: {count}",
        "",
        "Top 5 merchants:",
    ]
    for i, (merchant, total) in enumerate(spending.most_common(5), 1):
        lines.append(f"  {i}. {merchant}: R{total:,.2f}")
    return "\n".join(lines)


def cmd_merchant_search(query: str) -> str:
    matches = []
    for t in TRANSACTIONS:
        merchant = extract_merchant(t["description"])
        if query in merchant.lower() or query in t["description"].lower():
            matches.append((t, merchant))

    if not matches:
        return (
            f'No transactions found matching "{query}".\n'
            "Try a merchant name like *uber*, *bolt*, *checkers*, or type *help*."
        )

    total = sum(abs(t["amount"]) for t, _ in matches if t["amount"] < 0)
    count = sum(1 for t, _ in matches if t["amount"] < 0)

    by_merchant: Counter[str] = Counter()
    for t, merchant in matches:
        if t["amount"] < 0:
            by_merchant[merchant] += abs(t["amount"])

    if len(by_merchant) == 1:
        merchant_name = list(by_merchant.keys())[0]
        return f"Spending at {merchant_name}: R{total:,.2f}\n({count} transactions, Feb-May 2024)"

    lines = [f'Spending matching "{query}": R{total:,.2f} total\n']
    for merchant, amt in by_merchant.most_common(10):
        lines.append(f"  - {merchant}: R{amt:,.2f}")
    if len(by_merchant) > 10:
        lines.append(f"  ... and {len(by_merchant) - 10} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Message router
# ---------------------------------------------------------------------------


def _keyword_result(content: str) -> dict:
    return {"content": content, "used_llm": False}


def handle_message(body: str) -> dict:
    """Route an incoming message to the appropriate handler. Returns a result dict."""
    text = body.strip().lower()

    if text in ("help", "commands", "menu", "hi", "hello", "?"):
        return _keyword_result(cmd_help())

    if text in ("total spending", "total spend", "total debits", "total"):
        return _keyword_result(cmd_total_spending())

    if text in ("income", "credits", "total income", "salary", "total credits"):
        return _keyword_result(cmd_income())

    if text in ("balance", "closing balance", "current balance"):
        return _keyword_result(cmd_balance())

    if text in ("top merchants", "top spend", "top", "biggest", "top 10"):
        return _keyword_result(cmd_top_merchants())

    for keyword, month_num in MONTH_KEYWORDS.items():
        if keyword in text:
            return _keyword_result(cmd_month_spending(month_num))

    # Try merchant keyword search first
    merchant_result = cmd_merchant_search(text)
    if not merchant_result.startswith("No transactions found"):
        return {"content": merchant_result, "used_llm": False}

    # Fallback: ask the LLM
    result = ask_ollama(body.strip())
    result["used_llm"] = True
    return result


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Metrics database
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent / "data" / "metrics.db"


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_hash TEXT NOT NULL,
            message_text TEXT NOT NULL,
            used_llm INTEGER NOT NULL DEFAULT 0,
            llm_response_time REAL,
            total_response_time REAL NOT NULL,
            success INTEGER NOT NULL DEFAULT 1,
            tokens_used INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


_init_db()


def _hash_user(phone: str) -> str:
    return hashlib.sha256(phone.encode()).hexdigest()[:12]


def _log_metric(
    user_hash: str,
    message_text: str,
    used_llm: bool,
    llm_time: float | None,
    total_time: float,
    success: bool,
    tokens: int,
) -> None:
    conn = _get_db()
    conn.execute(
        """INSERT INTO metrics
           (timestamp, user_hash, message_text, used_llm,
            llm_response_time, total_response_time, success, tokens_used)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            user_hash,
            message_text,
            int(used_llm),
            llm_time,
            round(total_time, 3),
            int(success),
            tokens,
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


@app.route("/webhook", methods=["POST"])
def webhook():
    """Receive incoming WhatsApp messages from Twilio and respond."""
    start = time.monotonic()
    incoming_msg = request.form.get("Body", "").strip()
    user_phone = request.form.get("From", "anonymous")
    user_hash = _hash_user(user_phone)

    result = handle_message(incoming_msg)
    total_time = time.monotonic() - start

    _log_metric(
        user_hash=user_hash,
        message_text=incoming_msg,
        used_llm=result.get("used_llm", False),
        llm_time=result.get("llm_time"),
        total_time=total_time,
        success=not result.get("error", False),
        tokens=result.get("tokens", 0),
    )

    resp = MessagingResponse()
    resp.message(result["content"])
    return str(resp), 200, {"Content-Type": "application/xml"}


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>Spending Bot Dashboard</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto;
         padding: 0 1rem; background: #f8f9fa; color: #212529; }
  h1 { border-bottom: 2px solid #dee2e6; padding-bottom: .5rem; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
           gap: 1rem; margin: 1.5rem 0; }
  .card { background: #fff; border-radius: 8px; padding: 1.2rem;
          box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  .card .value { font-size: 1.8rem; font-weight: 700; color: #0d6efd; }
  .card .label { font-size: .85rem; color: #6c757d; margin-top: .3rem; }
  table { width: 100%%; border-collapse: collapse; background: #fff;
          border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  th, td { text-align: left; padding: .6rem .8rem; border-bottom: 1px solid #dee2e6; }
  th { background: #e9ecef; font-size: .85rem; text-transform: uppercase; }
  td { font-size: .9rem; }
  .section { margin: 2rem 0; }
  .bar { background: #0d6efd; height: 20px; border-radius: 3px; min-width: 2px; }
  .bar-row { display: flex; align-items: center; gap: .5rem; margin: .3rem 0; }
  .bar-label { font-size: .8rem; width: 80px; text-align: right; }
  .bar-value { font-size: .8rem; color: #6c757d; }
</style>
</head><body>
<h1>Spending Bot Dashboard</h1>

<div class="cards">
  <div class="card"><div class="value">%(today)s</div>
    <div class="label">Messages today</div></div>
  <div class="card"><div class="value">%(week)s</div>
    <div class="label">Messages this week</div></div>
  <div class="card"><div class="value">%(total)s</div>
    <div class="label">Total messages</div></div>
  <div class="card"><div class="value">%(avg_llm)s</div>
    <div class="label">Avg LLM response (s)</div></div>
  <div class="card"><div class="value">%(avg_total)s</div>
    <div class="label">Avg total response (s)</div></div>
  <div class="card"><div class="value">%(error_rate)s</div>
    <div class="label">Error rate</div></div>
</div>

<div class="section">
<h2>Messages per user</h2>
<table><tr><th>User (hashed)</th><th>Messages</th><th>LLM queries</th></tr>
%(user_rows)s
</table></div>

<div class="section">
<h2>Common question patterns</h2>
<table><tr><th>Message</th><th>Count</th></tr>
%(pattern_rows)s
</table></div>

<div class="section">
<h2>Messages over time (last 7 days)</h2>
%(timeseries)s
</div>

</body></html>"""


@app.route("/dashboard")
def dashboard():
    conn = _get_db()
    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")
    week_ago = (now - timedelta(days=7)).isoformat()

    # Counts
    today = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE timestamp >= ?", (today_str,)
    ).fetchone()[0]
    week = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE timestamp >= ?", (week_ago,)
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]

    # Averages
    avg_row = conn.execute(
        "SELECT AVG(llm_response_time), AVG(total_response_time) FROM metrics WHERE used_llm = 1"
    ).fetchone()
    avg_llm = f"{avg_row[0]:.2f}" if avg_row[0] else "-"
    avg_total_row = conn.execute("SELECT AVG(total_response_time) FROM metrics").fetchone()
    avg_total = f"{avg_total_row[0]:.2f}" if avg_total_row[0] else "-"

    # Error rate
    errors = conn.execute("SELECT COUNT(*) FROM metrics WHERE success = 0").fetchone()[0]
    error_rate = f"{errors / total * 100:.1f}%" if total > 0 else "0%"

    # Users
    users = conn.execute(
        "SELECT user_hash, COUNT(*) as cnt, SUM(used_llm) as llm_cnt "
        "FROM metrics GROUP BY user_hash ORDER BY cnt DESC LIMIT 20"
    ).fetchall()
    user_rows = "".join(
        f"<tr><td><code>{escape(r['user_hash'])}</code></td>"
        f"<td>{r['cnt']}</td><td>{r['llm_cnt']}</td></tr>"
        for r in users
    )

    # Patterns
    patterns = conn.execute(
        "SELECT LOWER(message_text) as msg, COUNT(*) as cnt "
        "FROM metrics GROUP BY msg ORDER BY cnt DESC LIMIT 15"
    ).fetchall()
    pattern_rows = "".join(
        f"<tr><td>{escape(r['msg'])}</td><td>{r['cnt']}</td></tr>" for r in patterns
    )

    # Time series (last 7 days, by day)
    days = conn.execute(
        "SELECT DATE(timestamp) as day, COUNT(*) as cnt "
        "FROM metrics WHERE timestamp >= ? "
        "GROUP BY day ORDER BY day",
        (week_ago,),
    ).fetchall()
    max_cnt = max((r["cnt"] for r in days), default=1)
    timeseries = "".join(
        f'<div class="bar-row">'
        f'<span class="bar-label">{r["day"][5:]}</span>'
        f'<div class="bar" style="width:{r["cnt"] / max_cnt * 400}px"></div>'
        f'<span class="bar-value">{r["cnt"]}</span></div>'
        for r in days
    )
    if not timeseries:
        timeseries = "<p>No data yet.</p>"

    conn.close()

    html = DASHBOARD_HTML % {
        "today": today,
        "week": week,
        "total": total,
        "avg_llm": avg_llm,
        "avg_total": avg_total,
        "error_rate": error_rate,
        "user_rows": user_rows or "<tr><td colspan=3>No data yet</td></tr>",
        "pattern_rows": pattern_rows or "<tr><td colspan=2>No data yet</td></tr>",
        "timeseries": timeseries,
    }
    return html


if __name__ == "__main__":
    app.run(debug=True, port=5001)
