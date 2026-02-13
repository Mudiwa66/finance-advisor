#!/usr/bin/env python3
"""Flask webhook for answering spending questions via WhatsApp (Twilio)."""

import hashlib
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import requests as http_requests
from dotenv import load_dotenv
from flask import Flask, request
from markupsafe import escape
from supabase import Client, create_client
from twilio.twiml.messaging_response import MessagingResponse

load_dotenv()

# LLM Configuration - Use Hugging Face Inference API (works on Railway!)
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HUGGINGFACE_API_URL = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}"

# ---------------------------------------------------------------------------
# Supabase configuration
# ---------------------------------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Debug logging for Railway deployment
if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: Missing Supabase environment variables!", file=sys.stderr)
    print(f"SUPABASE_URL: {'SET' if SUPABASE_URL else 'NOT SET'}", file=sys.stderr)
    print(f"SUPABASE_KEY: {'SET' if SUPABASE_KEY else 'NOT SET'}", file=sys.stderr)
    print("Available env vars:", list(os.environ.keys()), file=sys.stderr)
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------------------------
# Load transaction data from Supabase
# ---------------------------------------------------------------------------


def _load_user_transactions(user_id: str) -> list[dict]:
    """Load all transactions for a user from Supabase."""
    try:
        response = (
            supabase.table("transactions")
            .select("date, description, amount, balance, merchant")
            .eq("user_id", user_id)
            .order("date")
            .execute()
        )
        return response.data
    except Exception as e:
        print(f"Error loading transactions: {e}")
        return []


# For backward compatibility, load default user's transactions at startup
# TODO: Replace with actual user_id from migration script output
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "REPLACE_WITH_USER_ID_FROM_MIGRATION")
TRANSACTIONS: list[dict] = _load_user_transactions(DEFAULT_USER_ID)

# ---------------------------------------------------------------------------
# Ollama LLM integration
# ---------------------------------------------------------------------------


def build_spending_summary() -> str:
    """Pre-compute a text summary of all transactions for the LLM system prompt."""
    # Handle empty transactions gracefully
    if not TRANSACTIONS:
        return (
            "You are a helpful financial assistant for a South African FNB bank account.\n"
            "No transaction data is currently loaded. Please try again later or contact support."
        )

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


def ask_llm(question: str) -> dict:
    """Send a question to Mistral via Hugging Face Inference API. Returns dict with content, timing, tokens."""
    start = time.monotonic()

    if not HUGGINGFACE_API_TOKEN:
        # Fallback if no API token
        elapsed = time.monotonic() - start
        return {
            "content": (
                "LLM is not configured. "
                "Try a keyword like *total*, *uber*, *march*, or type *help*."
            ),
            "llm_time": round(elapsed, 3),
            "tokens": 0,
            "error": True,
        }

    try:
        # Format prompt for Mistral Instruct model
        prompt = f"<s>[INST] {SPENDING_SUMMARY}\n\n{question} [/INST]"

        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.95,
                "return_full_text": False,
            },
        }

        resp = http_requests.post(
            HUGGINGFACE_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.monotonic() - start

        # Handle different response formats
        if isinstance(data, list) and len(data) > 0:
            content = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            content = data.get("generated_text", data.get("error", "No response"))
        else:
            content = str(data)

        # Estimate tokens (rough approximation: ~4 chars per token)
        tokens = len(prompt + content) // 4

        return {
            "content": content.strip(),
            "llm_time": round(elapsed, 3),
            "tokens": tokens,
            "error": False,
        }
    except http_requests.exceptions.HTTPError as e:
        elapsed = time.monotonic() - start
        # Handle rate limiting
        if e.response.status_code == 429:
            error_msg = "Rate limit exceeded. Please try again in a moment."
        elif e.response.status_code == 503:
            error_msg = "Model is loading. Please wait ~20 seconds and try again."
        else:
            error_msg = f"API error ({e.response.status_code}). Try keywords instead."

        return {
            "content": f"{error_msg} Try *total*, *uber*, *march*, or *help*.",
            "llm_time": round(elapsed, 3),
            "tokens": 0,
            "error": True,
        }
    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"LLM error: {e}", file=sys.stderr)
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
    word_count = len(text.split())

    # Exact keyword matches
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

    # Month keywords (exact matches only)
    for keyword, month_num in MONTH_KEYWORDS.items():
        if keyword == text or text == f"{keyword} spending":
            return _keyword_result(cmd_month_spending(month_num))

    # For natural language questions (more than 3 words or contains question words), use LLM
    if word_count > 3 or "?" in text or any(
        word in text for word in ["how", "what", "when", "where", "why", "much", "many", "did", "do", "can"]
    ):
        result = ask_llm(body.strip())
        result["used_llm"] = True
        return result

    # Try merchant keyword search (only for short queries)
    if word_count <= 3:
        merchant_result = cmd_merchant_search(text)
        if not merchant_result.startswith("No transactions found"):
            return {"content": merchant_result, "used_llm": False}

    # Fallback: ask the LLM
    result = ask_llm(body.strip())
    result["used_llm"] = True
    return result


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Metrics database (now using Supabase)
# ---------------------------------------------------------------------------


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
    """Log interaction metrics to Supabase."""
    try:
        # Get or create user
        user_response = (
            supabase.table("users")
            .select("id, total_messages")
            .eq("phone_hash", user_hash)
            .execute()
        )

        if user_response.data:
            user_id = user_response.data[0]["id"]
            current_total = user_response.data[0]["total_messages"]
            # Update last_seen and increment message count
            supabase.table("users").update(
                {
                    "last_seen_at": datetime.now(timezone.utc).isoformat(),
                    "total_messages": current_total + 1,
                }
            ).eq("id", user_id).execute()
        else:
            # Create new user
            user_response = (
                supabase.table("users")
                .insert({"phone_hash": user_hash, "total_messages": 1})
                .execute()
            )
            user_id = user_response.data[0]["id"]

        # Insert metric
        supabase.table("metrics").insert(
            {
                "user_id": user_id,
                "message_text": message_text,
                "used_llm": used_llm,
                "llm_response_time": llm_time,
                "total_response_time": round(total_time, 3),
                "success": success,
                "tokens_used": tokens,
            }
        ).execute()

    except Exception as e:
        # Log error but don't fail the webhook response
        print(f"Error logging metric: {e}", file=sys.stderr)


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
# PDF Upload Endpoint
# ---------------------------------------------------------------------------


@app.route("/upload-statement", methods=["POST"])
def upload_statement():
    """Upload bank statement PDF to Supabase Storage."""
    if "file" not in request.files:
        return {"error": "No file provided"}, 400

    file = request.files["file"]
    user_hash = request.form.get("user_hash")

    if not user_hash or not file.filename or not file.filename.endswith(".pdf"):
        return {"error": "Invalid file or user_hash"}, 400

    try:
        # Validate user exists
        user_response = (
            supabase.table("users").select("id").eq("phone_hash", user_hash).single().execute()
        )

        if not user_response.data:
            return {"error": "User not found"}, 404

        # Upload to Supabase Storage
        storage_path = f"{user_hash}/{file.filename}"
        file_bytes = file.read()

        supabase.storage.from_("bank-statements").upload(
            path=storage_path, file=file_bytes, file_options={"content-type": "application/pdf"}
        )

        return {
            "success": True,
            "storage_path": storage_path,
            "message": "PDF uploaded successfully. Process manually with parse_statement.py",
        }, 200

    except Exception as e:
        return {"error": str(e)}, 500


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
    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")
    week_ago = (now - timedelta(days=7)).isoformat()

    try:
        # Counts - today's messages
        today_response = (
            supabase.table("metrics")
            .select("*", count="exact")
            .gte("timestamp", today_str)
            .execute()
        )
        today = today_response.count

        # Count - this week
        week_response = (
            supabase.table("metrics")
            .select("*", count="exact")
            .gte("timestamp", week_ago)
            .execute()
        )
        week = week_response.count

        # Count - total
        total_response = supabase.table("metrics").select("*", count="exact").execute()
        total = total_response.count

        # Averages - fetch all metrics with LLM for client-side aggregation
        metrics_with_llm = (
            supabase.table("metrics")
            .select("llm_response_time")
            .eq("used_llm", True)
            .not_.is_("llm_response_time", "null")
            .execute()
        )

        if metrics_with_llm.data:
            llm_times = [m["llm_response_time"] for m in metrics_with_llm.data]
            avg_llm_value = sum(llm_times) / len(llm_times)
            avg_llm = f"{avg_llm_value:.2f}"
        else:
            avg_llm = "-"

        # Average total response time
        all_metrics = supabase.table("metrics").select("total_response_time").execute()

        if all_metrics.data:
            total_times = [m["total_response_time"] for m in all_metrics.data]
            avg_total_value = sum(total_times) / len(total_times)
            avg_total = f"{avg_total_value:.2f}"
        else:
            avg_total = "-"

        # Error rate
        errors_response = (
            supabase.table("metrics").select("*", count="exact").eq("success", False).execute()
        )
        errors = errors_response.count
        error_rate = f"{errors / total * 100:.1f}%" if total > 0 else "0%"

        # Users - fetch all metrics and group client-side
        all_metrics_for_users = supabase.table("metrics").select("user_id, used_llm").execute()

        user_stats = defaultdict(lambda: {"cnt": 0, "llm_cnt": 0})
        for m in all_metrics_for_users.data:
            user_stats[m["user_id"]]["cnt"] += 1
            user_stats[m["user_id"]]["llm_cnt"] += 1 if m["used_llm"] else 0

        # Get user phone_hashes for top 20 users
        sorted_users = sorted(
            user_stats.keys(), key=lambda uid: user_stats[uid]["cnt"], reverse=True
        )
        top_user_ids = sorted_users[:20]

        if top_user_ids:
            users_data = (
                supabase.table("users").select("id, phone_hash").in_("id", top_user_ids).execute()
            )

            # Create lookup dict
            user_hash_map = {u["id"]: u["phone_hash"] for u in users_data.data}

            # Format for template
            users = [
                {
                    "user_hash": user_hash_map.get(uid, "unknown"),
                    "cnt": user_stats[uid]["cnt"],
                    "llm_cnt": user_stats[uid]["llm_cnt"],
                }
                for uid in top_user_ids
            ]
        else:
            users = []

        user_rows = "".join(
            f"<tr><td><code>{escape(u['user_hash'])}</code></td>"
            f"<td>{u['cnt']}</td><td>{u['llm_cnt']}</td></tr>"
            for u in users
        )

        # Patterns - fetch all message texts and group client-side
        all_messages = supabase.table("metrics").select("message_text").execute()

        pattern_counter = defaultdict(int)
        for m in all_messages.data:
            pattern_counter[m["message_text"].lower()] += 1

        patterns = sorted(pattern_counter.items(), key=lambda x: x[1], reverse=True)[:15]
        pattern_rows = "".join(
            f"<tr><td>{escape(msg)}</td><td>{cnt}</td></tr>" for msg, cnt in patterns
        )

        # Time series - fetch metrics from last 7 days and group by day
        week_metrics = (
            supabase.table("metrics").select("timestamp").gte("timestamp", week_ago).execute()
        )

        day_counter = defaultdict(int)
        for m in week_metrics.data:
            day = m["timestamp"][:10]  # Extract YYYY-MM-DD
            day_counter[day] += 1

        days = sorted(day_counter.items())
        max_cnt = max((cnt for _, cnt in days), default=1)
        timeseries = "".join(
            f'<div class="bar-row">'
            f'<span class="bar-label">{day[5:]}</span>'
            f'<div class="bar" style="width:{cnt / max_cnt * 400}px"></div>'
            f'<span class="bar-value">{cnt}</span></div>'
            for day, cnt in days
        )
        if not timeseries:
            timeseries = "<p>No data yet.</p>"

    except Exception as e:
        print(f"Error fetching dashboard data: {e}", file=sys.stderr)
        return (
            f"<html><body><h1>Error loading dashboard</h1><p>{escape(str(e))}</p></body></html>",
            500,
        )

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
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)
