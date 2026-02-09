#!/usr/bin/env python3
"""Flask webhook for answering spending questions via WhatsApp (Twilio)."""

import json
import re
from collections import Counter
from pathlib import Path

import requests as http_requests
from flask import Flask, request
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


def ask_ollama(question: str) -> str:
    """Send a question to llama3.2 via Ollama with the spending summary as context."""
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
        return resp.json()["message"]["content"]
    except Exception:
        return (
            "Sorry, I couldn't process that right now. "
            "Try a keyword like *total*, *uber*, *march*, or type *help*."
        )


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


def handle_message(body: str) -> str:
    """Route an incoming message to the appropriate handler."""
    text = body.strip().lower()

    if text in ("help", "commands", "menu", "hi", "hello", "?"):
        return cmd_help()

    if text in ("total spending", "total spend", "total debits", "total"):
        return cmd_total_spending()

    if text in ("income", "credits", "total income", "salary", "total credits"):
        return cmd_income()

    if text in ("balance", "closing balance", "current balance"):
        return cmd_balance()

    if text in ("top merchants", "top spend", "top", "biggest", "top 10"):
        return cmd_top_merchants()

    for keyword, month_num in MONTH_KEYWORDS.items():
        if keyword in text:
            return cmd_month_spending(month_num)

    # Try merchant keyword search first
    merchant_result = cmd_merchant_search(text)
    if not merchant_result.startswith("No transactions found"):
        return merchant_result

    # Fallback: ask the LLM
    return ask_ollama(body.strip())


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/webhook", methods=["POST"])
def webhook():
    """Receive incoming WhatsApp messages from Twilio and respond."""
    incoming_msg = request.form.get("Body", "").strip()
    reply_text = handle_message(incoming_msg)

    resp = MessagingResponse()
    resp.message(reply_text)
    return str(resp), 200, {"Content-Type": "application/xml"}


if __name__ == "__main__":
    app.run(debug=True, port=5001)
