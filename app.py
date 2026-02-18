#!/usr/bin/env python3
"""Flask webhook for answering spending questions via WhatsApp (Twilio)."""

import hashlib
import os
import random
import re
import sys
import time
from calendar import monthrange
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from flask import Flask, request
from markupsafe import escape
from supabase import Client, create_client
from twilio.twiml.messaging_response import MessagingResponse

load_dotenv()

# Twilio credentials (needed to download media from WhatsApp)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# LLM Configuration - Use Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_MAIN = "llama-3.3-70b-versatile"  # For financial advice and complex questions
GROQ_MODEL_FAST = "llama-3.1-8b-instant"     # For intent classification (faster)

# Try to import and configure Groq
groq_client = None
try:
    from groq import Groq
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("[LLM CONFIG] Groq client configured successfully", file=sys.stderr)
    else:
        print("[LLM CONFIG] Groq API Key: NOT SET", file=sys.stderr)
except ImportError as e:
    print(f"[LLM CONFIG] Failed to import groq: {e}", file=sys.stderr)
    print("[LLM CONFIG] LLM features will be disabled", file=sys.stderr)
except Exception as e:
    print(f"[LLM CONFIG] Error configuring Groq: {e}", file=sys.stderr)
    groq_client = None

# Debug logging for LLM configuration
print(f"[LLM CONFIG] Groq API Key: {'SET' if GROQ_API_KEY else 'NOT SET'}", file=sys.stderr)
print(f"[LLM CONFIG] Main Model: {GROQ_MODEL_MAIN}", file=sys.stderr)
print(f"[LLM CONFIG] Fast Model: {GROQ_MODEL_FAST}", file=sys.stderr)
print(f"[LLM CONFIG] Client object: {'READY' if groq_client else 'NOT READY'}", file=sys.stderr)

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
        print(f"[STARTUP] Loading transactions for user_id: {user_id}...", file=sys.stderr)
        response = (
            supabase.table("transactions")
            .select("date, description, amount, balance, merchant")
            .eq("user_id", user_id)
            .order("date")
            .execute()
        )
        transaction_count = len(response.data) if response.data else 0
        print(f"[STARTUP] Loaded {transaction_count} transactions", file=sys.stderr)
        return response.data if response.data else []
    except Exception as e:
        print(f"[STARTUP ERROR] Failed to load transactions: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []


# For backward compatibility, load default user's transactions at startup
# TODO: Replace with actual user_id from migration script output
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "REPLACE_WITH_USER_ID_FROM_MIGRATION")
print(f"[STARTUP] DEFAULT_USER_ID: {DEFAULT_USER_ID}", file=sys.stderr)
TRANSACTIONS: list[dict] = _load_user_transactions(DEFAULT_USER_ID)

if not TRANSACTIONS:
    print("[STARTUP WARNING] No transactions loaded! LLM will have no spending data.", file=sys.stderr)

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


def ask_llm(
    question: str,
    max_words: int = 50,
    history: list[dict] | None = None,
) -> dict:
    """
    Send a question to Groq with conversation history and word limit.

    Args:
        question:  Current user message
        max_words: Maximum words in response
        history:   List of {role, content} dicts from chat_history (oldest first)

    Returns:
        dict with content, timing, tokens
    """
    start = time.monotonic()

    print(f"[LLM] question='{question[:50]}' max_words={max_words} history={len(history or [])} turns", file=sys.stderr)

    if not groq_client or not GROQ_API_KEY:
        print("[LLM ERROR] GROQ_API_KEY not set!", file=sys.stderr)
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
        # System message: financial context + word limit instruction
        system_msg = (
            f"{SPENDING_SUMMARY}\n\n"
            f"Keep responses under {max_words} words. "
            "This is WhatsApp — be concise. "
            "Use conversation history to resolve pronouns and follow-ups "
            "('it', 'that', 'which one', 'what about last month')."
        )

        messages = [{"role": "system", "content": system_msg}]

        # Inject conversation history so the LLM has context
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": question})

        # Call Groq API
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL_MAIN,
            messages=messages,
            max_tokens=max_words * 2,
        )

        elapsed = time.monotonic() - start

        content = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else len(content) // 4
        print(f"[LLM] Success! Response length: {len(content)} chars, tokens: {tokens}", file=sys.stderr)

        return {
            "content": content,
            "llm_time": round(elapsed, 3),
            "tokens": tokens,
            "error": False,
        }
    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"[LLM ERROR] Exception: {type(e).__name__}: {e}", file=sys.stderr)

        # Handle specific errors
        error_msg = str(e)
        if "quota" in error_msg.lower() or "rate" in error_msg.lower():
            user_msg = "Rate limit exceeded. Please try again in a moment."
        elif "invalid" in error_msg.lower() and "key" in error_msg.lower():
            user_msg = "API key is invalid. Please check your configuration."
        else:
            user_msg = "Sorry, I couldn't process that right now."

        return {
            "content": f"{user_msg} Try a keyword like *total*, *uber*, *march*, or type *help*.",
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
# Date filtering helpers
# ---------------------------------------------------------------------------

_DATE_TOKEN_RE = re.compile(
    r"\b(in|on|for|during|last|this|past|the|next|"
    r"january|jan|february|feb|march|mar|april|apr|june|jun|"
    r"july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec|"
    r"weeks?|months?|years?|quarter|yesterday|today|ago|before|"
    r"q[1-4]|first|second|third|fourth)\b"
    r"|\b20\d{2}\b"
    r"|\b\d+\s+days?\b"
    # Word numbers only when directly followed by a time unit ("two weeks", "three months")
    r"|\b(one|two|three|four|five|six|seven|eight|nine|ten)(?=\s+(?:week|month|day|year)s?\b)",
    re.IGNORECASE,
)
# "may" excluded — too ambiguous (modal verb)


def _filter_transactions(txns: list[dict], start: datetime, end: datetime) -> list[dict]:
    """Return transactions within [start, end] inclusive."""
    s = start.strftime("%Y-%m-%d")
    e = end.strftime("%Y-%m-%d")
    return [t for t in txns if s <= t["date"] <= e]


def _date_label(start: datetime, end: datetime) -> str:
    """Human-readable label for a date range, e.g. 'in March', 'last month'."""
    today = datetime.now().date()
    s, e = start.date(), end.date()

    if s == e:
        if s == today:
            return "today"
        if s == today - timedelta(days=1):
            return "yesterday"
        return f"on {s.strftime('%-d %b %Y')}"

    # Full calendar month
    if s.day == 1 and e.day == monthrange(e.year, e.month)[1] and s.month == e.month and s.year == e.year:
        return f"in {s.strftime('%B')}" if s.year == today.year else f"in {s.strftime('%B %Y')}"

    # Full calendar year
    if s == s.replace(month=1, day=1) and e == e.replace(month=12, day=31) and s.year == e.year:
        return f"in {s.year}"

    # Relative to today
    if e == today:
        delta = (e - s).days + 1
        if delta == 7:
            return "in the past week"
        if delta == 30:
            return "in the past 30 days"
        return f"in the past {delta} days"

    return f"from {s.strftime('%-d %b')} to {e.strftime('%-d %b %Y')}"


_FILLER_RE = re.compile(
    r"\b(how|much|what|did|i|me|my|tell|show|give|about|is|are|was|were|"
    r"do|does|have|had|can|get|see|total|all|any|of|at|a|an|the|spend|spending|and)\b",
    re.IGNORECASE,
)


def _strip_date_tokens(text: str) -> str:
    """Remove date-related words so 'uber in march' → 'uber'."""
    cleaned = _DATE_TOKEN_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _extract_merchant_query(core: str) -> str:
    """Strip filler question words from core to isolate merchant/category."""
    cleaned = _FILLER_RE.sub(" ", core)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_help() -> str:
    return random.choice([
        "Hi! How can I help?",
        "Hey! What would you like to know?",
        "Hi there! Ask me anything about your spending.",
        "Hello! What can I look up for you?",
    ])


def cmd_total_spending(txns: list[dict] | None = None, date_label: str = "all time") -> str:
    data = txns if txns is not None else TRANSACTIONS
    total = sum(t["amount"] for t in data if t["amount"] < 0)
    count = sum(1 for t in data if t["amount"] < 0)
    if count == 0:
        if data is not TRANSACTIONS and TRANSACTIONS:
            latest = TRANSACTIONS[-1]["date"]
            return f"No spending found {date_label}. My data ends {latest}."
        return f"No spending found {date_label}."
    return f"Total spending {date_label}: R{abs(total):,.2f}\n({count} transactions)"


def cmd_income() -> str:
    total = sum(t["amount"] for t in TRANSACTIONS if t["amount"] > 0)
    count = sum(1 for t in TRANSACTIONS if t["amount"] > 0)
    return f"Total income: R{total:,.2f}\n({count} transactions)"


def cmd_balance() -> str:
    last = TRANSACTIONS[-1]
    return f"Closing balance: R{last['balance']:,.2f}\n(as of {last['date']})"


def cmd_top_merchants(txns: list[dict] | None = None, date_label: str = "all time") -> str:
    data = txns if txns is not None else TRANSACTIONS
    spending: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for t in data:
        if t["amount"] < 0:
            merchant = extract_merchant(t["description"])
            spending[merchant] += abs(t["amount"])
            counts[merchant] += 1

    if not spending:
        return f"No spending found {date_label}."

    lines = [f"Top 10 merchants {date_label}:\n"]
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


def cmd_merchant_search(
    query: str,
    txns: list[dict] | None = None,
    date_label: str = "all time",
) -> str:
    data = txns if txns is not None else TRANSACTIONS
    matches = []
    for t in data:
        merchant = extract_merchant(t["description"])
        if query in merchant.lower() or query in t["description"].lower():
            matches.append((t, merchant))

    if not matches:
        if txns is not None:
            # Date filter was active — give targeted "no results" message
            suffix = ""
            if not txns and TRANSACTIONS:
                suffix = f" My data ends {TRANSACTIONS[-1]['date']}."
            return f'No {query} spending {date_label}.{suffix}'
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
        return f"Spending at {merchant_name} {date_label}: R{total:,.2f}\n({count} transactions)"

    lines = [f'Spending matching "{query}" {date_label}: R{total:,.2f} total\n']
    for merchant, amt in by_merchant.most_common(10):
        lines.append(f"  - {merchant}: R{amt:,.2f}")
    if len(by_merchant) > 10:
        lines.append(f"  ... and {len(by_merchant) - 10} more")
    return "\n".join(lines)


def _cmd_period_summary(txns: list[dict], date_label: str) -> str:
    """Full spending summary for a date period (used when query is just a date expression)."""
    if not txns:
        # Hint at the actual data coverage
        if TRANSACTIONS:
            latest = TRANSACTIONS[-1]["date"]
            earliest = TRANSACTIONS[0]["date"]
            return (
                f"No transactions found {date_label}.\n"
                f"My data covers {earliest} to {latest}."
            )
        return f"No transactions found {date_label}."

    debits = sum(t["amount"] for t in txns if t["amount"] < 0)
    credits = sum(t["amount"] for t in txns if t["amount"] > 0)
    debit_count = sum(1 for t in txns if t["amount"] < 0)

    spending: Counter[str] = Counter()
    for t in txns:
        if t["amount"] < 0:
            spending[extract_merchant(t["description"])] += abs(t["amount"])

    lines = [
        f"Summary {date_label}:",
        f"  Spent: R{abs(debits):,.2f} ({debit_count} transactions)",
        f"  Income: R{credits:,.2f}",
    ]
    if spending:
        lines.append("\nTop merchants:")
        for i, (merchant, amt) in enumerate(spending.most_common(5), 1):
            lines.append(f"  {i}. {merchant}: R{amt:,.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Intent Classification System
# ---------------------------------------------------------------------------

from core.date_parser import parse_date_range as _parse_date_range
from core.intent_classifier import Intent, IntentClassifier
from core.pdf_processor import process_pdf_upload

# Wire up Groq fast model for LLM-based intent classification fallback
IntentClassifier.setup_llm(groq_client, GROQ_MODEL_FAST)


# ---------------------------------------------------------------------------
# Message router
# ---------------------------------------------------------------------------


def _keyword_result(content: str) -> dict:
    return {"content": content, "used_llm": False, "intent": None, "confidence": 1.0}


# Intent-specific handler functions
def handle_greeting(message: str, confidence: float, history: list[dict] | None = None) -> dict:
    """Handle greeting intents."""
    return _keyword_result(cmd_help())


def handle_account_balance(message: str, confidence: float, history: list[dict] | None = None) -> dict:
    """Handle balance check requests."""
    text = message.strip().lower()
    if text in ("balance", "closing balance", "current balance"):
        return _keyword_result(cmd_balance())
    max_words = IntentClassifier.get_max_words(Intent.ACCOUNT_BALANCE)
    result = ask_llm(message, max_words=max_words, history=history)
    result["used_llm"] = True
    return result


def handle_spending_query(message: str, confidence: float, history: list[dict] | None = None) -> dict:
    """Handle spending-related queries, with optional date filtering."""
    text = message.strip().lower()
    # Strip leading "and"/"and then" — context continuation from prior message
    text = re.sub(r"^\s*and\s+(then\s+)?", "", text).strip()
    today = datetime.now()

    # --- 1. Parse date range from the message ---
    date_range = _parse_date_range(text)
    filtered_txns: list[dict] | None = None
    date_label = "all time"
    start = end = None

    if date_range:
        start, end = date_range
        if start.date() > today.date():
            return _keyword_result("That date is in the future! Try asking about past spending.")
        # Cap end at today
        if end.date() > today.date():
            end = today
        filtered_txns = _filter_transactions(TRANSACTIONS, start, end)
        date_label = _date_label(start, end)
        print(f"[SPENDING] Date range: {start.date()} → {end.date()} ({date_label})", file=sys.stderr)
    else:
        print(f"[SPENDING] No date range found, using all time. text='{text}'", file=sys.stderr)

    txns = filtered_txns if filtered_txns is not None else TRANSACTIONS

    # --- 2. Strip date tokens to isolate the core query ---
    core = _strip_date_tokens(text)
    # Also remove any leftover punctuation that isn't part of a merchant name
    core = re.sub(r"[^\w\s]", "", core).strip()
    print(f"[SPENDING] core='{core}'", file=sys.stderr)

    # --- 3. Exact keyword matches on core query ---
    if core in ("total spending", "total spend", "total debits", "total", "spending", "spend", ""):
        if not core or core in ("spending", "spend"):
            # Bare date expression (e.g. "march", "last month") → period summary
            return _keyword_result(_cmd_period_summary(txns, date_label))
        return _keyword_result(cmd_total_spending(txns, date_label))

    if core in ("top merchants", "top spend", "top", "biggest", "top 10"):
        return _keyword_result(cmd_top_merchants(txns, date_label))

    # --- 4. Merchant/category search ---
    # Strip filler question words ("how much uber" → "uber")
    merchant_query = _extract_merchant_query(core)
    # Drop any remaining punctuation-only content
    merchant_query = re.sub(r"[^\w\s]", "", merchant_query).strip()
    words = merchant_query.split()
    print(f"[SPENDING] merchant_query='{merchant_query}'", file=sys.stderr)

    # If no specific merchant/category remains after stripping, treat as period summary
    if not merchant_query:
        return _keyword_result(_cmd_period_summary(txns, date_label))

    if merchant_query and len(words) <= 3:
        merchant_result = cmd_merchant_search(merchant_query, txns, date_label)
        # Return if date was specified (even "no results") or a match was found
        if filtered_txns is not None or not merchant_result.startswith("No transactions found"):
            return {
                "content": merchant_result,
                "used_llm": False,
                "intent": "spending_query",
                "confidence": confidence,
            }

    # --- 5. LLM fallback for complex queries ---
    llm_message = message
    if date_range:
        llm_message += f"\n[Date filter: {date_label} ({start.date()} to {end.date()})]"
    max_words = IntentClassifier.get_max_words(Intent.SPENDING_QUERY)
    result = ask_llm(llm_message, max_words=max_words, history=history)
    result["used_llm"] = True
    return result


def handle_debt_advice(message: str, confidence: float, history: list[dict] | None = None) -> dict:
    """Handle debt and credit-related advice."""
    max_words = IntentClassifier.get_max_words(Intent.DEBT_ADVICE)
    result = ask_llm(message, max_words=max_words, history=history)
    result["used_llm"] = True
    return result


def handle_budget_check(message: str, confidence: float, history: list[dict] | None = None) -> dict:
    """Handle budget and affordability checks."""
    max_words = IntentClassifier.get_max_words(Intent.BUDGET_CHECK)
    result = ask_llm(message, max_words=max_words, history=history)
    result["used_llm"] = True
    return result


def handle_document_upload(message: str, confidence: float, history: list[dict] | None = None) -> dict:
    """Handle document upload requests."""
    return {
        "content": "📄 Document upload coming soon! Contact support for manual uploads.",
        "used_llm": False,
        "intent": "document_upload",
        "confidence": confidence,
    }


def handle_general_financial_advice(message: str, confidence: float, history: list[dict] | None = None) -> dict:
    """Handle general financial advice requests."""
    max_words = IntentClassifier.get_max_words(Intent.GENERAL_FINANCIAL_ADVICE)
    result = ask_llm(message, max_words=max_words, history=history)
    result["used_llm"] = True
    return result


def handle_unknown(message: str, confidence: float, history: list[dict] | None = None) -> dict:
    """Handle unknown intents - fallback to LLM or help."""
    if len(message.strip()) < 5:
        return _keyword_result(cmd_help())
    max_words = IntentClassifier.DEFAULT_MAX_WORDS
    result = ask_llm(message, max_words=max_words, history=history)
    result["used_llm"] = True
    return result


# Intent router mapping
INTENT_HANDLERS = {
    Intent.GREETING: handle_greeting,
    Intent.ACCOUNT_BALANCE: handle_account_balance,
    Intent.SPENDING_QUERY: handle_spending_query,
    Intent.DEBT_ADVICE: handle_debt_advice,
    Intent.BUDGET_CHECK: handle_budget_check,
    Intent.DOCUMENT_UPLOAD: handle_document_upload,
    Intent.GENERAL_FINANCIAL_ADVICE: handle_general_financial_advice,
    Intent.UNKNOWN: handle_unknown,
}


def handle_message(body: str, history: list[dict] | None = None) -> dict:
    """
    Route an incoming message using intent classification.

    Args:
        body:    Raw message text from the user
        history: Recent conversation turns from chat_history (oldest first)

    Returns a result dict with: content, used_llm, intent, confidence
    """
    text = body.strip()

    # Classify intent
    intent, confidence, reasoning = IntentClassifier.classify(text)

    print(
        f"[INTENT] Detected: {intent.value} | Confidence: {confidence:.2f} | Reason: {reasoning}",
        file=sys.stderr
    )

    # Handle low confidence - ask for clarification
    if confidence < IntentClassifier.CONFIDENCE_THRESHOLD_LOW:
        print(f"[INTENT] Low confidence ({confidence:.2f}), asking for clarification", file=sys.stderr)
        return {
            "content": "Not sure what you meant. Try: *total*, *balance*, *top*, *march*, *uber*, or *help*",
            "used_llm": False,
            "intent": intent.value,
            "confidence": confidence,
        }

    # If classified as greeting but there's prior history and the message isn't
    # an actual greeting word, treat it as a contextual follow-up instead so the
    # LLM can answer using conversation context (e.g. "really?", "wow", "seriously")
    REAL_GREETINGS = {"hi", "hello", "hey", "help", "helo", "howzit", "sup", "yo"}
    if (
        intent == Intent.GREETING
        and history
        and text.lower().strip("?!.") not in REAL_GREETINGS
    ):
        intent = Intent.GENERAL_FINANCIAL_ADVICE
        print("[INTENT] Rerouted greeting→general (follow-up detected)", file=sys.stderr)

    # Route to appropriate handler, passing history for LLM context
    handler = INTENT_HANDLERS.get(intent, handle_unknown)
    result = handler(text, confidence, history=history or [])

    # Add intent metadata to result
    result["intent"] = intent.value
    result["confidence"] = confidence

    # Enforce word limit on response
    max_words = IntentClassifier.get_max_words(intent)
    result["content"] = IntentClassifier.truncate_response(result["content"], max_words)

    return result


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Health Check Routes
# ---------------------------------------------------------------------------


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint for Railway."""
    return {
        "status": "ok",
        "service": "whatsapp-financial-advisor",
        "transactions_loaded": len(TRANSACTIONS),
        "llm_configured": groq_client is not None,
    }


@app.route("/health", methods=["GET"])
def health():
    """Alternative health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


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
    intent: str | None = None,
    confidence: float | None = None,
) -> None:
    """Log interaction metrics to Supabase with intent classification data."""
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
        metric_data = {
            "user_id": user_id,
            "message_text": message_text,
            "used_llm": used_llm,
            "llm_response_time": llm_time,
            "total_response_time": round(total_time, 3),
            "success": success,
            "tokens_used": tokens,
        }

        # Add intent classification data if available
        if intent:
            metric_data["intent"] = intent
        if confidence is not None:
            metric_data["confidence"] = confidence

        supabase.table("metrics").insert(metric_data).execute()

    except Exception as e:
        # Log error but don't fail the webhook response
        print(f"Error logging metric: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Chat history (conversational memory)
# ---------------------------------------------------------------------------

def _get_user_id(user_hash: str) -> str | None:
    """Return Supabase user_id for a phone hash, or None if not found."""
    try:
        r = supabase.table("users").select("id").eq("phone_hash", user_hash).execute()
        return r.data[0]["id"] if r.data else None
    except Exception:
        return None


def _load_chat_history(user_hash: str, limit: int = 5) -> list[dict]:
    """
    Load the last `limit` exchanges for a user as a list of
    {role: 'user'|'assistant', content: str} dicts, oldest first.
    Returns [] on any error so the bot degrades gracefully.
    """
    try:
        user_id = _get_user_id(user_hash)
        if not user_id:
            return []
        rows = (
            supabase.table("chat_history")
            .select("user_message, bot_response")
            .eq("user_id", user_id)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        ).data
        messages: list[dict] = []
        for row in reversed(rows):          # oldest first
            messages.append({"role": "user",      "content": row["user_message"]})
            messages.append({"role": "assistant",  "content": row["bot_response"]})
        return messages
    except Exception as e:
        print(f"[HISTORY] Load failed (non-fatal): {e}", file=sys.stderr)
        return []


def _save_chat_message(
    user_hash: str,
    user_message: str,
    bot_response: str,
    intent: str | None = None,
) -> None:
    """Persist a message exchange to chat_history. Silently skips on error."""
    try:
        user_id = _get_user_id(user_hash)
        if not user_id:
            return
        supabase.table("chat_history").insert({
            "user_id": user_id,
            "user_message": user_message,
            "bot_response": bot_response,
            "intent_detected": intent,
        }).execute()
    except Exception as e:
        print(f"[HISTORY] Save failed (non-fatal): {e}", file=sys.stderr)


def _clear_chat_history(user_hash: str) -> None:
    """Delete all chat history for a user."""
    try:
        user_id = _get_user_id(user_hash)
        if user_id:
            supabase.table("chat_history").delete().eq("user_id", user_id).execute()
            print(f"[HISTORY] Cleared for {user_hash[:8]}", file=sys.stderr)
    except Exception as e:
        print(f"[HISTORY] Clear failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


@app.route("/webhook", methods=["POST"])
def webhook():
    """Receive incoming WhatsApp messages from Twilio and respond."""
    start = time.monotonic()

    try:
        user_phone = request.form.get("From", "anonymous")
        user_hash = _hash_user(user_phone)
        num_media = int(request.form.get("NumMedia", 0))

        # --- PDF upload path ---
        if num_media > 0:
            media_url = request.form.get("MediaUrl0", "")
            media_type = request.form.get("MediaContentType0", "")
            print(f"[WEBHOOK] Media received from {user_hash[:8]}: {media_type}", file=sys.stderr)

            # Send immediate acknowledgement
            ack = MessagingResponse()
            ack.message("⏳ Processing your statement, please wait...")
            # Note: we still process synchronously; ack is sent as the response
            # For true async we'd need a background worker - this is good enough for now

            pdf_result = process_pdf_upload(
                supabase=supabase,
                media_url=media_url,
                media_content_type=media_type,
                phone_hash=user_hash,
                account_sid=TWILIO_ACCOUNT_SID,
                auth_token=TWILIO_AUTH_TOKEN,
            )

            total_time = time.monotonic() - start
            _log_metric(
                user_hash=user_hash,
                message_text=f"[PDF UPLOAD] {media_type}",
                used_llm=False,
                llm_time=None,
                total_time=total_time,
                success=pdf_result["success"],
                tokens=0,
                intent="document_upload",
                confidence=1.0,
            )

            resp = MessagingResponse()
            resp.message(pdf_result["message"])
            print(f"[WEBHOOK] PDF result: {pdf_result['message'][:80]}", file=sys.stderr)
            return str(resp), 200, {"Content-Type": "application/xml"}

        # --- Text message path ---
        incoming_msg = request.form.get("Body", "").strip()
        print(f"[WEBHOOK] Received message from {user_hash[:8]}: '{incoming_msg[:50]}'", file=sys.stderr)

        # Handle "clear history" before anything else
        if incoming_msg.lower() in ("clear history", "start fresh", "forget everything"):
            _clear_chat_history(user_hash)
            resp = MessagingResponse()
            resp.message("Got it — I've cleared our conversation history. Fresh start!")
            return str(resp), 200, {"Content-Type": "application/xml"}

        # Load conversation history for LLM context
        history = _load_chat_history(user_hash)

        result = handle_message(incoming_msg, history=history)
        total_time = time.monotonic() - start

        # Save this exchange to chat history (best-effort)
        _save_chat_message(
            user_hash,
            incoming_msg,
            result["content"],
            result.get("intent"),
        )

        _log_metric(
            user_hash=user_hash,
            message_text=incoming_msg,
            used_llm=result.get("used_llm", False),
            llm_time=result.get("llm_time"),
            total_time=total_time,
            success=not result.get("error", False),
            tokens=result.get("tokens", 0),
            intent=result.get("intent"),
            confidence=result.get("confidence"),
        )

        resp = MessagingResponse()
        resp.message(result["content"])
        print(f"[WEBHOOK] Sending response: '{result['content'][:50]}'", file=sys.stderr)
        return str(resp), 200, {"Content-Type": "application/xml"}

    except Exception as e:
        print(f"[WEBHOOK ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

        resp = MessagingResponse()
        resp.message("Sorry, an error occurred. Please try again or type *help*.")
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


# ---------------------------------------------------------------------------
# Startup Logging
# ---------------------------------------------------------------------------

# Log all registered routes for debugging
print("\n" + "=" * 60, file=sys.stderr)
print("[STARTUP] Flask app initialized successfully!", file=sys.stderr)
print("[STARTUP] Registered routes:", file=sys.stderr)
for rule in app.url_map.iter_rules():
    methods = ",".join(sorted(rule.methods - {"HEAD", "OPTIONS"}))
    print(f"  {rule.rule:30s} {methods}", file=sys.stderr)
print("=" * 60 + "\n", file=sys.stderr)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)
