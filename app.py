#!/usr/bin/env python3
"""Flask webhook for answering spending questions via WhatsApp (Twilio)."""

import hashlib
import json
import os
import random
import re
import sys
import time
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
GROQ_MODEL_MAIN = "llama-3.3-70b-versatile"

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

print(f"[LLM CONFIG] Groq API Key: {'SET' if GROQ_API_KEY else 'NOT SET'}", file=sys.stderr)
print(f"[LLM CONFIG] Main Model: {GROQ_MODEL_MAIN}", file=sys.stderr)
print(f"[LLM CONFIG] Client object: {'READY' if groq_client else 'NOT READY'}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Supabase configuration
# ---------------------------------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

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


DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "REPLACE_WITH_USER_ID_FROM_MIGRATION")
print(f"[STARTUP] DEFAULT_USER_ID: {DEFAULT_USER_ID}", file=sys.stderr)
TRANSACTIONS: list[dict] = _load_user_transactions(DEFAULT_USER_ID)

if not TRANSACTIONS:
    print("[STARTUP WARNING] No transactions loaded! LLM will have no spending data.", file=sys.stderr)

# ---------------------------------------------------------------------------
# Merchant extraction
# ---------------------------------------------------------------------------


def _load_description_prefixes() -> dict[str, list[str]]:
    """
    Load transaction description prefixes from Supabase, keyed by bank.
    Sorted longest-first so more specific prefixes match before shorter ones.
    Falls back to an empty dict on error (extract_merchant still works, just
    won't strip prefixes).
    """
    try:
        response = (
            supabase.table("transaction_prefixes")
            .select("bank, prefix")
            .order("prefix", desc=False)
            .execute()
        )
        rows = response.data or []
        prefixes: dict[str, list[str]] = {}
        for row in rows:
            bank = row["bank"]
            prefixes.setdefault(bank, []).append(row["prefix"])
        # Sort each bank's list longest-first for correct prefix matching
        for bank in prefixes:
            prefixes[bank].sort(key=len, reverse=True)
        total = sum(len(v) for v in prefixes.values())
        print(f"[STARTUP] Loaded {total} transaction prefixes for {list(prefixes.keys())}", file=sys.stderr)
        return prefixes
    except Exception as e:
        print(f"[STARTUP WARNING] Could not load transaction prefixes: {e}", file=sys.stderr)
        return {}


DESCRIPTION_PREFIXES: dict[str, list[str]] = _load_description_prefixes()

CARD_RE = re.compile(r"\d{6}\*\d{4}")
AMOUNT_PREFIX_RE = re.compile(r"^[\d,]+\.\d{2}\s+")
TRAILING_DATE_RE = re.compile(r"\s+\d{2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$")
TRAILING_NUMBERS_RE = re.compile(r"\s+\d{8,}$")


def extract_merchant(description: str, bank: str = "FNB") -> str:
    """Extract a human-readable merchant/payee name from a transaction description."""
    if not description.strip():
        return "Bank Fees"

    text = description
    matched_prefix = None
    for prefix in DESCRIPTION_PREFIXES.get(bank, []):
        if text.startswith(prefix):
            matched_prefix = prefix
            text = text[len(prefix):].strip()
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


# ---------------------------------------------------------------------------
# Spending summary (injected into LLM system prompt)
# ---------------------------------------------------------------------------


def build_spending_summary() -> str:
    """Pre-compute a text summary of all transactions for the LLM system prompt."""
    if not TRANSACTIONS:
        return (
            "You are a helpful financial assistant for a South African.\n"
            "No transaction data is currently loaded. Please try again later or contact support."
        )

    total_debits = sum(t["amount"] for t in TRANSACTIONS if t["amount"] < 0)
    total_credits = sum(t["amount"] for t in TRANSACTIONS if t["amount"] > 0)
    debit_count = sum(1 for t in TRANSACTIONS if t["amount"] < 0)
    credit_count = sum(1 for t in TRANSACTIONS if t["amount"] > 0)
    closing_balance = TRANSACTIONS[-1]["balance"]
    first_date = TRANSACTIONS[0]["date"]
    last_date = TRANSACTIONS[-1]["date"]

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

    monthly: dict[str, dict] = {}
    for t in TRANSACTIONS:
        month_key = t["date"][:7]
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
        f"You are a helpful financial assistant.\n"
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


SPENDING_SUMMARY = build_spending_summary()

# ---------------------------------------------------------------------------
# Simple command handlers (no LLM needed)
# ---------------------------------------------------------------------------


def cmd_help() -> str:
    return random.choice([
        "Hi! How can I help?",
        "Hey! What would you like to know?",
        "Hi there! Ask me anything about your spending.",
        "Hello! What can I look up for you?",
    ])


def cmd_balance() -> str:
    last = TRANSACTIONS[-1]
    return f"Closing balance: R{last['balance']:,.2f}\n(as of {last['date']})"


# ---------------------------------------------------------------------------
# Tool functions (called by the LLM via tool calling)
# ---------------------------------------------------------------------------


def _filter_by_dates(
    txns: list[dict],
    date_from: str | None,
    date_to: str | None,
) -> list[dict]:
    result = txns
    if date_from:
        result = [t for t in result if t["date"] >= date_from]
    if date_to:
        result = [t for t in result if t["date"] <= date_to]
    return result


def _data_coverage_hint() -> str:
    if not TRANSACTIONS:
        return ""
    return f" Data covers {TRANSACTIONS[0]['date']} to {TRANSACTIONS[-1]['date']}."


def _tool_get_balance() -> str:
    if not TRANSACTIONS:
        return "No transaction data available."
    last = TRANSACTIONS[-1]
    return f"Closing balance: R{last['balance']:,.2f} (as of {last['date']})"


def _tool_get_total_spending(
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    txns = _filter_by_dates(TRANSACTIONS, date_from, date_to)
    debits = [t for t in txns if t["amount"] < 0]
    if not debits:
        return f"No spending found in that date range.{_data_coverage_hint()}"
    total = sum(t["amount"] for t in debits)
    return f"Total spending: R{abs(total):,.2f} ({len(debits)} transactions)"


def _tool_get_merchant_spending(
    merchant: str,
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    txns = _filter_by_dates(TRANSACTIONS, date_from, date_to)
    query = merchant.lower()
    matches = [
        (t, extract_merchant(t["description"]))
        for t in txns
        if query in extract_merchant(t["description"]).lower()
        or query in t["description"].lower()
    ]
    spending_matches = [(t, m) for t, m in matches if t["amount"] < 0]
    if not spending_matches:
        return f"No spending found for '{merchant}'.{_data_coverage_hint()}"

    total = sum(abs(t["amount"]) for t, _ in spending_matches)
    count = len(spending_matches)

    by_merchant: Counter[str] = Counter()
    for t, m in spending_matches:
        by_merchant[m] += abs(t["amount"])

    if len(by_merchant) == 1:
        name = list(by_merchant.keys())[0]
        return f"Spending at {name}: R{total:,.2f} ({count} transactions)"

    lines = [f"Spending matching '{merchant}': R{total:,.2f} total ({count} transactions)"]
    for name, amt in by_merchant.most_common(10):
        lines.append(f"  - {name}: R{amt:,.2f}")
    return "\n".join(lines)


def _tool_get_top_merchants(
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 10,
) -> str:
    txns = _filter_by_dates(TRANSACTIONS, date_from, date_to)
    spending: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for t in txns:
        if t["amount"] < 0:
            m = extract_merchant(t["description"])
            spending[m] += abs(t["amount"])
            counts[m] += 1

    if not spending:
        return f"No spending found in that date range.{_data_coverage_hint()}"

    lines = [f"Top {limit} merchants:"]
    for i, (m, total) in enumerate(spending.most_common(limit), 1):
        lines.append(f"{i}. {m}: R{total:,.2f} ({counts[m]}x)")
    return "\n".join(lines)


def _tool_get_period_summary(
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    txns = _filter_by_dates(TRANSACTIONS, date_from, date_to)
    if not txns:
        return f"No transactions found in that date range.{_data_coverage_hint()}"

    debits = sum(t["amount"] for t in txns if t["amount"] < 0)
    credits = sum(t["amount"] for t in txns if t["amount"] > 0)
    debit_count = sum(1 for t in txns if t["amount"] < 0)

    spending: Counter[str] = Counter()
    for t in txns:
        if t["amount"] < 0:
            spending[extract_merchant(t["description"])] += abs(t["amount"])

    period = f"{date_from or 'all'} to {date_to or 'all'}"
    lines = [
        f"Period summary ({period}):",
        f"  Spent: R{abs(debits):,.2f} ({debit_count} transactions)",
        f"  Income: R{credits:,.2f}",
    ]
    if spending:
        lines.append("Top merchants:")
        for i, (m, amt) in enumerate(spending.most_common(5), 1):
            lines.append(f"  {i}. {m}: R{amt:,.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Budget helpers
# ---------------------------------------------------------------------------


def _current_period_dates(period: str) -> tuple[str, str]:
    """Return (date_from, date_to) for the current budget period."""
    today = datetime.now().date()
    if period == "weekly":
        date_from = (today - timedelta(days=today.weekday())).isoformat()
    else:  # monthly
        date_from = today.replace(day=1).isoformat()
    return date_from, today.isoformat()


def _get_category_spending(category: str, date_from: str, date_to: str) -> float:
    """Sum debits matching a category keyword in a date range."""
    txns = _filter_by_dates(TRANSACTIONS, date_from, date_to)
    q = category.lower()
    return sum(
        abs(t["amount"])
        for t in txns
        if t["amount"] < 0
        and (q in extract_merchant(t["description"]).lower() or q in t["description"].lower())
    )


def _budget_status_line(cat: str, spent: float, limit: float, period: str) -> str:
    pct = (spent / limit * 100) if limit > 0 else 0
    remaining = limit - spent
    period_label = "month" if period == "monthly" else "week"
    if remaining < 0:
        status = f"OVER by R{abs(remaining):,.2f}"
    else:
        status = f"R{remaining:,.2f} left"
    return f"{cat.title()} ({period_label}): R{spent:,.2f} / R{limit:,.2f} ({pct:.0f}%) — {status}"


# ---------------------------------------------------------------------------
# Budget tool functions
# ---------------------------------------------------------------------------


def _tool_set_budget(user_id: str, category: str, amount: float, period: str = "monthly") -> str:
    category = category.lower().strip()
    if period not in ("monthly", "weekly"):
        period = "monthly"
    try:
        supabase.table("user_budgets").upsert(
            {
                "user_id": user_id,
                "category": category,
                "amount": amount,
                "period": period,
                "active": True,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="user_id,category,period",
        ).execute()
        date_from, date_to = _current_period_dates(period)
        spent = _get_category_spending(category, date_from, date_to)
        period_label = "month" if period == "monthly" else "week"
        pct = (spent / amount * 100) if amount > 0 else 0
        return (
            f"{category.title()} budget set: R{amount:,.2f}/{period_label}. "
            f"Currently spent: R{spent:,.2f} ({pct:.0f}%). "
            f"R{max(amount - spent, 0):,.2f} remaining."
        )
    except Exception as e:
        return f"Failed to set budget: {e}"


def _tool_get_budget_status(user_id: str, category: str | None = None) -> str:
    try:
        query = (
            supabase.table("user_budgets")
            .select("category, amount, period")
            .eq("user_id", user_id)
            .eq("active", True)
        )
        if category:
            query = query.eq("category", category.lower().strip())
        budgets = query.execute().data or []

        if not budgets:
            msg = f"No budget set for '{category}'." if category else "No budgets set yet."
            return f"{msg} Use set_budget to create one."

        lines = []
        for b in budgets:
            date_from, date_to = _current_period_dates(b["period"])
            spent = _get_category_spending(b["category"], date_from, date_to)
            lines.append(_budget_status_line(b["category"], spent, b["amount"], b["period"]))
        return "\n".join(lines)
    except Exception as e:
        return f"Failed to get budget status: {e}"


def _tool_list_budgets(user_id: str) -> str:
    try:
        budgets = (
            supabase.table("user_budgets")
            .select("category, amount, period")
            .eq("user_id", user_id)
            .eq("active", True)
            .order("category")
            .execute()
        ).data or []

        if not budgets:
            return "No active budgets. Use set_budget to create one."

        lines = ["Active budgets:"]
        for b in budgets:
            period_label = "month" if b["period"] == "monthly" else "week"
            lines.append(f"  - {b['category'].title()}: R{b['amount']:,.2f}/{period_label}")
        return "\n".join(lines)
    except Exception as e:
        return f"Failed to list budgets: {e}"


def _tool_delete_budget(user_id: str, category: str, period: str = "monthly") -> str:
    category = category.lower().strip()
    try:
        result = (
            supabase.table("user_budgets")
            .update({"active": False, "updated_at": datetime.now(timezone.utc).isoformat()})
            .eq("user_id", user_id)
            .eq("category", category)
            .eq("period", period)
            .execute()
        )
        if result.data:
            return f"{category.title()} {period} budget deleted."
        return f"No active {period} budget found for '{category}'."
    except Exception as e:
        return f"Failed to delete budget: {e}"


# ---------------------------------------------------------------------------
# Tool registry (Groq / OpenAI function calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Return the current closing account balance.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_total_spending",
            "description": (
                "Return total spending (debits) and transaction count for a date range. "
                "Pass date_from and date_to as YYYY-MM-DD strings. Omit both for all-time total."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from": {
                        "type": "string",
                        "description": "Start date YYYY-MM-DD (inclusive). Omit for no lower bound.",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date YYYY-MM-DD (inclusive). Omit for no upper bound.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_merchant_spending",
            "description": (
                "Return total spending at a specific merchant or category "
                "(e.g. 'uber', 'woolworths', 'fuel', 'checkers'). "
                "Searches both merchant names and raw transaction descriptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant": {
                        "type": "string",
                        "description": "Merchant name or keyword to search for (case-insensitive).",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start date YYYY-MM-DD (inclusive). Omit for no lower bound.",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date YYYY-MM-DD (inclusive). Omit for no upper bound.",
                    },
                },
                "required": ["merchant"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_merchants",
            "description": "Return the top 10 merchants ranked by total spend for a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from": {
                        "type": "string",
                        "description": "Start date YYYY-MM-DD (inclusive). Omit for no lower bound.",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date YYYY-MM-DD (inclusive). Omit for no upper bound.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_period_summary",
            "description": (
                "Return a full spending summary for a date range: total debits, "
                "total credits, and top 5 merchants. Use when the user asks about "
                "a general time period without specifying a merchant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_from": {
                        "type": "string",
                        "description": "Start date YYYY-MM-DD (inclusive). Omit for no lower bound.",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date YYYY-MM-DD (inclusive). Omit for no upper bound.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_budget",
            "description": (
                "Create or update a spending budget for a category. "
                "ONLY call this when the user has explicitly stated BOTH a category AND a specific rand amount. "
                "If either is missing from the user's message, ask for it — do NOT guess or use defaults. "
                "The category matches merchant names (e.g. 'uber', 'woolworths', 'fuel') "
                "or broad labels ('food', 'transport', 'entertainment'). "
                "Example trigger: 'set food budget to R2000' or 'budget R500 for uber'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category name explicitly stated by the user (e.g. 'food', 'uber', 'transport'). Must come from user input, not assumed.",
                    },
                    "amount": {
                        "description": "Rand amount explicitly stated by the user (e.g. 2000 for 'R2000'). Must come from user input, not assumed.",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["monthly", "weekly"],
                        "description": "Budget period stated by the user. Default: monthly.",
                    },
                },
                "required": ["category", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_budget_status",
            "description": (
                "Check spending vs budget limit for the current period. "
                "Pass a category to check one budget, or omit to check all budgets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category to check. Omit to check all active budgets.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_budgets",
            "description": "List all active budgets with their limits and periods.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_budget",
            "description": "Remove (deactivate) a budget for a category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category name to delete.",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["monthly", "weekly"],
                        "description": "Period of the budget to delete. Default: monthly.",
                    },
                },
                "required": ["category"],
            },
        },
    },
]


def _parse_failed_generation(exc) -> tuple[str, dict] | None:
    """
    Groq sometimes produces correct args but malformed wrapper syntax:
        <function=set_budget{"category": "food", "amount": 1500}</function>

    Access e.body['error']['failed_generation'] directly (structured API
    response) rather than parsing the stringified exception with regex.
    """
    try:
        fg: str = exc.body["error"]["failed_generation"]
    except (AttributeError, KeyError, TypeError):
        return None
    try:
        # fg format: <function=TOOLNAME{JSON}</function>
        tool_name = fg.split("=", 1)[1].split(">")[0].split("{")[0]
        args = json.loads(fg[fg.index("{") : fg.rindex("}") + 1])
        return tool_name, args
    except (ValueError, json.JSONDecodeError):
        return None


def _execute_tool(name: str, args: dict, user_id: str | None = None) -> str:
    """Dispatch a tool call by name and return the string result."""
    if name == "get_balance":
        return _tool_get_balance()
    elif name == "get_total_spending":
        return _tool_get_total_spending(
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
        )
    elif name == "get_merchant_spending":
        return _tool_get_merchant_spending(
            merchant=args["merchant"],
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
        )
    elif name == "get_top_merchants":
        return _tool_get_top_merchants(
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
            limit=int(args.get("limit", 10)),
        )
    elif name == "get_period_summary":
        return _tool_get_period_summary(
            date_from=args.get("date_from"),
            date_to=args.get("date_to"),
        )
    elif name == "set_budget":
        if not user_id:
            return "Cannot set budget: user not identified."
        return _tool_set_budget(
            user_id=user_id,
            category=args["category"],
            amount=float(args["amount"]),
            period=args.get("period", "monthly"),
        )
    elif name == "get_budget_status":
        if not user_id:
            return "Cannot get budget status: user not identified."
        return _tool_get_budget_status(
            user_id=user_id,
            category=args.get("category"),
        )
    elif name == "list_budgets":
        if not user_id:
            return "Cannot list budgets: user not identified."
        return _tool_list_budgets(user_id=user_id)
    elif name == "delete_budget":
        if not user_id:
            return "Cannot delete budget: user not identified."
        return _tool_delete_budget(
            user_id=user_id,
            category=args["category"],
            period=args.get("period", "monthly"),
        )
    else:
        return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# LLM with tool calling
# ---------------------------------------------------------------------------


def ask_llm_with_tools(
    message: str,
    history: list[dict] | None = None,
    user_id: str | None = None,
) -> dict:
    """
    Send a message to Groq with tool calling enabled.

    The LLM chooses which tool(s) to call. Results are fed back for a
    natural-language response. Falls back to direct answer when no tools needed.

    Returns:
        dict with content, llm_time, tokens, used_llm, error
    """
    start = time.monotonic()

    if not groq_client or not GROQ_API_KEY:
        elapsed = time.monotonic() - start
        return {
            "content": "LLM is not configured. Type *balance* or *help*.",
            "llm_time": round(elapsed, 3),
            "tokens": 0,
            "used_llm": True,
            "error": True,
        }

    today = datetime.now().strftime("%Y-%m-%d")
    data_start = TRANSACTIONS[0]["date"] if TRANSACTIONS else "N/A"
    data_end = TRANSACTIONS[-1]["date"] if TRANSACTIONS else "N/A"

    system_msg = (
        f"TODAY'S DATE: {today}. ALWAYS use this as the reference for ALL relative date "
        f"calculations ('last week', 'past 4 weeks', 'this month', etc.). "
        f"The statement period end date ({data_end}) is NOT today — do not use it as today.\n\n"
        f"{SPENDING_SUMMARY}\n\n"
        f"Transaction data available: {data_start} to {data_end}.\n\n"
        "TOOL CALLING RULES:\n"
        "- Only call a tool when the user has EXPLICITLY provided all required parameters.\n"
        "- NEVER guess, invent, or assume parameter values.\n"
        "- For set_budget: ONLY call if the user stated BOTH a category AND a specific rand "
        "amount in the CURRENT conversation turn. Do NOT reuse amounts from previous budget "
        "discussions about different categories — each budget needs its own explicit amount.\n"
        "- If the user provides only a category (e.g. 'work transport'), ask: "
        "'What monthly amount for [category]?' before calling set_budget.\n"
        "- For help/guidance requests ('can you help me', 'how do I', 'what should I'), "
        "respond conversationally — do NOT call any tool.\n"
        "- If a budget was just confirmed and the user adds a time qualifier "
        "('for March', 'starting next month', 'for this month'), do NOT call set_budget again. "
        "Explain conversationally: monthly budgets automatically apply to each calendar month.\n"
        "Good: 'Set food budget R2000' → call set_budget(category='food', amount=2000)\n"
        "Good: 'work transport' (after being asked category) + 'R800' (after being asked amount) → call set_budget\n"
        "Bad:  'work transport' alone → DO NOT call set_budget, ask for the amount first.\n"
        "Bad:  'for month of march' after budget confirmed → DO NOT call set_budget again.\n\n"
        "When the user mentions a time period, resolve it to exact YYYY-MM-DD dates relative "
        "to TODAY before calling tools. "
        "Keep responses concise — this is WhatsApp, under 80 words. "
        "Use conversation history to resolve pronouns and follow-ups. "
        "When a tool returns data, report ONLY what the tool returned — never compare or "
        "reconcile tool results against conversation history. Tool output is always authoritative."
    )

    messages: list[dict] = [{"role": "system", "content": system_msg}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})

    print(
        f"[LLM] message='{message[:60]}' history={len(history or [])} turns",
        file=sys.stderr,
    )

    try:
        # First call — LLM decides whether and which tools to use
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL_MAIN,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=500,
        )

        assistant_msg = response.choices[0].message
        tokens = response.usage.total_tokens if response.usage else 0

        # No tool calls — return direct response (conversational / general advice)
        if not assistant_msg.tool_calls:
            elapsed = time.monotonic() - start
            content = assistant_msg.content.strip() if assistant_msg.content else ""
            print(f"[LLM] Direct response ({len(content)} chars)", file=sys.stderr)
            return {
                "content": content,
                "llm_time": round(elapsed, 3),
                "tokens": tokens,
                "used_llm": True,
                "error": False,
            }

        # Execute each tool call
        messages.append(assistant_msg)

        for tool_call in assistant_msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            print(f"[TOOL] {tool_name}({tool_args})", file=sys.stderr)

            tool_result = _execute_tool(tool_name, tool_args, user_id=user_id)
            print(f"[TOOL] Result: {tool_result[:120]}", file=sys.stderr)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

        # Second call — format tool results as a natural language response
        final_response = groq_client.chat.completions.create(
            model=GROQ_MODEL_MAIN,
            messages=messages,
            max_tokens=300,
        )

        elapsed = time.monotonic() - start
        content = final_response.choices[0].message.content.strip()
        tokens += final_response.usage.total_tokens if final_response.usage else 0

        print(f"[LLM] Tool-call response ({len(content)} chars, {tokens} tokens)", file=sys.stderr)

        return {
            "content": content,
            "llm_time": round(elapsed, 3),
            "tokens": tokens,
            "used_llm": True,
            "error": False,
        }

    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"[LLM ERROR] {type(e).__name__}: {e}", file=sys.stderr)

        error_msg = str(e)
        if "quota" in error_msg.lower() or "rate" in error_msg.lower():
            user_msg = "Rate limit exceeded. Please try again in a moment."
        elif "invalid" in error_msg.lower() and "key" in error_msg.lower():
            user_msg = "API key is invalid. Please check your configuration."
        elif getattr(e, "body", {}).get("error", {}).get("code") == "tool_use_failed":
            # Groq sometimes generates correct args but malformed wrapper syntax.
            # Try to rescue the call by parsing e.body['error']['failed_generation'].
            parsed = _parse_failed_generation(e)
            if parsed:
                tool_name, tool_args = parsed
                print(f"[TOOL] Recovering malformed call: {tool_name}({tool_args})", file=sys.stderr)
                try:
                    tool_result = _execute_tool(tool_name, tool_args, user_id=user_id)
                    elapsed = time.monotonic() - start
                    return {
                        "content": tool_result,
                        "llm_time": round(elapsed, 3),
                        "tokens": 0,
                        "used_llm": True,
                        "error": False,
                    }
                except Exception as exec_err:
                    print(f"[TOOL] Recovery execution failed: {exec_err}", file=sys.stderr)
            # Parsing failed or execution failed — give contextual guidance.
            failed_gen = e.body.get("error", {}).get("failed_generation", "")
            if "set_budget" in failed_gen:
                user_msg = (
                    "To set a budget I need a category and amount. "
                    "Try: 'Set food budget to R2000' or 'R500 weekly uber budget'."
                )
            elif "delete_budget" in failed_gen:
                user_msg = "Which budget should I delete? E.g. 'Delete food budget'."
            elif "get_budget" in failed_gen or "list_budget" in failed_gen:
                user_msg = "I couldn't retrieve your budgets right now. Try 'list budgets'."
            else:
                user_msg = "I need more details. Could you be more specific?"
        else:
            user_msg = "Sorry, I couldn't process that right now."

        return {
            "content": f"{user_msg} Type *help* for commands.",
            "llm_time": round(elapsed, 3),
            "tokens": 0,
            "used_llm": True,
            "error": True,
        }


# ---------------------------------------------------------------------------
# Message router
# ---------------------------------------------------------------------------

GREETING_WORDS = {"hi", "hello", "hey", "help", "helo", "howzit", "sup", "yo", "hiya"}
BALANCE_WORDS = {"balance", "closing balance", "current balance", "my balance"}


def handle_message(
    body: str,
    history: list[dict] | None = None,
    user_id: str | None = None,
) -> dict:
    """
    Route an incoming message.

    Fast-path for greetings and balance checks; everything else goes to
    the LLM with tool calling so it can query spending data directly.
    """
    text = body.strip()
    lower = text.lower().strip("?!. ")

    # Fast path: greetings (or very short ambiguous inputs)
    if lower in GREETING_WORDS or (len(text) <= 3 and not text.isdigit()):
        return {
            "content": cmd_help(),
            "used_llm": False,
            "intent": "greeting",
            "confidence": 1.0,
            "tokens": 0,
        }

    # Fast path: balance checks
    if lower in BALANCE_WORDS:
        return {
            "content": cmd_balance(),
            "used_llm": False,
            "intent": "account_balance",
            "confidence": 1.0,
            "tokens": 0,
        }

    # Tool calling for all spending queries and general questions
    result = ask_llm_with_tools(text, history=history, user_id=user_id)
    result["intent"] = "tool_call"
    result["confidence"] = 1.0
    return result


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

from core.pdf_processor import process_pdf_upload


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
# Metrics database (Supabase)
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
    """Log interaction metrics to Supabase."""
    try:
        user_response = (
            supabase.table("users")
            .select("id, total_messages")
            .eq("phone_hash", user_hash)
            .execute()
        )

        if user_response.data:
            user_id = user_response.data[0]["id"]
            current_total = user_response.data[0]["total_messages"]
            supabase.table("users").update(
                {
                    "last_seen_at": datetime.now(timezone.utc).isoformat(),
                    "total_messages": current_total + 1,
                }
            ).eq("id", user_id).execute()
        else:
            user_response = (
                supabase.table("users")
                .insert({"phone_hash": user_hash, "total_messages": 1})
                .execute()
            )
            user_id = user_response.data[0]["id"]

        metric_data: dict = {
            "user_id": user_id,
            "message_text": message_text,
            "used_llm": used_llm,
            "llm_response_time": llm_time,
            "total_response_time": round(total_time, 3),
            "success": success,
            "tokens_used": tokens,
        }
        if intent:
            metric_data["intent"] = intent
        if confidence is not None:
            metric_data["confidence"] = confidence

        supabase.table("metrics").insert(metric_data).execute()

    except Exception as e:
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
        for row in reversed(rows):
            messages.append({"role": "user", "content": row["user_message"]})
            messages.append({"role": "assistant", "content": row["bot_response"]})
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
        print(f"[WEBHOOK] Received from {user_hash[:8]}: '{incoming_msg[:80]}'", file=sys.stderr)

        # Handle "clear history" before anything else
        if incoming_msg.lower() in ("clear history", "start fresh", "forget everything"):
            _clear_chat_history(user_hash)
            resp = MessagingResponse()
            resp.message("Got it — I've cleared our conversation history. Fresh start!")
            return str(resp), 200, {"Content-Type": "application/xml"}

        # Load conversation history and resolve user_id for budget tools
        user_id = _get_user_id(user_hash)
        history = _load_chat_history(user_hash)

        result = handle_message(incoming_msg, history=history, user_id=user_id)
        total_time = time.monotonic() - start

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
        print(f"[WEBHOOK] Sending: '{result['content'][:120]}'", file=sys.stderr)
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
        user_response = (
            supabase.table("users").select("id").eq("phone_hash", user_hash).single().execute()
        )

        if not user_response.data:
            return {"error": "User not found"}, 404

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
        today_response = (
            supabase.table("metrics")
            .select("*", count="exact")
            .gte("timestamp", today_str)
            .execute()
        )
        today = today_response.count

        week_response = (
            supabase.table("metrics")
            .select("*", count="exact")
            .gte("timestamp", week_ago)
            .execute()
        )
        week = week_response.count

        total_response = supabase.table("metrics").select("*", count="exact").execute()
        total = total_response.count

        metrics_with_llm = (
            supabase.table("metrics")
            .select("llm_response_time")
            .eq("used_llm", True)
            .not_.is_("llm_response_time", "null")
            .execute()
        )

        if metrics_with_llm.data:
            llm_times = [m["llm_response_time"] for m in metrics_with_llm.data]
            avg_llm = f"{sum(llm_times) / len(llm_times):.2f}"
        else:
            avg_llm = "-"

        all_metrics = supabase.table("metrics").select("total_response_time").execute()
        if all_metrics.data:
            total_times = [m["total_response_time"] for m in all_metrics.data]
            avg_total = f"{sum(total_times) / len(total_times):.2f}"
        else:
            avg_total = "-"

        errors_response = (
            supabase.table("metrics").select("*", count="exact").eq("success", False).execute()
        )
        errors = errors_response.count
        error_rate = f"{errors / total * 100:.1f}%" if total > 0 else "0%"

        all_metrics_for_users = supabase.table("metrics").select("user_id, used_llm").execute()
        user_stats = defaultdict(lambda: {"cnt": 0, "llm_cnt": 0})
        for m in all_metrics_for_users.data:
            user_stats[m["user_id"]]["cnt"] += 1
            user_stats[m["user_id"]]["llm_cnt"] += 1 if m["used_llm"] else 0

        sorted_users = sorted(
            user_stats.keys(), key=lambda uid: user_stats[uid]["cnt"], reverse=True
        )
        top_user_ids = sorted_users[:20]

        if top_user_ids:
            users_data = (
                supabase.table("users").select("id, phone_hash").in_("id", top_user_ids).execute()
            )
            user_hash_map = {u["id"]: u["phone_hash"] for u in users_data.data}
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

        all_messages = supabase.table("metrics").select("message_text").execute()
        pattern_counter: defaultdict[str, int] = defaultdict(int)
        for m in all_messages.data:
            pattern_counter[m["message_text"].lower()] += 1

        patterns = sorted(pattern_counter.items(), key=lambda x: x[1], reverse=True)[:15]
        pattern_rows = "".join(
            f"<tr><td>{escape(msg)}</td><td>{cnt}</td></tr>" for msg, cnt in patterns
        )

        week_metrics = (
            supabase.table("metrics").select("timestamp").gte("timestamp", week_ago).execute()
        )
        day_counter: defaultdict[str, int] = defaultdict(int)
        for m in week_metrics.data:
            day_counter[m["timestamp"][:10]] += 1

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
