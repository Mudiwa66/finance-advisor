#!/usr/bin/env python3
"""Migrate SQLite metrics and JSON transactions to Supabase."""

import json
import os
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Merchant extraction logic (copied from app.py to avoid circular imports)
DESCRIPTION_PREFIXES = [
    "Card Purchase",
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

def migrate():
    print("🚀 Starting Supabase migration...")

    # Step 1: Load existing data
    print("\n📊 Loading existing data...")

    # Load transactions from JSON
    transactions_file = Path("data/transactions.json")
    with open(transactions_file) as f:
        transactions = json.load(f)
    print(f"  ✓ Loaded {len(transactions)} transactions")

    # Load metrics from SQLite
    metrics_db = Path("data/metrics.db")
    conn = sqlite3.connect(str(metrics_db))
    conn.row_factory = sqlite3.Row
    metrics = conn.execute("SELECT * FROM metrics").fetchall()
    metrics = [dict(m) for m in metrics]
    print(f"  ✓ Loaded {len(metrics)} metrics")

    # Step 2: Create default user (current single user)
    print("\n👤 Creating default user...")
    # Use the most common user_hash from metrics, or create a test user
    if metrics:
        default_hash = metrics[0]["user_hash"]
    else:
        import hashlib
        default_hash = hashlib.sha256("default_user".encode()).hexdigest()[:12]

    user_response = supabase.table("users").insert({
        "phone_hash": default_hash,
        "total_messages": len(metrics)
    }).execute()
    user_id = user_response.data[0]["id"]
    print(f"  ✓ Created user with hash {default_hash}, ID: {user_id}")

    # Step 3: Migrate transactions with merchant extraction
    print("\n💰 Migrating transactions...")
    enriched_transactions = []
    for txn in transactions:
        # Extract merchant for spending transactions (amount < 0)
        merchant = None
        if float(txn.get("amount", 0)) < 0:
            merchant = extract_merchant(txn.get("description", ""))

        enriched_transactions.append({
            "user_id": user_id,
            "date": txn["date"],
            "description": txn.get("description", ""),
            "amount": txn["amount"],
            "balance": txn["balance"],
            "merchant": merchant,
            "statement_source": "FNB_ASPIRE_CURRENT_ACCOUNT_27.pdf"
        })

    # Bulk insert transactions
    batch_size = 100
    for i in range(0, len(enriched_transactions), batch_size):
        batch = enriched_transactions[i:i+batch_size]
        supabase.table("transactions").insert(batch).execute()
        print(f"  ✓ Inserted transactions {i+1}-{min(i+batch_size, len(enriched_transactions))}")

    # Step 4: Migrate metrics
    print("\n📈 Migrating metrics...")
    migrated_metrics = []
    for m in metrics:
        migrated_metrics.append({
            "user_id": user_id,
            "timestamp": m["timestamp"],
            "message_text": m["message_text"],
            "used_llm": bool(m["used_llm"]),
            "llm_response_time": m["llm_response_time"],
            "total_response_time": m["total_response_time"],
            "success": bool(m["success"]),
            "tokens_used": m["tokens_used"]
        })

    if migrated_metrics:
        supabase.table("metrics").insert(migrated_metrics).execute()
        print(f"  ✓ Inserted {len(migrated_metrics)} metrics")

    # Step 5: Verification
    print("\n✅ Verifying migration...")
    db_users = supabase.table("users").select("*", count="exact").execute()
    db_transactions = supabase.table("transactions").select("*", count="exact").execute()
    db_metrics = supabase.table("metrics").select("*", count="exact").execute()

    print(f"  Users: {db_users.count}")
    print(f"  Transactions: {db_transactions.count} (expected {len(transactions)})")
    print(f"  Metrics: {db_metrics.count} (expected {len(metrics)})")

    if db_transactions.count == len(transactions) and db_metrics.count == len(metrics):
        print("\n🎉 Migration completed successfully!")

        # Backup old files
        print("\n💾 Backing up old data files...")
        import shutil
        shutil.copy(metrics_db, metrics_db.with_suffix(".db.bak"))
        shutil.copy(transactions_file, transactions_file.with_suffix(".json.bak"))
        print("  ✓ Backed up metrics.db → metrics.db.bak")
        print("  ✓ Backed up transactions.json → transactions.json.bak")
    else:
        print("\n⚠️  Migration counts don't match! Please review.")

    conn.close()
    print(f"\n📝 Save this user_id for configuration: {user_id}")
    print(f"📝 Save this phone_hash for configuration: {default_hash}")

if __name__ == "__main__":
    migrate()
