"""
PDF Bank Statement Processor for WhatsApp uploads.

Handles downloading, parsing, deduplication, and Supabase storage
of FNB bank statement PDFs sent via WhatsApp/Twilio.
"""

import re
import sys
from datetime import datetime, timezone
from io import BytesIO

import requests


# ---------------------------------------------------------------------------
# PDF Text Extraction (reuses parse_statement.py logic)
# ---------------------------------------------------------------------------

MONTH_MAP = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes preserving column layout.

    Uses pdfminer with position-aware reconstruction so that amounts
    appear on the same line as their descriptions (like pdftotext -layout).
    Pure Python — no system dependencies required.
    """
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTTextBox, LTTextLine

    try:
        params = LAParams(line_margin=0.5, word_margin=0.1, char_margin=2.0)
        all_lines = []

        for page in extract_pages(BytesIO(pdf_bytes), laparams=params):
            h = page.height
            elements = []
            for elem in page:
                if isinstance(elem, LTTextBox):
                    for line in elem:
                        if isinstance(line, LTTextLine):
                            text = line.get_text().strip()
                            if text:
                                elements.append((round(h - line.y0), round(line.x0), text))

            # Sort top-to-bottom, then left-to-right
            elements.sort(key=lambda e: (e[0], e[1]))
            if not elements:
                continue

            # Group elements within 4px vertically into the same row
            rows, current_y, current_row = [], elements[0][0], []
            for y, x, text in elements:
                if abs(y - current_y) <= 4:
                    current_row.append((x, text))
                else:
                    rows.append(sorted(current_row, key=lambda e: e[0]))
                    current_row = [(x, text)]
                    current_y = y
            if current_row:
                rows.append(sorted(current_row, key=lambda e: e[0]))

            for row in rows:
                all_lines.append("  ".join(t for _, t in row))

        return "\n".join(all_lines)

    except Exception as e:
        raise ValueError(f"Could not read this PDF. Please make sure it's a valid FNB bank statement. ({e})")


def is_fnb_statement(text: str) -> bool:
    """Check if the extracted text looks like an FNB statement."""
    fnb_markers = [
        "FNB", "First National Bank",
        "Statement Period",
        "ASPIRE", "CHEQUE", "SAVINGS",
    ]
    return any(marker in text for marker in fnb_markers)


def parse_statement_year(text: str) -> int:
    """Extract the year from the statement period header."""
    match = re.search(r"Statement Period.*?(\d{4})", text)
    return int(match.group(1)) if match else datetime.now().year


def parse_transactions(text: str) -> list[dict]:
    """Parse all transaction lines from extracted statement text."""
    year = parse_statement_year(text)

    date_re = re.compile(r"^(\d{2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b")
    tail_re = re.compile(
        r"([\d,]+\.\d{2})\s*(Cr)?"
        r"\s+"
        r"([\d,]+\.\d{2})\s*(Cr)?"
        r"(?:\s+([\d,]+\.\d{2}))?"
        r"\s*$"
    )

    transactions = []
    for line in text.splitlines():
        line = line.rstrip()
        date_match = date_re.match(line)
        if not date_match:
            continue
        tail_match = tail_re.search(line)
        if not tail_match:
            continue

        day = date_match.group(1)
        month = MONTH_MAP[date_match.group(2)]
        date = f"{year}-{month}-{day}"

        description = re.sub(
            r"\s{2,}", " ", line[date_match.end():tail_match.start()]
        ).strip()

        amount_str = tail_match.group(1)
        is_credit = bool(tail_match.group(2))
        amount = round(float(amount_str.replace(",", "")) * (1 if is_credit else -1), 2)

        balance_str = tail_match.group(3)
        balance_credit = bool(tail_match.group(4))
        balance = round(float(balance_str.replace(",", "")) * (1 if balance_credit else -1), 2)

        transactions.append({
            "date": date,
            "description": description,
            "amount": amount,
            "balance": balance,
        })

    return transactions


# ---------------------------------------------------------------------------
# Download from Twilio
# ---------------------------------------------------------------------------

def download_pdf_from_twilio(media_url: str, account_sid: str, auth_token: str) -> bytes:
    """Download a PDF from Twilio's media URL using basic auth."""
    response = requests.get(
        media_url,
        auth=(account_sid, auth_token),
        timeout=30,
    )
    response.raise_for_status()
    return response.content


# ---------------------------------------------------------------------------
# Supabase Storage and Deduplication
# ---------------------------------------------------------------------------

def get_or_create_user(supabase, phone_hash: str) -> str:
    """Get existing user ID or create new user. Returns user_id."""
    response = (
        supabase.table("users")
        .select("id")
        .eq("phone_hash", phone_hash)
        .execute()
    )
    if response.data:
        return response.data[0]["id"]

    # Create new user
    new_user = (
        supabase.table("users")
        .insert({"phone_hash": phone_hash, "total_messages": 0})
        .execute()
    )
    return new_user.data[0]["id"]


def check_duplicate_statement(supabase, user_id: str, date_from: str, date_to: str) -> bool:
    """Check if a statement covering this date range already exists."""
    response = (
        supabase.table("statement_uploads")
        .select("id")
        .eq("user_id", user_id)
        .eq("date_from", date_from)
        .eq("date_to", date_to)
        .execute()
    )
    return bool(response.data)


def save_transactions_to_supabase(supabase, user_id: str, transactions: list[dict]) -> int:
    """Insert transactions for a user, skipping duplicates. Returns count inserted."""
    if not transactions:
        return 0

    rows = [
        {
            "user_id": user_id,
            "date": t["date"],
            "description": t["description"],
            "amount": t["amount"],
            "balance": t["balance"],
            "merchant": None,  # Can be enriched later
        }
        for t in transactions
    ]

    # Use upsert to avoid duplicates on (user_id, date, description, amount)
    response = supabase.table("transactions").upsert(
        rows,
        on_conflict="user_id,date,description,amount"
    ).execute()

    return len(response.data) if response.data else 0


def record_statement_upload(
    supabase,
    user_id: str,
    filename: str,
    date_from: str,
    date_to: str,
    transaction_count: int,
    storage_path: str | None = None,
) -> None:
    """Record metadata about an uploaded statement."""
    supabase.table("statement_uploads").insert({
        "user_id": user_id,
        "filename": filename,
        "date_from": date_from,
        "date_to": date_to,
        "transaction_count": transaction_count,
        "storage_path": storage_path,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


def store_pdf_in_supabase(supabase, user_id: str, pdf_bytes: bytes, filename: str) -> str | None:
    """Upload PDF to Supabase Storage. Returns storage path or None on failure."""
    try:
        storage_path = f"{user_id}/{filename}"
        supabase.storage.from_("bank-statements").upload(
            path=storage_path,
            file=pdf_bytes,
            file_options={"content-type": "application/pdf"},
        )
        return storage_path
    except Exception as e:
        print(f"[PDF] Storage upload failed (non-fatal): {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_pdf_upload(
    supabase,
    media_url: str,
    media_content_type: str,
    phone_hash: str,
    account_sid: str,
    auth_token: str,
) -> dict:
    """
    Full pipeline: download → validate → parse → store → respond.

    Returns:
        dict with keys: success (bool), message (str), transaction_count (int)
    """
    print(f"[PDF] Processing upload from {phone_hash[:8]}", file=sys.stderr)

    # 1. Validate content type
    if "pdf" not in media_content_type.lower():
        return {
            "success": False,
            "message": "That doesn't look like a PDF. Please send your FNB statement as a PDF file.",
            "transaction_count": 0,
        }

    # 2. Download from Twilio
    try:
        pdf_bytes = download_pdf_from_twilio(media_url, account_sid, auth_token)
        print(f"[PDF] Downloaded {len(pdf_bytes):,} bytes", file=sys.stderr)
    except Exception as e:
        print(f"[PDF] Download failed: {e}", file=sys.stderr)
        return {
            "success": False,
            "message": "Could not download your file. Please try sending it again.",
            "transaction_count": 0,
        }

    # 3. Extract text
    try:
        text = extract_text_from_bytes(pdf_bytes)
    except ValueError as e:
        return {"success": False, "message": str(e), "transaction_count": 0}
    except RuntimeError as e:
        print(f"[PDF] Runtime error: {e}", file=sys.stderr)
        return {
            "success": False,
            "message": "PDF processing is unavailable right now. Please try again later.",
            "transaction_count": 0,
        }

    # 4. Validate it's an FNB statement
    if not is_fnb_statement(text):
        return {
            "success": False,
            "message": "Only FNB bank statements are supported for now.",
            "transaction_count": 0,
        }

    # 5. Parse transactions
    transactions = parse_transactions(text)
    if not transactions:
        return {
            "success": False,
            "message": "Could not read this PDF. Make sure it's an FNB bank statement and try again.",
            "transaction_count": 0,
        }

    date_from = transactions[0]["date"]
    date_to = transactions[-1]["date"]
    closing_balance = transactions[-1]["balance"]
    print(f"[PDF] Parsed {len(transactions)} transactions ({date_from} to {date_to})", file=sys.stderr)

    # 6. Get or create user
    try:
        user_id = get_or_create_user(supabase, phone_hash)
    except Exception as e:
        print(f"[PDF] User lookup failed: {e}", file=sys.stderr)
        return {
            "success": False,
            "message": "Account error. Please try again.",
            "transaction_count": 0,
        }

    # 7. Check for duplicate statement
    if check_duplicate_statement(supabase, user_id, date_from, date_to):
        return {
            "success": False,
            "message": f"This statement ({date_from} to {date_to}) has already been uploaded.",
            "transaction_count": 0,
        }

    # 8. Store PDF in Supabase Storage (best effort)
    filename = f"statement_{date_from}_{date_to}.pdf"
    storage_path = store_pdf_in_supabase(supabase, user_id, pdf_bytes, filename)

    # 9. Save transactions
    try:
        inserted = save_transactions_to_supabase(supabase, user_id, transactions)
    except Exception as e:
        print(f"[PDF] Transaction save failed: {e}", file=sys.stderr)
        return {
            "success": False,
            "message": "Could not save your transactions. Please try again.",
            "transaction_count": 0,
        }

    # 10. Record statement upload metadata
    try:
        record_statement_upload(
            supabase, user_id, filename, date_from, date_to, inserted, storage_path
        )
    except Exception as e:
        print(f"[PDF] Metadata record failed (non-fatal): {e}", file=sys.stderr)

    print(f"[PDF] Success: {inserted} transactions saved for user {user_id}", file=sys.stderr)

    return {
        "success": True,
        "message": (
            f"✅ Added {len(transactions)} transactions from {date_from} to {date_to}.\n"
            f"Your closing balance is R{closing_balance:,.2f}"
        ),
        "transaction_count": inserted,
    }
