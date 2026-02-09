#!/usr/bin/env python3
"""Parse FNB bank statement PDF and extract transactions to CSV and JSON."""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

MONTH_MAP = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12',
}


def extract_text(pdf_path: str) -> str:
    """Extract text from PDF using pdftotext with layout preservation."""
    result = subprocess.run(
        ['pdftotext', '-layout', pdf_path, '-'],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def parse_statement_year(text: str) -> int:
    """Extract the year from the statement period header."""
    match = re.search(r'Statement Period.*?(\d{4})', text)
    return int(match.group(1)) if match else 2024


def parse_amount(value: str, is_credit: bool) -> float:
    """Parse monetary string to float. Credits positive, debits negative."""
    num = float(value.replace(',', ''))
    return num if is_credit else -num


def parse_transactions(text: str) -> list[dict]:
    """Parse all transaction lines from the extracted statement text."""
    year = parse_statement_year(text)

    date_re = re.compile(
        r'^(\d{2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b'
    )
    # Match the monetary tail of a transaction line:
    #   amount [Cr]  balance [Cr]  [bank_charges]
    # Anchored to end-of-line so description numbers (e.g. "99.00 Netflix")
    # don't cause false matches.
    tail_re = re.compile(
        r'([\d,]+\.\d{2})\s*(Cr)?'   # amount
        r'\s+'
        r'([\d,]+\.\d{2})\s*(Cr)?'   # balance
        r'(?:\s+([\d,]+\.\d{2}))?'    # optional accrued bank charges
        r'\s*$'
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

        # Collapse multi-space gaps left over from the fixed-width PDF layout
        description = re.sub(r'\s{2,}', ' ', line[date_match.end():tail_match.start()]).strip()

        amount = round(parse_amount(tail_match.group(1), bool(tail_match.group(2))), 2)
        balance = round(parse_amount(tail_match.group(3), bool(tail_match.group(4))), 2)

        transactions.append({
            'date': date,
            'description': description,
            'amount': amount,
            'balance': balance,
        })

    return transactions


def main():
    pdf_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else 'data/FNB_ASPIRE_CURRENT_ACCOUNT_27.pdf'
    )
    output_dir = Path('data')

    text = extract_text(pdf_path)
    transactions = parse_transactions(text)

    print(f"Extracted {len(transactions)} transactions")

    # Save CSV
    csv_path = output_dir / 'transactions.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['date', 'description', 'amount', 'balance']
        )
        writer.writeheader()
        writer.writerows(transactions)
    print(f"Saved: {csv_path}")

    # Save JSON
    json_path = output_dir / 'transactions.json'
    with open(json_path, 'w') as f:
        json.dump(transactions, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == '__main__':
    main()
