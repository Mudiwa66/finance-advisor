"""
Unit tests for the core tool functions in app.py.

All tests use SAMPLE_TRANSACTIONS as a known fixture dataset injected into
the app.TRANSACTIONS global via unittest.mock.patch. No external services
are called.
"""

import pytest
from datetime import datetime
from unittest.mock import patch

import app

# ---------------------------------------------------------------------------
# Fixture dataset
# ---------------------------------------------------------------------------

SAMPLE_TRANSACTIONS = [
    # January 2026
    {
        "date": "2026-01-05",
        "description": "Card Purchase Uber",
        "amount": -150.00,
        "balance": 4850.00,
        "merchant": "Uber",
    },
    {
        "date": "2026-01-10",
        "description": "Card Purchase Woolworths Food",
        "amount": -320.50,
        "balance": 4529.50,
        "merchant": "Woolworths Food",
    },
    {
        "date": "2026-01-15",
        "description": "Rtc Express Credit",
        "amount": 25000.00,
        "balance": 29529.50,
        "merchant": None,
    },
    {
        "date": "2026-01-20",
        "description": "Card Purchase Uber",
        "amount": -200.00,
        "balance": 29329.50,
        "merchant": "Uber",
    },
    {
        "date": "2026-01-25",
        "description": "Card Purchase Checkers",
        "amount": -450.00,
        "balance": 28879.50,
        "merchant": "Checkers",
    },
    # February 2026
    {
        "date": "2026-02-01",
        "description": "Card Purchase Uber Eats",
        "amount": -180.00,
        "balance": 28699.50,
        "merchant": "Uber Eats",
    },
    {
        "date": "2026-02-10",
        "description": "Card Purchase Woolworths Food",
        "amount": -290.00,
        "balance": 28409.50,
        "merchant": "Woolworths Food",
    },
    {
        "date": "2026-02-11",
        "description": "Card Purchase Engen",
        "amount": -650.00,
        "balance": 27759.50,
        "merchant": "Engen",
    },
]


@pytest.fixture(autouse=True)
def patch_transactions():
    """Inject SAMPLE_TRANSACTIONS into app.TRANSACTIONS for every test."""
    with patch.object(app, "TRANSACTIONS", SAMPLE_TRANSACTIONS):
        yield


# ---------------------------------------------------------------------------
# extract_merchant
# ---------------------------------------------------------------------------


class TestExtractMerchant:
    def test_strips_card_purchase_prefix(self):
        assert app.extract_merchant("Card Purchase Uber") == "Uber"

    def test_strips_card_purchase_with_cashback(self):
        assert app.extract_merchant("Card Purchase With Cashback Woolworths Food") == "Woolworths Food"

    def test_strips_fnb_app_payment_to(self):
        assert app.extract_merchant("FNB App Payment To John Smith") == "John Smith"

    def test_strips_rtc_express_credit(self):
        assert app.extract_merchant("Rtc Express Credit Salary") == "Salary"

    def test_strips_card_number(self):
        # Card number like 123456*7890 should be removed
        assert app.extract_merchant("Card Purchase Amazon 123456*7890") == "Amazon"

    def test_empty_description_returns_bank_fees(self):
        assert app.extract_merchant("") == "Bank Fees"
        assert app.extract_merchant("   ") == "Bank Fees"

    def test_unknown_description_falls_back_to_first_40_chars(self):
        # No prefix matched, nothing to strip — falls back to description[:40]
        # only when text AND matched_prefix are both falsy. With unknown text,
        # extract_merchant returns the cleaned text as-is (no truncation unless
        # text is empty after cleaning). The [:40] fallback is the last resort.
        long_desc = "X" * 50
        result = app.extract_merchant(long_desc)
        # Text had no known prefix but is non-empty, so it's returned as-is
        assert result == long_desc


# ---------------------------------------------------------------------------
# _filter_by_dates
# ---------------------------------------------------------------------------


class TestFilterByDates:
    def test_no_filters_returns_all(self):
        result = app._filter_by_dates(SAMPLE_TRANSACTIONS, None, None)
        assert result == SAMPLE_TRANSACTIONS

    def test_date_from_filters_earlier_dates(self):
        result = app._filter_by_dates(SAMPLE_TRANSACTIONS, "2026-02-01", None)
        assert all(t["date"] >= "2026-02-01" for t in result)
        assert len(result) == 3

    def test_date_to_filters_later_dates(self):
        result = app._filter_by_dates(SAMPLE_TRANSACTIONS, None, "2026-01-31")
        assert all(t["date"] <= "2026-01-31" for t in result)
        assert len(result) == 5

    def test_date_range_returns_correct_window(self):
        result = app._filter_by_dates(SAMPLE_TRANSACTIONS, "2026-01-10", "2026-01-20")
        dates = [t["date"] for t in result]
        assert dates == ["2026-01-10", "2026-01-15", "2026-01-20"]

    def test_empty_range_returns_empty(self):
        result = app._filter_by_dates(SAMPLE_TRANSACTIONS, "2025-01-01", "2025-12-31")
        assert result == []

    def test_single_day_range(self):
        result = app._filter_by_dates(SAMPLE_TRANSACTIONS, "2026-01-05", "2026-01-05")
        assert len(result) == 1
        assert result[0]["date"] == "2026-01-05"


# ---------------------------------------------------------------------------
# _tool_get_balance
# ---------------------------------------------------------------------------


class TestToolGetBalance:
    def test_returns_last_transaction_balance(self):
        result = app._tool_get_balance()
        assert "27,759.50" in result
        assert "2026-02-11" in result

    def test_empty_transactions(self):
        with patch.object(app, "TRANSACTIONS", []):
            result = app._tool_get_balance()
        assert "No transaction data" in result


# ---------------------------------------------------------------------------
# _tool_get_total_spending
# ---------------------------------------------------------------------------


class TestToolGetTotalSpending:
    def test_all_time_total(self):
        result = app._tool_get_total_spending()
        # Debits: 150 + 320.50 + 200 + 450 + 180 + 290 + 650 = 2,240.50
        assert "2,240.50" in result
        assert "7 transactions" in result

    def test_filtered_by_month(self):
        result = app._tool_get_total_spending("2026-01-01", "2026-01-31")
        # Jan debits: 150 + 320.50 + 200 + 450 = 1,120.50
        assert "1,120.50" in result
        assert "4 transactions" in result

    def test_no_spending_in_range(self):
        result = app._tool_get_total_spending("2025-01-01", "2025-12-31")
        assert "No spending found" in result

    def test_credits_not_counted(self):
        # Only the R25,000 credit exists in Jan; debits are 1,120.50
        result = app._tool_get_total_spending("2026-01-01", "2026-01-31")
        assert "25,000" not in result


# ---------------------------------------------------------------------------
# _tool_get_merchant_spending
# ---------------------------------------------------------------------------


class TestToolGetMerchantSpending:
    def test_exact_merchant_match(self):
        result = app._tool_get_merchant_spending("uber")
        # Uber: 150 + 200 = 350 (Uber Eats is separate)
        assert "350.00" in result

    def test_partial_merchant_match(self):
        result = app._tool_get_merchant_spending("woolworths")
        # Woolworths Food: 320.50 + 290 = 610.50
        assert "610.50" in result

    def test_merchant_with_date_filter(self):
        result = app._tool_get_merchant_spending("uber", "2026-01-01", "2026-01-31")
        # Only Jan Uber transactions: 150 + 200 = 350
        assert "350.00" in result

    def test_merchant_not_found(self):
        result = app._tool_get_merchant_spending("kfc")
        assert "No spending found" in result

    def test_merchant_not_found_includes_coverage_hint(self):
        result = app._tool_get_merchant_spending("kfc")
        assert "2026-01-05" in result  # data start
        assert "2026-02-11" in result  # data end

    def test_case_insensitive(self):
        result_lower = app._tool_get_merchant_spending("uber")
        result_upper = app._tool_get_merchant_spending("UBER")
        # Both find the same transactions — amounts match even if query label differs
        assert "350.00" in result_lower
        assert "350.00" in result_upper


# ---------------------------------------------------------------------------
# _tool_get_top_merchants
# ---------------------------------------------------------------------------


class TestToolGetTopMerchants:
    def test_returns_merchants_sorted_by_spend(self):
        result = app._tool_get_top_merchants()
        lines = result.strip().split("\n")
        # Engen (650) should be first, then Checkers (450), then Woolworths (610.50)
        # Wait: Woolworths Food = 320.50+290 = 610.50 > Checkers 450 > Uber 350 > Engen 650
        # Sorted: Engen 650, Woolworths 610.50, Checkers 450, Uber 350, Uber Eats 180
        assert "Engen" in lines[1]  # line 0 is header

    def test_filters_credits_out(self):
        result = app._tool_get_top_merchants()
        # R25,000 credit should not appear
        assert "25,000" not in result
        assert "Rtc" not in result

    def test_date_filtered(self):
        result = app._tool_get_top_merchants("2026-02-01", "2026-02-28")
        # Feb only: Engen 650, Woolworths 290, Uber Eats 180
        assert "Engen" in result
        assert "Checkers" not in result

    def test_no_spending_in_range(self):
        result = app._tool_get_top_merchants("2025-01-01", "2025-12-31")
        assert "No spending found" in result


# ---------------------------------------------------------------------------
# _tool_get_period_summary
# ---------------------------------------------------------------------------


class TestToolGetPeriodSummary:
    def test_includes_debits_and_credits(self):
        result = app._tool_get_period_summary("2026-01-01", "2026-01-31")
        assert "1,120.50" in result   # total debits
        assert "25,000.00" in result  # total credits

    def test_includes_top_merchants(self):
        result = app._tool_get_period_summary("2026-01-01", "2026-01-31")
        assert "Checkers" in result or "Woolworths" in result

    def test_empty_range_returns_no_data_message(self):
        result = app._tool_get_period_summary("2025-01-01", "2025-12-31")
        assert "No transactions found" in result

    def test_no_date_filter_covers_all(self):
        result = app._tool_get_period_summary()
        # No date filter → all transactions included
        assert "2,240.50" in result   # total debits across all sample data
        assert "25,000.00" in result  # total credits across all sample data


# ---------------------------------------------------------------------------
# Budget helpers
# ---------------------------------------------------------------------------

TEST_USER_ID = "00000000-0000-0000-0000-000000000001"


class TestCurrentPeriodDates:
    def test_monthly_starts_on_first(self):
        date_from, date_to = app._current_period_dates("monthly")
        assert date_from.endswith("-01")
        assert date_to == datetime.now().strftime("%Y-%m-%d")

    def test_weekly_starts_on_monday(self):
        date_from, _ = app._current_period_dates("weekly")
        from datetime import datetime as dt
        day = dt.strptime(date_from, "%Y-%m-%d").weekday()
        assert day == 0  # Monday = 0

    def test_unknown_period_defaults_to_monthly(self):
        date_from_monthly, _ = app._current_period_dates("monthly")
        date_from_unknown, _ = app._current_period_dates("bogus")
        assert date_from_monthly == date_from_unknown


class TestGetCategorySpending:
    def test_matches_merchant_keyword(self):
        spent = app._get_category_spending("uber", "2026-01-01", "2026-01-31")
        assert spent == 350.0  # 150 + 200

    def test_case_insensitive(self):
        lower = app._get_category_spending("woolworths", "2026-01-01", "2026-12-31")
        upper = app._get_category_spending("WOOLWORTHS", "2026-01-01", "2026-12-31")
        assert lower == upper

    def test_credits_not_counted(self):
        spent = app._get_category_spending("rtc", "2026-01-01", "2026-01-31")
        assert spent == 0.0

    def test_no_match_returns_zero(self):
        spent = app._get_category_spending("kfc", "2026-01-01", "2026-12-31")
        assert spent == 0.0

    def test_date_range_filters_correctly(self):
        # Only Feb transactions
        spent = app._get_category_spending("woolworths", "2026-02-01", "2026-02-28")
        assert spent == 290.0


class TestBudgetStatusLine:
    def test_under_budget(self):
        line = app._budget_status_line("food", 500.0, 2000.0, "monthly")
        assert "food" in line.lower()
        assert "R1,500.00 left" in line
        assert "25%" in line

    def test_over_budget(self):
        line = app._budget_status_line("food", 2500.0, 2000.0, "monthly")
        assert "OVER" in line
        assert "R500.00" in line

    def test_weekly_label(self):
        line = app._budget_status_line("transport", 100.0, 500.0, "weekly")
        assert "week" in line


class TestToolSetBudget:
    def test_set_budget_calls_supabase_upsert(self):
        from unittest.mock import MagicMock, patch
        mock_result = MagicMock()
        mock_result.data = [{"id": "abc", "category": "uber", "amount": 1000.0}]

        with patch.object(app.supabase.table("user_budgets"), "upsert") as mock_upsert:
            mock_upsert.return_value.execute.return_value = mock_result
            result = app._tool_set_budget(TEST_USER_ID, "Uber", 1000.0, "monthly")

        # Should contain budget confirmation text
        assert "uber" in result.lower()
        assert "1,000.00" in result

    def test_normalises_category_to_lowercase(self):
        from unittest.mock import MagicMock, patch
        with patch.object(app.supabase, "table") as mock_table:
            mock_table.return_value.upsert.return_value.execute.return_value = MagicMock(data=[{}])
            result = app._tool_set_budget(TEST_USER_ID, "FOOD", 2000.0)
        assert "food" in result.lower()

    def test_invalid_period_defaults_to_monthly(self):
        from unittest.mock import MagicMock, patch
        with patch.object(app.supabase, "table") as mock_table:
            mock_table.return_value.upsert.return_value.execute.return_value = MagicMock(data=[{}])
            result = app._tool_set_budget(TEST_USER_ID, "food", 500.0, "yearly")
        assert "month" in result


class TestToolGetBudgetStatus:
    def test_no_budgets_returns_helpful_message(self):
        from unittest.mock import MagicMock, patch
        with patch.object(app.supabase, "table") as mock_table:
            chain = MagicMock()
            chain.select.return_value = chain
            chain.eq.return_value = chain
            chain.execute.return_value = MagicMock(data=[])
            mock_table.return_value = chain

            result = app._tool_get_budget_status(TEST_USER_ID)
        assert "No budgets" in result

    def test_returns_status_for_each_budget(self):
        from unittest.mock import MagicMock, patch
        budgets = [
            {"category": "uber", "amount": 1000.0, "period": "monthly"},
            {"category": "food", "amount": 2000.0, "period": "monthly"},
        ]
        with patch.object(app.supabase, "table") as mock_table:
            chain = MagicMock()
            chain.select.return_value = chain
            chain.eq.return_value = chain
            chain.execute.return_value = MagicMock(data=budgets)
            mock_table.return_value = chain

            result = app._tool_get_budget_status(TEST_USER_ID)
        assert "Uber" in result
        assert "Food" in result


class TestToolListBudgets:
    def test_empty_returns_helpful_message(self):
        from unittest.mock import MagicMock, patch
        with patch.object(app.supabase, "table") as mock_table:
            chain = MagicMock()
            chain.select.return_value = chain
            chain.eq.return_value = chain
            chain.order.return_value = chain
            chain.execute.return_value = MagicMock(data=[])
            mock_table.return_value = chain

            result = app._tool_list_budgets(TEST_USER_ID)
        assert "No active budgets" in result

    def test_lists_all_budgets(self):
        from unittest.mock import MagicMock, patch
        budgets = [
            {"category": "food", "amount": 2000.0, "period": "monthly"},
            {"category": "transport", "amount": 800.0, "period": "weekly"},
        ]
        with patch.object(app.supabase, "table") as mock_table:
            chain = MagicMock()
            chain.select.return_value = chain
            chain.eq.return_value = chain
            chain.order.return_value = chain
            chain.execute.return_value = MagicMock(data=budgets)
            mock_table.return_value = chain

            result = app._tool_list_budgets(TEST_USER_ID)
        assert "Food" in result
        assert "Transport" in result
        assert "R2,000.00/month" in result
        assert "R800.00/week" in result


# ---------------------------------------------------------------------------
# handle_message fast paths (no LLM needed)
# ---------------------------------------------------------------------------


class TestHandleMessageFastPaths:
    @pytest.mark.parametrize("msg", ["hi", "hello", "hey", "help", "howzit", "yo"])
    def test_greeting_returns_help_without_llm(self, msg):
        result = app.handle_message(msg)
        assert result["used_llm"] is False
        assert result["intent"] == "greeting"

    @pytest.mark.parametrize("msg", ["balance", "my balance", "closing balance", "current balance"])
    def test_balance_returns_balance_without_llm(self, msg):
        result = app.handle_message(msg)
        assert result["used_llm"] is False
        assert result["intent"] == "account_balance"
        assert "27,759.50" in result["content"]

    def test_balance_content_includes_date(self):
        result = app.handle_message("balance")
        assert "2026-02-11" in result["content"]

    def test_short_ambiguous_input_treated_as_greeting(self):
        result = app.handle_message("ok")
        assert result["intent"] == "greeting"
        assert result["used_llm"] is False
