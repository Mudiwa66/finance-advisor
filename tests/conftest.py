"""
Pytest configuration: mock external services before app is imported.

app.py has module-level side effects (Supabase connection, transaction loading)
so we must inject mocks into sys.modules BEFORE the first import of app.
"""

import os
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Environment variables (must be set before app import)
# ---------------------------------------------------------------------------

os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiJ9.test.test")
os.environ.setdefault("DEFAULT_USER_ID", "00000000-0000-0000-0000-000000000001")
os.environ.setdefault("GROQ_API_KEY", "gsk_test_not_real")

# ---------------------------------------------------------------------------
# Mock supabase — create_client returns a MagicMock with .data = []
# ---------------------------------------------------------------------------

_mock_execute = MagicMock()
_mock_execute.data = []

# MagicMock returns a new MagicMock for any attribute/call by default,
# so chaining .select().eq().order() works automatically.
# We only need to wire .execute() to return our controlled result.
_mock_chain = MagicMock()
_mock_chain.execute.return_value = _mock_execute
_mock_chain.select.return_value = _mock_chain
_mock_chain.eq.return_value = _mock_chain
_mock_chain.order.return_value = _mock_chain
_mock_chain.limit.return_value = _mock_chain

_mock_sb_client = MagicMock()
_mock_sb_client.table.return_value = _mock_chain

_mock_supabase_mod = MagicMock()
_mock_supabase_mod.create_client.return_value = _mock_sb_client
_mock_supabase_mod.Client = MagicMock()

sys.modules["supabase"] = _mock_supabase_mod

# ---------------------------------------------------------------------------
# Mock groq
# ---------------------------------------------------------------------------

_mock_groq_mod = MagicMock()
_mock_groq_mod.Groq = MagicMock()
sys.modules["groq"] = _mock_groq_mod

# ---------------------------------------------------------------------------
# Mock core modules (pdf_processor triggers heavy deps)
# ---------------------------------------------------------------------------

sys.modules.setdefault("core", MagicMock())
sys.modules.setdefault("core.pdf_processor", MagicMock())

# ---------------------------------------------------------------------------
# Add project root to path and import app (with mocks already in place)
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402 — intentionally after sys.modules setup
