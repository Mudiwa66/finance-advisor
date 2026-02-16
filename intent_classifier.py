"""
Intent Classification System for WhatsApp Financial Advisor Bot.

Provides hybrid intent classification using rule-based matching with
optional LLM fallback for ambiguous cases.
"""

import re
from enum import Enum
from typing import Tuple


class Intent(Enum):
    """Supported user intents."""
    GREETING = "greeting"
    ACCOUNT_BALANCE = "account_balance"
    SPENDING_QUERY = "spending_query"
    DEBT_ADVICE = "debt_advice"
    BUDGET_CHECK = "budget_check"
    DOCUMENT_UPLOAD = "document_upload"
    GENERAL_FINANCIAL_ADVICE = "general_financial_advice"
    UNKNOWN = "unknown"


class IntentClassifier:
    """Hybrid intent classifier using rule-based + LLM fallback."""

    # Rule-based patterns for each intent
    INTENT_PATTERNS = {
        Intent.GREETING: {
            "keywords": ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "help"],
            "exact_matches": ["hi", "hello", "hey", "sup", "yo", "help"],
        },
        Intent.ACCOUNT_BALANCE: {
            "keywords": ["balance", "how much", "money left", "remaining", "account", "available", "income", "credits"],
            "exact_matches": ["balance", "closing balance", "current balance", "income"],
            "patterns": ["how much.*left", "how much.*have", "what.*balance", "total.*income", "how much.*income"],
        },
        Intent.SPENDING_QUERY: {
            "keywords": [
                "spent", "spend", "spending", "bought", "purchase", "paid", "debits",
                # Common merchants
                "uber", "bolt", "takealot", "checkers", "woolworths", "pep", "clicks",
                # Month names
                "january", "jan", "february", "feb", "march", "mar", "april", "apr",
                "may", "june", "jun", "july", "jul", "august", "aug", "september", "sep",
                "october", "oct", "november", "nov", "december", "dec",
            ],
            "exact_matches": [
                "total", "total spending", "total spend", "total debits",
                "top", "top merchants", "top spend", "top 10", "biggest",
            ],
            "patterns": [
                "how much.*spent", "what.*spent", ".*spending", "spent.*on",
                "how much.*at", "spent.*at", ".*merchant",
            ],
        },
        Intent.DEBT_ADVICE: {
            "keywords": ["debt", "owe", "loan", "credit", "borrow", "repay", "overdraft", "owing"],
            "patterns": ["how.*pay.*debt", "reduce.*debt", "debt.*advice", "manage.*debt"],
        },
        Intent.BUDGET_CHECK: {
            "keywords": ["budget", "afford", "save", "savings", "limit", "over budget"],
            "patterns": ["can.*afford", "within.*budget", "budget.*check", "over.*budget"],
        },
        Intent.DOCUMENT_UPLOAD: {
            "keywords": ["upload", "statement", "document", "pdf", "file", "attach"],
            "patterns": ["upload.*statement", "send.*statement", "attach.*document"],
        },
        Intent.GENERAL_FINANCIAL_ADVICE: {
            "keywords": ["advice", "recommend", "should i", "invest", "financial", "tips"],
            "patterns": ["what.*should.*do", "how.*improve", "advice.*on", ".*recommend"],
        },
    }

    CONFIDENCE_THRESHOLD_HIGH = 0.85
    CONFIDENCE_THRESHOLD_LOW = 0.4

    @classmethod
    def classify(cls, message: str) -> Tuple[Intent, float, str]:
        """
        Classify user intent from message.

        Args:
            message: User's message text

        Returns:
            Tuple of (intent, confidence, reasoning)
        """
        text = message.strip().lower()

        # Try rule-based classification first
        intent, confidence, reasoning = cls._rule_based_classify(text)

        # If confidence is medium/low, optionally use LLM for disambiguation
        # (Currently disabled for speed - can enable if needed)
        if confidence < cls.CONFIDENCE_THRESHOLD_HIGH and False:  # Set to True to enable LLM
            intent, confidence, reasoning = cls._llm_classify(message)

        return intent, confidence, reasoning

    @classmethod
    def _rule_based_classify(cls, text: str) -> Tuple[Intent, float, str]:
        """Fast rule-based classification using keywords and patterns."""
        scores = {}
        reasons = {}

        for intent, patterns in cls.INTENT_PATTERNS.items():
            score = 0
            matched_terms = []

            # Check exact matches (highest confidence)
            if "exact_matches" in patterns:
                for exact in patterns["exact_matches"]:
                    if text == exact:
                        return intent, 0.95, f"exact match: '{exact}'"

            # Check keywords
            if "keywords" in patterns:
                for keyword in patterns["keywords"]:
                    if keyword in text:
                        score += 0.3
                        matched_terms.append(keyword)

            # Check regex patterns
            if "patterns" in patterns:
                for pattern in patterns["patterns"]:
                    if re.search(pattern, text):
                        score += 0.5
                        matched_terms.append(f"pattern:{pattern}")

            if score > 0:
                scores[intent] = min(score, 1.0)  # Cap at 1.0
                reasons[intent] = f"matched: {', '.join(matched_terms[:3])}"

        # Return highest scoring intent
        if scores:
            best_intent = max(scores.keys(), key=lambda k: scores[k])
            return best_intent, scores[best_intent], reasons[best_intent]

        # No matches - unknown intent
        return Intent.UNKNOWN, 0.0, "no patterns matched"

    @classmethod
    def _llm_classify(cls, message: str) -> Tuple[Intent, float, str]:
        """Use LLM for intent classification (fallback for ambiguous cases)."""
        # This would call Gemini to classify the intent
        # For now, return unknown with medium confidence
        # TODO: Implement LLM-based classification if needed
        return Intent.UNKNOWN, 0.5, "LLM classification not implemented"
