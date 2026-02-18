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

    # Rule-based patterns for each intent with max response lengths
    INTENT_PATTERNS = {
        Intent.GREETING: {
            "keywords": ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "help"],
            "exact_matches": ["hi", "hello", "hey", "sup", "yo", "help"],
            "max_words": 20,  # Short, friendly
        },
        Intent.ACCOUNT_BALANCE: {
            "keywords": ["balance", "how much", "money left", "remaining", "account", "available", "income", "credits"],
            "exact_matches": ["balance", "closing balance", "current balance", "income"],
            "patterns": ["how much.*left", "how much.*have", "what.*balance", "total.*income", "how much.*income"],
            "max_words": 30,  # Just the number + context
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
            "max_words": 40,  # Amount + brief insight
        },
        Intent.DEBT_ADVICE: {
            "keywords": ["debt", "owe", "loan", "credit", "borrow", "repay", "overdraft", "owing"],
            "patterns": ["how.*pay.*debt", "reduce.*debt", "debt.*advice", "manage.*debt"],
            "max_words": 80,  # Needs more context and guidance
        },
        Intent.BUDGET_CHECK: {
            "keywords": ["budget", "afford", "save", "savings", "limit", "over budget"],
            "patterns": ["can.*afford", "within.*budget", "budget.*check", "over.*budget"],
            "max_words": 50,  # Status + one recommendation
        },
        Intent.DOCUMENT_UPLOAD: {
            "keywords": ["upload", "statement", "document", "pdf", "file", "attach"],
            "patterns": ["upload.*statement", "send.*statement", "attach.*document"],
            "max_words": 20,  # Confirmation only
        },
        Intent.GENERAL_FINANCIAL_ADVICE: {
            "keywords": ["advice", "recommend", "should i", "invest", "financial", "tips"],
            "patterns": ["what.*should.*do", "how.*improve", "advice.*on", ".*recommend"],
            "max_words": 80,  # Conversational but concise
        },
    }

    # Default max words for unknown intent
    DEFAULT_MAX_WORDS = 30

    CONFIDENCE_THRESHOLD_HIGH = 0.85
    CONFIDENCE_THRESHOLD_LOW = 0.4

    # Groq client for LLM-based classification (set via setup_llm)
    _llm_client = None
    _llm_model = None

    @classmethod
    def setup_llm(cls, client, model: str) -> None:
        """Configure Groq client for LLM-based intent classification fallback."""
        cls._llm_client = client
        cls._llm_model = model

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

        # For low-confidence cases, use the fast LLM model to disambiguate
        if confidence < cls.CONFIDENCE_THRESHOLD_LOW:
            llm_intent, llm_confidence, llm_reasoning = cls._llm_classify(message)
            if llm_confidence > confidence:
                intent, confidence, reasoning = llm_intent, llm_confidence, llm_reasoning

        return intent, confidence, reasoning

    @classmethod
    def get_max_words(cls, intent: Intent) -> int:
        """Get the maximum word limit for a given intent."""
        pattern = cls.INTENT_PATTERNS.get(intent)
        if pattern and "max_words" in pattern:
            return pattern["max_words"]
        return cls.DEFAULT_MAX_WORDS

    @staticmethod
    def truncate_response(text: str, max_words: int) -> str:
        """
        Truncate response to max words while preserving sentence structure.

        Args:
            text: The response text to truncate
            max_words: Maximum number of words allowed

        Returns:
            Truncated text with "..." appended if truncated
        """
        words = text.split()
        if len(words) <= max_words:
            return text

        # Truncate and add ellipsis
        truncated = " ".join(words[:max_words])

        # Try to end at a sentence boundary if possible
        last_period = truncated.rfind(".")
        last_exclaim = truncated.rfind("!")
        last_question = truncated.rfind("?")
        last_punct = max(last_period, last_exclaim, last_question)

        if last_punct > len(truncated) * 0.7:  # If we're past 70%, use that sentence
            return truncated[:last_punct + 1]

        # Otherwise just truncate with ellipsis
        return truncated.rstrip(".,;:") + "..."

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
        """Use Groq llama-3.1-8b-instant to classify ambiguous intents."""
        if cls._llm_client is None or cls._llm_model is None:
            return Intent.UNKNOWN, 0.5, "LLM classification not configured"

        intent_names = [i.value for i in Intent if i != Intent.UNKNOWN]
        prompt = (
            f"Classify this financial chatbot message into exactly one intent.\n"
            f"Message: '{message}'\n"
            f"Intents: {', '.join(intent_names)}\n"
            f"Reply with ONLY the intent name, nothing else."
        )

        try:
            response = cls._llm_client.chat.completions.create(
                model=cls._llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0,
            )
            intent_str = response.choices[0].message.content.strip().lower()

            for intent in Intent:
                if intent.value == intent_str or intent.value.replace("_", " ") in intent_str:
                    return intent, 0.75, f"LLM classified as: {intent_str}"

            return Intent.UNKNOWN, 0.5, f"LLM returned unrecognized: {intent_str}"
        except Exception as e:
            return Intent.UNKNOWN, 0.5, f"LLM classification error: {e}"
