#!/usr/bin/env python3
"""List available Gemini models using google-genai package."""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set in .env")
    exit(1)

try:
    from google import genai

    print("Initializing Gemini client...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    print("\n" + "=" * 60)
    print("AVAILABLE GEMINI MODELS")
    print("=" * 60)

    # List all available models
    models = client.models.list()

    for model in models:
        print(f"\nModel: {model.name}")
        print(f"  Display Name: {model.display_name if hasattr(model, 'display_name') else 'N/A'}")
        print(f"  Description: {model.description if hasattr(model, 'description') else 'N/A'}")
        if hasattr(model, 'supported_generation_methods'):
            print(f"  Supported Methods: {model.supported_generation_methods}")

    print("\n" + "=" * 60)
    print("RECOMMENDED MODELS FOR YOUR USE CASE:")
    print("=" * 60)
    print("1. models/gemini-1.5-flash - Fast, cost-effective (FREE tier)")
    print("2. models/gemini-1.5-pro - Better quality, more expensive")
    print("3. models/gemini-pro - Standard quality (if available)")

except ImportError as e:
    print(f"ERROR: Failed to import google.genai: {e}")
    print("\nMake sure google-genai is installed:")
    print("  pip install google-genai")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
