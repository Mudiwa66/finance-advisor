#!/usr/bin/env python3
"""Test webhook functionality without Flask server."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Test 1: Check environment variables
print("=" * 60)
print("TEST 1: Environment Variables")
print("=" * 60)
print(f"SUPABASE_URL: {'✓ SET' if os.getenv('SUPABASE_URL') else '✗ NOT SET'}")
print(f"SUPABASE_KEY: {'✓ SET' if os.getenv('SUPABASE_KEY') else '✗ NOT SET'}")
print(f"DEFAULT_USER_ID: {os.getenv('DEFAULT_USER_ID', 'NOT SET')}")
print(f"GEMINI_API_KEY: {'✓ SET' if os.getenv('GEMINI_API_KEY') else '✗ NOT SET'}")
print(f"GEMINI_MODEL: {os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')}")
print()

# Test 2: Import Gemini
print("=" * 60)
print("TEST 2: Gemini Import")
print("=" * 60)
try:
    import google.generativeai as genai
    print("✓ google.generativeai imported successfully")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        print("✓ Gemini model configured")
    else:
        print("✗ GEMINI_API_KEY not set")
        model = None
except Exception as e:
    print(f"✗ Failed to import/configure Gemini: {e}")
    model = None
print()

# Test 3: Supabase connection
print("=" * 60)
print("TEST 3: Supabase Connection")
print("=" * 60)
try:
    from supabase import create_client

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Supabase client created")

    # Test loading transactions
    DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")
    response = supabase.table("transactions").select("*").eq("user_id", DEFAULT_USER_ID).limit(5).execute()
    print(f"✓ Transactions query successful: {len(response.data)} transactions found (showing first 5)")

except Exception as e:
    print(f"✗ Supabase error: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 4: Simple Gemini API call
print("=" * 60)
print("TEST 4: Gemini API Call")
print("=" * 60)
if model and GEMINI_API_KEY:
    try:
        print("Sending test question to Gemini...")
        response = model.generate_content(
            "What is 2+2? Answer in one sentence.",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=100,
                temperature=0.7,
            ),
        )

        # THIS IS THE CRITICAL PART - accessing response.text can fail
        print(f"Response object: {response}")
        print(f"Response type: {type(response)}")

        # Check if response is blocked
        if hasattr(response, 'prompt_feedback'):
            print(f"Prompt feedback: {response.prompt_feedback}")

        if hasattr(response, 'candidates'):
            print(f"Candidates: {len(response.candidates) if response.candidates else 0}")
            if response.candidates:
                print(f"First candidate: {response.candidates[0]}")

        # Try to access text
        try:
            text = response.text
            print(f"✓ Gemini responded: {text}")
        except Exception as text_error:
            print(f"✗ Failed to access response.text: {text_error}")
            print(f"Response parts: {response.parts if hasattr(response, 'parts') else 'N/A'}")

    except Exception as e:
        print(f"✗ Gemini API call failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
else:
    print("✗ Skipping - Gemini not configured")
print()

# Test 5: Simulate webhook flow
print("=" * 60)
print("TEST 5: Simulate Webhook Flow")
print("=" * 60)
try:
    # Import the handle_message function
    sys.path.insert(0, os.path.dirname(__file__))
    from app import handle_message

    print("Testing with keyword 'total'...")
    result = handle_message("total")
    print(f"✓ Result: {result['content'][:100]}...")
    print(f"  Used LLM: {result.get('used_llm', False)}")
    print()

    print("Testing with natural language 'how much did I spend?'...")
    result = handle_message("how much did I spend?")
    print(f"✓ Result: {result['content'][:100]}...")
    print(f"  Used LLM: {result.get('used_llm', False)}")

except Exception as e:
    print(f"✗ Webhook simulation failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
