#!/usr/bin/env python3
"""List available Gemini models - simplified version."""

# Replace with your actual API key
GEMINI_API_KEY = "AIzaSyC_7UfsUaCMpRif7LehctaRdN7ef668Icg"

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
        print(f"\n✓ {model.name}")
        if hasattr(model, 'display_name'):
            print(f"  Display: {model.display_name}")
        if hasattr(model, 'supported_generation_methods'):
            methods = ', '.join(model.supported_generation_methods)
            print(f"  Methods: {methods}")

    print("\n" + "=" * 60)

except ImportError:
    print("ERROR: google-genai not installed")
    print("Install: pip install google-genai")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
