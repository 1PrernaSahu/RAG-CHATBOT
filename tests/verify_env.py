"""
Environment Verification Script
Checks that all required libraries are installed and verifies .env configuration.
"""

import sys
import os
from dotenv import load_dotenv

def test_imports():
    print("=" * 50)
    print("Testing Package Imports...")
    print("=" * 50)

    packages = [
        ("PyMuPDF", "pymupdf"),
        ("Sentence Transformers", "sentence_transformers"),
        ("FAISS", "faiss"),
        ("NumPy", "numpy"),
        ("Groq SDK", "groq"),
        ("Python Dotenv", "dotenv"),
        ("Streamlit", "streamlit"),
    ]

    all_passed = True
    for display_name, import_name in packages:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "installed")
            print(f" [PASS] {display_name:<25} (version: {version})")
        except ImportError as e:
            print(f" [FAIL] {display_name:<25} Error: {e}")
            all_passed = False

    return all_passed


def test_groq_api_key():
    print("\n" + "=" * 50)
    print("Testing Groq API Key Configuration...")
    print("=" * 50)

    # Load environment variables from .env
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print(" [WARN] GROQ_API_KEY not found in .env file.")
        print("        Create a .env file and add: GROQ_API_KEY=your_actual_key")
        return False

    if api_key == "gsk_your_groq_api_key_here" or "your_" in api_key:
        print(" [WARN] GROQ_API_KEY is still using the placeholder value!")
        print("        Please replace it with your real key from https://console.groq.com/keys")
        return False

    if api_key.startswith("gsk_"):
        masked = api_key[:7] + "..." + api_key[-4:]
        print(f" [PASS] GROQ_API_KEY detected and formatted correctly ({masked})")
        return True
    else:
        print(" [WARN] GROQ_API_KEY does not start with expected prefix 'gsk_'")
        return False


if __name__ == "__main__":
    imports_ok = test_imports()
    key_ok = test_groq_api_key()

    print("\n" + "=" * 50)
    if imports_ok:
        print("Environment dependencies are ready!")
    else:
        print("Some dependencies failed to import. Check error messages above.")
    print("=" * 50)
