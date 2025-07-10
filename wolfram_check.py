import os
import sys

print("WOLFRAM QUICK CHECK")
print("=" * 40)

# Add backend to path
sys.path.insert(0, r"D:\MCP Mods\HAK_GAL_SUITE\backend")

# Check 1: Is wolframalpha installed?
try:
    import wolframalpha
    print("✅ wolframalpha package is installed")
except ImportError:
    print("❌ wolframalpha package NOT installed!")
    print("   Run: pip install wolframalpha")
    sys.exit(1)

# Check 2: Load .env
try:
    os.chdir(r"D:\MCP Mods\HAK_GAL_SUITE")
    from dotenv import load_dotenv
    load_dotenv()
    app_id = os.getenv("WOLFRAM_APP_ID")
    print(f"✅ App ID loaded: {app_id[:10]}..." if app_id else "❌ No App ID")
except Exception as e:
    print(f"❌ Error loading .env: {e}")

# Check 3: Import backend and check WOLFRAM_INTEGRATION
try:
    # First check what happens during import
    print("\nImporting k_assistant_main_v7_wolfram...")
    
    # Temporarily redirect stdout to capture prints
    from io import StringIO
    import contextlib
    
    captured_output = StringIO()
    with contextlib.redirect_stdout(captured_output):
        from k_assistant_main_v7_wolfram import WOLFRAM_INTEGRATION
    
    output = captured_output.getvalue()
    if output:
        print("Import output:")
        print(output)
    
    print(f"\nWOLFRAM_INTEGRATION = {WOLFRAM_INTEGRATION}")
    
    if not WOLFRAM_INTEGRATION:
        print("\n❌ PROBLEM: WOLFRAM_INTEGRATION is False!")
        print("This means either:")
        print("- wolframalpha module import failed")
        print("- WOLFRAM_APP_ID is not set correctly")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")
