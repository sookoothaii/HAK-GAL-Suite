#!/usr/bin/env python3
"""
WOLFRAM INTEGRATION INSTALLER & FIXER
Installiert und aktiviert die Wolfram-Integration
"""

import subprocess
import sys
import os

print("=" * 50)
print("🔧 HAK-GAL WOLFRAM INTEGRATION FIXER")
print("=" * 50)

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Erfolgreich!")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} - Fehlgeschlagen!")
            if result.stderr:
                print(f"Fehler: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

# Step 1: Install wolframalpha
print("\n📦 SCHRITT 1: Installiere wolframalpha Paket")
if not run_command(f"{sys.executable} -m pip install wolframalpha", "Installation"):
    print("\nAlternative: Versuche mit --user flag...")
    if not run_command(f"{sys.executable} -m pip install --user wolframalpha", "Installation mit --user"):
        print("\n❌ Installation fehlgeschlagen. Bitte manuell installieren:")
        print(f"   {sys.executable} -m pip install wolframalpha")
        sys.exit(1)

# Step 2: Verify import
print("\n🔍 SCHRITT 2: Verifiziere Installation")
try:
    import wolframalpha
    print("✅ wolframalpha erfolgreich importiert!")
except ImportError:
    print("❌ Import fehlgeschlagen!")
    sys.exit(1)

# Step 3: Check .env
print("\n📄 SCHRITT 3: Prüfe .env Konfiguration")
try:
    from dotenv import load_dotenv
    load_dotenv()
    app_id = os.getenv("WOLFRAM_APP_ID")
    if app_id and app_id != "your_wolfram_app_id_here":
        print(f"✅ Wolfram App ID gefunden: {app_id[:10]}...")
    else:
        print("⚠️ Wolfram App ID nicht konfiguriert!")
        print("\nBitte füge deine App ID zur .env Datei hinzu:")
        print("WOLFRAM_APP_ID=deine_app_id_hier")
        print("\nKostenlose App ID: https://developer.wolframalpha.com/portal/myapps/")
except Exception as e:
    print(f"❌ .env Fehler: {e}")

# Step 4: Test backend import
print("\n🧪 SCHRITT 4: Teste Backend-Integration")
try:
    # Add backend to path
    backend_path = os.path.join(os.path.dirname(__file__), 'backend')
    if os.path.exists(backend_path):
        sys.path.insert(0, backend_path)
    
    from k_assistant_main_v7_wolfram import WOLFRAM_INTEGRATION, KAssistant
    
    if WOLFRAM_INTEGRATION:
        print("✅ WOLFRAM_INTEGRATION = True")
        
        # Check if WolframProver is loaded
        assistant = KAssistant()
        wolfram_loaded = any(p.name == "Wolfram|Alpha Orakel" for p in assistant.core.provers)
        
        if wolfram_loaded:
            print("✅ WolframProver erfolgreich geladen!")
            print("\n🎉 WOLFRAM INTEGRATION IST AKTIV!")
        else:
            print("⚠️ WolframProver wurde nicht zum Portfolio hinzugefügt")
    else:
        print("❌ WOLFRAM_INTEGRATION = False")
        print("   Mögliche Gründe:")
        print("   - Wolfram App ID fehlt in .env")
        print("   - Import-Fehler beim Start")
        
except Exception as e:
    print(f"❌ Backend-Test fehlgeschlagen: {e}")

print("\n" + "=" * 50)
print("📊 ZUSAMMENFASSUNG")
print("=" * 50)

print("\n✅ Was zu tun ist:")
print("1. Stelle sicher dass deine WOLFRAM_APP_ID in .env korrekt ist")
print("2. Starte das HAK-GAL Backend neu")
print("3. Du solltest jetzt 'Loading: YES' sehen")
print("4. Wolfram-Befehle sind verfügbar:")
print("   - wolfram_stats")
print("   - ask wie ist das wetter in berlin")
print("   - ask_raw HauptstadtVon(Deutschland).")

print("\n💡 Tipp: Führe 'python test_wolfram_integration.py' aus zum Testen")
