#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WOLFRAM INTEGRATION DIAGNOSE SCRIPT
Findet heraus warum "Loading: NO" angezeigt wird
"""

import os
import sys

print("=" * 60)
print("🔍 WOLFRAM INTEGRATION DIAGNOSE")
print("=" * 60)

# 1. Check Python Version
print(f"\n1️⃣ Python Version: {sys.version}")

# 2. Check .env file
print("\n2️⃣ Checking .env file...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    app_id = os.getenv("WOLFRAM_APP_ID")
    debug_mode = os.getenv("WOLFRAM_DEBUG", "false")
    print(f"   ✅ .env geladen")
    print(f"   WOLFRAM_APP_ID = {app_id[:10]}..." if app_id else "   ❌ WOLFRAM_APP_ID nicht gesetzt")
    print(f"   WOLFRAM_DEBUG = {debug_mode}")
except Exception as e:
    print(f"   ❌ Fehler beim Laden der .env: {e}")

# 3. Check wolframalpha installation
print("\n3️⃣ Checking wolframalpha package...")
try:
    import wolframalpha
    print(f"   ✅ wolframalpha installiert (Version: {wolframalpha.__version__ if hasattr(wolframalpha, '__version__') else 'unknown'})")
except ImportError as e:
    print(f"   ❌ wolframalpha NICHT installiert!")
    print(f"      Fehler: {e}")
    print(f"      Lösung: pip install wolframalpha")

# 4. Test Wolfram Client
print("\n4️⃣ Testing Wolfram Client initialization...")
try:
    import wolframalpha
    app_id = os.getenv("WOLFRAM_APP_ID")
    if app_id and app_id != "your_wolfram_app_id_here":
        client = wolframalpha.Client(app_id)
        print(f"   ✅ Wolfram Client erstellt")
        
        # Quick API test
        print("\n5️⃣ Testing Wolfram API...")
        try:
            import urllib.parse
            import urllib.request
            import xml.etree.ElementTree as ET
            
            query = "capital of germany"
            encoded_query = urllib.parse.quote(query)
            url = f"http://api.wolframalpha.com/v2/query?input={encoded_query}&appid={app_id}&format=plaintext"
            
            print(f"   Testing query: '{query}'")
            with urllib.request.urlopen(url, timeout=5) as response:
                xml_data = response.read().decode('utf-8')
                root = ET.fromstring(xml_data)
                
                success = root.get('success') == 'true'
                if success:
                    print(f"   ✅ API Response: SUCCESS")
                    # Extract first answer
                    for pod in root.findall('.//pod'):
                        for subpod in pod.findall('.//subpod'):
                            plaintext_elem = subpod.find('plaintext')
                            if plaintext_elem is not None and plaintext_elem.text:
                                answer = plaintext_elem.text.strip()
                                if answer:
                                    print(f"   Answer: {answer[:50]}...")
                                    break
                else:
                    print(f"   ❌ API Response: FAILED")
                    error = root.get('error')
                    if error:
                        print(f"   Error: {error}")
                        
        except Exception as e:
            print(f"   ❌ API Test failed: {type(e).__name__}: {str(e)}")
            
    else:
        print(f"   ❌ Wolfram App ID nicht konfiguriert")
except Exception as e:
    print(f"   ❌ Client-Erstellung fehlgeschlagen: {e}")

# 6. Check backend import
print("\n6️⃣ Testing backend import...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
    from k_assistant_main_v7_wolfram import KAssistant, WOLFRAM_INTEGRATION
    print(f"   ✅ Backend import erfolgreich")
    print(f"   WOLFRAM_INTEGRATION = {WOLFRAM_INTEGRATION}")
    
    # Check if WolframProver is loaded
    assistant = KAssistant()
    wolfram_loaded = any(p.name == "Wolfram|Alpha Orakel" for p in assistant.core.provers)
    print(f"   Wolfram Prover geladen: {'✅ JA' if wolfram_loaded else '❌ NEIN'}")
    
    if wolfram_loaded:
        print("\n   🎉 WOLFRAM IST AKTIV!")
    else:
        print("\n   ⚠️ WOLFRAM PROVER WURDE NICHT GELADEN!")
        
except Exception as e:
    print(f"   ❌ Backend import fehlgeschlagen: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("📊 DIAGNOSE ABGESCHLOSSEN")
print("=" * 60)

# Recommendations
print("\n💡 EMPFEHLUNGEN:")
print("1. Stelle sicher dass 'pip install wolframalpha' ausgeführt wurde")
print("2. Prüfe ob die WOLFRAM_APP_ID in .env korrekt ist")
print("3. Starte das Backend neu nach der Installation")
print("4. Führe 'python start_wolfram.py' aus für Setup-Hilfe")
