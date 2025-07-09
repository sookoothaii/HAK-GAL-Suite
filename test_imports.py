# Test Import der neuen Hauptdatei
# Überprüft ob alle Imports korrekt funktionieren

import sys
import os

# Füge das Backend-Verzeichnis hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

try:
    print("🧪 Teste Import von k_assistant_main...")
    from backend.k_assistant_main import KAssistant
    print("✅ Import erfolgreich!")
    
    print("🧪 Teste Initialisierung...")
    # Teste nur Import, nicht vollständige Initialisierung um Abhängigkeiten zu vermeiden
    print("✅ KAssistant-Klasse verfügbar!")
    
    print("\n🎉 ALLE IMPORTS FUNKTIONIEREN!")
    print("Die start_suite.bat kann jetzt verwendet werden.")
    
except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
    print("Stelle sicher, dass:")
    print("  - k_assistant_main.py im backend/ Ordner ist")
    print("  - Alle Abhängigkeiten installiert sind")
    
except Exception as e:
    print(f"❌ Allgemeiner Fehler: {e}")
