#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAK-GAL Suite Starter Script mit Wolfram-Integration
Automatisiert den Setup-Prozess und führt Benutzer durch die erste Nutzung

Features:
- Automatische Abhängigkeits-Prüfung
- Interaktives Setup
- Konfigurationshilfe
- Erste-Schritte-Tutorial
- Troubleshooting-Assistent

Version: 1.0 - Benutzerfreundlicher Einstieg
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class Colors:
    """ANSI-Farbcodes für bessere Darstellung"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def colored_print(message, color=Colors.ENDC):
    """Druckt farbige Nachrichten"""
    print(f"{color}{message}{Colors.ENDC}")

def print_banner():
    """Druckt das HAK-GAL Banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    🚀 HAK-GAL Suite mit Wolfram|Alpha Integration           ║
    ║                                                              ║
    ║    Hybrid AI Framework für verifizierbares Reasoning        ║
    ║    + Realwelt-Wissen durch Wolfram|Alpha                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    colored_print(banner, Colors.CYAN)

def check_python_version():
    """Prüft Python-Version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        colored_print("❌ Python 3.8+ erforderlich", Colors.RED)
        colored_print(f"   Aktuelle Version: {version.major}.{version.minor}.{version.micro}", Colors.YELLOW)
        return False
    
    colored_print(f"✅ Python {version.major}.{version.minor}.{version.micro}", Colors.GREEN)
    return True

def check_dependencies():
    """Prüft installierte Abhängigkeiten"""
    colored_print("\n🔍 Prüfe Abhängigkeiten...", Colors.BLUE)
    
    required_packages = {
        'z3-solver': 'z3',
        'lark': 'lark',
        'openai': 'openai',
        'python-dotenv': 'dotenv',
        'wolframalpha': 'wolframalpha'
    }
    
    optional_packages = {
        'sentence-transformers': 'sentence_transformers',
        'faiss-cpu': 'faiss',
        'numpy': 'numpy',
        'google-generativeai': 'google.generativeai'
    }
    
    missing_required = []
    missing_optional = []
    
    # Erforderliche Pakete prüfen
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            colored_print(f"✅ {package_name}", Colors.GREEN)
        except ImportError:
            colored_print(f"❌ {package_name}", Colors.RED)
            missing_required.append(package_name)
    
    # Optionale Pakete prüfen
    for package_name, import_name in optional_packages.items():
        try:
            __import__(import_name)
            colored_print(f"✅ {package_name} (optional)", Colors.GREEN)
        except ImportError:
            colored_print(f"⚠️ {package_name} (optional)", Colors.YELLOW)
            missing_optional.append(package_name)
    
    return missing_required, missing_optional

def install_dependencies(missing_packages):
    """Installiert fehlende Abhängigkeiten"""
    if not missing_packages:
        return True
    
    colored_print(f"\n📦 Installiere {len(missing_packages)} fehlende Pakete...", Colors.BLUE)
    
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + missing_packages
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            colored_print("✅ Installation erfolgreich", Colors.GREEN)
            return True
        else:
            colored_print("❌ Installation fehlgeschlagen", Colors.RED)
            colored_print(f"Fehler: {result.stderr}", Colors.RED)
            return False
            
    except Exception as e:
        colored_print(f"❌ Installationsfehler: {e}", Colors.RED)
        return False

def check_env_file():
    """Prüft .env Datei und hilft bei der Konfiguration"""
    colored_print("\n⚙️ Prüfe Konfiguration...", Colors.BLUE)
    
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if not env_path.exists():
        if env_example_path.exists():
            colored_print("📄 Erstelle .env Datei aus Vorlage...", Colors.YELLOW)
            try:
                with open(env_example_path, 'r') as f:
                    content = f.read()
                with open(env_path, 'w') as f:
                    f.write(content)
                colored_print("✅ .env Datei erstellt", Colors.GREEN)
            except Exception as e:
                colored_print(f"❌ Fehler beim Erstellen der .env Datei: {e}", Colors.RED)
                return False
        else:
            colored_print("❌ Keine .env.example Vorlage gefunden", Colors.RED)
            return False
    
    # .env Datei prüfen
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        wolfram_app_id = os.getenv("WOLFRAM_APP_ID")
        if not wolfram_app_id or wolfram_app_id == "your_wolfram_app_id_here":
            colored_print("⚠️ Wolfram App ID nicht konfiguriert", Colors.YELLOW)
            return "partial"
        
        colored_print("✅ Wolfram App ID konfiguriert", Colors.GREEN)
        return True
        
    except Exception as e:
        colored_print(f"❌ Fehler beim Lesen der .env Datei: {e}", Colors.RED)
        return False

def interactive_wolfram_setup():
    """Interaktives Wolfram-Setup"""
    colored_print("\n🔮 Wolfram|Alpha Setup", Colors.CYAN)
    print("Um die Wolfram-Integration zu nutzen, benötigen Sie eine kostenlose App ID.")
    print("Diese erhalten Sie auf: https://developer.wolframalpha.com/portal/myapps/")
    print()
    
    response = input("Haben Sie bereits eine Wolfram App ID? (j/n): ").lower()
    
    if response == 'j':
        app_id = input("Bitte geben Sie Ihre App ID ein: ").strip()
        if app_id and len(app_id) > 10:
            try:
                # .env Datei aktualisieren
                env_path = Path(".env")
                with open(env_path, 'r') as f:
                    content = f.read()
                
                # App ID ersetzen
                updated_content = content.replace(
                    "WOLFRAM_APP_ID=your_wolfram_app_id_here",
                    f"WOLFRAM_APP_ID={app_id}"
                )
                
                with open(env_path, 'w') as f:
                    f.write(updated_content)
                
                colored_print("✅ Wolfram App ID gespeichert", Colors.GREEN)
                return True
                
            except Exception as e:
                colored_print(f"❌ Fehler beim Speichern: {e}", Colors.RED)
                return False
        else:
            colored_print("❌ Ungültige App ID", Colors.RED)
            return False
    else:
        colored_print("\n📘 So erhalten Sie eine kostenlose Wolfram App ID:", Colors.BLUE)
        print("1. Besuchen Sie: https://developer.wolframalpha.com/portal/myapps/")
        print("2. Erstellen Sie ein kostenloses Konto")
        print("3. Klicken Sie auf 'Get an AppID'")
        print("4. Wählen Sie 'Personal Use'")
        print("5. Kopieren Sie die generierte App ID")
        print("6. Starten Sie dieses Script erneut")
        return False

def run_basic_test():
    """Führt einen Basis-Test des Systems durch"""
    colored_print("\n🧪 Führe Basis-Test durch...", Colors.BLUE)
    
    try:
        # Prüfe ob Hauptmodul importierbar ist
        sys.path.insert(0, str(Path("backend")))
        
        # Versuche neue Version zu importieren
        try:
            from k_assistant_main_v7_wolfram import KAssistant
            colored_print("✅ Wolfram-Integration verfügbar", Colors.GREEN)
            wolfram_available = True
        except ImportError:
            from k_assistant_main import KAssistant
            colored_print("⚠️ Fallback auf Standard-Version", Colors.YELLOW)
            wolfram_available = False
        
        # Kurzer Initialisierungstest
        test_kb_path = Path("startup_test.kb")
        assistant = KAssistant(str(test_kb_path))
        
        prover_count = len(assistant.core.provers)
        colored_print(f"✅ System initialisiert mit {prover_count} Provern", Colors.GREEN)
        
        # Test-KB aufräumen
        if test_kb_path.exists():
            test_kb_path.unlink()
        
        return wolfram_available
        
    except Exception as e:
        colored_print(f"❌ Basis-Test fehlgeschlagen: {e}", Colors.RED)
        return False

def show_usage_examples(wolfram_available):
    """Zeigt Nutzungsbeispiele"""
    colored_print("\n📚 Erste Schritte", Colors.CYAN)
    
    if wolfram_available:
        script_name = "k_assistant_main_v7_wolfram.py"
        colored_print("🔮 Wolfram-Integration aktiv!", Colors.GREEN)
    else:
        script_name = "k_assistant_main.py" 
        colored_print("🧠 Standard-Modus (ohne Wolfram)", Colors.YELLOW)
    
    print(f"""
🚀 System starten:
   python backend/{script_name}

💡 Beispiel-Anfragen:""")
    
    if wolfram_available:
        print("""
   🌍 Realwelt-Wissen:
   k-assistant> ask was ist die hauptstadt von deutschland
   k-assistant> ask_raw HauptstadtVon(Deutschland, x).
   
   🧮 Mathematik:
   k-assistant> ask was ist das integral von x^2
   k-assistant> ask_raw Integral(x^2, x).
   
   🌤️ Wetter:
   k-assistant> ask wie ist das wetter in berlin""")
    
    print("""
   🧠 Logisches Reasoning:
   k-assistant> add_raw IstKritisch(System).
   k-assistant> ask_raw IstKritisch(System).
   
   📊 System-Status:
   k-assistant> status
   k-assistant> help""")
    
    if wolfram_available:
        print("""
   🔮 Wolfram-spezifisch:
   k-assistant> wolfram_stats
   k-assistant> add_oracle MeinPrädikat""")

def main():
    """Haupt-Setup-Funktion"""
    print_banner()
    
    # Python-Version prüfen
    if not check_python_version():
        sys.exit(1)
    
    # Abhängigkeiten prüfen
    missing_required, missing_optional = check_dependencies()
    
    if missing_required:
        colored_print(f"\n⚠️ {len(missing_required)} erforderliche Pakete fehlen", Colors.YELLOW)
        response = input("Sollen diese automatisch installiert werden? (j/n): ").lower()
        
        if response == 'j':
            if not install_dependencies(missing_required):
                colored_print("❌ Installation fehlgeschlagen. Setup abgebrochen.", Colors.RED)
                sys.exit(1)
        else:
            colored_print("Setup abgebrochen. Installieren Sie die Pakete manuell:", Colors.YELLOW)
            colored_print(f"pip install {' '.join(missing_required)}", Colors.CYAN)
            sys.exit(1)
    
    # Optionale Pakete
    if missing_optional:
        colored_print(f"\n💡 {len(missing_optional)} optionale Pakete verfügbar", Colors.BLUE)
        print("Diese verbessern die Funktionalität (RAG, erweiterte LLM-Provider)")
        response = input("Sollen diese installiert werden? (j/n): ").lower()
        
        if response == 'j':
            install_dependencies(missing_optional)
    
    # Konfiguration prüfen
    env_status = check_env_file()
    
    if env_status == "partial":
        response = input("Möchten Sie die Wolfram-Integration jetzt konfigurieren? (j/n): ").lower()
        if response == 'j':
            interactive_wolfram_setup()
    
    # Basis-Test
    wolfram_available = run_basic_test()
    
    # Nutzungsbeispiele
    show_usage_examples(wolfram_available)
    
    # Abschluss
    colored_print("\n🎉 Setup abgeschlossen!", Colors.GREEN)
    
    if wolfram_available:
        colored_print("🔮 Wolfram|Alpha Integration ist einsatzbereit", Colors.CYAN)
        print("   Testen Sie mit: python demo_wolfram_integration.py")
    
    print("\n📖 Weitere Ressourcen:")
    print("   • Installationsanleitung: WOLFRAM_INSTALLATION.md")
    print("   • Feature-Übersicht: WOLFRAM_FEATURES.md")
    print("   • Tests ausführen: python test_wolfram_integration.py")
    
    colored_print("\nViel Spaß mit HAK-GAL! 🚀", Colors.BOLD)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        colored_print("\n\n⏹️ Setup von Benutzer abgebrochen", Colors.YELLOW)
        sys.exit(0)
    except Exception as e:
        colored_print(f"\n❌ Unerwarteter Fehler: {e}", Colors.RED)
        import traceback
        traceback.print_exc()
        sys.exit(1)
