#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wolfram-Integration Demo Script
Zeigt die Leistungsfähigkeit der neuen HAK-GAL + Wolfram|Alpha Integration

Dieses Script demonstriert:
1. Automatische Oracle-Erkennung
2. Intelligente Prover-Auswahl
3. Realwelt-Wissensabfragen
4. Mathematische Berechnungen
5. Multi-Step Reasoning mit Wolfram

Version: 1.0 - Interaktive Demonstration
"""

import os
import sys
import time
from pathlib import Path

# Pfad-Setup
script_dir = Path(__file__).parent
backend_dir = script_dir / "backend"
sys.path.insert(0, str(backend_dir))

def print_section(title):
    """Druckt eine formatierte Sektion"""
    print("\n" + "=" * 60)
    print(f"🎯 {title}")
    print("=" * 60)

def print_query(description, query, is_raw=False):
    """Druckt eine formatierte Anfrage"""
    query_type = "ask_raw" if is_raw else "ask"
    print(f"\n💡 {description}")
    print(f"➤ {query_type} {query}")
    print("-" * 40)

def wait_for_enter(message="Drücken Sie Enter um fortzufahren..."):
    """Wartet auf Benutzer-Eingabe"""
    input(f"\n⏸️  {message}")

class WolframDemo:
    """Haupt-Demonstrationsklasse"""
    
    def __init__(self):
        self.assistant = None
        self.demo_queries = []
        self._setup_demo_queries()
    
    def _setup_demo_queries(self):
        """Vorbereitung der Demo-Anfragen"""
        self.demo_queries = [
            # Geografische Wissensabfragen
            {
                "category": "Geografisches Wissen",
                "queries": [
                    ("Hauptstadt von Deutschland", "was ist die hauptstadt von deutschland"),
                    ("Hauptstadt (roh)", "HauptstadtVon(Deutschland, x).", True),
                    ("Bevölkerung von Berlin", "wie viele einwohner hat berlin"),
                    ("Bevölkerung (roh)", "Bevölkerung(Berlin, x).", True),
                ]
            },
            
            # Mathematische Berechnungen  
            {
                "category": "Mathematische Berechnungen",
                "queries": [
                    ("Integration", "was ist das integral von x^2"),
                    ("Integration (roh)", "Integral(x^2, x).", True),
                    ("Ableitung", "was ist die ableitung von sin(x)"),
                    ("Gleichung lösen", "löse die gleichung x^2 - 4 = 0"),
                ]
            },
            
            # Realzeit-Daten
            {
                "category": "Realzeit & Wetter",
                "queries": [
                    ("Wetter in Berlin", "wie ist das wetter in berlin"),
                    ("Wetter (roh)", "WetterIn(Berlin, x).", True),
                    ("Aktuelle Zeit", "wie spät ist es in new york"),
                    ("Zeitzone", "ZeitzoneVon(Tokyo, x).", True),
                ]
            },
            
            # Vergleiche und Berechnungen
            {
                "category": "Vergleiche & Logik",
                "queries": [
                    ("Zahlenvergleich", "ist 10 größer als 5"),
                    ("Vergleich (roh)", "IstGroesserAls(10, 5).", True),
                    ("Währungsumrechnung", "wie viel sind 100 dollar in euro"),
                    ("Umrechnung (roh)", "Umrechnung(100_Dollar, Euro, x).", True),
                ]
            }
        ]
    
    def initialize_system(self):
        """Initialisiert das HAK-GAL System mit Wolfram-Integration"""
        print_section("System-Initialisierung")
        
        try:
            from k_assistant_main_v7_wolfram import KAssistant
            print("✅ Lade HAK-GAL System mit Wolfram-Integration...")
            
            # Temporäre KB für Demo
            demo_kb_path = script_dir / "demo_wolfram.kb"
            self.assistant = KAssistant(str(demo_kb_path))
            
            print(f"✅ System initialisiert mit {len(self.assistant.core.provers)} Provern")
            
            # Prover anzeigen
            prover_names = [p.name for p in self.assistant.core.provers]
            print(f"📋 Verfügbare Prover: {', '.join(prover_names)}")
            
            # Wolfram-Status prüfen
            wolfram_available = any("Wolfram" in name for name in prover_names)
            if wolfram_available:
                print("🔮 Wolfram|Alpha Integration: AKTIV")
            else:
                print("⚠️ Wolfram|Alpha Integration: NICHT KONFIGURIERT")
                print("   Hinweis: Stellen Sie sicher, dass WOLFRAM_APP_ID in .env gesetzt ist")
            
            return True
            
        except ImportError as e:
            print(f"❌ Import-Fehler: {e}")
            print("   Stellen Sie sicher, dass die Wolfram-Integration installiert ist")
            return False
        except Exception as e:
            print(f"❌ Initialisierungsfehler: {e}")
            return False
    
    def demonstrate_oracle_detection(self):
        """Demonstriert die automatische Oracle-Erkennung"""
        print_section("Automatische Oracle-Erkennung")
        
        print("Der ComplexityAnalyzer erkennt automatisch, welche Anfragen")
        print("externes Wissen (Oracle) benötigen:")
        
        # Teste verschiedene Formel-Typen
        test_formulas = [
            ("HauptstadtVon(Deutschland, x).", "Geografisches Wissen"),
            ("IstKritisch(System).", "Reine Logik"),
            ("Integral(x^2, x).", "Mathematik"),
            ("WetterIn(Berlin, x).", "Realzeit-Daten"),
            ("all x (IstSystem(x) -> IstLegacy(x)).", "Quantifizierte Logik")
        ]
        
        for formula, category in test_formulas:
            try:
                report = self.assistant.core.complexity_analyzer.analyze(formula)
                
                oracle_status = "🔮 ORACLE" if report.requires_oracle else "🧠 LOGIK"
                print(f"\n{oracle_status} | {formula}")
                print(f"   Kategorie: {category}")
                print(f"   Query-Typ: {report.query_type.value}")
                print(f"   Begründung: {report.reasoning}")
                print(f"   Empfohlene Prover: {', '.join(report.recommended_provers)}")
                
            except Exception as e:
                print(f"❌ Fehler bei Analyse von {formula}: {e}")
    
    def run_interactive_demo(self):
        """Führt interaktive Demonstration durch"""
        print_section("Interaktive Wolfram-Demo")
        
        for category_data in self.demo_queries:
            category = category_data["category"]
            queries = category_data["queries"]
            
            print(f"\n🎪 Kategorie: {category}")
            print("=" * 40)
            
            for description, query, *args in queries:
                is_raw = args[0] if args else False
                
                print_query(description, query, is_raw)
                
                # Benutzer fragen ob Query ausgeführt werden soll
                response = input("Möchten Sie diese Anfrage ausführen? (j/n/s=skip category): ").lower()
                
                if response == 's':
                    print("⏭️ Kategorie übersprungen")
                    break
                elif response == 'n':
                    print("⏭️ Anfrage übersprungen")
                    continue
                
                # Query ausführen
                try:
                    start_time = time.time()
                    
                    if is_raw:
                        self.assistant.ask_raw(query)
                    else:
                        self.assistant.ask(query)
                    
                    duration = time.time() - start_time
                    print(f"\n⏱️ Ausführungszeit: {duration:.2f} Sekunden")
                    
                except Exception as e:
                    print(f"❌ Fehler bei Anfrage: {e}")
                
                wait_for_enter()
    
    def show_system_stats(self):
        """Zeigt System-Statistiken"""
        print_section("System-Statistiken")
        
        try:
            # Basis-Status
            self.assistant.status()
            
            # Portfolio-Performance
            portfolio_stats = self.assistant.core.get_portfolio_stats()
            if portfolio_stats["performance"]:
                print("\n📊 Portfolio-Performance:")
                for prover_name, perf in portfolio_stats["performance"].items():
                    usage = portfolio_stats["usage_count"].get(prover_name, 0)
                    print(f"   {prover_name}: {perf['success_rate']:.1%} Erfolg, "
                          f"{perf['avg_duration']:.2f}s ⌀, {usage}x verwendet")
            
            # Wolfram-spezifische Stats
            print("\n🔮 Wolfram|Alpha Statistiken:")
            self.assistant.wolfram_stats()
            
        except Exception as e:
            print(f"❌ Fehler beim Abrufen der Statistiken: {e}")
    
    def cleanup(self):
        """Räumt Demo-Dateien auf"""
        try:
            demo_kb_path = script_dir / "demo_wolfram.kb"
            if demo_kb_path.exists():
                demo_kb_path.unlink()
                print("✅ Demo-Dateien aufgeräumt")
        except Exception as e:
            print(f"⚠️ Fehler beim Aufräumen: {e}")

def main():
    """Haupt-Demo-Funktion"""
    print("🚀 HAK-GAL + Wolfram|Alpha Integration Demo")
    print("=" * 60)
    print("Diese Demo zeigt die neuen Fähigkeiten der Wolfram-Integration:")
    print("• Automatische Oracle-Erkennung")
    print("• Intelligente Prover-Auswahl")  
    print("• Realwelt-Wissensabfragen")
    print("• Mathematische Berechnungen")
    print("• Performance-Optimierung")
    
    demo = WolframDemo()
    
    try:
        # System initialisieren
        if not demo.initialize_system():
            print("\n❌ Demo kann nicht fortgesetzt werden.")
            return
        
        wait_for_enter("System bereit. Drücken Sie Enter um die Demo zu starten...")
        
        # Oracle-Erkennung demonstrieren
        demo.demonstrate_oracle_detection()
        wait_for_enter()
        
        # Interaktive Demo
        demo.run_interactive_demo()
        
        # System-Statistiken anzeigen
        demo.show_system_stats()
        
        print_section("Demo abgeschlossen")
        print("🎉 Die Wolfram-Integration ist erfolgreich demonstriert!")
        print("💡 Sie können jetzt eigene Anfragen mit dem System testen.")
        print("\nStarten Sie das System mit:")
        print("   python backend/k_assistant_main_v7_wolfram.py")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo von Benutzer abgebrochen")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.cleanup()

if __name__ == "__main__":
    main()
