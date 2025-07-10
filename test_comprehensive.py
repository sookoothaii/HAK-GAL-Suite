#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 VOLLSTÄNDIGE TEST-SUITE für HAK-GAL FUNKTIONALE CONSTRAINTS
===============================================================

Diese Test-Suite validiert die Lösung für das ursprüngliche Problem:
- KB: Einwohner(Rom,2873000).
- Anfrage: Einwohner(Rom,283000).
- Alte Antwort: "Unbekannt" ❌
- Neue Antwort: "FALSCH" ✅

NEUE FEATURES:
1. FunctionalConstraintProver - Spezialisierter Prover für funktionale Constraints
2. Erweiterte Z3-Integration mit Gleichheits-Handling  
3. Automatische funktionale Constraints beim Start
4. Archon-Prime Portfolio-Management

GETESTETE SZENARIEN:
✅ Einwohner-Funktionalität
✅ Hauptstadt-Funktionalität  
✅ Constraint-Verletzungs-Erkennung
✅ Portfolio-Prover-Reihenfolge
✅ Cache-Performance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def run_comprehensive_test():
    print("🚀 HAK-GAL FUNKTIONALE CONSTRAINTS - VOLLTEST")
    print("=" * 55)
    
    try:
        from k_assistant_main_v7_wolfram import KAssistant
        print("✅ System erfolgreich importiert")
    except Exception as e:
        print(f"❌ Import-Fehler: {e}")
        return
    
    # Erstelle Test-Instanz
    print("\n📋 Erstelle Test-Wissensbasis...")
    assistant = KAssistant("test_comprehensive.kb")
    
    print("🔍 Prover-Portfolio:")
    for i, prover in enumerate(assistant.core.provers):
        print(f"   [{i}] {prover.name}")
    
    # TEST 1: EINWOHNER-FUNKTIONALITÄT
    print("\n" + "="*50)
    print("🧪 TEST 1: EINWOHNER-FUNKTIONALITÄT")
    print("="*50)
    
    print("\n✅ Schritt 1.1: Füge korrekten Einwohner-Fakt hinzu")
    assistant.add_raw("Einwohner(Rom,2873000).")
    
    print("\n🔍 Schritt 1.2: Teste korrekten Fakt (Erwartet: JA)")
    print("Anfrage: Einwohner(Rom,2873000).")
    assistant.ask_raw("Einwohner(Rom,2873000).")
    
    print("\n🚨 Schritt 1.3: KRITISCHER TEST - Teste widersprüchlichen Fakt")
    print("Anfrage: Einwohner(Rom,283000).")
    print("Erwartet: NEIN/FALSCH (wegen funktionalem Constraint)")
    assistant.ask_raw("Einwohner(Rom,283000).")
    
    # TEST 2: HAUPTSTADT-FUNKTIONALITÄT  
    print("\n" + "="*50)
    print("🧪 TEST 2: HAUPTSTADT-FUNKTIONALITÄT")
    print("="*50)
    
    print("\n✅ Schritt 2.1: Füge Hauptstadt hinzu")
    assistant.add_raw("Hauptstadt(Deutschland,Berlin).")
    
    print("\n🔍 Schritt 2.2: Teste korrekten Fakt (Erwartet: JA)")
    assistant.ask_raw("Hauptstadt(Deutschland,Berlin).")
    
    print("\n🚨 Schritt 2.3: Teste widersprüchlichen Fakt (Erwartet: NEIN)")
    assistant.ask_raw("Hauptstadt(Deutschland,München).")
    
    # TEST 3: CONSTRAINT-VERLETZUNG BEI HINZUFÜGUNG
    print("\n" + "="*50)
    print("🧪 TEST 3: CONSTRAINT-VERLETZUNG BEI HINZUFÜGUNG")
    print("="*50)
    
    print("\n🚨 Schritt 3.1: Versuche inkonsistenten Fakt hinzuzufügen")
    print("Versuch: Einwohner(Rom,999999).")
    print("Erwartet: ABLEHNUNG wegen Inkonsistenz")
    assistant.add_raw("Einwohner(Rom,999999).")
    
    # TEST 4: PORTFOLIO-PERFORMANCE
    print("\n" + "="*50)
    print("🧪 TEST 4: PORTFOLIO-PERFORMANCE")
    print("="*50)
    
    assistant.status()
    
    # TEST 5: WISSENSBASIS-ANALYSE
    print("\n" + "="*50)
    print("🧪 TEST 5: WISSENSBASIS-ANALYSE")
    print("="*50)
    
    data = assistant.show()
    print("\n📊 FUNKTIONALE CONSTRAINTS in der KB:")
    functional_count = 0
    for fact in data['permanent_knowledge']:
        if (' = ' in fact and 'all x all y all z' in fact):
            print(f"   ✅ {fact}")
            functional_count += 1
    
    print(f"\n📈 STATISTIKEN:")
    print(f"   - Funktionale Constraints: {functional_count}")
    print(f"   - Gesamte Kernregeln: {len(data['permanent_knowledge'])}")
    print(f"   - Prover im Portfolio: {len(assistant.core.provers)}")
    
    # Cleanup
    try:
        os.remove("test_comprehensive.kb")
        print("\n🧹 Test-Datei gelöscht")
    except:
        pass
    
    print("\n" + "="*50)
    print("✅ VOLLSTÄNDIGER TEST ABGESCHLOSSEN!")
    print("="*50)
    
    print("\n🎯 ERGEBNIS-ZUSAMMENFASSUNG:")
    print("   Wenn Einwohner(Rom,283000) NEIN/FALSCH zurückgibt:")
    print("   ✅ PROBLEM GELÖST - Funktionale Constraints funktionieren!")
    print("\n   Wenn Einwohner(Rom,283000) UNBEKANNT zurückgibt:")
    print("   ❌ PROBLEM BESTEHT NOCH - Weitere Debugging erforderlich")

if __name__ == "__main__":
    run_comprehensive_test()
