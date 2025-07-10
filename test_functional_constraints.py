#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TEST-SUITE FÜR FUNKTIONALE CONSTRAINTS
Testet das ursprüngliche Problem: Einwohner(Rom, 2873000) vs Einwohner(Rom, 283000)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from k_assistant_main_v7_wolfram import KAssistant

def test_functional_constraints():
    print("🧪 TESTE FUNKTIONALE CONSTRAINTS")
    print("=" * 50)
    
    # Erstelle temporäre Wissensbasis
    assistant = KAssistant("test_functional.kb")
    
    print("\n1️⃣ TESTE EINWOHNER-FUNKTIONALITÄT")
    print("-" * 30)
    
    # Füge einen korrekten Einwohner-Fakt hinzu
    print("✅ Füge hinzu: Einwohner(Rom,2873000).")
    assistant.add_raw("Einwohner(Rom,2873000).")
    
    # Teste korrekten Fakt
    print("\n🔍 Teste: Einwohner(Rom,2873000).")
    assistant.ask_raw("Einwohner(Rom,2873000).")
    
    # KRITISCHER TEST: Teste widersprüchlichen Fakt
    print("\n🚨 KRITISCHER TEST: Teste: Einwohner(Rom,283000).")
    print("   Erwartet: FALSCH (wegen funktionalem Constraint)")
    assistant.ask_raw("Einwohner(Rom,283000).")
    
    print("\n2️⃣ TESTE HAUPTSTADT-FUNKTIONALITÄT")
    print("-" * 30)
    
    # Füge Hauptstadt hinzu
    print("✅ Füge hinzu: Hauptstadt(Deutschland,Berlin).")
    assistant.add_raw("Hauptstadt(Deutschland,Berlin).")
    
    # Teste korrekten Fakt
    print("\n🔍 Teste: Hauptstadt(Deutschland,Berlin).")
    assistant.ask_raw("Hauptstadt(Deutschland,Berlin).")
    
    # Teste widersprüchlichen Fakt
    print("\n🚨 Teste widersprüchlichen Fakt: Hauptstadt(Deutschland,München).")
    print("   Erwartet: FALSCH (wegen funktionalem Constraint)")
    assistant.ask_raw("Hauptstadt(Deutschland,München).")
    
    print("\n3️⃣ TESTE CONSTRAINT-HINZUFÜGUNG")
    print("-" * 30)
    
    # Versuche widersprüchlichen Fakt hinzuzufügen
    print("🚨 Versuche hinzuzufügen: Einwohner(Rom,999999).")
    print("   Erwartet: ABLEHNUNG wegen Inkonsistenz")
    assistant.add_raw("Einwohner(Rom,999999).")
    
    print("\n4️⃣ ZEIGE WISSENSBASIS")
    print("-" * 30)
    
    data = assistant.show()
    print("📊 Aktuelle Kernregeln:")
    for i, fact in enumerate(data['permanent_knowledge']):
        if 'Einwohner' in fact or 'Hauptstadt' in fact or ' = ' in fact:
            print(f"   [{i}] {fact}")
    
    # Cleanup
    try:
        os.remove("test_functional.kb")
        print("\n🧹 Test-Datei gelöscht")
    except:
        pass
    
    print("\n✅ TEST ABGESCHLOSSEN!")

if __name__ == "__main__":
    test_functional_constraints()
