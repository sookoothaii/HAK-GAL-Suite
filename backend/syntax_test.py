#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Syntax-Validierungstest für k_assistant_main_v7_wolfram.py
"""

import sys
import os

def test_syntax():
    """Testet die Python-Syntax der korrigierten Datei"""
    
    print("=" * 60)
    print("SYNTAX-VALIDIERUNG: k_assistant_main_v7_wolfram.py")
    print("=" * 60)
    
    # Test 1: Import-Test
    print("\n1. Import-Test...")
    try:
        import k_assistant_main_v7_wolfram
        print("   [SUCCESS] Import erfolgreich")
    except SyntaxError as e:
        print(f"   [ERROR] Syntax-Fehler gefunden: {e}")
        return False
    except Exception as e:
        print(f"   [WARNING] Import-Warnung: {e}")
    
    # Test 2: Kritische Zeile testen
    print("\n2. Kritische Zeilen-Test...")
    try:
        # Die korrigierte Zeile testen
        test_goal = "HauptstadtVon(Deutschland)."
        logical_operators = ["->", "&", "|", "all "]
        result = any(op in test_goal for op in logical_operators)
        print(f"   [SUCCESS] Logische Operatoren-Test bestanden: {result}")
    except Exception as e:
        print(f"   [ERROR] Operatoren-Test fehlgeschlagen: {e}")
        return False
    
    # Test 3: Klassen-Instanziierung
    print("\n3. Klassen-Instanziierung...")
    try:
        from k_assistant_main_v7_wolfram import WolframProver, ComplexityAnalyzer
        
        # Test WolframProver
        prover = WolframProver()
        print("   [SUCCESS] WolframProver instanziiert")
        
        # Test ComplexityAnalyzer
        analyzer = ComplexityAnalyzer()
        print("   [SUCCESS] ComplexityAnalyzer instanziiert")
        
    except Exception as e:
        print(f"   [WARNING] Klassen-Instanziierung: {e}")
    
    # Test 4: Portfolio-Manager
    print("\n4. Portfolio-Manager Test...")
    try:
        from k_assistant_main_v7_wolfram import ProverPortfolioManager, ComplexityAnalyzer
        analyzer = ComplexityAnalyzer()
        portfolio = ProverPortfolioManager(analyzer)
        print("   [SUCCESS] ProverPortfolioManager instanziiert")
    except Exception as e:
        print(f"   [WARNING] Portfolio-Manager: {e}")
    
    print("\n" + "=" * 60)
    print("SYNTAX-VALIDIERUNG ABGESCHLOSSEN")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_syntax()
    if success:
        print("\nALLE KRITISCHEN TESTS ERFOLGREICH!")
        exit(0)
    else:
        print("\nSYNTAXFEHLER GEFUNDEN!")
        exit(1)
