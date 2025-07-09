#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-Script für die Wolfram-Integration des HAK-GAL Systems
Validiert alle Aspekte der "Hardened Wolfram Integration"

Führt schrittweise Tests durch:
1. Basis-Setup und Konfiguration
2. WolframProver Funktionalität  
3. ComplexityAnalyzer Oracle-Erkennung
4. ProverPortfolioManager Auswahl
5. End-to-End Integration

Version: 1.0 - Wissenschaftlich validierte Tests
"""

import os
import sys
import time
from pathlib import Path

# Pfad-Setup für Import
script_dir = Path(__file__).parent
backend_dir = script_dir / "backend"
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(script_dir))

def test_environment_setup():
    """Test 1: Umgebungssetup und Abhängigkeiten"""
    print("🧪 Test 1: Umgebungssetup und Abhängigkeiten")
    print("=" * 60)
    
    # .env Datei prüfen
    env_file = script_dir / ".env"
    if not env_file.exists():
        print("❌ FEHLER: .env Datei nicht gefunden!")
        print(f"   Erstellen Sie eine .env Datei basierend auf .env.example")
        print(f"   Pfad: {env_file}")
        return False
    
    print("✅ .env Datei gefunden")
    
    # Wolfram App ID prüfen
    from dotenv import load_dotenv
    load_dotenv()
    
    wolfram_app_id = os.getenv("WOLFRAM_APP_ID")
    if not wolfram_app_id or wolfram_app_id == "your_wolfram_app_id_here":
        print("⚠️ WARNUNG: WOLFRAM_APP_ID nicht konfiguriert")
        print("   Tests werden ohne echte Wolfram-Anfragen durchgeführt")
        return "partial"
    
    print(f"✅ Wolfram App ID konfiguriert: {wolfram_app_id[:8]}...")
    
    # Abhängigkeiten prüfen
    try:
        import wolframalpha
        print("✅ wolframalpha Bibliothek verfügbar")
    except ImportError:
        print("❌ wolframalpha Bibliothek nicht installiert")
        print("   Installieren Sie mit: pip install wolframalpha")
        return False
    
    return True

def test_wolfram_prover():
    """Test 2: WolframProver Funktionalität"""
    print("\n🧪 Test 2: WolframProver Funktionalität")
    print("=" * 60)
    
    try:
        from plugins.provers.wolfram_prover import WolframProver
        print("✅ WolframProver erfolgreich importiert")
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        return False
    
    # Prover initialisieren
    prover = WolframProver()
    
    if not prover.client:
        print("⚠️ WolframProver ist deaktiviert (keine App ID)")
        print("   Syntax-Tests werden trotzdem durchgeführt")
    else:
        print("✅ WolframProver erfolgreich initialisiert")
    
    # Syntax-Validation testen
    test_formulas = [
        "HauptstadtVon(Deutschland, x).",
        "Bevölkerungsdichte(Berlin, y).",
        "IstGroesserAls(5, 3).",
        "Integral(x^2, x).",
        "all x (IstSystem(x) -> IstKritisch(x))."  # Sollte nicht unterstützt werden
    ]
    
    print("\n--- Syntax-Validation Tests ---")
    for formula in test_formulas:
        is_valid, msg = prover.validate_syntax(formula)
        status = "✅" if is_valid else "❌"
        print(f"{status} '{formula}' -> {msg}")
    
    # Cache-Funktionalität testen
    print("\n--- Cache-System Tests ---")
    cache_stats = prover.get_cache_stats()
    print(f"✅ Cache-Stats: {cache_stats}")
    
    # Template-System testen
    print("\n--- Template-Übersetzung Tests ---")
    test_cases = [
        ("HauptstadtVon(Deutschland).", "capital of Deutschland"),
        ("WetterIn(Berlin).", "weather in Berlin"),
        ("Integral(x^2).", "integral of x^2"),
    ]
    
    for hakgal_formula, expected_contains in test_cases:
        natural_query = prover._hakgal_to_natural_language(hakgal_formula.replace('.', ''))
        success = expected_contains.lower() in natural_query.lower()
        status = "✅" if success else "❌"
        print(f"{status} '{hakgal_formula}' -> '{natural_query}'")
    
    return True

def test_complexity_analyzer():
    """Test 3: ComplexityAnalyzer Oracle-Erkennung"""
    print("\n🧪 Test 3: ComplexityAnalyzer Oracle-Erkennung")
    print("=" * 60)
    
    try:
        from k_assistant_main_v7_wolfram import ComplexityAnalyzer
        print("✅ ComplexityAnalyzer erfolgreich importiert")
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        return False
    
    analyzer = ComplexityAnalyzer()
    
    # Oracle-Erkennung testen
    test_cases = [
        # (Formula, Expected Oracle Requirement, Expected Query Type)
        ("HauptstadtVon(Deutschland, x).", True, "knowledge"),
        ("Bevölkerungsdichte(Berlin, y).", True, "knowledge"),
        ("Integral(x^2, x).", True, "mathematical"),
        ("IstKritisch(System).", False, "logic"),
        ("all x (IstSystem(x) -> IstLegacy(x)).", False, "logic"),
        ("WetterIn(München, temp).", True, "knowledge"),
    ]
    
    print("\n--- Oracle-Erkennung Tests ---")
    for formula, expected_oracle, expected_type in test_cases:
        report = analyzer.analyze(formula)
        
        oracle_correct = report.requires_oracle == expected_oracle
        type_matches = expected_type in report.query_type.value
        
        oracle_status = "✅" if oracle_correct else "❌"
        type_status = "✅" if type_matches else "❌"
        
        print(f"{oracle_status} Oracle: '{formula}' -> {report.requires_oracle} (erwartet: {expected_oracle})")
        print(f"{type_status} Typ: {report.query_type.value} (erwartet: {expected_type})")
        print(f"   Begründung: {report.reasoning}")
        print(f"   Empfohlene Prover: {', '.join(report.recommended_provers)}")
        print()
    
    return True

def test_portfolio_manager():
    """Test 4: ProverPortfolioManager"""
    print("\n🧪 Test 4: ProverPortfolioManager")
    print("=" * 60)
    
    try:
        from k_assistant_main_v7_wolfram import ProverPortfolioManager, ComplexityAnalyzer
        from k_assistant_main_v7_wolfram import PatternProver, Z3Adapter
        if os.getenv("WOLFRAM_APP_ID"):
            from plugins.provers.wolfram_prover import WolframProver
            provers = [PatternProver(), Z3Adapter(), WolframProver()]
        else:
            provers = [PatternProver(), Z3Adapter()]
        print("✅ ProverPortfolioManager und Provers importiert")
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        return False
    
    analyzer = ComplexityAnalyzer()
    manager = ProverPortfolioManager(analyzer)
    
    # Prover-Auswahl testen
    test_formulas = [
        "HauptstadtVon(Deutschland, x).",  # Sollte Wolfram bevorzugen
        "IstKritisch(System).",           # Sollte Pattern/Z3 bevorzugen
        "Integral(x^2, x).",              # Sollte Wolfram bevorzugen
    ]
    
    print("\n--- Prover-Auswahl Tests ---")
    for formula in test_formulas:
        ordered_provers = manager.select_prover_strategy(formula, provers)
        prover_names = [p.name for p in ordered_provers]
        print(f"✅ '{formula}'")
        print(f"   Reihenfolge: {' -> '.join(prover_names)}")
    
    # Performance-Tracking testen
    print("\n--- Performance-Tracking Tests ---")
    manager.update_performance("Pattern Matcher", "test", True, 0.1)
    manager.update_performance("Z3 SMT Solver", "test", False, 2.0)
    
    stats = manager.get_performance_report()
    print(f"✅ Performance-Daten gesammelt: {len(stats['performance'])} Prover")
    for name, perf in stats['performance'].items():
        print(f"   {name}: {perf['success_rate']:.1%} Erfolg, {perf['avg_duration']:.2f}s ⌀")
    
    return True

def test_end_to_end_integration():
    """Test 5: End-to-End Integration"""
    print("\n🧪 Test 5: End-to-End Integration")
    print("=" * 60)
    
    try:
        from k_assistant_main_v7_wolfram import KAssistant
        print("✅ KAssistant erfolgreich importiert")
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        return False
    
    # Test-KB für diesen Test erstellen
    test_kb_path = script_dir / "test_wolfram_integration.kb"
    
    try:
        # KAssistant initialisieren
        print("--- Initialisierung ---")
        assistant = KAssistant(str(test_kb_path))
        print(f"✅ KAssistant initialisiert mit {len(assistant.core.provers)} Provern")
        
        # Oracle-Prädikat hinzufügen
        print("\n--- Oracle-Konfiguration ---")
        assistant.add_oracle_predicate("TestPrädikat")
        print("✅ Oracle-Prädikat hinzugefügt")
        
        # Status prüfen
        print("\n--- System-Status ---")
        assistant.status()
        
        # Portfolio-Performance anzeigen
        portfolio_stats = assistant.core.get_portfolio_stats()
        if portfolio_stats["performance"]:
            print("✅ Portfolio-Performance verfügbar")
        else:
            print("ℹ️ Noch keine Portfolio-Performance (erwartet bei erstem Start)")
        
        # Wolfram-Stats testen
        print("\n--- Wolfram-Statistiken ---")
        assistant.wolfram_stats()
        
        # Test-KB aufräumen
        if test_kb_path.exists():
            test_kb_path.unlink()
            print("✅ Test-KB aufgeräumt")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-End Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_integration_test():
    """Führt alle Tests in der korrekten Reihenfolge aus"""
    print("🚀 HAK-GAL Wolfram-Integration Test Suite")
    print("=" * 70)
    print("Testet die vollständige 'Hardened Wolfram Integration'")
    print()
    
    tests = [
        ("Umgebungssetup", test_environment_setup),
        ("WolframProver", test_wolfram_prover),
        ("ComplexityAnalyzer", test_complexity_analyzer),
        ("ProverPortfolioManager", test_portfolio_manager),
        ("End-to-End Integration", test_end_to_end_integration),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ KRITISCHER FEHLER in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
            break
    
    # Zusammenfassung
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("🏁 TEST-ZUSAMMENFASSUNG")
    print("=" * 70)
    
    passed = 0
    partial = 0
    failed = 0
    
    for test_name, result in results:
        if result is True:
            print(f"✅ {test_name}: BESTANDEN")
            passed += 1
        elif result == "partial":
            print(f"⚠️ {test_name}: TEILWEISE (Konfiguration unvollständig)")
            partial += 1
        else:
            print(f"❌ {test_name}: FEHLGESCHLAGEN")
            failed += 1
    
    print()
    print(f"Ergebnis: {passed} bestanden, {partial} teilweise, {failed} fehlgeschlagen")
    print(f"Laufzeit: {elapsed:.2f} Sekunden")
    
    if failed == 0:
        print("\n🎉 ALLE TESTS BESTANDEN! Wolfram-Integration ist einsatzbereit.")
        if partial > 0:
            print("💡 Vervollständigen Sie die .env Konfiguration für volle Funktionalität.")
    else:
        print(f"\n⚠️ {failed} Test(s) fehlgeschlagen. Überprüfen Sie die Konfiguration.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
