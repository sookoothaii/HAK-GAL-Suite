# -*- coding: utf-8 -*-
"""
Hardened Wolfram Integration für HAK-GAL System
Implementiert gemäß dem finalen Arbeitsplan mit allen Verbesserungen:
- Caching für Wolfram-Anfragen
- Erweiterte Übersetzung von HAK-GAL zu natürlicher Sprache  
- Robuste Antwort-Interpretation
- Graceful Degradation bei Fehlern

Version: 1.0 - Wissenschaftlich validierte Implementation
"""

import os
import re
import time
import logging
from typing import Optional, Tuple, Dict, Any

# Wolfram Alpha API Import mit Fehlerbehandlung
try:
    import wolframalpha
    WOLFRAM_AVAILABLE = True
except ImportError:
    WOLFRAM_AVAILABLE = False
    print("⚠️ WARNUNG: wolframalpha nicht installiert. WolframProver deaktiviert.")

# Import der BaseProver-Klasse - ROBUSTER IMPORT
try:
    # Versuch 1: Relativer Import vom backend
    import sys
    import os
    backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    
    from k_assistant_main_v7_wolfram import BaseProver
    print("✅ BaseProver erfolgreich importiert (v7_wolfram)")
except ImportError:
    try:
        # Versuch 2: Direkter Import
        from backend.k_assistant_main_v7_wolfram import BaseProver
        print("✅ BaseProver erfolgreich importiert (backend.v7_wolfram)")
    except ImportError:
        try:
            # Versuch 3: Fallback auf Standard-Version
            from k_assistant_main import BaseProver
            print("✅ BaseProver erfolgreich importiert (standard)")
        except ImportError:
            try:
                # Versuch 4: Backend-Standard
                from backend.k_assistant_main import BaseProver
                print("✅ BaseProver erfolgreich importiert (backend.standard)")
            except ImportError:
                # Finale Fallback-Implementierung
                print("⚠️ WARNUNG: BaseProver nicht gefunden - nutze Fallback-Implementierung")
                from abc import ABC, abstractmethod
                
                class BaseProver(ABC):
                    def __init__(self, name: str): 
                        self.name = name
                    
                    @abstractmethod
                    def prove(self, assumptions: list, goal: str) -> tuple:
                        pass
                    
                    @abstractmethod
                    def validate_syntax(self, formula: str) -> tuple:
                        pass

class WolframProver(BaseProver):
    """
    Gehärteter Wolfram|Alpha Prover mit erweiterten Funktionen:
    - In-Memory Caching mit konfigurierbarem Timeout
    - Intelligente HAK-GAL zu natürlicher Sprache Übersetzung
    - Robuste Antwort-Interpretation und Klassifikation
    - Graceful Degradation bei API-Fehlern
    """
    
    def __init__(self):
        if BaseProver is None:
            raise ImportError("BaseProver konnte nicht importiert werden")
        
        super().__init__("Wolfram|Alpha Orakel")
        
        # Wolfram Client Initialisierung
        app_id = os.getenv("WOLFRAM_APP_ID")
        if not app_id:
            self.client = None
            print("⚠️ WARNUNG: WolframProver deaktiviert. WOLFRAM_APP_ID nicht gefunden.")
            print("   Setzen Sie WOLFRAM_APP_ID in der .env Datei.")
            return
        
        if not WOLFRAM_AVAILABLE:
            self.client = None
            print("⚠️ WARNUNG: WolframProver deaktiviert. wolframalpha Bibliothek nicht verfügbar.")
            print("   Installieren Sie mit: pip install wolframalpha")
            return
            
        self.client = wolframalpha.Client(app_id)
        
        # Cache-System für Wolfram-Anfragen
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_timeout = int(os.getenv("WOLFRAM_CACHE_TIMEOUT", "3600"))  # 1 Stunde Standard
        
        # Debug-Modus
        self.debug = os.getenv("WOLFRAM_DEBUG", "false").lower() == "true"
        
        # Logging-Setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if self.client:
            print("✅ WolframProver erfolgreich initialisiert")
            if self.debug:
                print(f"   [Debug] Cache-Timeout: {self.cache_timeout}s")

    def prove(self, assumptions: list, goal: str) -> Tuple[Optional[bool], str]:
        """
        Hauptmethode für das Beweisen mit Wolfram|Alpha.
        
        Args:
            assumptions: Liste der Annahmen (wird für Wolfram-Anfragen nicht verwendet)
            goal: Das zu beweisende Ziel in HAK-GAL Format
            
        Returns:
            Tuple[Optional[bool], str]: (Beweis-Ergebnis, Begründung)
        """
        if not self.client:
            return None, "WolframProver ist nicht konfiguriert."

        # Validierung: Logische Operatoren werden nicht von Wolfram behandelt
        logical_operators = ["->", "&", "|", "all "]
        if any(op in goal for op in logical_operators):
            return None, "WolframProver unterstützt nur atomare Fakten, keine logischen Operatoren."

        # Cache-Prüfung
        cached_result = self._check_cache(goal)
        if cached_result is not None:
            return cached_result[0], f"[Cached] {cached_result[1]}"

        # HAK-GAL zu natürlicher Sprache Übersetzung
        query_string = self._hakgal_to_natural_language(goal)
        if not query_string:
            return None, "Konnte Formel nicht in eine verständliche Wolfram-Anfrage übersetzen."

        try:
            self._debug_log(f"Sende Anfrage: '{query_string}'")
            
            # Wolfram API-Aufruf
            res = self.client.query(query_string)
            
            # Erste verfügbare Antwort extrahieren
            try:
                answer = next(res.results).text
                self._debug_log(f"Antwort erhalten: '{answer}'")
            except StopIteration:
                reason = "Wolfram|Alpha konnte keine definitive Antwort finden."
                self._update_cache(goal, (False, reason))
                return False, reason
            
            # Robuste Antwort-Interpretation
            is_proven, reason = self._interpret_wolfram_response(goal, answer)

            # Ergebnis cachen (nur definitive Antworten)
            if is_proven is not None:
                self._update_cache(goal, (is_proven, reason))

            return is_proven, reason

        except Exception as e:
            error_msg = f"Fehler bei der Wolfram|Alpha API-Anfrage: {e}"
            self.logger.error(error_msg)
            return None, error_msg

    def _check_cache(self, goal: str) -> Optional[Tuple[Optional[bool], str]]:
        """
        Prüft den Cache für bereits beantwortete Anfragen.
        
        Args:
            goal: Das zu überprüfende Ziel
            
        Returns:
            Cached result or None if not found/expired
        """
        if goal in self.cache:
            cached_result, timestamp = self.cache[goal]
            if time.time() - timestamp < self.cache_timeout:
                self._debug_log(f"Cache-Treffer für: '{goal}'")
                return cached_result
            else:
                # Abgelaufenen Cache-Eintrag entfernen
                del self.cache[goal]
                self._debug_log(f"Cache-Eintrag abgelaufen für: '{goal}'")
        return None

    def _update_cache(self, goal: str, result: Tuple[Optional[bool], str]):
        """
        Aktualisiert den Cache mit neuen Ergebnissen.
        
        Args:
            goal: Das Ziel
            result: Das Ergebnis-Tupel
        """
        self.cache[goal] = (result, time.time())
        self._debug_log(f"Cache aktualisiert für: '{goal}'")
        
    def _hakgal_to_natural_language(self, formula: str) -> str:
        """
        Erweiterte Übersetzung von HAK-GAL Formeln zu natürlicher Sprache.
        Implementiert Template-basierte Konvertierung mit Fallback-Mechanismus.
        
        Args:
            formula: HAK-GAL Formel
            
        Returns:
            Natürlichsprachliche Anfrage für Wolfram|Alpha
        """
        # Punkt am Ende entfernen für die Verarbeitung
        formula = formula.strip().removesuffix('.')
        
        # Regex-Pattern für Prädikat(Argumente) Struktur
        match = re.match(r'(\w+)\((.*)\)', formula)
        if not match:
            return self._simple_conversion(formula)
        
        predicate, args_str = match.groups()
        args = [a.strip() for a in args_str.split(',') if a.strip()]
        
        # Template-System für bekannte Prädikate
        templates = {
            # Geografische Prädikate
            'HauptstadtVon': 'capital of {0}',
            'Bevölkerungsdichte': 'population density of {0}',
            'Bevölkerung': 'population of {0}',
            'FlächeVon': 'area of {0}',
            'WährungVon': 'currency of {0}',
            
            # Vergleichsprädikate
            'IstGroesserAls': 'is {0} greater than {1}',
            'IstKleinerAls': 'is {0} less than {1}',
            'IstGleich': 'is {0} equal to {1}',
            
            # Wetter-Prädikate
            'WetterIn': 'weather in {0}',
            'TemperaturIn': 'temperature in {0}',
            
            # Mathematische Prädikate
            'Integral': 'integral of {0}',
            'AbleitungVon': 'derivative of {0}',
            'Lösung': 'solve {0}',
            'Faktorisierung': 'factor {0}',
            
            # Zeit-Prädikate
            'ZeitzoneVon': 'timezone of {0}',
            'AktuelleZeit': 'current time in {0}',
            
            # Einheiten-Prädikate
            'Umrechnung': 'convert {0} to {1}',
            'Einheit': 'unit of {0}',
        }
        
        # Template anwenden wenn verfügbar
        if predicate in templates:
            try:
                # Unterstriche in Argumenten durch Leerzeichen ersetzen
                clean_args = [arg.replace('_', ' ') for arg in args]
                return templates[predicate].format(*clean_args)
            except IndexError:
                self.logger.warning(f"Template '{predicate}' erwartet andere Anzahl Argumente: {args}")
        
        # Fallback für unbekannte Prädikate
        return self._simple_conversion(formula)

    def _simple_conversion(self, formula: str) -> str:
        """
        Fallback-Methode für einfache Konvertierung.
        Wandelt CamelCase zu Leerzeichen und entfernt Sonderzeichen.
        
        Args:
            formula: Original-Formel
            
        Returns:
            Vereinfachte natürlichsprachliche Anfrage
        """
        # Sonderzeichen entfernen/ersetzen
        formula = formula.replace('.', '').replace('(', ' of ').replace(')', '').replace(',', ' and ')
        
        # CamelCase zu Leerzeichen
        formula = re.sub(r'(?<!^)(?=[A-Z])', ' ', formula).lower()
        
        # Mehrfache Leerzeichen entfernen
        formula = re.sub(r'\s+', ' ', formula).strip()
        
        return formula

    def _interpret_wolfram_response(self, goal: str, answer: str) -> Tuple[Optional[bool], str]:
        """
        Robuste Interpretation der Wolfram|Alpha Antwort.
        Klassifiziert die Antwort und bestimmt den Wahrheitswert.
        
        Args:
            goal: Original HAK-GAL Ziel
            answer: Wolfram|Alpha Antwort
            
        Returns:
            Tuple[Optional[bool], str]: (Wahrheitswert, formatierte Antwort)
        """
        answer_lower = answer.lower()
        
        # 1. Variable-Bindung erkennen (z.B. HauptstadtVon(Deutschland, x))
        var_match = re.search(r'\b([a-z])\b', goal)
        if var_match:
            var_name = var_match.group(1)
            return True, f"{var_name} = {answer}"
        
        # 2. Explizite Ja/Nein Antworten
        positive_indicators = ['yes', 'true', 'correct', 'is ']
        negative_indicators = ['no', 'false', 'incorrect', 'is not', 'not ']
        
        if any(word in answer_lower for word in positive_indicators):
            return True, f"Bestätigt: {answer}"
        elif any(word in answer_lower for word in negative_indicators):
            return False, f"Verneint: {answer}"
        
        # 3. Numerische Daten erkennen
        if re.search(r'\d', answer):
            return True, f"Daten gefunden: {answer}"
        
        # 4. Einheiten und Messungen
        unit_patterns = [r'\d+\s*(km|kg|m|°C|°F|mph|kph|\$|€|%)', r'\d+([.,]\d+)?\s*(million|billion|trillion)']
        if any(re.search(pattern, answer, re.IGNORECASE) for pattern in unit_patterns):
            return True, f"Messwert ermittelt: {answer}"
        
        # 5. Mathematische Ausdrücke
        math_indicators = ['=', '+', '-', '*', '/', '^', 'sqrt', 'log']
        if any(indicator in answer_lower for indicator in math_indicators):
            return True, f"Mathematische Lösung: {answer}"
        
        # 6. Fallback für unklare Antworten
        return None, f"Unklare Antwort von Wolfram|Alpha: {answer}"

    def validate_syntax(self, formula: str) -> Tuple[bool, str]:
        """
        Validiert die Syntax einer HAK-GAL Formel für Wolfram-Kompatibilität.
        
        Args:
            formula: Zu validierende Formel
            
        Returns:
            Tuple[bool, str]: (Ist gültig, Fehlermeldung)
        """
        # Grundlegende Syntax-Checks
        if not formula.strip():
            return False, "Leere Formel"
        
        if not formula.strip().endswith('.'):
            return False, "Formel muss mit '.' enden"
        
        # Logische Operatoren prüfen (nicht unterstützt)
        logical_ops = ["->", "&", "|", "all "]
        if any(op in formula for op in logical_ops):
            return False, "Wolfram unterstützt keine logischen Operatoren"
        
        # Klammern-Balance prüfen
        if formula.count('(') != formula.count(')'):
            return False, "Unbalancierte Klammern"
        
        # Prädikat-Struktur prüfen
        formula_clean = formula.strip().removesuffix('.')
        if not re.match(r'^[A-ZÄÖÜ][a-zA-ZÄÖÜäöüß0-9_]*(\([^)]*\))?$', formula_clean):
            return False, "Ungültige Prädikat-Struktur"
        
        return True, "Syntax für Wolfram-Anfrage OK"

    def _debug_log(self, message: str):
        """
        Debug-Logging wenn aktiviert.
        
        Args:
            message: Debug-Nachricht
        """
        if self.debug:
            print(f"   [Wolfram Debug] {message}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Gibt Cache-Statistiken zurück.
        
        Returns:
            Dictionary mit Cache-Statistiken
        """
        current_time = time.time()
        valid_entries = sum(1 for _, (_, timestamp) in self.cache.items() 
                          if current_time - timestamp < self.cache_timeout)
        
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "cache_timeout": self.cache_timeout,
            "cache_hit_potential": valid_entries / max(len(self.cache), 1) * 100
        }

    def clear_cache(self):
        """
        Leert den Wolfram-Cache.
        """
        self.cache.clear()
        print("   [Wolfram] Cache geleert.")

# Für Kompatibilität mit älteren Versionen
WolframAlphaProver = WolframProver
