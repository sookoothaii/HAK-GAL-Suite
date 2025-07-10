# -*- coding: utf-8 -*-
# k_assistant.py - Version 7 mit Wolfram-Integration
# Implementiert den "Hardened Wolfram Integration" Plan mit:
# - ComplexityAnalyzer für Oracle-Erkennung
# - ProverPortfolioManager für intelligente Prover-Auswahl
# - WolframProver Integration
# - Erweiterte Archon-Prime Architektur

import re
import pickle
import os
import time
import subprocess
import platform
from abc import ABC, abstractmethod
from collections import Counter
import threading
from typing import Optional, Tuple, List, Dict, Any
import concurrent.futures
import json
from dataclasses import dataclass
from enum import Enum

# ==============================================================================
# IMPORTS
# ==============================================================================

try:
    from dotenv import load_dotenv
    if load_dotenv(): print("✅ .env Datei geladen.")
except ImportError: pass

try:
    from openai import OpenAI
except ImportError: print("❌ FEHLER: 'openai' nicht gefunden. Bitte mit 'pip install openai' installieren."); exit()
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError: GEMINI_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    from pypdf import PdfReader
    RAG_ENABLED = True
except ImportError: RAG_ENABLED = False
try:
    import z3
except ImportError: print("❌ FEHLER: 'z3-solver' nicht gefunden. Bitte mit 'pip install z3-solver' installieren."); exit()
try:
    import lark
    LARK_AVAILABLE = True
except ImportError:
    LARK_AVAILABLE = False
    print("⚠️ WARNUNG: 'lark' nicht gefunden. Der Parser wird im Fallback-Modus ausgeführt.")

# Wolfram Integration versuchen - VEREINFACHTE LÖSUNG
WOLFRAM_INTEGRATION = False
try:
    import wolframalpha
    import os
    
    # Prüfe ob App ID verfügbar
    app_id = os.getenv("WOLFRAM_APP_ID")
    if app_id and app_id != "your_wolfram_app_id_here":
        WOLFRAM_INTEGRATION = True
        print("✅ Wolfram-Integration aktiviert")
    else:
        print("⚠️ Wolfram App ID nicht konfiguriert")
except ImportError:
    print("⚠️ wolframalpha Bibliothek nicht installiert")

# WolframProver wird später nach BaseProver definiert

# ==============================================================================
# NEUE DATENSTRUKTUREN FÜR ARCHON-PRIME
# ==============================================================================

class QueryType(Enum):
    LOGIC = "logic"
    KNOWLEDGE = "knowledge"
    MATHEMATICAL = "mathematical"
    MIXED = "mixed"

class ComplexityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"

@dataclass
class ComplexityReport:
    """Analyse-Bericht für Query-Komplexität und Ressourcen-Anforderungen"""
    query_type: QueryType
    complexity_level: ComplexityLevel
    requires_oracle: bool
    estimated_time: float
    confidence: float
    reasoning: str
    recommended_provers: List[str]

# ==============================================================================
# GRAMMAR DEFINITION
# ==============================================================================

HAKGAL_GRAMMAR = r"""
    ?start: formula
    formula: expression "."
    ?expression: quantified_formula | implication
    ?implication: disjunction ( "->" implication )?
    ?disjunction: conjunction ( "|" disjunction )?
    ?conjunction: negation ( "&" conjunction )?
    ?negation: "-" atom_expression | atom_expression
    ?atom_expression: atom | "(" expression ")"
    quantified_formula: "all" VAR "(" expression ")"
    atom: PREDICATE ("(" [arg_list] ")")?
    arg_list: term ("," term)*
    ?term: PREDICATE | VAR | NUMBER
    PREDICATE: /[A-ZÄÖÜ][a-zA-ZÄÖÜäöüß0-9_-]*/
    VAR: /[a-z][a-zA-Z0-9_]*/
    NUMBER: /[0-9]+([_][0-9]+)*/
    %import common.WS
    %ignore WS
"""

# ==============================================================================
# Cache-Klassen
# ==============================================================================
class BaseCache(ABC):
    def __init__(self):
        self.cache: Dict[Any, Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[Any]:
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: Any, value: Any):
        self.cache[key] = value

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        print("   [Cache] Cache geleert.")

    @property
    def size(self) -> int: return len(self.cache)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

class ProofCache(BaseCache):
    def get(self, query_str: str, key: Tuple) -> Optional[Tuple[bool, str, float]]: return super().get(key)
    def put(self, query_str: str, key: Tuple, success: bool, reason: str): super().put(key, (success, reason, time.time()))

class PromptCache(BaseCache):
    def get(self, prompt: str) -> Optional[str]: return super().get(prompt)
    def put(self, prompt: str, response: str): super().put(prompt, response)

#==============================================================================
# 1. ABSTRAKTE BASISKLASSEN
#==============================================================================
class BaseLLMProvider(ABC):
    def __init__(self, model_name: str): self.model_name = model_name
    @abstractmethod
    def query(self, prompt: str, system_prompt: str, temperature: float) -> str: pass

class BaseProver(ABC):
    def __init__(self, name: str): self.name = name
    @abstractmethod
    def prove(self, assumptions: list, goal: str) -> tuple[Optional[bool], str]: pass
    @abstractmethod
    def validate_syntax(self, formula: str) -> tuple[bool, str]: pass

# ==============================================================================
# WOLFRAM PROVER - NACH BASEPROVER DEFINIERT
# ==============================================================================
class WolframProver(BaseProver):
    def __init__(self):
        super().__init__("Wolfram|Alpha Orakel")
        
        app_id = os.getenv("WOLFRAM_APP_ID")
        if not app_id or app_id == "your_wolfram_app_id_here":
            self.client = None
            return
            
        # Debug-Modus aus .env lesen
        self.debug = os.getenv("WOLFRAM_DEBUG", "false").lower() == "true"
        
        try:
            import wolframalpha
            self.client = wolframalpha.Client(app_id)
            self.cache = {}
            self.cache_timeout = 3600
            print("✅ WolframProver erfolgreich initialisiert")
            if self.debug:
                print(f"   🐛 Debug-Modus aktiviert für Wolfram-Prover")
        except ImportError:
            self.client = None
            print("⚠️ wolframalpha nicht verfügbar")
    
    def prove(self, assumptions: list, goal: str) -> tuple:
        if not self.client:
            return None, "WolframProver ist nicht konfiguriert."
            
        # Logische Operatoren werden nicht unterstützt
        if any(op in goal for op in ["->" , "&", "|", "all "]):
            return None, "WolframProver unterstützt nur atomare Fakten."
            
        # Einfache Übersetzung
        query = self._simple_translate(goal)
        if not query:
            return None, "Konnte Formel nicht übersetzen."
            
        # OPTIMIERT: Direkt HTTP-Aufruf (umgeht AssertionError-Problem)
        try:
            import urllib.parse
            import urllib.request
            import xml.etree.ElementTree as ET
            
            app_id = os.getenv("WOLFRAM_APP_ID")
            encoded_query = urllib.parse.quote(query)
            url = f"http://api.wolframalpha.com/v2/query?input={encoded_query}&appid={app_id}&format=plaintext"
            
            # Schneller HTTP-Aufruf mit kurzem Timeout
            with urllib.request.urlopen(url, timeout=5) as response:
                xml_data = response.read().decode('utf-8')
                root = ET.fromstring(xml_data)
                
                # Extrahiere erste brauchbare Antwort
                for pod in root.findall('.//pod'):
                    for subpod in pod.findall('.//subpod'):
                        plaintext_elem = subpod.find('plaintext')
                        if plaintext_elem is not None and plaintext_elem.text:
                            answer = plaintext_elem.text.strip()
                            if answer and len(answer) > 2:  # Mindestlänge für sinnvolle Antworten
                                # Intelligente Interpretation
                                answer_lower = answer.lower()
                                query_lower = query.lower()
                                
                                # Hauptstadt-Anfragen
                                if 'capital' in query_lower:
                                    if any(city in answer_lower for city in ['london', 'berlin', 'paris', 'madrid', 'rome', 'moscow']):
                                        return True, f"Hauptstadt: {answer}"
                                
                                # Bevölkerungs-Anfragen
                                if 'population' in query_lower:
                                    if any(char.isdigit() for char in answer):
                                        return True, f"Bevölkerung: {answer}"
                                
                                # Allgemeine Datenantworten
                                if len(answer) > 5:  # Substanzielle Antwort
                                    return True, f"Wolfram: {answer}"
                
                return None, "Keine verwertbare Antwort von Wolfram Alpha"
                
        except Exception as e:
            return None, f"Wolfram-Fehler: {type(e).__name__}"
    
    def validate_syntax(self, formula: str) -> tuple:
        if not formula.strip().endswith('.'):
            return False, "Formel muss mit '.' enden"
        if any(op in formula for op in ["->", "&", "|", "all "]):
            return False, "Wolfram unterstützt keine logischen Operatoren"
        return True, "Syntax für Wolfram-Anfrage OK"
    
    def _simple_translate(self, formula: str) -> str:
        """Einfache HAK-GAL zu natürlicher Sprache Übersetzung"""
        formula = formula.strip().removesuffix('.')
        
        # SPEZIAL: Variable-Handling für Wolfram
        if ', x)' in formula or ', y)' in formula or ', z)' in formula:
            # Einwohner(Wien, x) -> "population of Vienna"
            if 'Einwohner(' in formula:
                match = re.search(r'Einwohner\(([^,]+),', formula)
                if match:
                    city = match.group(1).lower()
                    return f"population of {city}"
            # Hauptstadt(x, Berlin) -> "country with capital Berlin"
            if 'Hauptstadt(' in formula and 'Hauptstadt(x' in formula:
                match = re.search(r'Hauptstadt\(x,\s*([^)]+)\)', formula)
                if match:
                    city = match.group(1)
                    return f"country with capital {city}"
        
        # SPEZIAL-BEHANDLUNG für fehlerhafte Eingaben
        if 'HauptstadtvonEnglandist(' in formula:
            # Extrahiere den Parameter aus HauptstadtvonEnglandist(London)
            match = re.search(r'HauptstadtvonEnglandist\(([^)]+)\)', formula)
            if match:
                city = match.group(1)
                return f"capital of england"
        
        # Bekannte Muster
        if 'HauptstadtVon(' in formula:
            match = re.search(r'HauptstadtVon\(([^)]+)\)', formula)
            if match:
                country = match.group(1).lower()
                # Übersetze bekannte Länder
                country_map = {
                    'deutschland': 'germany',
                    'frankreich': 'france',
                    'italien': 'italy',
                    'spanien': 'spain',
                    'england': 'england',
                    'großbritannien': 'united kingdom'
                }
                country = country_map.get(country, country)
                return f"capital of {country}"
        
        if 'Bevölkerung(' in formula:
            match = re.search(r'Bevölkerung\(([^)]+)\)', formula)
            if match:
                place = match.group(1).lower()
                return f"population of {place}"
                
        if 'WetterIn(' in formula:
            match = re.search(r'WetterIn\(([^)]+)\)', formula)
            if match:
                place = match.group(1).lower()
                return f"weather in {place}"
                
        if 'WährungVon(' in formula:
            match = re.search(r'WährungVon\(([^)]+)\)', formula)
            if match:
                country = match.group(1).lower()
                return f"currency of {country}"
                
        if 'FlächeVon(' in formula:
            match = re.search(r'FlächeVon\(([^)]+)\)', formula)
            if match:
                place = match.group(1).lower()
                return f"area of {place}"
                
        if 'ZeitzoneVon(' in formula:
            match = re.search(r'ZeitzoneVon\(([^)]+)\)', formula)
            if match:
                place = match.group(1).lower()
                return f"timezone of {place}"
                
        if 'Integral(' in formula:
            match = re.search(r'Integral\(([^)]+)\)', formula)
            if match:
                expr = match.group(1)
                return f"integral of {expr}"
        
        # Fallback: einfache Konvertierung
        # Aber vorsichtiger - nur wenn keine Spezialbehandlung greift
        clean_formula = formula.replace('(', ' of ').replace(')', '').replace(',', ' and ')
        clean_formula = re.sub(r'(?<!^)(?=[A-Z])', ' ', clean_formula).lower()
        
        # Deutsch-Englisch Übersetzung für häufige Begriffe
        translations = {
            'bevölkerung': 'population',
            'hauptstadt': 'capital',
            'wetter': 'weather',
            'währung': 'currency',
            'fläche': 'area',
            'temperatur': 'temperature',
            'zeitzone': 'timezone',
            'deutschland': 'germany',
            'frankreich': 'france',
            'italien': 'italy',
            'spanien': 'spain',
            'großbritannien': 'united kingdom',
            'england': 'england'
        }
        
        for german, english in translations.items():
            clean_formula = clean_formula.replace(german, english)
            
        return clean_formula

#==============================================================================
# 2. COMPLEXITY ANALYZER - KERN DER ARCHON-PRIME ARCHITEKTUR
#==============================================================================
class ComplexityAnalyzer:
    """
    Erweiterte Komplexitätsanalyse für intelligente Oracle-Erkennung
    Implementiert Pattern-basierte Erkennung für Wolfram-geeignete Queries
    """
    
    def __init__(self):
        # Oracle-Prädikate: Bekannte Wissensprädikate für externe Abfragen
        self.oracle_predicates = {
            "Bevölkerungsdichte", "HauptstadtVon", "WetterIn", "TemperaturIn",
            "Integral", "AbleitungVon", "WährungVon", "FlächeVon", "Bevölkerung",
            "ZeitzoneVon", "AktuelleZeit", "Umrechnung", "Einheit", "Lösung",
            "Faktorisierung", "IstGroesserAls", "IstKleinerAls",
            "Einwohner", "Hauptstadt"  # NEU: Explizit als Oracle-Prädikate
        }
        
        # Pattern für Oracle-Erkennung via Regex
        self.oracle_patterns = [
            r'.*[Vv]on$',           # Endet mit "Von"
            r'.*[Ii]n$',            # Endet mit "In" 
            r'Berechne.*',          # Startet mit "Berechne"
            r'.*temperatur.*',      # Enthält Temperatur
            r'.*wetter.*',          # Enthält Wetter
            r'.*hauptstadt.*',      # Enthält Hauptstadt
            r'.*währung.*',         # Enthält Währung
            r'.*bevölkerung.*',     # Enthält Bevölkerung
        ]
        
        # Komplexitätsmuster
        self.high_complexity_patterns = [
            r'all\s+\w+',          # Quantifizierte Formeln
            r'->\s*all',           # Verschachtelte Implikationen
            r'&.*&.*&',            # Mehrfache Konjunktionen
        ]
        
        self.mathematical_patterns = [
            r'[Ii]ntegral',         # Integration
            r'[Aa]bleitung',        # Differentiation
            r'[Ll]ösung',           # Gleichungslösung
            r'[Ff]aktor',           # Faktorisierung
            r'[Gg]renze',           # Grenzwerte
        ]

    def analyze(self, formula: str) -> ComplexityReport:
        """
        Hauptanalyse-Methode für Formeln
        
        Args:
            formula: HAK-GAL Formel zur Analyse
            
        Returns:
            ComplexityReport mit detaillierter Analyse
        """
        # Prädikat extrahieren
        predicate_match = re.match(r'([A-ZÄÖÜ][a-zA-ZÄÖÜäöüß0-9_]*)', formula.strip())
        predicate = predicate_match.group(1) if predicate_match else ""
        
        # Oracle-Bedarf analysieren
        requires_oracle = self._requires_oracle_analysis(predicate, formula)
        
        # Query-Typ bestimmen
        query_type = self._determine_query_type(predicate, formula)
        
        # Komplexität schätzen
        complexity_level = self._estimate_complexity(formula)
        
        # Zeitschätzung (vereinfacht)
        estimated_time = self._estimate_time(complexity_level, requires_oracle)
        
        # Empfohlene Prover
        recommended_provers = self._recommend_provers(query_type, complexity_level, requires_oracle)
        
        # Konfidenz und Reasoning
        confidence = self._calculate_confidence(predicate, formula)
        reasoning = self._generate_reasoning(predicate, requires_oracle, query_type, complexity_level)
        
        return ComplexityReport(
            query_type=query_type,
            complexity_level=complexity_level,
            requires_oracle=requires_oracle,
            estimated_time=estimated_time,
            confidence=confidence,
            reasoning=reasoning,
            recommended_provers=recommended_provers
        )

    def _requires_oracle_analysis(self, predicate: str, formula: str) -> bool:
        """Analysiert, ob ein Oracle (Wolfram) benötigt wird"""
        
        # 1. Bekannte Oracle-Prädikate
        if predicate in self.oracle_predicates:
            return True
        
        # 2. Pattern-basierte Erkennung
        if any(re.match(pattern, predicate, re.IGNORECASE) for pattern in self.oracle_patterns):
            return True
        
        # 3. Einheiten und Maßangaben erkennen
        if re.search(r'\d+.*(?:km|kg|€|\$|°C|°F|%|meter|grad)', formula, re.IGNORECASE):
            return True
        
        # 4. Mathematische Ausdrücke
        if any(re.search(pattern, formula, re.IGNORECASE) for pattern in self.mathematical_patterns):
            return True
        
        return False

    def _determine_query_type(self, predicate: str, formula: str) -> QueryType:
        """Bestimmt den Typ der Anfrage"""
        
        if any(re.search(pattern, formula, re.IGNORECASE) for pattern in self.mathematical_patterns):
            return QueryType.MATHEMATICAL
        
        if self._requires_oracle_analysis(predicate, formula):
            return QueryType.KNOWLEDGE
        
        if any(op in formula for op in ['->', '&', '|', 'all ']):
            return QueryType.LOGIC
        
        return QueryType.MIXED

    def _estimate_complexity(self, formula: str) -> ComplexityLevel:
        """Schätzt die Komplexität der Formel"""
        
        # Hohe Komplexität
        if any(re.search(pattern, formula) for pattern in self.high_complexity_patterns):
            return ComplexityLevel.HIGH
        
        # Mittlere Komplexität
        if len(re.findall(r'[&|]|->|-', formula)) > 1:
            return ComplexityLevel.MEDIUM
        
        # Niedrige Komplexität für einfache atomare Formeln
        if re.match(r'^[A-ZÄÖÜ][a-zA-ZÄÖÜäöüß0-9_]*\([^)]*\)\.$', formula):
            return ComplexityLevel.LOW
        
        return ComplexityLevel.UNKNOWN

    def _estimate_time(self, complexity: ComplexityLevel, requires_oracle: bool) -> float:
        """Schätzt die Ausführungszeit in Sekunden"""
        
        base_time = {
            ComplexityLevel.LOW: 0.1,
            ComplexityLevel.MEDIUM: 0.5,
            ComplexityLevel.HIGH: 2.0,
            ComplexityLevel.UNKNOWN: 1.0
        }
        
        time_estimate = base_time[complexity]
        
        # Oracle-Anfragen dauern länger
        if requires_oracle:
            time_estimate += 1.5
        
        return time_estimate

    def _recommend_provers(self, query_type: QueryType, complexity: ComplexityLevel, requires_oracle: bool) -> List[str]:
        """Empfiehlt geeignete Prover für die Anfrage"""
        
        recommended = []
        
        # Oracle-Anfragen -> Wolfram zuerst
        if requires_oracle and WOLFRAM_INTEGRATION:
            recommended.append("Wolfram|Alpha Orakel")
        
        # Funktionale Constraints -> Spezialisierter Prover zuerst
        recommended.append("Functional Constraint Prover")
        
        # Logische Anfragen -> Z3
        if query_type in [QueryType.LOGIC, QueryType.MIXED]:
            recommended.append("Z3 SMT Solver")
        
        # Pattern Matcher als Fallback
        recommended.append("Pattern Matcher")
        
        return recommended

    def _calculate_confidence(self, predicate: str, formula: str) -> float:
        """Berechnet Konfidenz der Analyse (0.0 - 1.0)"""
        
        confidence = 0.5  # Basis-Konfidenz
        
        # Bekannte Prädikate erhöhen Konfidenz
        if predicate in self.oracle_predicates:
            confidence += 0.3
        
        # Klare Muster erhöhen Konfidenz
        if any(re.match(pattern, predicate, re.IGNORECASE) for pattern in self.oracle_patterns):
            confidence += 0.2
        
        # Einfache Strukturen sind sicherer
        if re.match(r'^[A-ZÄÖÜ][a-zA-ZÄÖÜäöüß0-9_]*\([^)]*\)\.$', formula):
            confidence += 0.1
        
        return min(confidence, 1.0)

    def _generate_reasoning(self, predicate: str, requires_oracle: bool, query_type: QueryType, complexity: ComplexityLevel) -> str:
        """Generiert Begründung für die Analyse"""
        
        reasons = []
        
        if requires_oracle:
            if predicate in self.oracle_predicates:
                reasons.append(f"'{predicate}' ist bekanntes Wissensprädikat")
            else:
                reasons.append("Pattern deutet auf Wissensabfrage hin")
        
        reasons.append(f"Query-Typ: {query_type.value}")
        reasons.append(f"Komplexität: {complexity.value}")
        
        return "; ".join(reasons)

#==============================================================================
# 3. PROVER PORTFOLIO MANAGER
#==============================================================================
class ProverPortfolioManager:
    """
    Intelligenter Manager für Prover-Auswahl basierend auf Komplexitätsanalyse
    Implementiert Multi-Armed Bandit-ähnliche Strategien
    """
    
    def __init__(self, complexity_analyzer: ComplexityAnalyzer):
        self.complexity_analyzer = complexity_analyzer
        self.prover_performance: Dict[str, Dict[str, float]] = {}
        self.prover_usage_count: Dict[str, int] = {}
        
    def select_prover_strategy(self, formula: str, available_provers: List[BaseProver]) -> List[BaseProver]:
        """
        Wählt optimale Prover-Reihenfolge basierend auf Komplexitätsanalyse
        
        Args:
            formula: Zu analysierende Formel
            available_provers: Verfügbare Prover
            
        Returns:
            Geordnete Liste der Prover nach Priorität
        """
        # Komplexitätsanalyse durchführen
        report = self.complexity_analyzer.analyze(formula)
        
        print(f"   [Portfolio] Analyse: {report.reasoning}")
        print(f"   [Portfolio] Empfohlene Prover: {', '.join(report.recommended_provers)}")
        
        # Prover nach Empfehlung sortieren
        prover_by_name = {p.name: p for p in available_provers}
        ordered_provers = []
        
        # Zuerst empfohlene Prover in der empfohlenen Reihenfolge
        for recommended_name in report.recommended_provers:
            if recommended_name in prover_by_name:
                ordered_provers.append(prover_by_name[recommended_name])
        
        # Dann restliche Prover
        for prover in available_provers:
            if prover not in ordered_provers:
                ordered_provers.append(prover)
        
        return ordered_provers
    
    def update_performance(self, prover_name: str, formula: str, success: bool, duration: float):
        """
        Aktualisiert Performance-Metriken für adaptive Optimierung
        
        Args:
            prover_name: Name des Provers
            formula: Verwendete Formel
            success: Erfolg des Beweises
            duration: Ausführungszeit
        """
        if prover_name not in self.prover_performance:
            self.prover_performance[prover_name] = {"success_rate": 0.0, "avg_duration": 0.0}
        
        if prover_name not in self.prover_usage_count:
            self.prover_usage_count[prover_name] = 0
        
        # Einfache gleitende Durchschnitte
        current_count = self.prover_usage_count[prover_name]
        current_success_rate = self.prover_performance[prover_name]["success_rate"]
        current_avg_duration = self.prover_performance[prover_name]["avg_duration"]
        
        # Update success rate
        new_success_rate = (current_success_rate * current_count + (1.0 if success else 0.0)) / (current_count + 1)
        
        # Update average duration
        new_avg_duration = (current_avg_duration * current_count + duration) / (current_count + 1)
        
        self.prover_performance[prover_name]["success_rate"] = new_success_rate
        self.prover_performance[prover_name]["avg_duration"] = new_avg_duration
        self.prover_usage_count[prover_name] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Gibt Performance-Bericht zurück"""
        return {
            "performance": self.prover_performance.copy(),
            "usage_count": self.prover_usage_count.copy()
        }

#==============================================================================
# Parser-Implementierung
#==============================================================================
class HAKGALParser:
    def __init__(self):
        self.parser_available = LARK_AVAILABLE
        if self.parser_available:
            try:
                self.parser = lark.Lark(HAKGAL_GRAMMAR, parser='lalr', debug=False)
                print("✅ Lark-Parser initialisiert")
            except Exception as e:
                print(f"❌ Parser-Initialisierung fehlgeschlagen: {e}")
                self.parser_available = False
        else:
            print("⚠️ Lark-Parser nicht verfügbar, nutze Regex-Fallback")

    def parse(self, formula: str) -> Tuple[bool, Optional['lark.Tree'], str]:
        if not self.parser_available:
            return self._regex_fallback(formula)
        try:
            tree = self.parser.parse(formula)
            return (True, tree, "Syntax OK")
        except lark.exceptions.LarkError as e:
            return (False, None, f"Parser-Fehler: {e}")

    def _regex_fallback(self, formula: str) -> Tuple[bool, None, str]:
        if not formula.strip().endswith('.'): return (False, None, "Formel muss mit '.' enden")
        if not re.match(r'^[A-Za-zÄÖÜäöüß0-9\s\(\),\->&|._-]+$', formula): return (False, None, "Ungültige Zeichen")
        if formula.count('(') != formula.count(')'): return (False, None, "Unbalancierte Klammern")
        return (True, None, "Syntax OK (Regex-Fallback)")

    def extract_predicates(self, tree: 'lark.Tree') -> List[str]:
        predicates = [str(node.children[0]) for node in tree.find_data('atom') if node.children and isinstance(node.children[0], lark.Token) and node.children[0].type == 'PREDICATE']
        return list(dict.fromkeys(predicates))

#==============================================================================
# 4. KONKRETE PROVIDER- UND ADAPTER-IMPLEMENTIERUNGEN
#==============================================================================
class DeepSeekProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str = "deepseek-chat"):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    def query(self, prompt: str, system_prompt: str, temperature: float) -> str:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=temperature)
        return response.choices[0].message.content.strip()

class MistralProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str = "mistral-large-latest"):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1/")
    def query(self, prompt: str, system_prompt: str, temperature: float) -> str:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=temperature)
        return response.choices[0].message.content.strip()

class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro-latest"):
        super().__init__(model_name)
        if not GEMINI_AVAILABLE: raise ImportError("Google Gemini Bibliothek nicht installiert.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    def query(self, prompt: str, system_prompt: str, temperature: float) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}"
        response = self.model.generate_content(full_prompt, generation_config=genai.types.GenerationConfig(temperature=temperature))
        return response.text.strip()

class FunctionalConstraintProver(BaseProver):
    """
    Spezialisierter Prover für funktionale Constraints
    Behandelt Fälle wie Einwohner(X, Y) wo Y eindeutig sein muss
    """
    
    def __init__(self):
        super().__init__("Functional Constraint Prover")
        self.functional_predicates = {
            'Einwohner', 'Hauptstadt', 'Bevölkerung', 'Fläche', 
            'Temperatur', 'Geburtsjahr', 'LiegtIn'
        }
    
    def prove(self, assumptions: list, goal: str) -> tuple[Optional[bool], str]:
        # Extrahiere Prädikat und Argumente aus dem Ziel
        goal_clean = goal.strip().rstrip('.')
        goal_match = re.match(r'([A-ZÄÖÜ][\w]*)\(([^)]+)\)', goal_clean)
        if not goal_match:
            return None, f"{self.name}: Kein atomares Prädikat erkannt in '{goal_clean}'"
        
        goal_predicate, goal_args = goal_match.groups()
        goal_arg_list = [arg.strip() for arg in goal_args.split(',')]
        
        # Nur funktionale Prädikate behandeln
        if goal_predicate not in self.functional_predicates:
            return None, f"{self.name}: '{goal_predicate}' ist nicht funktional"
        
        # Suche nach widersprüchlichen Fakten in Assumptions
        for assumption in assumptions:
            assume_clean = assumption.strip().rstrip('.')
            assume_match = re.match(r'([A-ZÄÖÜ][\w]*)\(([^)]+)\)', assume_clean)
            if not assume_match:
                continue
            
            assume_predicate, assume_args = assume_match.groups()
            assume_arg_list = [arg.strip() for arg in assume_args.split(',')]
            
            # Gleicher Prädikatname und gleiche erste Argumente?
            if (assume_predicate == goal_predicate and 
                len(assume_arg_list) == len(goal_arg_list) and
                len(assume_arg_list) >= 2):
                
                # Prüfe ob erste n-1 Argumente gleich sind, aber letztes unterschiedlich
                if (assume_arg_list[:-1] == goal_arg_list[:-1] and
                    assume_arg_list[-1] != goal_arg_list[-1]):
                    
                    # FUNKTIONALER WIDERSPRUCH!
                    return False, f"{self.name}: Funktionaler Widerspruch - {goal_predicate} kann für {assume_arg_list[:-1]} nicht sowohl {assume_arg_list[-1]} als auch {goal_arg_list[-1]} sein"
        
        return None, f"{self.name}: Kein funktionaler Widerspruch gefunden"
    
    def validate_syntax(self, formula: str) -> tuple[bool, str]:
        return HAKGALParser().parse(formula)[0::2]

class PatternProver(BaseProver):
    def __init__(self):
        super().__init__("Pattern Matcher")
    def prove(self, assumptions: list, goal: str) -> tuple[Optional[bool], str]:
        if goal in assumptions: return (True, f"{self.name} fand exakten Match für '{goal}'.")
        neg_goal = f"-{goal}" if not goal.startswith('-') else goal[1:]
        if neg_goal in assumptions: return (False, f"{self.name} fand Widerspruch für '{goal}'.")
        return (None, f"{self.name} fand keine Übereinstimmung.")
    def validate_syntax(self, formula: str) -> tuple[bool, str]: return HAKGALParser().parse(formula)[0::2]

class Z3Adapter(BaseProver):
    def __init__(self):
        super().__init__("Z3 SMT Solver")
        z3.set_param('proof', True)
        self.parser = HAKGALParser()
        self.func_cache = {}

    def _parse_hakgal_formula_to_z3_expr(self, formula_str: str, quantified_vars: set = None):
        if quantified_vars is None: quantified_vars = set()
        formula_str = formula_str.strip().removesuffix('.')
        def _find_top_level_operator(s: str, operators: list[str]) -> tuple[int, str | None]:
            balance = 0
            for i in range(len(s) - 1, -1, -1):
                if s[i] == ')': balance += 1
                elif s[i] == '(': balance -= 1
                elif balance == 0:
                    for op in operators:
                        if s.startswith(op, i): return i, op
            return -1, None
        expr = formula_str.strip()
        
        # GLEICHHEIT: Spezialbehandlung für = Operator
        if '=' in expr and not any(op in expr for op in ['->', '|', '&']):
            parts = expr.split('=')
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                # Konvertiere beide Seiten zu Z3 Integers
                left_z3 = z3.Int(left) if left in quantified_vars else z3.Int(left)
                right_z3 = z3.Int(right) if right in quantified_vars else z3.Int(right)
                return left_z3 == right_z3
        
        op_map = {'->': z3.Implies, '|': z3.Or, '&': z3.And}
        for op_str, op_func in op_map.items():
            idx, op = _find_top_level_operator(expr, [op_str])
            if idx != -1: return op_func(self._parse_hakgal_formula_to_z3_expr(expr[:idx], quantified_vars), self._parse_hakgal_formula_to_z3_expr(expr[idx + len(op):], quantified_vars))
        if expr.startswith('-'): return z3.Not(self._parse_hakgal_formula_to_z3_expr(expr[1:], quantified_vars))
        if expr.startswith('all '):
            match = re.match(r"all\s+([\w]+)\s+\((.*)\)$", expr, re.DOTALL)
            if not match:
                raise ValueError(f"Ungültiges Quantifikator-Format: '{expr}'")
            var_name, body = match.groups()
            z3_var = z3.Int(var_name)
            return z3.ForAll([z3_var], self._parse_hakgal_formula_to_z3_expr(body, quantified_vars | {var_name}))
        if expr.startswith('(') and expr.endswith(')'): return self._parse_hakgal_formula_to_z3_expr(expr[1:-1], quantified_vars)
        match = re.match(r"([A-ZÄÖÜ][\w]*)\s*\((.*?)\)", expr)
        if match:
            pred_name, args_str = match.groups()
            args = [a.strip() for a in args_str.split(',') if a.strip()]
            z3_args = []
            for arg in args:
                if arg in quantified_vars:
                    z3_args.append(z3.Int(arg))
                elif re.match(r'^[0-9]+([_][0-9]+)*$', arg):  # NUMBER pattern
                    z3_args.append(z3.IntVal(int(arg.replace('_', ''))))
                else:
                    z3_args.append(z3.Int(arg))
            func_sig = (pred_name, len(z3_args))
            if func_sig not in self.func_cache:
                self.func_cache[func_sig] = z3.Function(pred_name, *([z3.IntSort()] * len(z3_args)), z3.BoolSort())
            return self.func_cache[func_sig](*z3_args)
        if re.match(r"^[A-ZÄÖÜ][\w]*$", expr):
            func_sig = (expr, 0)
            if func_sig not in self.func_cache:
                self.func_cache[func_sig] = z3.Function(expr, z3.BoolSort())
            return self.func_cache[func_sig]()
        raise ValueError(f"Konnte Formelteil nicht parsen: '{expr}'")

    def validate_syntax(self, formula: str) -> tuple[bool, str]:
        success, _, msg = self.parser.parse(formula)
        if not success: return (False, f"Parser: {msg}")
        try:
            self.func_cache = {}
            self._parse_hakgal_formula_to_z3_expr(formula)
            return (True, "✅ Syntax OK (Lark + Z3)")
        except (ValueError, z3.Z3Exception) as e:
            return (False, f"Z3-Konvertierung fehlgeschlagen: {e}")

    def prove(self, assumptions: list, goal: str) -> tuple[Optional[bool], str]:
        solver = z3.Tactic('smt').solver(); solver.set(model=True)
        self.func_cache = {}
        try:
            for fact_str in assumptions: solver.add(self._parse_hakgal_formula_to_z3_expr(fact_str))
            goal_expr = self._parse_hakgal_formula_to_z3_expr(goal); solver.add(z3.Not(goal_expr))
        except (ValueError, z3.Z3Exception) as e: return (None, f"Fehler beim Parsen: {e}")
        check_result = solver.check()
        if check_result == z3.unsat: return (True, "Z3 hat das Ziel bewiesen.")
        if check_result == z3.sat: return (False, f"Z3 konnte das Ziel nicht beweisen (Gegenmodell):\n{solver.model()}")
        return (None, f"Z3 konnte das Ziel nicht beweisen (Grund: {solver.reason_unknown()}).")

#==============================================================================
# 5. Shell-Manager
#==============================================================================
class ShellManager:
    def __init__(self):
        self.system = platform.system()
        self.shell = self._detect_shell()
        print(f"🖥️ System: {self.system}, Shell: {self.shell}")

    def _detect_shell(self) -> str:
        if self.system == "Windows":
            if os.path.exists(r"C:\Windows\System32\wsl.exe"): return "wsl"
            if os.path.exists(r"C:\Program Files\Git\bin\bash.exe"): return "git-bash"
            return "powershell"
        return "bash"

    def execute(self, command: str, timeout: int = 30) -> tuple[bool, str, str]:
        try:
            if self.shell == "wsl":
                proc = subprocess.run(["wsl.exe", "bash", "-c", command], capture_output=True, text=True, timeout=timeout, encoding='utf-8')
            elif self.shell == "git-bash":
                proc = subprocess.run([r"C:\Program Files\Git\bin\bash.exe", "-c", command], capture_output=True, text=True, timeout=timeout, encoding='utf-8')
            elif self.shell == "powershell":
                proc = subprocess.run(["powershell", "-Command", command], capture_output=True, text=True, timeout=timeout, encoding='utf-8')
            else: # bash/sh
                proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout, encoding='utf-8')
            return (proc.returncode == 0, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired: return (False, "", f"Timeout: Befehl überschritt {timeout}s")
        except Exception as e: return (False, "", f"Fehler: {e}")

    def analyze_system_facts(self) -> list[str]:
        return [
            f"LäuftAuf({self.system}).",
            f"VerwendetShell({self.shell}).",
            f"PythonVersion({platform.python_version().replace('.', '_')})."
        ]

#==============================================================================
# 6. MANAGER- UND KERN-KLASSEN
#==============================================================================
class EnsembleManager:
    def __init__(self):
        self.providers: List[BaseLLMProvider] = []
        self._initialize_providers()
        self.prompt_cache = PromptCache()
        self.system_prompt_logicalize = """
        You are a hyper-precise, non-conversational logic translator. Your ONLY function is to translate user input into a single HAK/GAL first-order logic formula. You MUST adhere to these rules without exception.

        **HAK/GAL SYNTAX RULES:**
        1.  **Structure:** `Predicate(Argument).` or `all x (...)`.
        2.  **Predicates & Constants:** `PascalCase`. Examples: `IstUrgent`, `UserManagement`.
        3.  **Variables:** `lowercase`. Example: `x`.
        4.  **Operators:** `&` (AND), `|` (OR), `->` (IMPLIES), `-` (NOT).
        5.  **Termination:** Every formula MUST end with a period `.`.
        6.  **NO CONVERSATION:** Do not ask for clarification. Do not explain yourself. Do not add any text before or after the formula. Your response must contain ONLY the formula.
        7.  **VAGUE INPUT RULE:** If the user input is short, vague, or a single word (like "system", "test", "help"), translate it to a generic query about its properties.
            - "system" -> `Eigenschaften(System).`
            - "hakgal" -> `Eigenschaften(HAKGAL).`
            - "test" -> `IstTest().`
        8.  **IDEMPOTENCY:** If the input is already a valid formula, return it UNCHANGED. `IstKritisch(X).` -> `IstKritisch(X).`

        **Translate the following sentence into a single HAK/GAL formula and nothing else:**
        """
        self.fact_extraction_prompt = """
        You are a precise logic extractor. Your task is to extract all facts and rules from the provided text and format them as a Python list of strings. Each string must be a valid HAK/GAL first-order logic formula.

        **HAK/GAL SYNTAX RULES (MUST BE FOLLOWED EXACTLY):**
        1.  **Structure:** `Predicate(Argument).` or `all x (Rule(x)).`
        2.  **Predicates & Constants:** `PascalCase`. (e.g., `IstLegacy`, `UserManagement`)
        3.  **Variables:** Lowercase. (e.g., `x`)
        4.  **Quantifiers:** Rules with variables MUST use `all x (...)`.
        5.  **Operators:** `&` (AND), `|` (OR), `->` (IMPLIES), `-` (NOT).
        6.  **Termination:** Every formula MUST end with a period `.`.
        7.  **Output Format:** A single Python list of strings, and nothing else.

        **Example Extraction:**
        - Text: "The billing system is a legacy system. All legacy systems are critical."
        - Output: `["IstLegacySystem(BillingSystem).", "all x (IstLegacySystem(x) -> IstKritisch(x))."]`

        - Text: "The server is not responding."
        - Output: `["-IstErreichbar(Server)."]`

        **Text to analyze:**
        {context}

        **Output (Python list of strings only):**
        """

    def _initialize_providers(self):
        print("🤖 Initialisiere LLM-Provider-Ensemble...")
        if api_key := os.getenv("DEEPSEEK_API_KEY"): self.providers.append(DeepSeekProvider(api_key=api_key)); print("   ✅ DeepSeek")
        if api_key := os.getenv("MISTRAL_API_KEY"): self.providers.append(MistralProvider(api_key=api_key)); print("   ✅ Mistral")
        if GEMINI_AVAILABLE and (api_key := os.getenv("GEMINI_API_KEY")):
            try: self.providers.append(GeminiProvider(api_key=api_key)); print("   ✅ Gemini")
            except Exception as e: print(f"   ❌ Fehler Gemini: {e}")
        if not self.providers: print("   ⚠️ Keine LLM-Provider aktiv.")

    def logicalize(self, sentence: str) -> str | dict | None:
        if not self.providers: return None
        full_prompt = f"{self.system_prompt_logicalize}\n\n{sentence}"
        if cached_response := self.prompt_cache.get(full_prompt):
            print("   [Cache] ✅ Treffer im Prompt-Cache!")
            try: return json.loads(cached_response)
            except json.JSONDecodeError: return cached_response
        try:
            response_text = self.providers[0].query(sentence, self.system_prompt_logicalize, 0)
            self.prompt_cache.put(full_prompt, response_text)
            try: return json.loads(response_text)
            except json.JSONDecodeError: return response_text
        except Exception as e:
            print(f"   [Warnung] Logik-Übersetzung: {e}")
            return None

    def extract_facts_with_ensemble(self, context: str) -> list[str]:
        if not self.providers: return []
        results: List[Optional[List[str]]] = [None] * len(self.providers)
        def worker(provider: BaseLLMProvider, index: int):
            try:
                prompt = self.fact_extraction_prompt.format(context=context)
                raw_output = provider.query(prompt, "", 0.1)
                if match := re.search(r'\[.*\]', raw_output, re.DOTALL):
                    try:
                        if isinstance(fact_list := eval(match.group(0)), list): results[index] = list(dict.fromkeys(fact_list))
                    except: pass
            except Exception as e: print(f"   [Warnung] {provider.model_name}: {e}")
        
        threads = [threading.Thread(target=worker, args=(p, i)) for i, p in enumerate(self.providers)]
        for t in threads: t.start()
        for t in threads: t.join()
        
        mistral_result = next((results[i] for i, p in enumerate(self.providers) if isinstance(p, MistralProvider) and results[i]), None)
        if mistral_result:
            print(f"   [Ensemble] ✅ Mistral-Veto: {len(mistral_result)} Fakten.")
            return mistral_result

        other_results = [res for i, res in enumerate(results) if res and not isinstance(self.providers[i], MistralProvider)]
        if not other_results: return []
            
        print("   [Ensemble] ⚠️ Fallback auf Mehrheitsentscheid...")
        fact_counts = Counter(fact for res in other_results for fact in res)
        threshold = len(other_results) // 2 + 1 if len(other_results) > 1 else 1
        consistent_facts = [fact for fact, count in fact_counts.items() if count >= threshold]
        if consistent_facts: print(f"   [Ensemble] Konsens für {len(consistent_facts)} Fakten.")
        return consistent_facts

class WissensbasisManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if not RAG_ENABLED:
            self.model, self.index, self.chunks, self.doc_paths = None, None, [], {}
            print("   ℹ️  RAG-Funktionen deaktiviert.")
            return
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks: List[Dict[str, str]] = []
        self.doc_paths: Dict[str, str] = {}
        
    def add_document(self, file_path: str):
        if not RAG_ENABLED: return
        doc_id = os.path.basename(file_path)
        if doc_id in self.doc_paths:
            print(f"   ℹ️ '{file_path}' bereits indiziert.")
            return
        try:
            if file_path.endswith(".pdf"):
                reader = PdfReader(file_path)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            else:
                with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
        except Exception as e:
            print(f"   ❌ Fehler beim Lesen von '{file_path}': {e}")
            return
        
        text_chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk.strip()) > 30]
        if not text_chunks:
            print(f"   ℹ️ Keine Chunks in '{file_path}' gefunden.")
            return
        
        embeddings = self.model.encode(text_chunks, convert_to_tensor=False, show_progress_bar=True)
        self.index.add(np.array(embeddings).astype('float32'))
        for chunk in text_chunks: self.chunks.append({'text': chunk, 'source': doc_id})
        self.doc_paths[doc_id] = file_path
        print(f"   ✅ {len(text_chunks)} Chunks aus '{doc_id}' indiziert.")
        
    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        if not RAG_ENABLED or self.index.ntotal == 0: return []
        query_embedding = self.model.encode([query])
        _, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        return [self.chunks[i] for i in indices[0] if i != -1 and i < len(self.chunks)]

class HAKGAL_Core_FOL:
    def __init__(self):
        self.K: List[str] = []
        self.provers: List[BaseProver] = []
        self._initialize_provers()
        
        # Neue Archon-Prime Komponenten
        self.complexity_analyzer = ComplexityAnalyzer()
        self.portfolio_manager = ProverPortfolioManager(self.complexity_analyzer)
        
        self.parser_stats = {"total": 0, "success": 0, "failed": 0}
        self.proof_cache = ProofCache()

    def _initialize_provers(self):
        """Initialisiert verfügbare Prover"""
        self.provers = [FunctionalConstraintProver(), PatternProver(), Z3Adapter()]
        
        # Wolfram-Prover hinzufügen wenn verfügbar
        if WOLFRAM_INTEGRATION:
            try:
                wolfram_prover = WolframProver()
                if wolfram_prover.client:  # Nur hinzufügen wenn konfiguriert
                    self.provers.append(wolfram_prover)
                    print("✅ WolframProver zum Portfolio hinzugefügt")
            except Exception as e:
                print(f"⚠️ Wolfram-Prover Initialisierung fehlgeschlagen: {e}")

    def add_fact(self, formula_str: str):
        if formula_str not in self.K:
            self.K.append(formula_str)
            self.proof_cache.clear()
            return True
        return False

    def retract_fact(self, fact_to_remove: str) -> bool:
        if fact_to_remove in self.K:
            self.K.remove(fact_to_remove)
            self.proof_cache.clear()
            return True
        return False

    def check_consistency(self, new_fact: str) -> Tuple[bool, Optional[str]]:
        # 1. Prüfe ob Negation bereits beweisbar ist
        negated_fact = f"-{new_fact}" if not new_fact.startswith('-') else new_fact[1:]
        is_contradictory, reason = self.verify_logical(negated_fact, self.K)
        if is_contradictory:
            return (False, f"Widerspruch! Neuer Fakt '{new_fact}' widerspricht KB ({reason})")
        
        # 2. NEUE PRÜFUNG: Teste funktionale Widersprüche direkt
        functional_prover = next((p for p in self.provers if p.name == "Functional Constraint Prover"), None)
        if functional_prover:
            success, reason = functional_prover.prove(self.K, new_fact)
            if success is False:  # Funktionaler Widerspruch erkannt
                return (False, f"Funktionaler Widerspruch! {reason}")
        
        return (True, None)

    def verify_logical(self, query_str: str, full_kb: list) -> tuple[Optional[bool], str]:
        cache_key = (tuple(sorted(full_kb)), query_str)
        if cached_result := self.proof_cache.get(query_str, cache_key):
            print("   [Cache] ✅ Treffer im Proof-Cache!")
            return cached_result[0], cached_result[1]

        # NEUE ARCHON-PRIME LOGIK: Intelligente Prover-Auswahl
        ordered_provers = self.portfolio_manager.select_prover_strategy(query_str, self.provers)
        
        # Sequenzielle Ausführung nach Priorität (nicht parallel)
        for prover in ordered_provers:
            start_time = time.time()
            try:
                success, reason = prover.prove(full_kb, query_str)
                duration = time.time() - start_time
                
                # Performance-Update für Portfolio-Manager
                self.portfolio_manager.update_performance(prover.name, query_str, success is not None, duration)
                
                if success is not None:
                    if success: 
                        self.proof_cache.put(query_str, cache_key, success, reason)
                        print(f"   [✅ Erfolg] {prover.name} nach {duration:.2f}s")
                    return success, reason
                else:
                    print(f"   [⏭️ Weiter] {prover.name} nach {duration:.2f}s - {reason}")
                    
            except Exception as e: 
                duration = time.time() - start_time
                self.portfolio_manager.update_performance(prover.name, query_str, False, duration)
                print(f"   [❌ Fehler] {prover.name}: {e}")
        
        return (None, "Kein Prover konnte eine definitive Antwort finden.")

    def update_parser_stats(self, success: bool):
        self.parser_stats["total"] += 1
        if success: self.parser_stats["success"] += 1
        else: self.parser_stats["failed"] += 1

    def get_portfolio_stats(self) -> Dict[str, Any]:
        """Gibt Portfolio-Performance-Statistiken zurück"""
        return self.portfolio_manager.get_performance_report()

#==============================================================================
# 7. K-ASSISTANT MIT WOLFRAM-INTEGRATION
#==============================================================================
class KAssistant:
    def __init__(self, kb_filepath="k_assistant.kb"):
        self.kb_filepath = kb_filepath
        self.core = HAKGAL_Core_FOL()
        self.ensemble_manager = EnsembleManager()
        self.wissensbasis_manager = WissensbasisManager()
        self.shell_manager = ShellManager()
        self.parser = HAKGALParser()
        self.potential_new_facts: List[str] = []
        self.load_kb(kb_filepath)
        self._add_system_facts()
        self._add_functional_constraints()  # NEU: Funktionale Constraints
        prover_names = ', '.join([p.name for p in self.core.provers])
        print(f"--- Prover-Portfolio (Archon-Prime): {prover_names} ---")
        print(f"--- Parser-Modus: {'Lark' if self.parser.parser_available else 'Regex-Fallback'} ---")
        if WOLFRAM_INTEGRATION:
            print("--- Wolfram|Alpha Integration: Aktiv ---")
    
    def _normalize_and_correct_syntax(self, formula: str) -> str:
        original_formula = formula
        # Wichtig: Bindestriche in Entitäten entfernen, um Konsistenz zu gewährleisten
        corrected = re.sub(r'\b([A-ZÄÖÜ][a-zA-ZÄÖÜ0-9]*)-([a-zA-ZÄÖÜ0-9]+)\b', r'\1\2', formula)

        synonym_map = {
            r'IstTechnischesLegacy(System)?': 'IstLegacy',
            r'SollteRefactoringInBetrachtGezo[h|e]genWerden': 'SollteRefactoredWerden',
            r'SollteIdentifiziertUndRefactoredWerden': 'SollteRefactoredWerden',
            r'IstBasierendAufCobolMainframe': 'IstCobolMainframe',
            r'BasiertAufCobolMainframe': 'IstCobolMainframe',
            r'BasiertAufModernerJavaMicroservice': 'IstJavaMicroservice',
            r'IstBasierendAufJavaMicroservice': 'IstJavaMicroservice',
            r'HatGeringeBetriebskosten': 'HatNiedrigeBetriebskosten',
        }
        
        for pattern, canonical in synonym_map.items():
            corrected = re.sub(pattern, canonical, corrected)
            
        corrected = corrected.strip().replace(':-', '->').replace('~', '-')
        while corrected.startswith('--'): corrected = corrected[2:]
        if re.match(r"^[A-ZÄÖÜ][a-zA-Z0-9_]*\.$", corrected): corrected = corrected.replace('.', '().')
        
        if corrected != original_formula: 
            print(f"   [Normalisierung] '{original_formula.strip()}' -> '{corrected.strip()}'")
        return corrected

    def _add_system_facts(self):
        system_facts = self.shell_manager.analyze_system_facts()
        for fact in system_facts:
            if fact not in self.core.K: self.core.K.append(fact)
        print(f"   ✅ {len(system_facts)} Systemfakten hinzugefügt.")
    
    def _add_functional_constraints(self):
        """Fügt funktionale Constraints für eindeutige Relationen hinzu"""
        functional_constraints = [
            # Eine Stadt hat nur EINE Einwohnerzahl
            "all x all y all z ((Einwohner(x, y) & Einwohner(x, z)) -> (y = z)).",
            # Ein Land hat nur EINE Hauptstadt  
            "all x all y all z ((Hauptstadt(x, y) & Hauptstadt(x, z)) -> (y = z)).",
            # Eine Stadt liegt nur in EINEM Land
            "all x all y all z ((LiegtIn(x, y) & LiegtIn(x, z)) -> (y = z)).",
            # Ein Objekt hat nur EINE Fläche
            "all x all y all z ((Fläche(x, y) & Fläche(x, z)) -> (y = z)).",
            # Bevölkerung ist auch funktional
            "all x all y all z ((Bevölkerung(x, y) & Bevölkerung(x, z)) -> (y = z)).",
            # Eine Person hat nur EIN Geburtsjahr
            "all x all y all z ((Geburtsjahr(x, y) & Geburtsjahr(x, z)) -> (y = z)).",
            # Eine Stadt hat nur EINE Temperatur (zu einem Zeitpunkt)
            "all x all y all z ((Temperatur(x, y) & Temperatur(x, z)) -> (y = z))."
        ]
        
        added = 0
        for constraint in functional_constraints:
            if constraint not in self.core.K:
                self.core.K.append(constraint)
                added += 1
        
        if added > 0:
            print(f"   ✅ {added} funktionale Constraints hinzugefügt.")
    
    def _ask_or_explain(self, q: str, explain: bool, is_raw: bool):
        print(f"\n> {'Erklärung für' if explain else 'Frage'}{' (roh)' if is_raw else ''}: '{q}'")
        self.potential_new_facts = []
        temp_assumptions = []
        
        logical_form = ""
        if is_raw:
            logical_form = self._normalize_and_correct_syntax(q)
        else:
            if RAG_ENABLED and self.wissensbasis_manager.index.ntotal > 0:
                print("🧠 RAG-Pipeline wird für Kontext angereichert...")
                relevant_chunks = self.wissensbasis_manager.retrieve_relevant_chunks(q)
                if relevant_chunks:
                    context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
                    print(f"   [RAG] Relevanter Kontext gefunden. Extrahiere Fakten...")
                    extracted_facts = self.ensemble_manager.extract_facts_with_ensemble(context)
                    for fact in extracted_facts:
                        corrected_fact = self._normalize_and_correct_syntax(fact)
                        is_valid, _, _ = self.parser.parse(corrected_fact)
                        if is_valid:
                            temp_assumptions.append(corrected_fact)
                            if corrected_fact not in self.core.K and corrected_fact not in self.potential_new_facts:
                                self.potential_new_facts.append(corrected_fact)
                    if temp_assumptions: print(f"   [RAG] {len(temp_assumptions)} temporäre Fakten hinzugefügt.")
            
            print("🔮 Übersetze Anfrage in Logik...")
            logical_form_raw = self.ensemble_manager.logicalize(q)
            
            if not logical_form_raw or not isinstance(logical_form_raw, str) or " " in logical_form_raw.split("(")[0]:
                print(f"   ❌ FEHLER: LLM hat eine konversationelle oder ungültige Antwort gegeben. Antwort war: '{logical_form_raw}'")
                print("   ℹ️ Tipp: Versuchen Sie, die Frage präziser zu formulieren.")
                return

            logical_form = self._normalize_and_correct_syntax(logical_form_raw)
        
        print(f"   -> Logische Form: '{logical_form}'")
        is_valid, _, msg = self.parser.parse(logical_form)
        self.core.update_parser_stats(is_valid)
        if not is_valid: 
            print(f"   ❌ FEHLER: Ungültige Syntax. {msg}")
            return
        
        print(f"🧠 Archon-Prime Portfolio-Manager übernimmt...")
        success, reason = self.core.verify_logical(logical_form, self.core.K + temp_assumptions)
        
        print("\n--- ERGEBNIS ---")
        if not explain:
            print("✅ Antwort: Ja." if success else "❔ Antwort: Nein/Unbekannt.")
            print(f"   [Begründung] {reason}")
        else:
            success_status_text = "Ja (bewiesen)" if success else "Nein (nicht bewiesen)"
            if not self.ensemble_manager.providers: print("   ❌ Keine Erklärung (keine LLMs)."); return
            print("🗣️  Generiere einfache Erklärung...")
            explanation_prompt = f"Anfrage: '{q}', Ergebnis: {success_status_text}, Grund: '{reason}'. Übersetze dies in eine einfache Erklärung."
            explanation = self.ensemble_manager.providers[0].query(explanation_prompt, "Du bist ein Logik-Experte, der formale Beweise in einfache Sprache übersetzt.", 0.2)
            print(f"--- Erklärung ---\n{explanation}\n-------------------\n")

        if self.potential_new_facts:
            print(f"💡 INFO: {len(self.potential_new_facts)} neue Fakten gefunden. Benutze 'learn', um sie zu speichern.")

    def add_raw(self, formula: str):
        print(f"\n> Füge KERNREGEL hinzu: '{formula}'")
        normalized_formula = self._normalize_and_correct_syntax(formula)
        
        is_valid, _, msg = self.parser.parse(normalized_formula)
        self.core.update_parser_stats(is_valid)
        if not is_valid:
            print(f"   ❌ FEHLER nach Normalisierung: Ungültige Syntax. {msg}")
            return

        is_consistent, reason = self.core.check_consistency(normalized_formula)
        if not is_consistent:
            print(f"   🛡️  WARNUNG: {reason}")
            return
        
        if self.core.add_fact(normalized_formula):
            print("   -> Erfolgreich hinzugefügt.")
        else:
            print("   -> Fakt bereits vorhanden.")
        
    def retract(self, formula_to_retract: str):
        print(f"\n> Entferne KERNREGEL: '{formula_to_retract}'")
        normalized_target = self._normalize_and_correct_syntax(formula_to_retract)
        # Suche nach der normalisierten Form in der KB (die jetzt auch normalisiert ist)
        if self.core.retract_fact(normalized_target):
            print(f"   -> Fakt '{normalized_target}' entfernt.")
        else:
            print(f"   -> Fakt '{normalized_target}' nicht gefunden.")

    def learn_facts(self):
        if not self.potential_new_facts: print("🧠 Nichts Neues zu lernen."); return
        print(f"🧠 Lerne {len(self.potential_new_facts)} neue Fakten...")
        added_count = 0
        for fact in self.potential_new_facts:
            # Fakten aus RAG sind bereits normalisiert
            if self.core.check_consistency(fact)[0] and self.core.add_fact(fact):
                added_count += 1
        
        if added_count > 0: print(f"✅ {added_count} neue Fakten gespeichert.")
        else: print("ℹ️ Alle Fakten waren bereits bekannt oder inkonsistent.")
        self.potential_new_facts = []

    def clear_cache(self): 
        self.core.proof_cache.clear()
        self.ensemble_manager.prompt_cache.clear()
        # Auch Wolfram-Cache leeren wenn verfügbar
        for prover in self.core.provers:
            if hasattr(prover, 'clear_cache'):
                prover.clear_cache()

    def status(self): 
        print(f"\n--- System Status ---")
        pc, pmc = self.core.proof_cache, self.ensemble_manager.prompt_cache
        stats = self.core.parser_stats
        success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  Prover: {', '.join([p.name for p in self.core.provers])} (Archon-Prime Portfolio)")
        print(f"  Parser: {'Lark' if self.parser.parser_available else 'Regex'} | Versuche: {stats['total']} | Erfolg: {success_rate:.1f}%")
        print(f"  Wissen: {len(self.core.K)} Kernfakten | {len(self.potential_new_facts)} lernbare Fakten")
        print(f"  Caches: Beweise={pc.size} (Rate {pc.hit_rate:.1f}%) | Prompts={pmc.size} (Rate {pmc.hit_rate:.1f}%)")
        if RAG_ENABLED: print(f"  RAG: {self.wissensbasis_manager.index.ntotal} Chunks aus {len(self.wissensbasis_manager.doc_paths)} Docs")
        
        # Portfolio-Performance anzeigen
        portfolio_stats = self.core.get_portfolio_stats()
        if portfolio_stats["performance"]:
            print(f"\n--- Portfolio Performance ---")
            for prover_name, perf in portfolio_stats["performance"].items():
                usage = portfolio_stats["usage_count"].get(prover_name, 0)
                print(f"  {prover_name}: {perf['success_rate']:.1%} Erfolg, {perf['avg_duration']:.2f}s ⌀, {usage}x verwendet")

    def show(self) -> Dict[str, Any]:
        permanent_knowledge = sorted(self.core.K)
        learnable_facts = sorted(self.potential_new_facts)
        rag_chunks_summary = []
        if RAG_ENABLED and self.wissensbasis_manager.chunks:
            for i, chunk_info in enumerate(self.wissensbasis_manager.chunks):
                rag_chunks_summary.append({
                    "id": i, "source": chunk_info.get('source', 'Unbekannt'),
                    "text_preview": chunk_info.get('text', '')[:80] + "..."
                })
        return {
            "permanent_knowledge": permanent_knowledge, "learnable_facts": learnable_facts,
            "rag_chunks": rag_chunks_summary,
            "rag_stats": {
                "doc_count": len(self.wissensbasis_manager.doc_paths) if RAG_ENABLED else 0,
                "chunk_count": self.wissensbasis_manager.index.ntotal if RAG_ENABLED and self.wissensbasis_manager.index else 0,
            },
            "portfolio_stats": self.core.get_portfolio_stats()
        }
        
    def save_kb(self, filepath: str):
        try:
            rag_data = {'chunks': self.wissensbasis_manager.chunks, 'doc_paths': self.wissensbasis_manager.doc_paths} if RAG_ENABLED else {}
            portfolio_data = self.core.get_portfolio_stats()
            data = {
                'facts': self.core.K, 
                'rag_data': rag_data, 
                'parser_stats': self.core.parser_stats, 
                'proof_cache': self.core.proof_cache.cache,
                'portfolio_stats': portfolio_data
            }
            with open(filepath, 'wb') as f: pickle.dump(data, f)
            print(f"✅ Wissensbasis in '{filepath}' gespeichert.")
        except Exception as e: print(f"❌ Fehler beim Speichern: {e}")
        
    def load_kb(self, filepath: str):
        if not os.path.exists(filepath): return
        try:
            with open(filepath, 'rb') as f: data = pickle.load(f)
            self.core.K = data.get('facts', [])
            self.core.parser_stats = data.get('parser_stats', {"total": 0, "success": 0, "failed": 0})
            self.core.proof_cache.cache = data.get('proof_cache', {})
            
            # Portfolio-Daten laden
            if 'portfolio_stats' in data:
                portfolio_data = data['portfolio_stats']
                if 'performance' in portfolio_data:
                    self.core.portfolio_manager.prover_performance = portfolio_data['performance']
                if 'usage_count' in portfolio_data:
                    self.core.portfolio_manager.prover_usage_count = portfolio_data['usage_count']
                print(f"✅ Portfolio-Performance geladen")
            
            print(f"✅ {len(self.core.K)} Kernregeln und {len(self.core.proof_cache.cache)} gecachte Beweise geladen.")
            
            if RAG_ENABLED and 'rag_data' in data and data['rag_data']:
                rag_chunks_with_meta = data['rag_data'].get('chunks', [])
                converted_chunks = []
                if rag_chunks_with_meta and isinstance(rag_chunks_with_meta[0], tuple):
                    print("   [Migration] Altes KB-Format (Tupel) erkannt. Konvertiere RAG-Chunks...")
                    for chunk_tuple in rag_chunks_with_meta:
                        if len(chunk_tuple) == 2: converted_chunks.append({'text': chunk_tuple[0], 'source': chunk_tuple[1]})
                    self.wissensbasis_manager.chunks = converted_chunks
                else:
                    self.wissensbasis_manager.chunks = rag_chunks_with_meta
                
                self.wissensbasis_manager.doc_paths = data['rag_data'].get('doc_paths', {})
                if self.wissensbasis_manager.chunks:
                    just_chunks = [c['text'] for c in self.wissensbasis_manager.chunks]
                    embeddings = self.wissensbasis_manager.model.encode(just_chunks, convert_to_tensor=False, show_progress_bar=False)
                    self.wissensbasis_manager.index.add(np.array(embeddings).astype('float32'))
                    print(f"✅ {len(self.wissensbasis_manager.chunks)} RAG-Chunks aus Speicher geladen und indiziert.")
        except Exception as e: print(f"❌ Fehler beim Laden der KB: {e}")
        
    def what_is(self, entity: str):
        # Wichtig: Auch hier die normalisierte Entität verwenden
        normalized_entity = self._normalize_and_correct_syntax(entity).replace('.','').replace('(','').replace(')','')
        print(f"\n> Analysiere Wissen über Entität: '{normalized_entity}'")

        explicit_facts = [fact for fact in self.core.K if f"({normalized_entity})" in fact or f",{normalized_entity})" in fact or f"({normalized_entity}," in fact]
        unary_predicates_to_test = ["IstLegacy", "IstKritisch", "IstOnline", "IstStabil", "HatHoheBetriebskosten", "SollteRefactoredWerden", "MussÜberwacht"]
        derived_properties = []
        print("🧠 Leite Eigenschaften ab...")
        for pred in unary_predicates_to_test:
            positive_goal = f"{pred}({normalized_entity})."
            is_positive, _ = self.core.verify_logical(positive_goal, self.core.K)
            if is_positive and positive_goal not in explicit_facts:
                derived_properties.append(positive_goal)
                continue
            negative_goal = f"-{pred}({normalized_entity})."
            is_negative, _ = self.core.verify_logical(negative_goal, self.core.K)
            if is_negative and negative_goal not in explicit_facts:
                derived_properties.append(negative_goal)

        print("\n" + f"--- Profil für: {normalized_entity} ---".center(60, "-"))
        print("\n  [Explizite Fakten]")
        if explicit_facts: [print(f"   - {f}") for f in sorted(explicit_facts)]
        else: print("   (Keine)")
        print("\n  [Abgeleitete Eigenschaften]")
        if derived_properties: [print(f"   - {p}") for p in sorted(derived_properties)]
        else: print("   (Keine)")
        print("-" * 60)

    # NEUE METHODEN FÜR WOLFRAM-SPEZIFISCHE FUNKTIONEN
    def add_oracle_predicate(self, predicate: str):
        """Fügt ein neues Oracle-Prädikat zur Erkennung hinzu"""
        if hasattr(self.core, 'complexity_analyzer'):
            self.core.complexity_analyzer.oracle_predicates.add(predicate)
            print(f"✅ Oracle-Prädikat '{predicate}' hinzugefügt")

            
    def test_wolfram(self, query: str = "HauptstadtVon(Deutschland)."):
        """Testet die Wolfram-Integration mit einer einfachen Anfrage"""
        wolfram_prover = next((p for p in self.core.provers if p.name == "Wolfram|Alpha Orakel"), None)
        if not wolfram_prover:
            print("❌ Wolfram-Prover nicht im Portfolio gefunden")
            return
            
        if not hasattr(wolfram_prover, 'client') or not wolfram_prover.client:
            print("❌ Wolfram-Client nicht konfiguriert")
            return
            
        print(f"🧪 Teste Wolfram mit: {query}")
        try:
            success, reason = wolfram_prover.prove([], query)
            if success is True:
                print(f"✅ Wolfram-Test erfolgreich: {reason}")
            elif success is False:
                print(f"❌ Wolfram-Test negativ: {reason}")
            else:
                print(f"⚠️ Wolfram-Test unbestimmt: {reason}")
        except Exception as e:
            print(f"❌ Wolfram-Test fehlgeschlagen: {e}")

    def wolfram_stats(self):
        """Zeigt Wolfram-spezifische Statistiken"""
        wolfram_prover = next((p for p in self.core.provers if p.name == "Wolfram|Alpha Orakel"), None)
        if wolfram_prover and hasattr(wolfram_prover, 'client') and wolfram_prover.client:
            print(f"\n--- Wolfram|Alpha Statistiken ---")
            print(f"  Client: Aktiv")
            print(f"  Cache-Einträge: {len(getattr(wolfram_prover, 'cache', {}))}")  
            print(f"  Cache-Timeout: {getattr(wolfram_prover, 'cache_timeout', 3600)}s")
            print(f"  App ID: Konfiguriert")
        else:
            print("⚠️ Wolfram|Alpha nicht verfügbar oder nicht konfiguriert")
            print("   Lösungsvorschläge:")
            print("   1. Wolfram App ID in .env konfigurieren")
            print("   2. 'pip install wolframalpha' ausführen")
            print("   3. Backend neu starten")

    def ask(self, q: str): self._ask_or_explain(q, explain=False, is_raw=False)
    def explain(self, q: str): self._ask_or_explain(q, explain=True, is_raw=False)
    def ask_raw(self, formula: str): self._ask_or_explain(formula, explain=False, is_raw=True)
    def build_kb_from_file(self, filepath: str): self.wissensbasis_manager.add_document(filepath)
    def search(self, query: str):
        if not RAG_ENABLED: print("   ❌ RAG-Funktionen sind deaktiviert."); return
        print(f"\n> Suche Kontext für: '{query}'")
        if not (chunks := self.wissensbasis_manager.retrieve_relevant_chunks(query)):
            print("   [RAG] Keine relevanten Informationen gefunden."); return
        print(f"   [RAG] Relevanter Kontext:\n---")
        for i, chunk in enumerate(chunks, 1): print(f"[{i} from {chunk['source']}] {chunk['text']}\n")
    def sources(self):
        if not RAG_ENABLED: print("   ❌ RAG-Funktionen sind deaktiviert."); return
        print("\n📑 Indizierte Wissensquellen:")
        if not self.wissensbasis_manager.doc_paths: print("   (Keine)")
        else:
            for doc_id, path in self.wissensbasis_manager.doc_paths.items(): print(f"   - {doc_id} (aus {path})")
    def execute_shell(self, command: str): self.shell_manager.execute(command)
    def test_parser(self, formula: str):
        print(f"\n> Parser-Test für: '{formula}'")
        # Normalisierung auch hier für konsistente Tests
        normalized_formula = self._normalize_and_correct_syntax(formula)
        success, tree, msg = self.parser.parse(normalized_formula)
        self.core.update_parser_stats(success)
        if success:
            print(f"✅ Parse erfolgreich: {msg}")
            if tree: print(f"   Gefundene Prädikate: {', '.join(self.parser.extract_predicates(tree))}")
        else: print(f"❌ Parse fehlgeschlagen: {msg}")

#==============================================================================
# 8. MAIN LOOP MIT ERWEITERTEN BEFEHLEN
#==============================================================================
def print_help():
    print("\n" + " K-Assistant Hilfe (Archon-Prime mit Wolfram) ".center(70, "-"))
    print("  build_kb <pfad>      - Indiziert Dokument für RAG")
    print("  add_raw <formel>     - Fügt KERNREGEL hinzu")
    print("  retract <formel>     - Entfernt KERNREGEL")
    print("  learn                - Speichert gefundene Fakten")
    print("  show                 - Zeigt Wissensbasis an")
    print("  sources              - Zeigt Wissensquellen an")
    print("  search <anfrage>     - Findet Text in der KB (RAG)")
    print("  ask <frage>          - Beantwortet Frage (mit RAG + Wolfram)")
    print("  explain <frage>      - Erklärt eine Antwort")
    print("  ask_raw <formel>     - Stellt rohe logische Frage")
    print("  what_is <entity>     - Zeigt Profil einer Entität an")
    print("  status               - Zeigt Systemstatus und Portfolio-Metriken")
    print("  wolfram_stats        - Zeigt Wolfram|Alpha Cache-Statistiken")
    print("  add_oracle <pred>    - Fügt Oracle-Prädikat hinzu")
    print("  test_wolfram [query] - Testet Wolfram-Integration")
    print("  shell <befehl>       - Führt Shell-Befehl aus")
    print("  parse <formel>       - Testet Parser mit Formel")
    print("  clearcache           - Leert alle Caches")
    print("  exit                 - Beendet und speichert die KB")
    print("-" * 70 + "\n")

def main_loop():
    try:
        assistant = KAssistant()
        print_help()
        
        def show_in_console(assistant):
            data = assistant.show()
            print("\n--- Permanente Wissensbasis (Kernregeln) ---")
            if not data['permanent_knowledge']: print("   (Leer)")
            else:
                for i, fact in enumerate(data['permanent_knowledge']): print(f"   [{i}] {fact}")
            
            if data['learnable_facts']:
                print("\n--- Vorgeschlagene Fakten (mit 'learn' übernehmen) ---")
                for i, fact in enumerate(data['learnable_facts']): print(f"   [{i}] {fact}")

            print("\n--- Indizierte Wissens-Chunks ---")
            stats = data['rag_stats']
            print(f"   (Dokumente: {stats['doc_count']}, Chunks: {stats['chunk_count']})")
            if not data['rag_chunks']: print("   (Leer oder RAG deaktiviert)")
            else:
                for chunk in data['rag_chunks'][:3]:
                    print(f"   [{chunk['id']} from {chunk['source']}] {chunk['text_preview']}")
                if len(data['rag_chunks']) > 3:
                    print(f"   ... und {len(data['rag_chunks']) - 3} weitere Chunks.")

        command_map = {
            "exit": lambda a, args: a.save_kb(a.kb_filepath),
            "help": lambda a, args: print_help(),
            "build_kb": lambda a, args: a.build_kb_from_file(args),
            "add_raw": lambda a, args: a.add_raw(args), 
            "retract": lambda a, args: a.retract(args),
            "learn": lambda a, args: a.learn_facts(), 
            "clearcache": lambda a, args: a.clear_cache(),
            "ask": lambda a, args: a.ask(args), 
            "explain": lambda a, args: a.explain(args),
            "ask_raw": lambda a, args: a.ask_raw(args), 
            "status": lambda a, args: a.status(),
            "show": lambda a, args: show_in_console(a),
            "search": lambda a, args: a.search(args),
            "sources": lambda a, args: a.sources(), 
            "what_is": lambda a, args: a.what_is(args),
            "shell": lambda a, args: a.execute_shell(args), 
            "parse": lambda a, args: a.test_parser(args),
            # NEUE WOLFRAM-BEFEHLE
            "wolfram_stats": lambda a, args: a.wolfram_stats(),
            "add_oracle": lambda a, args: a.add_oracle_predicate(args),
            "test_wolfram": lambda a, args: a.test_wolfram(args) if args else a.test_wolfram(),
        }
        
        while True:
            try:
                user_input = input("k-assistant> ").strip()
                if not user_input: continue
                parts = user_input.split(" ", 1)
                command, args = parts[0].lower(), parts[1].strip('"\'') if len(parts) > 1 else ""
                
                no_args_commands = ["exit", "help", "learn", "clearcache", "status", "show", "sources", "wolfram_stats"]
                
                if command in command_map:
                    if command in no_args_commands and args:
                        print(f"Befehl '{command}' erwartet keine Argumente.")
                    elif command not in no_args_commands and not args:
                         print(f"Befehl '{command}' benötigt ein Argument.")
                    else:
                        command_map[command](assistant, args)
                        if command == "exit": break
                else: print(f"Unbekannter Befehl: '{command}'. Geben Sie 'help' für Hilfe ein.")
            except (KeyboardInterrupt, EOFError):
                print("\nBeende... Speichere Wissensbasis.")
                assistant.save_kb(assistant.kb_filepath)
                break
            except Exception as e:
                import traceback
                print(f"\n🚨 Unerwarteter Fehler: {e}"); traceback.print_exc()
    except Exception as e:
        import traceback
        print(f"\n🚨 Kritischer Startfehler: {e}"); traceback.print_exc()

if __name__ == "__main__":
    main_loop()
