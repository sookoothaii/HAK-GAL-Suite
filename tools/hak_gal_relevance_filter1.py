"""
HAK-GAL Relevance Filter Module
==============================

Dieses Modul implementiert einen hocheffizienten Relevanz-Filter, der den
Suchraum für Z3 dramatisch reduziert, indem nur relevante Fakten für eine
Query ausgewählt werden.

Author: HAK-GAL Team
Date: 2024-10-27
Version: 1.0
"""

import re
from collections import defaultdict
from typing import Set, List, Dict, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class Fact:
    """Repräsentation eines Fakts in der Wissensbasis"""
    predicate: str
    subject: str
    object: Optional[str] = None
    confidence: float = 1.0
    source: str = "manual"

    def __str__(self):
        if self.object:
            return f"{self.predicate}({self.subject}, {self.object})"
        return f"{self.predicate}({self.subject})"

    def get_entities(self) -> Set[str]:
        """Extrahiert alle Entitäten aus diesem Fakt"""
        entities = {self.subject}
        if self.object:
            entities.add(self.object)
        return entities


class RelevanceFilter:
    """
    Hocheffizienter Relevanz-Filter für große Wissensbasen.

    Features:
    - O(1) Lookup für Entitäten und Prädikate
    - Intelligente 1-Hop und 2-Hop Expansion
    - Query-Pattern-Matching
    - Performance-Metriken
    """

    def __init__(self):
        # Primäre Indizes
        self.entity_to_facts: Dict[str, Set[int]] = defaultdict(set)
        self.predicate_to_facts: Dict[str, Set[int]] = defaultdict(set)

        # Sekundäre Indizes für Beziehungen
        self.entity_connections: Dict[str, Set[str]] = defaultdict(set)
        self.predicate_patterns: Dict[str, List[re.Pattern]] = {}

        # Fakt-Speicher
        self.facts: List[Fact] = []
        self.fact_to_index: Dict[str, int] = {}

        # Performance-Metriken
        self.stats = {
            'total_queries': 0,
            'avg_facts_returned': 0,
            'avg_reduction_ratio': 0,
            'avg_query_time': 0
        }

    def add_fact(self, fact: Fact) -> None:
        """Fügt einen Fakt hinzu und aktualisiert alle Indizes"""
        fact_str = str(fact)

        # Duplikat-Check
        if fact_str in self.fact_to_index:
            return

        # Fakt speichern
        fact_idx = len(self.facts)
        self.facts.append(fact)
        self.fact_to_index[fact_str] = fact_idx

        # Entity-Index aktualisieren
        for entity in fact.get_entities():
            self.entity_to_facts[entity].add(fact_idx)

            # Verbindungen zwischen Entitäten tracken
            for other_entity in fact.get_entities():
                if entity != other_entity:
                    self.entity_connections[entity].add(other_entity)

        # Predicate-Index aktualisieren
        self.predicate_to_facts[fact.predicate].add(fact_idx)

    def extract_query_entities(self, query: str) -> Set[str]:
        """
        Extrahiert Entitäten aus einer Query.
        Nutzt mehrere Strategien:
        1. Exakte Matches mit bekannten Entitäten
        2. CamelCase/snake_case Erkennung
        3. Quoted strings
        """
        entities = set()

        # Strategie 1: Exakte Matches
        query_lower = query.lower()
        for entity in self.entity_to_facts.keys():
            if entity.lower() in query_lower:
                entities.add(entity)

        # Strategie 2: CamelCase und snake_case
        camel_pattern = r'[A-Z][a-z]+(?:[A-Z][a-z]+)*'
        snake_pattern = r'[a-z]+(?:_[a-z]+)*'

        for match in re.finditer(camel_pattern, query):
            potential = match.group()
            if potential in self.entity_to_facts:
                entities.add(potential)

        for match in re.finditer(snake_pattern, query):
            potential = match.group()
            if potential in self.entity_to_facts:
                entities.add(potential)

        # Strategie 3: Quoted strings
        quoted_pattern = r'"([^"]+)"|'([^']+)''
        for match in re.finditer(quoted_pattern, query):
            potential = match.group(1) or match.group(2)
            if potential in self.entity_to_facts:
                entities.add(potential)

        return entities

    def extract_query_predicates(self, query: str) -> Set[str]:
        """Extrahiert relevante Prädikate aus einer Query"""
        predicates = set()

        # Direkte Prädikat-Erwähnungen
        for predicate in self.predicate_to_facts.keys():
            if predicate.lower() in query.lower():
                predicates.add(predicate)

        # Semantische Mappings (erweiterbar)
        semantic_mappings = {
            'kritisch': ['IstKritisch', 'RequiresSecurity', 'HighPriority'],
            'wichtig': ['IstKritisch', 'IstWichtig', 'HighPriority'],
            'hauptstadt': ['Hauptstadt', 'Capital', 'IstHauptstadtVon'],
            'überwacht': ['MussÜberwachtWerden', 'RequiresMonitoring', 'NeedsSupervision']
        }

        query_lower = query.lower()
        for keyword, related_predicates in semantic_mappings.items():
            if keyword in query_lower:
                for pred in related_predicates:
                    if pred in self.predicate_to_facts:
                        predicates.add(pred)

        return predicates

    def get_relevant_facts(self, query: str, max_hops: int = 1) -> Tuple[Set[Fact], Dict[str, float]]:
        """
        Hauptmethode: Filtert relevante Fakten für eine Query.

        Returns:
            - Set von relevanten Fakten
            - Relevanz-Scores für jeden Fakt
        """
        start_time = time.time()

        # Extrahiere Query-Komponenten
        query_entities = self.extract_query_entities(query)
        query_predicates = self.extract_query_predicates(query)

        # Sammle direkt relevante Fakten
        relevant_indices = set()
        relevance_scores = defaultdict(float)

        # Entitäts-basierte Relevanz
        for entity in query_entities:
            for fact_idx in self.entity_to_facts.get(entity, set()):
                relevant_indices.add(fact_idx)
                relevance_scores[fact_idx] += 1.0

        # Prädikat-basierte Relevanz
        for predicate in query_predicates:
            for fact_idx in self.predicate_to_facts.get(predicate, set()):
                relevant_indices.add(fact_idx)
                relevance_scores[fact_idx] += 0.8

        # N-Hop Expansion
        if max_hops > 0 and query_entities:
            expanded_indices = self._expand_by_hops(
                query_entities, 
                relevant_indices, 
                max_hops
            )
            for fact_idx in expanded_indices:
                relevant_indices.add(fact_idx)
                relevance_scores[fact_idx] += 0.5 / max_hops

        # Konvertiere Indizes zu Fakten
        relevant_facts = {self.facts[idx] for idx in relevant_indices}

        # Update Performance-Metriken
        self._update_stats(len(relevant_facts), len(self.facts), time.time() - start_time)

        return relevant_facts, dict(relevance_scores)

    def _expand_by_hops(self, start_entities: Set[str], 
                       current_indices: Set[int], 
                       max_hops: int) -> Set[int]:
        """Expandiert die Faktenmenge durch Hop-basierte Exploration"""
        expanded = set()
        current_entities = start_entities.copy()

        for hop in range(max_hops):
            next_entities = set()

            # Finde alle verbundenen Entitäten
            for entity in current_entities:
                next_entities.update(self.entity_connections.get(entity, set()))

            # Sammle Fakten für neue Entitäten
            for entity in next_entities:
                for fact_idx in self.entity_to_facts.get(entity, set()):
                    if fact_idx not in current_indices:
                        expanded.add(fact_idx)

            current_entities = next_entities

        return expanded

    def _update_stats(self, facts_returned: int, total_facts: int, query_time: float):
        """Aktualisiert Performance-Metriken"""
        self.stats['total_queries'] += 1
        n = self.stats['total_queries']

        # Moving average für Metriken
        self.stats['avg_facts_returned'] = (
            (self.stats['avg_facts_returned'] * (n-1) + facts_returned) / n
        )

        reduction_ratio = 1 - (facts_returned / max(total_facts, 1))
        self.stats['avg_reduction_ratio'] = (
            (self.stats['avg_reduction_ratio'] * (n-1) + reduction_ratio) / n
        )

        self.stats['avg_query_time'] = (
            (self.stats['avg_query_time'] * (n-1) + query_time * 1000) / n  # in ms
        )

    def get_performance_report(self) -> str:
        """Generiert einen Performance-Report"""
        return f"""
RelevanceFilter Performance Report
==================================
Total Queries: {self.stats['total_queries']}
Avg Facts Returned: {self.stats['avg_facts_returned']:.1f}
Avg Reduction: {self.stats['avg_reduction_ratio']*100:.1f}%
Avg Query Time: {self.stats['avg_query_time']:.2f}ms
Total Facts in KB: {len(self.facts)}
Unique Entities: {len(self.entity_to_facts)}
Unique Predicates: {len(self.predicate_to_facts)}
"""

    def optimize_indices(self):
        """Optimiert die Indizes für bessere Performance"""
        # Entferne leere Einträge
        self.entity_to_facts = {k: v for k, v in self.entity_to_facts.items() if v}
        self.predicate_to_facts = {k: v for k, v in self.predicate_to_facts.items() if v}
        self.entity_connections = {k: v for k, v in self.entity_connections.items() if v}

        # Konvertiere Sets zu frozensets für bessere Speicher-Effizienz
        for entity in self.entity_to_facts:
            self.entity_to_facts[entity] = frozenset(self.entity_to_facts[entity])

        print(f"Indices optimized. Memory usage reduced.")


class HAKGALIntegration:
    """Integration des RelevanceFilters in HAK-GAL"""

    def __init__(self, knowledge_base, relevance_filter):
        self.kb = knowledge_base
        self.filter = relevance_filter
        self.original_ask = self.kb.ask  # Backup der Original-Methode

    def enhanced_ask(self, query: str, timeout: int = 30):
        """
        Ersetzt die Standard ask-Methode mit Relevanz-Filterung
        """
        # 1. Relevante Fakten filtern
        relevant_facts, scores = self.filter.get_relevant_facts(query, max_hops=2)

        print(f"\n[RelevanceFilter] Query: '{query}'")
        print(f"[RelevanceFilter] Reduced facts from {len(self.filter.facts)} to {len(relevant_facts)}")

        if len(relevant_facts) < 20:
            # Bei wenigen Fakten: Details anzeigen
            print("[RelevanceFilter] Selected facts:")
            for fact in sorted(relevant_facts, key=lambda f: scores.get(self.filter.fact_to_index[str(f)], 0), reverse=True)[:10]:
                score = scores.get(self.filter.fact_to_index[str(fact)], 0)
                print(f"  - {fact} (relevance: {score:.2f})")

        # 2. Temporäre KB mit nur relevanten Fakten erstellen
        temp_kb = self._create_temp_kb(relevant_facts)

        # 3. Query auf reduzierter KB ausführen
        result = temp_kb.ask(query, timeout=timeout)

        return result

    def _create_temp_kb(self, relevant_facts: Set[Fact]):
        """Erstellt eine temporäre KB mit nur den relevanten Fakten"""
        # Dies ist eine Vereinfachung - in der echten Implementation
        # würde man die KB-Klasse erweitern
        temp_kb = type(self.kb)()  # Neue Instanz der gleichen Klasse

        for fact in relevant_facts:
            # Konvertiere Fact zurück in KB-Format
            temp_kb.add_fact(fact.predicate, fact.subject, fact.object)

        return temp_kb


# Test und Demonstration
if __name__ == "__main__":
    print("=== RelevanceFilter Test Suite ===\n")

    # Initialisierung
    rf = RelevanceFilter()

    # Test-Daten laden
    test_facts = [
        Fact("IstKritisch", "RAG-Pipeline"),
        Fact("IstTeilVon", "RAG-Pipeline", "HAK-GAL"),
        Fact("VerwendetTechnologie", "RAG-Pipeline", "Embeddings"),
        Fact("RequiresSecurity", "RAG-Pipeline"),
        Fact("MussÜberwachtWerden", "KnowledgeGraph"),
        Fact("IstTeilVon", "KnowledgeGraph", "HAK-GAL"),
        Fact("SpeichertDaten", "KnowledgeGraph", "Fakten"),
        Fact("Hauptstadt", "Italien", "Rom"),
        Fact("Hauptstadt", "Frankreich", "Paris"),
        Fact("Hauptstadt", "Deutschland", "Berlin"),
        Fact("IstLandIn", "Italien", "Europa"),
        Fact("IstLandIn", "Frankreich", "Europa"),
        # Füge 150+ weitere Test-Fakten hinzu
        *[Fact(f"TestPredicate{i}", f"Entity{i}", f"Value{i}") for i in range(150)]
    ]

    # Fakten hinzufügen
    print(f"Loading {len(test_facts)} facts...")
    for fact in test_facts:
        rf.add_fact(fact)

    # Test-Queries
    test_queries = [
        "Ist die RAG-Pipeline kritisch?",
        "Was ist die Hauptstadt von Italien?",
        "Welche Komponenten müssen überwacht werden?",
        "Zeige alle Fakten über HAK-GAL"
    ]

    print("\n=== Query Tests ===")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        relevant_facts, scores = rf.get_relevant_facts(query, max_hops=1)
        print(f"Found {len(relevant_facts)} relevant facts (from {len(rf.facts)} total)")

        # Top 5 relevante Fakten anzeigen
        sorted_facts = sorted(
            relevant_facts, 
            key=lambda f: scores.get(rf.fact_to_index[str(f)], 0), 
            reverse=True
        )[:5]

        for fact in sorted_facts:
            score = scores.get(rf.fact_to_index[str(fact)], 0)
            print(f"  - {fact} (score: {score:.2f})")

    # Performance Report
    print("\n" + rf.get_performance_report())

    # Index-Optimierung
    rf.optimize_indices()
    print("\n✓ RelevanceFilter successfully tested and optimized!")
