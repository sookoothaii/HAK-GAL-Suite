"""
HAK-GAL Learned Relevance Module
================================

Implementiert maschinelles Lernen für den RelevanceFilter.
Der Filter lernt aus User-Interaktionen und wird mit jeder
Query intelligenter.

Author: HAK-GAL Team
Date: 2024-10-27
Version: 3.0
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import time
from datetime import datetime
import math


@dataclass
class QuerySession:
    """Repräsentiert eine User-Query Session"""
    query: str
    timestamp: datetime
    returned_facts: List[int]
    clicked_facts: List[int] = field(default_factory=list)
    ignored_facts: List[int] = field(default_factory=list)
    query_time_ms: float = 0.0
    user_satisfied: Optional[bool] = None
    follow_up_query: Optional[str] = None


@dataclass
class FactRelevanceProfile:
    """Lern-Profil für jeden Fakt"""
    fact_id: int
    total_appearances: int = 0
    total_clicks: int = 0
    total_ignores: int = 0
    query_associations: Dict[str, float] = field(default_factory=dict)
    entity_associations: Dict[str, float] = field(default_factory=dict)
    temporal_relevance: List[Tuple[datetime, float]] = field(default_factory=list)

    @property
    def click_through_rate(self) -> float:
        """Berechnet Click-Through-Rate"""
        if self.total_appearances == 0:
            return 0.0
        return self.total_clicks / self.total_appearances

    @property
    def relevance_score(self) -> float:
        """Berechnet Gesamt-Relevanz Score"""
        ctr = self.click_through_rate
        recency_boost = self._calculate_recency_boost()
        confidence = min(self.total_appearances / 10, 1.0)  # Confidence steigt mit Daten

        return (ctr * 0.7 + recency_boost * 0.3) * confidence

    def _calculate_recency_boost(self) -> float:
        """Neuere Interaktionen haben mehr Gewicht"""
        if not self.temporal_relevance:
            return 0.0

        now = datetime.now()
        weighted_sum = 0.0
        weight_total = 0.0

        for timestamp, relevance in self.temporal_relevance[-10:]:  # Letzte 10
            age_days = (now - timestamp).days
            weight = math.exp(-age_days / 30)  # Exponentieller Decay
            weighted_sum += relevance * weight
            weight_total += weight

        return weighted_sum / weight_total if weight_total > 0 else 0.0


class LearnedRelevanceEngine:
    """
    Machine Learning Engine für Relevanz-Lernen

    Features:
    - Click-Through-Rate Learning
    - Query Pattern Recognition
    - Temporal Relevance Decay
    - A/B Testing für Verbesserungen
    - Online Learning (keine Retraining nötig)
    """

    def __init__(self):
        # Lern-Daten
        self.fact_profiles: Dict[int, FactRelevanceProfile] = {}
        self.query_sessions: List[QuerySession] = []

        # Query-Muster
        self.query_patterns: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.query_embeddings_cache: Dict[str, np.ndarray] = {}

        # A/B Testing
        self.ab_test_active = False
        self.ab_test_results = {
            'control': {'queries': 0, 'satisfaction': 0},
            'learned': {'queries': 0, 'satisfaction': 0}
        }

        # Performance Metriken
        self.metrics = {
            'total_queries': 0,
            'avg_ctr': 0.0,
            'learning_rate': 0.1,
            'satisfaction_rate': 0.0
        }

    def record_query_session(self, session: QuerySession):
        """Zeichnet eine Query-Session für Lernen auf"""
        self.query_sessions.append(session)
        self.metrics['total_queries'] += 1

        # Update Fact Profiles
        for fact_id in session.returned_facts:
            if fact_id not in self.fact_profiles:
                self.fact_profiles[fact_id] = FactRelevanceProfile(fact_id)

            profile = self.fact_profiles[fact_id]
            profile.total_appearances += 1

            # Click/Ignore Updates
            if fact_id in session.clicked_facts:
                profile.total_clicks += 1
                relevance = 1.0
            elif fact_id in session.ignored_facts:
                profile.total_ignores += 1
                relevance = -0.5
            else:
                relevance = 0.0  # Neutral

            # Temporal Update
            profile.temporal_relevance.append((session.timestamp, relevance))

            # Query Association Update
            profile.query_associations[session.query] = (
                profile.query_associations.get(session.query, 0.0) * 0.9 + 
                relevance * 0.1  # Exponential moving average
            )

        # Update Query Patterns
        self._update_query_patterns(session)

        # Update Global Metrics
        self._update_global_metrics()

    def _update_query_patterns(self, session: QuerySession):
        """Lernt Muster zwischen Queries und erfolgreichen Fakten"""
        if not session.clicked_facts:
            return

        # Extrahiere Query-Features (vereinfacht)
        query_tokens = set(session.query.lower().split())

        for clicked_fact in session.clicked_facts:
            for token in query_tokens:
                self.query_patterns[token][clicked_fact] = (
                    self.query_patterns[token].get(clicked_fact, 0.0) * 0.95 +
                    1.0 * 0.05
                )

    def predict_relevance(self, query: str, fact_ids: List[int]) -> Dict[int, float]:
        """
        Sagt Relevanz für gegebene Fakten voraus

        Returns:
            Dict[fact_id, predicted_relevance_score]
        """
        predictions = {}
        query_tokens = set(query.lower().split())

        for fact_id in fact_ids:
            score = 0.0

            # 1. Historical Performance
            if fact_id in self.fact_profiles:
                profile = self.fact_profiles[fact_id]
                score += profile.relevance_score * 0.4

                # Query-specific history
                if query in profile.query_associations:
                    score += profile.query_associations[query] * 0.3

            # 2. Query Pattern Matching
            pattern_score = 0.0
            for token in query_tokens:
                if token in self.query_patterns and fact_id in self.query_patterns[token]:
                    pattern_score += self.query_patterns[token][fact_id]

            if len(query_tokens) > 0:
                pattern_score /= len(query_tokens)
                score += pattern_score * 0.3

            predictions[fact_id] = min(score, 1.0)  # Cap at 1.0

        return predictions

    def get_learning_insights(self) -> Dict[str, Any]:
        """Gibt Einblicke in das Gelernte zurück"""
        # Top performing facts
        top_facts = sorted(
            self.fact_profiles.values(),
            key=lambda p: p.relevance_score,
            reverse=True
        )[:10]

        # Most associated query patterns
        pattern_strengths = {}
        for token, facts in self.query_patterns.items():
            pattern_strengths[token] = sum(facts.values()) / len(facts) if facts else 0

        top_patterns = sorted(
            pattern_strengths.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            'total_sessions': len(self.query_sessions),
            'facts_with_profiles': len(self.fact_profiles),
            'avg_click_through_rate': self.metrics['avg_ctr'],
            'top_performing_facts': [
                {
                    'fact_id': f.fact_id,
                    'ctr': f.click_through_rate,
                    'appearances': f.total_appearances,
                    'score': f.relevance_score
                }
                for f in top_facts
            ],
            'top_query_patterns': [
                {'pattern': pattern, 'strength': strength}
                for pattern, strength in top_patterns
            ],
            'satisfaction_rate': self.metrics['satisfaction_rate']
        }

    def _update_global_metrics(self):
        """Aktualisiert globale Metriken"""
        if not self.fact_profiles:
            return

        # Average CTR
        total_ctr = sum(p.click_through_rate for p in self.fact_profiles.values())
        self.metrics['avg_ctr'] = total_ctr / len(self.fact_profiles)

        # Satisfaction Rate
        satisfied_sessions = sum(
            1 for s in self.query_sessions[-100:]  # Letzte 100
            if s.user_satisfied is True
        )
        total_recent = min(len(self.query_sessions), 100)
        if total_recent > 0:
            self.metrics['satisfaction_rate'] = satisfied_sessions / total_recent

    def export_learning_data(self, filepath: str):
        """Exportiert Lern-Daten für Backup/Analyse"""
        data = {
            'fact_profiles': {
                fid: {
                    'total_appearances': p.total_appearances,
                    'total_clicks': p.total_clicks,
                    'total_ignores': p.total_ignores,
                    'query_associations': p.query_associations,
                    'ctr': p.click_through_rate,
                    'score': p.relevance_score
                }
                for fid, p in self.fact_profiles.items()
            },
            'query_patterns': dict(self.query_patterns),
            'metrics': self.metrics,
            'export_time': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[Export] Saved learning data to {filepath}")


class AdaptiveRelevanceFilter:
    """
    Integration von LearnedRelevance mit dem RelevanceFilter
    """

    def __init__(self, base_filter, learning_engine):
        self.base_filter = base_filter
        self.learning_engine = learning_engine
        self.session_tracking = {}

    def adaptive_query(self, query: str, user_id: str = "default") -> Tuple[List[Any], str]:
        """
        Adaptive Query mit Lernen

        Returns:
            (results, session_id)
        """
        session_id = f"{user_id}_{int(time.time() * 1000)}"
        start_time = time.time()

        # 1. Basis-Relevanz von Filter
        base_results, base_scores = self.base_filter.get_relevant_facts(query)
        fact_ids = [self.base_filter.facts.index(f) for f in base_results]

        # 2. Gelernte Relevanz
        learned_scores = self.learning_engine.predict_relevance(query, fact_ids)

        # 3. Kombiniere Scores
        combined_scores = {}
        for fact_id in fact_ids:
            base_score = base_scores.get(fact_id, 0.0)
            learned_score = learned_scores.get(fact_id, 0.0)

            # Adaptive Gewichtung basierend auf Konfidenz
            confidence = min(
                self.learning_engine.fact_profiles.get(
                    fact_id, 
                    FactRelevanceProfile(fact_id)
                ).total_appearances / 10,
                1.0
            )

            # Mehr Gewicht auf Gelerntes mit steigender Konfidenz
            combined_scores[fact_id] = (
                base_score * (1 - 0.5 * confidence) +
                learned_score * (0.5 * confidence)
            )

        # 4. Re-rank basierend auf kombinierten Scores
        ranked_facts = sorted(
            fact_ids,
            key=lambda fid: combined_scores[fid],
            reverse=True
        )

        # 5. Track Session
        session = QuerySession(
            query=query,
            timestamp=datetime.now(),
            returned_facts=ranked_facts[:20],  # Top 20
            query_time_ms=(time.time() - start_time) * 1000
        )

        self.session_tracking[session_id] = session

        # Return top results
        results = [self.base_filter.facts[fid] for fid in ranked_facts[:20]]

        print(f"[Adaptive] Query '{query}' processed with learning")
        print(f"[Adaptive] Confidence levels: "
              f"High={sum(1 for f in ranked_facts[:5] if f in self.learning_engine.fact_profiles)} "
              f"of top 5")

        return results, session_id

    def record_interaction(self, session_id: str, clicked_facts: List[int], 
                         satisfied: bool = None):
        """Zeichnet User-Interaktion auf"""
        if session_id not in self.session_tracking:
            print(f"[Warning] Unknown session: {session_id}")
            return

        session = self.session_tracking[session_id]
        session.clicked_facts = clicked_facts
        session.user_satisfied = satisfied

        # Bestimme ignorierte Fakten
        session.ignored_facts = [
            f for f in session.returned_facts 
            if f not in clicked_facts
        ]

        # Lerne aus der Session
        self.learning_engine.record_query_session(session)

        print(f"[Learning] Recorded interaction: "
              f"{len(clicked_facts)} clicks, "
              f"satisfied={satisfied}")

    def get_performance_report(self) -> str:
        """Generiert Performance-Report mit Lern-Statistiken"""
        insights = self.learning_engine.get_learning_insights()

        report = f"""
Adaptive Relevance Filter Report
================================
Total Queries Processed: {insights['total_sessions']}
Facts with Learning Profiles: {insights['facts_with_profiles']}
Average Click-Through Rate: {insights['avg_click_through_rate']:.2%}
User Satisfaction Rate: {insights['satisfaction_rate']:.2%}

Top Performing Facts (by CTR):
"""
        for fact in insights['top_performing_facts'][:5]:
            report += f"  - Fact #{fact['fact_id']}: "
            report += f"CTR={fact['ctr']:.2%}, "
            report += f"Appearances={fact['appearances']}, "
            report += f"Score={fact['score']:.3f}\n"

        report += "\nTop Query Patterns Learned:"
        for pattern in insights['top_query_patterns'][:5]:
            report += f"  - '{pattern['pattern']}': "
            report += f"strength={pattern['strength']:.3f}\n"

        return report


# Demo und Tests
if __name__ == "__main__":
    print("=== LEARNED RELEVANCE ENGINE DEMO ===\n")

    # Initialisierung
    engine = LearnedRelevanceEngine()

    # Simuliere User-Interaktionen
    print("Simulating user interactions...\n")

    # Session 1: User sucht nach kritischen Komponenten
    session1 = QuerySession(
        query="critical components",
        timestamp=datetime.now(),
        returned_facts=[1, 2, 3, 4, 5, 10, 15, 20],
        clicked_facts=[1, 3, 5],  # User klickt auf Fakt 1, 3, 5
        user_satisfied=True
    )
    engine.record_query_session(session1)

    # Session 2: Ähnliche Query
    session2 = QuerySession(
        query="wichtige komponenten",  # Deutsch
        timestamp=datetime.now(),
        returned_facts=[1, 2, 3, 6, 7, 8],
        clicked_facts=[1, 3],  # Wieder Fakt 1 und 3!
        user_satisfied=True
    )
    engine.record_query_session(session2)

    # Session 3: Andere Query, aber Fakt 1 wird ignoriert
    session3 = QuerySession(
        query="security modules",
        timestamp=datetime.now(),
        returned_facts=[1, 10, 11, 12],
        clicked_facts=[10, 11],  # Fakt 1 wird ignoriert
        ignored_facts=[1],
        user_satisfied=True
    )
    engine.record_query_session(session3)

    # Zeige Gelerntes
    print("\n=== LEARNING INSIGHTS ===")
    insights = engine.get_learning_insights()

    print(f"\nTotal Sessions: {insights['total_sessions']}")
    print(f"Average CTR: {insights['avg_click_through_rate']:.2%}")

    print("\nTop Performing Facts:")
    for fact in insights['top_performing_facts']:
        print(f"  Fact #{fact['fact_id']}: CTR={fact['ctr']:.2%}, Score={fact['score']:.3f}")

    # Teste Vorhersagen
    print("\n=== RELEVANCE PREDICTIONS ===")

    test_query = "critical system components"
    test_facts = [1, 2, 3, 4, 5, 10, 15, 20]

    predictions = engine.predict_relevance(test_query, test_facts)

    print(f"\nQuery: '{test_query}'")
    print("Predicted Relevance:")
    for fact_id, score in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
        print(f"  Fact #{fact_id}: {score:.3f}")

    print("\n✨ The system is LEARNING from every interaction!")
    print("✨ Facts that users click on become more relevant!")
    print("✨ Facts that are ignored become less relevant!")
