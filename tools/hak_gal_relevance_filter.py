"""
HAK-GAL Relevance Filter
========================

A high-performance relevance filtering system for the HAK-GAL knowledge base.
Addresses the "Operation Damocles" issues by implementing intelligent query analysis,
entity/predicate indexing, and N-hop graph expansion.

Key Features:
- Entity and predicate indexing using defaultdict
- Multiple query analysis strategies
- N-hop graph expansion for related facts
- Performance metrics tracking
- Integration-ready with HAK-GAL system
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional, Callable
from collections import defaultdict
import time
import re
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RelevanceStrategy(Enum):
    """Strategies for determining relevance"""
    EXACT_MATCH = "exact_match"
    ENTITY_OVERLAP = "entity_overlap"
    PREDICATE_MATCH = "predicate_match"
    GRAPH_EXPANSION = "graph_expansion"
    COMBINED = "combined"


@dataclass
class Fact:
    """Represents a fact in the knowledge base"""
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def entities(self) -> Set[str]:
        """Extract all entities from the fact"""
        return {self.subject, self.object}

    def __hash__(self):
        return hash(self.id)


@dataclass
class QueryContext:
    """Context for a relevance query"""
    raw_query: str
    entities: Set[str]
    predicates: Set[str]
    keywords: Set[str]
    strategy: RelevanceStrategy = RelevanceStrategy.COMBINED
    max_hops: int = 2
    max_results: int = 100
    min_confidence: float = 0.5


@dataclass
class RelevanceResult:
    """Result of a relevance query"""
    fact: Fact
    score: float
    reason: str
    hops: int = 0


class RelevanceFilter:
    """
    High-performance relevance filter for HAK-GAL knowledge base.
    Solves the Operation Damocles timeout issues through intelligent indexing.
    """

    def __init__(self):
        # Indexes for fast lookup
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # entity -> fact_ids
        self.predicate_index: Dict[str, Set[str]] = defaultdict(set)  # predicate -> fact_ids
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)  # keyword -> fact_ids
        self.fact_store: Dict[str, Fact] = {}  # fact_id -> Fact

        # Graph structure for expansion
        self.entity_graph: Dict[str, Set[str]] = defaultdict(set)  # entity -> connected entities

        # Performance metrics
        self.metrics = {
            'total_facts': 0,
            'total_queries': 0,
            'avg_query_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # Query cache
        self.query_cache: Dict[str, List[RelevanceResult]] = {}
        self.cache_size_limit = 1000

    def add_fact(self, fact: Fact) -> None:
        """Add a fact to the knowledge base with indexing"""
        start_time = time.time()

        # Store the fact
        self.fact_store[fact.id] = fact

        # Index by entities
        for entity in fact.entities():
            self.entity_index[entity.lower()].add(fact.id)
            # Build entity graph
            for other_entity in fact.entities():
                if entity != other_entity:
                    self.entity_graph[entity].add(other_entity)

        # Index by predicate
        self.predicate_index[fact.predicate.lower()].add(fact.id)

        # Index by keywords (simple tokenization)
        keywords = self._extract_keywords(f"{fact.subject} {fact.predicate} {fact.object}")
        for keyword in keywords:
            self.keyword_index[keyword].add(fact.id)

        self.metrics['total_facts'] += 1

        logger.debug(f"Added fact {fact.id} in {time.time() - start_time:.4f}s")

    def query(self, query: str, strategy: RelevanceStrategy = RelevanceStrategy.COMBINED,
              max_results: int = 100, max_hops: int = 2) -> List[RelevanceResult]:
        """
        Query for relevant facts using specified strategy.
        This is the main entry point for relevance filtering.
        """
        start_time = time.time()
        self.metrics['total_queries'] += 1

        # Check cache
        cache_key = f"{query}_{strategy}_{max_results}_{max_hops}"
        if cache_key in self.query_cache:
            self.metrics['cache_hits'] += 1
            return self.query_cache[cache_key]

        self.metrics['cache_misses'] += 1

        # Parse query into context
        context = self._parse_query(query, strategy, max_results, max_hops)

        # Execute query based on strategy
        if strategy == RelevanceStrategy.EXACT_MATCH:
            results = self._exact_match_query(context)
        elif strategy == RelevanceStrategy.ENTITY_OVERLAP:
            results = self._entity_overlap_query(context)
        elif strategy == RelevanceStrategy.PREDICATE_MATCH:
            results = self._predicate_match_query(context)
        elif strategy == RelevanceStrategy.GRAPH_EXPANSION:
            results = self._graph_expansion_query(context)
        else:  # COMBINED
            results = self._combined_query(context)

        # Sort by score and limit results
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:max_results]

        # Update cache
        self._update_cache(cache_key, results)

        # Update metrics
        query_time = time.time() - start_time
        self.metrics['avg_query_time'] = (
            (self.metrics['avg_query_time'] * (self.metrics['total_queries'] - 1) + query_time) /
            self.metrics['total_queries']
        )

        logger.info(f"Query completed in {query_time:.4f}s, returned {len(results)} results")

        return results

    def _parse_query(self, query: str, strategy: RelevanceStrategy,
                     max_results: int, max_hops: int) -> QueryContext:
        """Parse query string into structured context"""
        # Extract entities (simple heuristic: capitalized words)
        entities = set()
        for word in query.split():
            if word[0].isupper() and len(word) > 1:
                entities.add(word.lower())

        # Extract predicates (simple heuristic: verbs, relationships)
        predicates = set()
        predicate_patterns = [
            r'\bis\s+a\b', r'\bhas\b', r'\bborn\b', r'\blives\b',
            r'\bworks\b', r'\brelated\b', r'\bknows\b'
        ]
        for pattern in predicate_patterns:
            if re.search(pattern, query.lower()):
                predicates.add(pattern.strip('\\b').strip())

        # Extract keywords
        keywords = self._extract_keywords(query)

        return QueryContext(
            raw_query=query,
            entities=entities,
            predicates=predicates,
            keywords=keywords,
            strategy=strategy,
            max_hops=max_hops,
            max_results=max_results
        )

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text"""
        # Simple tokenization and filtering
        words = text.lower().split()
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were'}
        keywords = {w for w in words if w not in stop_words and len(w) > 2}
        return keywords

    def _exact_match_query(self, context: QueryContext) -> List[RelevanceResult]:
        """Find facts that exactly match query entities and predicates"""
        results = []
        fact_ids = set()

        # Get facts matching all entities
        if context.entities:
            entity_sets = [self.entity_index[entity] for entity in context.entities]
            if entity_sets:
                fact_ids = set.intersection(*entity_sets)

        # Filter by predicates if specified
        if context.predicates and fact_ids:
            predicate_sets = [self.predicate_index[pred] for pred in context.predicates]
            if predicate_sets:
                predicate_facts = set.union(*predicate_sets)
                fact_ids &= predicate_facts

        # Convert to results
        for fact_id in fact_ids:
            fact = self.fact_store[fact_id]
            if fact.confidence >= context.min_confidence:
                results.append(RelevanceResult(
                    fact=fact,
                    score=1.0,
                    reason="Exact match on entities and predicates",
                    hops=0
                ))

        return results

    def _entity_overlap_query(self, context: QueryContext) -> List[RelevanceResult]:
        """Find facts that have overlapping entities"""
        results = []
        fact_scores = defaultdict(float)

        # Score facts by entity overlap
        for entity in context.entities:
            for fact_id in self.entity_index[entity]:
                fact_scores[fact_id] += 1.0 / len(context.entities)

        # Also check keywords
        for keyword in context.keywords:
            for fact_id in self.keyword_index[keyword]:
                fact_scores[fact_id] += 0.5 / len(context.keywords) if context.keywords else 0

        # Convert to results
        for fact_id, score in fact_scores.items():
            fact = self.fact_store[fact_id]
            if fact.confidence >= context.min_confidence:
                results.append(RelevanceResult(
                    fact=fact,
                    score=score,
                    reason=f"Entity overlap score: {score:.2f}",
                    hops=0
                ))

        return results

    def _predicate_match_query(self, context: QueryContext) -> List[RelevanceResult]:
        """Find facts matching query predicates"""
        results = []
        fact_ids = set()

        # Get all facts matching any predicate
        for predicate in context.predicates:
            fact_ids.update(self.predicate_index[predicate])

        # Score by predicate match and keyword overlap
        for fact_id in fact_ids:
            fact = self.fact_store[fact_id]
            if fact.confidence >= context.min_confidence:
                # Calculate keyword overlap bonus
                fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
                keyword_score = sum(1 for kw in context.keywords if kw in fact_text) / (len(context.keywords) + 1)

                results.append(RelevanceResult(
                    fact=fact,
                    score=0.7 + 0.3 * keyword_score,
                    reason=f"Predicate match with keyword score: {keyword_score:.2f}",
                    hops=0
                ))

        return results

    def _graph_expansion_query(self, context: QueryContext) -> List[RelevanceResult]:
        """Find facts through N-hop graph expansion"""
        results = []
        visited_facts = set()

        # Start with direct matches
        seed_facts = set()
        for entity in context.entities:
            seed_facts.update(self.entity_index[entity])

        # Expand through N hops
        current_layer = seed_facts
        for hop in range(context.max_hops + 1):
            next_layer = set()

            for fact_id in current_layer:
                if fact_id in visited_facts:
                    continue

                visited_facts.add(fact_id)
                fact = self.fact_store[fact_id]

                if fact.confidence >= context.min_confidence:
                    # Score decreases with distance
                    base_score = 1.0 / (hop + 1)

                    # Bonus for keyword matches
                    fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
                    keyword_bonus = sum(0.1 for kw in context.keywords if kw in fact_text)

                    results.append(RelevanceResult(
                        fact=fact,
                        score=min(1.0, base_score + keyword_bonus),
                        reason=f"Graph expansion at {hop} hops",
                        hops=hop
                    ))

                # Find connected facts for next layer
                if hop < context.max_hops:
                    for entity in fact.entities():
                        next_layer.update(self.entity_index[entity.lower()])

            current_layer = next_layer

        return results

    def _combined_query(self, context: QueryContext) -> List[RelevanceResult]:
        """Combine multiple strategies with weighted scoring"""
        # Collect results from all strategies
        all_results = {}

        # Weight for each strategy
        weights = {
            RelevanceStrategy.EXACT_MATCH: 1.0,
            RelevanceStrategy.ENTITY_OVERLAP: 0.7,
            RelevanceStrategy.PREDICATE_MATCH: 0.6,
            RelevanceStrategy.GRAPH_EXPANSION: 0.5
        }

        # Run each strategy
        for strategy, weight in weights.items():
            if strategy != RelevanceStrategy.COMBINED:
                context.strategy = strategy
                strategy_results = self.query(
                    context.raw_query, 
                    strategy=strategy,
                    max_results=context.max_results * 2,  # Get more for merging
                    max_hops=context.max_hops
                )

                for result in strategy_results:
                    fact_id = result.fact.id
                    if fact_id not in all_results:
                        all_results[fact_id] = result
                        result.score *= weight
                    else:
                        # Combine scores
                        all_results[fact_id].score += result.score * weight
                        all_results[fact_id].reason += f"; {result.reason}"

        # Normalize scores
        max_score = max((r.score for r in all_results.values()), default=1.0)
        for result in all_results.values():
            result.score /= max_score

        return list(all_results.values())

    def _update_cache(self, key: str, results: List[RelevanceResult]) -> None:
        """Update query cache with size limit"""
        self.query_cache[key] = results

        # Evict oldest entries if cache is too large
        if len(self.query_cache) > self.cache_size_limit:
            # Simple FIFO eviction
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]

    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        return self.metrics.copy()

    def clear_cache(self) -> None:
        """Clear the query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")

    def bulk_add_facts(self, facts: List[Fact]) -> None:
        """Efficiently add multiple facts"""
        start_time = time.time()
        for fact in facts:
            self.add_fact(fact)

        logger.info(f"Added {len(facts)} facts in {time.time() - start_time:.2f}s")

    def remove_fact(self, fact_id: str) -> bool:
        """Remove a fact from the knowledge base"""
        if fact_id not in self.fact_store:
            return False

        fact = self.fact_store[fact_id]

        # Remove from entity index
        for entity in fact.entities():
            self.entity_index[entity.lower()].discard(fact_id)
            if not self.entity_index[entity.lower()]:
                del self.entity_index[entity.lower()]

        # Remove from predicate index
        self.predicate_index[fact.predicate.lower()].discard(fact_id)
        if not self.predicate_index[fact.predicate.lower()]:
            del self.predicate_index[fact.predicate.lower()]

        # Remove from keyword index
        keywords = self._extract_keywords(f"{fact.subject} {fact.predicate} {fact.object}")
        for keyword in keywords:
            self.keyword_index[keyword].discard(fact_id)
            if not self.keyword_index[keyword]:
                del self.keyword_index[keyword]

        # Remove from fact store
        del self.fact_store[fact_id]

        # Clear cache as results may have changed
        self.clear_cache()

        self.metrics['total_facts'] -= 1
        return True


# Example usage and testing
if __name__ == "__main__":
    # Create relevance filter
    rf = RelevanceFilter()

    # Add sample facts
    sample_facts = [
        Fact("f1", "Socrates", "is_a", "philosopher", confidence=0.95),
        Fact("f2", "Socrates", "is_a", "human", confidence=0.99),
        Fact("f3", "Plato", "is_a", "philosopher", confidence=0.95),
        Fact("f4", "Plato", "student_of", "Socrates", confidence=0.9),
        Fact("f5", "Aristotle", "student_of", "Plato", confidence=0.9),
        Fact("f6", "Alexander", "student_of", "Aristotle", confidence=0.85),
        Fact("f7", "human", "is_a", "mortal", confidence=1.0),
        Fact("f8", "philosopher", "studies", "wisdom", confidence=0.8),
        Fact("f9", "Socrates", "born_in", "Athens", confidence=0.9),
        Fact("f10", "Athens", "located_in", "Greece", confidence=1.0),
    ]

    rf.bulk_add_facts(sample_facts)

    # Test queries
    print("\n=== Testing Relevance Filter ===\n")

    # Test 1: Entity query
    print("Query: 'What do we know about Socrates?'")
    results = rf.query("What do we know about Socrates?", strategy=RelevanceStrategy.ENTITY_OVERLAP)
    for r in results[:5]:
        print(f"  - {r.fact.subject} {r.fact.predicate} {r.fact.object} (score: {r.score:.2f})")

    # Test 2: Graph expansion
    print("\nQuery: 'Connections to Socrates' (with graph expansion)")
    results = rf.query("Connections to Socrates", strategy=RelevanceStrategy.GRAPH_EXPANSION, max_hops=2)
    for r in results[:7]:
        print(f"  - {r.fact.subject} {r.fact.predicate} {r.fact.object} (hops: {r.hops}, score: {r.score:.2f})")

    # Test 3: Combined strategy
    print("\nQuery: 'philosophers students' (combined strategy)")
    results = rf.query("philosophers students", strategy=RelevanceStrategy.COMBINED)
    for r in results[:5]:
        print(f"  - {r.fact.subject} {r.fact.predicate} {r.fact.object} (score: {r.score:.2f})")

    # Show metrics
    print("\n=== Performance Metrics ===")
    metrics = rf.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
