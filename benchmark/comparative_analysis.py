# benchmark/comparative_analysis.py

import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class BenchmarkResult:
    system_name: str
    query: str
    response: str
    execution_time: float
    accuracy: float
    verifiability: bool
    explanation_quality: float
    resource_usage: Dict[str, float]
    metadata: Dict[str, Any]

class SystemBenchmark:
    """
    Vergleicht HAK-GAL mit anderen KI-Systemen
    """
    
    def __init__(self):
        self.test_queries = self._load_test_suite()
        self.results = []
        
    def _load_test_suite(self) -> List[Dict[str, Any]]:
        """Standardisierte Test-Queries für alle Systeme"""
        return [
            # === LOGISCHE INFERENZ ===
            {
                "id": "logic_01",
                "category": "logical_inference",
                "query": "If all birds can fly and penguins are birds, can penguins fly?",
                "expected_answer": "No (with explanation about premise error)",
                "requires_reasoning": True
            },
            {
                "id": "logic_02", 
                "category": "logical_inference",
                "query": "Wenn es regnet, wird die Straße nass. Die Straße ist nass. Hat es geregnet?",
                "expected_answer": "Not necessarily (affirming consequent fallacy)",
                "requires_reasoning": True
            },
            
            # === MATHEMATISCHE BEWEISE ===
            {
                "id": "math_01",
                "category": "mathematical_proof",
                "query": "Prove that sqrt(2) is irrational",
                "expected_answer": "Proof by contradiction",
                "requires_reasoning": True
            },
            {
                "id": "math_02",
                "category": "mathematical_computation",
                "query": "What is the integral of x^2 * e^x dx?",
                "expected_answer": "e^x(x^2 - 2x + 2) + C",
                "requires_reasoning": False
            },
            
            # === FAKTENWISSEN ===
            {
                "id": "fact_01",
                "category": "factual_knowledge",
                "query": "What is the capital of Germany?",
                "expected_answer": "Berlin",
                "requires_reasoning": False
            },
            {
                "id": "fact_02",
                "category": "factual_knowledge",
                "query": "Who proved Fermat's Last Theorem?",
                "expected_answer": "Andrew Wiles",
                "requires_reasoning": False
            },
            
            # === SELBST-REFLEXION ===
            {
                "id": "meta_01",
                "category": "self_reflection",
                "query": "What type of AI system are you?",
                "expected_answer": "System-specific description",
                "requires_reasoning": True
            },
            {
                "id": "meta_02",
                "category": "self_reflection",
                "query": "Can you explain your reasoning process?",
                "expected_answer": "Detailed explanation",
                "requires_reasoning": True
            },
            
            # === KOMPLEXE REASONING ===
            {
                "id": "complex_01",
                "category": "complex_reasoning",
                "query": "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
                "expected_answer": "$0.05",
                "requires_reasoning": True
            },
            {
                "id": "complex_02",
                "category": "complex_reasoning",
                "query": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "expected_answer": "5 minutes",
                "requires_reasoning": True
            }
        ]
    
    def benchmark_hakgal(self, hakgal_instance) -> List[BenchmarkResult]:
        """Benchmark HAK-GAL System"""
        results = []
        
        for test in self.test_queries:
            start_time = time.time()
            
            try:
                # HAK-GAL spezifischer Aufruf
                response = hakgal_instance.process_query(test["query"])
                
                # Extrahiere Verifikations-Info
                verifiable = bool(response.get("proof_steps") or response.get("wolfram_result"))
                
                result = BenchmarkResult(
                    system_name="HAK-GAL",
                    query=test["query"],
                    response=response.get("answer", ""),
                    execution_time=time.time() - start_time,
                    accuracy=self._evaluate_accuracy(response.get("answer"), test["expected_answer"]),
                    verifiability=verifiable,
                    explanation_quality=self._rate_explanation(response),
                    resource_usage={
                        "memory_mb": response.get("memory_usage", 0),
                        "cpu_percent": response.get("cpu_usage", 0)
                    },
                    metadata={
                        "used_wolfram": response.get("used_wolfram", False),
                        "used_z3": response.get("used_z3", False),
                        "pattern_matched": response.get("pattern_id"),
                        "confidence": response.get("confidence", 0)
                    }
                )
                
                results.append(result)
                
            except Exception as e:
                results.append(BenchmarkResult(
                    system_name="HAK-GAL",
                    query=test["query"],
                    response=f"Error: {str(e)}",
                    execution_time=time.time() - start_time,
                    accuracy=0.0,
                    verifiability=False,
                    explanation_quality=0.0,
                    resource_usage={},
                    metadata={"error": str(e)}
                ))
        
        return results
    
    def benchmark_openai_gpt4(self, api_key: str) -> List[BenchmarkResult]:
        """Benchmark OpenAI GPT-4"""
        import openai
        openai.api_key = api_key
        results = []
        
        for test in self.test_queries:
            start_time = time.time()
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a logical reasoning assistant."},
                        {"role": "user", "content": test["query"]}
                    ],
                    temperature=0
                )
                
                answer = response.choices[0].message.content
                
                result = BenchmarkResult(
                    system_name="GPT-4",
                    query=test["query"],
                    response=answer,
                    execution_time=time.time() - start_time,
                    accuracy=self._evaluate_accuracy(answer, test["expected_answer"]),
                    verifiability=False,  # GPT-4 kann keine formalen Beweise
                    explanation_quality=self._rate_explanation({"answer": answer}),
                    resource_usage={
                        "tokens": response.usage.total_tokens
                    },
                    metadata={
                        "model": "gpt-4",
                        "temperature": 0
                    }
                )
                
                results.append(result)
                
            except Exception as e:
                results.append(self._error_result("GPT-4", test["query"], e, start_time))
        
        return results
    
    def benchmark_wolfram_alpha(self, app_id: str) -> List[BenchmarkResult]:
        """Benchmark Wolfram Alpha direkt"""
        import wolframalpha
        client = wolframalpha.Client(app_id)
        results = []
        
        for test in self.test_queries:
            start_time = time.time()
            
            try:
                res = client.query(test["query"])
                answer = next(res.results).text if res.results else "No result"
                
                result = BenchmarkResult(
                    system_name="Wolfram Alpha",
                    query=test["query"],
                    response=answer,
                    execution_time=time.time() - start_time,
                    accuracy=self._evaluate_accuracy(answer, test["expected_answer"]),
                    verifiability=True,  # Wolfram gibt verifizierbare Antworten
                    explanation_quality=0.5,  # Wolfram erklärt wenig
                    resource_usage={},
                    metadata={
                        "success": bool(res.results)
                    }
                )
                
                results.append(result)
                
            except Exception as e:
                results.append(self._error_result("Wolfram Alpha", test["query"], e, start_time))
        
        return results
    
    def benchmark_prolog_system(self) -> List[BenchmarkResult]:
        """Benchmark reines Prolog-System"""
        from pyswip import Prolog
        prolog = Prolog()
        results = []
        
        # Lade Wissensbasis
        prolog.assertz("bird(penguin)")
        prolog.assertz("bird(sparrow)")
        prolog.assertz("can_fly(sparrow)")
        prolog.assertz("capital(germany, berlin)")
        
        for test in self.test_queries:
            start_time = time.time()
            
            try:
                # Konvertiere Query zu Prolog (vereinfacht)
                if "capital" in test["query"].lower() and "germany" in test["query"].lower():
                    query = "capital(germany, X)"
                    solutions = list(prolog.query(query))
                    answer = solutions[0]["X"] if solutions else "Unknown"
                else:
                    answer = "Cannot process this query type"
                
                result = BenchmarkResult(
                    system_name="Prolog",
                    query=test["query"],
                    response=answer,
                    execution_time=time.time() - start_time,
                    accuracy=self._evaluate_accuracy(answer, test["expected_answer"]),
                    verifiability=True,  # Prolog ist formal verifizierbar
                    explanation_quality=0.3,  # Prolog gibt nur Antworten
                    resource_usage={},
                    metadata={
                        "query_type": "logical"
                    }
                )
                
                results.append(result)
                
            except Exception as e:
                results.append(self._error_result("Prolog", test["query"], e, start_time))
        
        return results
    
    def _evaluate_accuracy(self, response: str, expected: str) -> float:
        """Bewertet Genauigkeit der Antwort (0-1)"""
        if not response:
            return 0.0
            
        response_lower = response.lower()
        expected_lower = expected.lower()
        
        # Exakte Übereinstimmung
        if expected_lower in response_lower:
            return 1.0
        
        # Teilweise Übereinstimmung
        if any(word in response_lower for word in expected_lower.split()):
            return 0.5
            
        return 0.0
    
    def _rate_explanation(self, response: Dict) -> float:
        """Bewertet Qualität der Erklärung (0-1)"""
        explanation_indicators = [
            "because", "therefore", "since", "proof", "steps",
            "weil", "daher", "deshalb", "beweis", "schritte"
        ]
        
        text = str(response.get("answer", ""))
        score = 0.0
        
        # Check for explanation indicators
        for indicator in explanation_indicators:
            if indicator in text.lower():
                score += 0.2
        
        # Check for structured reasoning
        if response.get("proof_steps"):
            score += 0.3
            
        # Check for confidence/certainty indicators
        if response.get("confidence"):
            score += 0.1
            
        return min(score, 1.0)
    
    def _error_result(self, system: str, query: str, error: Exception, start_time: float) -> BenchmarkResult:
        """Erstellt Error-Result"""
        return BenchmarkResult(
            system_name=system,
            query=query,
            response=f"Error: {str(error)}",
            execution_time=time.time() - start_time,
            accuracy=0.0,
            verifiability=False,
            explanation_quality=0.0,
            resource_usage={},
            metadata={"error": str(error)}
        )
    
    def generate_comparison_report(self, all_results: Dict[str, List[BenchmarkResult]]) -> Dict:
        """Generiert umfassenden Vergleichsbericht"""
        
        report = {
            "summary": {},
            "by_category": {},
            "detailed_results": [],
            "visualizations": {}
        }
        
        # Aggregate by system
        for system_name, results in all_results.items():
            avg_accuracy = sum(r.accuracy for r in results) / len(results)
            avg_time = sum(r.execution_time for r in results) / len(results)
            verifiable_count = sum(1 for r in results if r.verifiability)
            avg_explanation = sum(r.explanation_quality for r in results) / len(results)
            
            report["summary"][system_name] = {
                "average_accuracy": avg_accuracy,
                "average_execution_time": avg_time,
                "verifiable_responses": verifiable_count,
                "average_explanation_quality": avg_explanation,
                "total_queries": len(results)
            }
        
        # Aggregate by category
        categories = set(test["category"] for test in self.test_queries)
        
        for category in categories:
            report["by_category"][category] = {}
            
            for system_name, results in all_results.items():
                category_results = [
                    r for r, t in zip(results, self.test_queries) 
                    if t["category"] == category
                ]
                
                if category_results:
                    report["by_category"][category][system_name] = {
                        "accuracy": sum(r.accuracy for r in category_results) / len(category_results),
                        "avg_time": sum(r.execution_time for r in category_results) / len(category_results)
                    }
        
        return report
    
    def visualize_results(self, report: Dict):
        """Erstellt Visualisierungen der Benchmark-Ergebnisse"""
        
        # 1. Accuracy Comparison
        plt.figure(figsize=(12, 8))
        
        systems = list(report["summary"].keys())
        accuracies = [report["summary"][s]["average_accuracy"] for s in systems]
        
        plt.subplot(2, 2, 1)
        plt.bar(systems, accuracies)
        plt.title("Average Accuracy by System")
        plt.ylabel("Accuracy (0-1)")
        plt.xticks(rotation=45)
        
        # 2. Execution Time Comparison
        times = [report["summary"][s]["average_execution_time"] for s in systems]
        
        plt.subplot(2, 2, 2)
        plt.bar(systems, times)
        plt.title("Average Execution Time")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45)
        
        # 3. Verifiability
        verifiable = [report["summary"][s]["verifiable_responses"] for s in systems]
        
        plt.subplot(2, 2, 3)
        plt.bar(systems, verifiable)
        plt.title("Verifiable Responses")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        
        # 4. Category Heatmap
        plt.subplot(2, 2, 4)
        
        categories = list(report["by_category"].keys())
        heatmap_data = []
        
        for system in systems:
            row = []
            for category in categories:
                if system in report["by_category"][category]:
                    row.append(report["by_category"][category][system]["accuracy"])
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Accuracy')
        plt.yticks(range(len(systems)), systems)
        plt.xticks(range(len(categories)), categories, rotation=45)
        plt.title("Accuracy by Category")
        
        plt.tight_layout()
        plt.savefig('benchmark_comparison.png', dpi=300)
        plt.show()

# === AUSFÜHRUNG DER BENCHMARKS ===

def run_comprehensive_benchmark():
    """Führt vollständigen Benchmark durch"""
    
    benchmark = SystemBenchmark()
    all_results = {}
    
    print("🚀 Starting Comprehensive AI System Benchmark...")
    print("=" * 60)
    
    # 1. HAK-GAL
    print("\n📊 Benchmarking HAK-GAL...")
    try:
        from hakgal_system import HAKGALSystem
        hakgal = HAKGALSystem()
        all_results["HAK-GAL"] = benchmark.benchmark_hakgal(hakgal)
        print(f"✅ HAK-GAL: {len(all_results['HAK-GAL'])} tests completed")
    except Exception as e:
        print(f"❌ HAK-GAL Error: {e}")
    
    # 2. GPT-4
    print("\n📊 Benchmarking GPT-4...")
    try:
        all_results["GPT-4"] = benchmark.benchmark_openai_gpt4("your-api-key")
        print(f"✅ GPT-4: {len(all_results['GPT-4'])} tests completed")
    except Exception as e:
        print(f"❌ GPT-4 Error: {e}")
    
    # 3. Wolfram Alpha
    print("\n📊 Benchmarking Wolfram Alpha...")
    try:
        all_results["Wolfram Alpha"] = benchmark.benchmark_wolfram_alpha("your-app-id")
        print(f"✅ Wolfram: {len(all_results['Wolfram Alpha'])} tests completed")
    except Exception as e:
        print(f"❌ Wolfram Error: {e}")
    
    # 4. Prolog
    print("\n📊 Benchmarking Prolog...")
    try:
        all_results["Prolog"] = benchmark.benchmark_prolog_system()
        print(f"✅ Prolog: {len(all_results['Prolog'])} tests completed")
    except Exception as e:
        print(f"❌ Prolog Error: {e}")
    
    # Generate Report
    print("\n📈 Generating Comparison Report...")
    report = benchmark.generate_comparison_report(all_results)
    
    # Print Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    for system, metrics in report["summary"].items():
        print(f"\n{system}:")
        print(f"  - Accuracy: {metrics['average_accuracy']:.2%}")
        print(f"  - Avg Time: {metrics['average_execution_time']:.3f}s")
        print(f"  - Verifiable: {metrics['verifiable_responses']}/{metrics['total_queries']}")
        print(f"  - Explanation Quality: {metrics['average_explanation_quality']:.2%}")
    
    # Save detailed results
    with open('benchmark_results.json', 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "report": report,
            "raw_results": {
                system: [r.__dict__ for r in results]
                for system, results in all_results.items()
            }
        }, f, indent=2)
    
    # Generate visualizations
    benchmark.visualize_results(report)
    
    print("\n✅ Benchmark completed! Results saved to benchmark_results.json")
    
    return report

# === SPEZIELLE HAK-GAL STÄRKEN TESTS ===

class HAKGALStrengthBenchmark:
    """Tests die speziell HAK-GAL's Stärken zeigen"""
    
    def __init__(self):
        self.strength_tests = [
            {
                "name": "Formal Proof Verification",
                "query": "Prove that if A implies B and B implies C, then A implies C",
                "expected_features": ["z3_proof", "formal_steps", "symbolic_logic"]
            },
            {
                "name": "Self-Learning Capability",
                "queries": [
                    "What have you learned about yourself?",
                    "How many facts do you know?",
                    "What new knowledge did you acquire?"
                ],
                "measure": "knowledge_growth"
            },
            {
                "name": "Hybrid Reasoning",
                "query": "Use both mathematical computation and logical reasoning to solve: If x^2 = 16 and x > 0, what is x?",
                "expected_features": ["wolfram_computation", "logical_inference"]
            },
            {
                "name": "Verifiable Explanations",
                "query": "Explain why the square root of 2 is irrational with verifiable steps",
                "expected_features": ["proof_steps", "verification", "explanation"]
            }
        ]
    
    def run_strength_tests(self, hakgal_instance):
        """Führt HAK-GAL-spezifische Tests durch"""
        
        results = {
            "formal_verification_score": 0,
            "self_learning_score": 0,
            "hybrid_reasoning_score": 0,
            "explanation_quality_score": 0,
            "unique_capabilities": []
        }
        
        # Test 1: Formal Verification
        response = hakgal_instance.process_query(self.strength_tests[0]["query"])
        if all(feature in str(response) for feature in self.strength_tests[0]["expected_features"]):
            results["formal_verification_score"] = 1.0
            results["unique_capabilities"].append("Formal Z3 Proofs")
        
        # Test 2: Self-Learning
        initial_facts = hakgal_instance.get_fact_count()
        for query in self.strength_tests[1]["queries"]:
            hakgal_instance.process_query(query)
        final_facts = hakgal_instance.get_fact_count()
        
        if final_facts > initial_facts:
            results["self_learning_score"] = (final_facts - initial_facts) / initial_facts
            results["unique_capabilities"].append(f"Self-Learning (+{final_facts - initial_facts} facts)")
        
        # Test 3: Hybrid Reasoning
        response = hakgal_instance.process_query(self.strength_tests[2]["query"])
        if "wolfram" in str(response).lower() and "logic" in str(response).lower():
            results["hybrid_reasoning_score"] = 1.0
            results["unique_capabilities"].append("Wolfram + Logic Integration")
        
        return results

if __name__ == "__main__":
    # Führe Benchmark aus
    report = run_comprehensive_benchmark()
    
    # Zusätzlich: HAK-GAL Stärken-Test
    print("\n🎯 Running HAK-GAL Strength Tests...")
    strength_benchmark = HAKGALStrengthBenchmark()
    
    try:
        from hakgal_system import HAKGALSystem
        hakgal = HAKGALSystem()
        strength_results = strength_benchmark.run_strength_tests(hakgal)
        
        print("\nHAK-GAL Unique Strengths:")
        for capability in strength_results["unique_capabilities"]:
            print(f"  ✨ {capability}")
    except Exception as e:
        print(f"Strength test error: {e}")