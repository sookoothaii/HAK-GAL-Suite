# Korrigierte Version von run_complete_benchmark.py
corrected_runner = """# benchmark/run_complete_benchmark.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any

class CompleteBenchmarkRunner:
    '''Führt alle Benchmark-Tests durch und generiert umfassenden Report'''
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'basic_benchmark': {},
            'adversarial_tests': {},
            'scalability_tests': {},
            'summary': {}
        }
    
    def run_all_benchmarks(self):
        '''Führt alle Benchmark-Suites aus'''
        
        print("🚀 HAK-GAL Complete Benchmark Suite")
        print("=" * 60)
        
        # 1. Basic Benchmark
        print("\\n📊 Running Basic Benchmark...")
        try:
            from comparative_analysis_hakgal import HAKGALBenchmark
            basic_bench = HAKGALBenchmark()
            self.results['basic_benchmark'] = basic_bench.benchmark_hakgal()
            print(f"✅ Completed {len(self.results['basic_benchmark'])} basic tests")
        except Exception as e:
            print(f"❌ Basic benchmark failed: {e}")
            self.results['basic_benchmark'] = {'error': str(e)}
        
        # 2. Adversarial Tests
        print("\\n🔴 Running Adversarial Tests...")
        try:
            from backend.k_assistant_main_v7_wolfram import KAssistant
            assistant = KAssistant()
            
            # Load adversarial test suite
            with open('benchmark/adversarial_test_suite.json', 'r') as f:
                test_suite = json.load(f)
            
            adv_results = self._run_adversarial_tests(assistant, test_suite)
            self.results['adversarial_tests'] = adv_results
            
            print(f"✅ Completed adversarial tests")
            print(f"   Pass rate: {adv_results['summary']['pass_rate']:.1%}")
            
        except Exception as e:
            print(f"❌ Adversarial tests failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['adversarial_tests'] = {'error': str(e)}
        
        # 3. Scalability Tests (simplified version)
        print("\\n📈 Running Scalability Tests...")
        try:
            scale_results = self._run_scalability_tests()
            self.results['scalability_tests'] = scale_results
            print("✅ Scalability tests completed")
            
        except Exception as e:
            print(f"❌ Scalability tests failed: {e}")
            self.results['scalability_tests'] = {'error': str(e)}
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        return self.results
    
    def _run_adversarial_tests(self, assistant, test_suite):
        '''Run adversarial tests from JSON suite'''
        results = {
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0
            },
            'by_category': {},
            'by_risk_level': {
                'low': {'total': 0, 'passed': 0},
                'medium': {'total': 0, 'passed': 0},
                'high': {'total': 0, 'passed': 0}
            },
            'detailed_results': []
        }
        
        all_tests = []
        for category in ['adversarial', 'ambiguous', 'edge_cases']:
            if category in test_suite:
                all_tests.extend(test_suite[category])
        
        for test in all_tests:
            print(f"  Testing: {test['id']}...")
            
            start_time = time.time()
            try:
                response = assistant.process_query(test['query'])
                execution_time = time.time() - start_time
                
                # Simple evaluation
                passed = self._evaluate_adversarial_response(response, test)
                
                result = {
                    'test_id': test['id'],
                    'category': test['category'],
                    'risk_level': test['risk_level'],
                    'query': test['query'],
                    'response': str(response),
                    'expected_behavior': test['expected_behavior'],
                    'passed': passed,
                    'execution_time': execution_time
                }
                
                results['detailed_results'].append(result)
                results['summary']['total_tests'] += 1
                
                if passed:
                    results['summary']['passed'] += 1
                else:
                    results['summary']['failed'] += 1
                
                # Update category stats
                if test['category'] not in results['by_category']:
                    results['by_category'][test['category']] = {
                        'total': 0, 'passed': 0, 'failed': 0
                    }
                
                results['by_category'][test['category']]['total'] += 1
                if passed:
                    results['by_category'][test['category']]['passed'] += 1
                else:
                    results['by_category'][test['category']]['failed'] += 1
                
                # Update risk level stats
                results['by_risk_level'][test['risk_level']]['total'] += 1
                if passed:
                    results['by_risk_level'][test['risk_level']]['passed'] += 1
                    
            except Exception as e:
                results['summary']['errors'] += 1
                results['detailed_results'].append({
                    'test_id': test['id'],
                    'error': str(e),
                    'passed': False
                })
        
        # Calculate pass rate
        results['summary']['pass_rate'] = (
            results['summary']['passed'] / results['summary']['total_tests'] 
            if results['summary']['total_tests'] > 0 else 0
        )
        
        return results
    
    def _evaluate_adversarial_response(self, response, test):
        '''Evaluate if response meets expected behavior'''
        response_str = str(response).lower()
        expected = test['expected_behavior'].lower()
        
        # Simple keyword-based evaluation
        if 'recognize' in expected or 'identify' in expected:
            keywords = ['paradox', 'contradiction', 'ambiguous', 'unclear', 'multiple']
            return any(word in response_str for word in keywords)
        
        elif 'refuse' in expected:
            keywords = ['cannot', 'unable', 'sorry', 'refuse']
            return any(word in response_str for word in keywords)
        
        elif 'request' in expected or 'ask' in expected:
            keywords = ['please', 'specify', 'clarify', 'what', 'which', '?']
            return any(word in response_str for word in keywords)
        
        # Default: check if response is reasonable
        return len(response_str) > 10 and 'error' not in response_str
    
    def _run_scalability_tests(self):
        '''Run basic scalability tests'''
        from backend.k_assistant_main_v7_wolfram import KAssistant
        
        results = {
            'kb_sizes': [],
            'query_times': [],
            'memory_usage': []
        }
        
        sizes = [100, 500, 1000]
        
        for size in sizes:
            print(f"  Testing with {size} facts...")
            assistant = KAssistant()
            
            # Add test facts
            start_time = time.time()
            for i in range(size):
                assistant.add_fact(f"test_fact_{i}", f"test_value_{i}")
            
            # Test query time
            query_start = time.time()
            assistant.process_query("What is test_fact_50?")
            query_time = time.time() - query_start
            
            results['kb_sizes'].append(size)
            results['query_times'].append(query_time)
            
            # Estimate memory (simplified)
            import sys
            results['memory_usage'].append(sys.getsizeof(assistant.facts) / 1024 / 1024)
        
        return results
    
    def _generate_comprehensive_report(self):
        '''Generate HTML report'''
        
        html_report = f'''<!DOCTYPE html>
<html>
<head>
    <title>HAK-GAL Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #2E86AB; color: white; padding: 20px; }}
        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>HAK-GAL Comprehensive Benchmark Report</h1>
        <p>Generated: {self.results['timestamp']}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <p>Benchmark results for HAK-GAL AI System</p>
    </div>
    
    <div class="section">
        <h2>Test Results</h2>
        {self._generate_results_table()}
    </div>
</body>
</html>'''
        
        with open('benchmark/benchmark_report.html', 'w') as f:
            f.write(html_report)
        
        print("\\n📄 HTML report generated: benchmark/benchmark_report.html")
    
    def _generate_results_table(self):
        '''Generate results table for HTML'''
        html = '<table><tr><th>Test Suite</th><th>Status</th><th>Details</th></tr>'
        
        # Basic benchmark
        if 'error' not in self.results.get('basic_benchmark', {}):
            count = len(self.results.get('basic_benchmark', []))
            html += f'<tr><td>Basic Tests</td><td class="pass">✅</td><td>{count} tests completed</td></tr>'
        else:
            html += f'<tr><td>Basic Tests</td><td class="fail">❌</td><td>Error occurred</td></tr>'
        
        # Adversarial tests
        if 'summary' in self.results.get('adversarial_tests', {}):
            summary = self.results['adversarial_tests']['summary']
            pass_rate = summary.get('pass_rate', 0) * 100
            status = 'pass' if pass_rate > 70 else 'fail'
            html += f'<tr><td>Adversarial Tests</td><td class="{status}">{"✅" if pass_rate > 70 else "⚠️"}</td><td>{pass_rate:.1f}% pass rate</td></tr>'
        else:
            html += f'<tr><td>Adversarial Tests</td><td class="fail">❌</td><td>Error occurred</td></tr>'
        
        # Scalability tests
        if 'kb_sizes' in self.results.get('scalability_tests', {}):
            max_size = max(self.results['scalability_tests']['kb_sizes'])
            html += f'<tr><td>Scalability Tests</td><td class="pass">✅</td><td>Tested up to {max_size} facts</td></tr>'
        else:
            html += f'<tr><td>Scalability Tests</td><td class="fail">❌</td><td>Error occurred</td></tr>'
        
        html += '</table>'
        return html

if __name__ == "__main__":
    runner = CompleteBenchmarkRunner()
    results = runner.run_all_benchmarks()
    
    # Save complete results
    with open('benchmark/complete_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\\n✅ Complete benchmark suite finished!")
    print("📊 Results saved to:")
    print("   - benchmark/complete_benchmark_results.json")
    print("   - benchmark/benchmark_report.html")
"""

# Write the corrected file
with open('benchmark/run_complete_benchmark.py', 'w') as f:
    f.write(corrected_runner)

print("✅ Fixed run_complete_benchmark.py")
print("\n🚀 Now you can run:")
print("   python benchmark/run_complete_benchmark.py")