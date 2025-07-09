# k_assistant.py
# Phase 22.2: Final `what_is` Logic Edition
# - Verbessert die Ableitungslogik für negative Eigenschaften.
# - Entfernt Redundanzen in der Ausgabe.

import re
import pickle
import os
import time
import subprocess
import platform
from abc import ABC, abstractmethod
from collections import Counter
import threading
from typing import Optional, Tuple, List, Dict

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
    print("⚠️ WARNUNG: 'lark' nicht gefunden. Parser läuft im Fallback-Modus.")

try:
    from .hakgal_grammar import HAKGAL_GRAMMAR
except ImportError:
    print("❌ FEHLER: 'hakgal_grammar.py' nicht gefunden. Bitte stellen Sie sicher, dass die korrekte Version (v4.1) vorhanden ist.")
    exit()

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
    def prove(self, assumptions: list, goal: str) -> tuple[bool, str]: pass
    @abstractmethod
    def validate_syntax(self, formula: str) -> tuple[bool, str]: pass

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
        except lark.exceptions.UnexpectedCharacters as e:
            return (False, None, f"Unerwartetes Zeichen an Position {e.pos_in_stream}: '{e.char}'")
        except lark.exceptions.UnexpectedToken as e:
            expected = ", ".join(e.expected)
            return (False, None, f"Unerwartetes Token '{e.token}' an Position {e.pos_in_stream}. Erwartet: {expected}")
        except Exception as e:
            return (False, None, f"Parser-Fehler: {str(e)}")
    
    def _regex_fallback(self, formula: str) -> Tuple[bool, None, str]:
        if not formula.strip().endswith('.'):
            return (False, None, "Formel muss mit '.' enden")
        valid_chars = re.compile(r'^[A-Za-zÄÖÜäöüß0-9\s\(\),\->&|._-]+$')
        if not valid_chars.match(formula):
            return (False, None, "Ungültige Zeichen in der Formel")
        if formula.count('(') != formula.count(')'):
            return (False, None, "Unbalancierte Klammern")
        return (True, None, "Syntax OK (Regex-Fallback)")
    
    def extract_predicates(self, tree: 'lark.Tree') -> List[str]:
        predicates = []
        if tree:
            for node in tree.find_data('atom'):
                if node.children and isinstance(node.children[0], lark.Token) and node.children[0].type == 'PREDICATE':
                    predicates.append(str(node.children[0]))
        return list(dict.fromkeys(predicates))

#==============================================================================
# 2. KONKRETE PROVIDER- UND ADAPTER-IMPLEMENTIERUNGEN
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
        generation_config = genai.types.GenerationConfig(temperature=temperature)
        response = self.model.generate_content(full_prompt, generation_config=generation_config)
        return response.text.strip()

class PatternProver(BaseProver):
    def __init__(self): 
        super().__init__("Pattern Matcher")
        self.parser = HAKGALParser()

    def prove(self, assumptions: list, goal: str) -> tuple[bool, str]:
        if goal in assumptions:
            return (True, f"{self.name} hat einen exakten Match für '{goal}' gefunden.")
        
        if goal.startswith('-'):
            positive_form = goal[1:]
            if positive_form in assumptions:
                return (False, f"{self.name} hat einen Widerspruch für '{goal}' gefunden (positives Faktum existiert).")
        else:
            negative_form = f"-{goal}"
            if negative_form in assumptions:
                return (False, f"{self.name} hat einen Widerspruch für '{goal}' gefunden (negatives Faktum existiert).")

        return (None, f"{self.name} konnte keine exakte Übereinstimmung finden.")

    def validate_syntax(self, formula: str) -> tuple[bool, str]: 
        success, _, msg = self.parser.parse(formula)
        return (success, msg)

class Z3Adapter(BaseProver):
    def __init__(self):
        super().__init__("Z3 SMT Solver")
        z3.set_param('proof', True)
        self.parser = HAKGALParser()
        
    def _parse_hakgal_formula_to_z3_expr(self, formula_str: str, quantified_vars: set = None):
        if quantified_vars is None: quantified_vars = set()
        formula_str = formula_str.strip().removesuffix('.')
        def _find_top_level_operator(s: str, operators: list[str]) -> tuple[int, str | None]:
            balance = 0
            for i in range(len(s) - 1, -1, -1):
                char = s[i]
                if char == ')': balance += 1
                elif char == '(': balance -= 1
                elif balance == 0:
                    for op in operators:
                        if s.startswith(op, i): return i, op
            return -1, None
        expr = formula_str.strip()
        idx, op = _find_top_level_operator(expr, ['->'])
        if idx != -1: return z3.Implies(self._parse_hakgal_formula_to_z3_expr(expr[:idx], quantified_vars), self._parse_hakgal_formula_to_z3_expr(expr[idx + len(op):], quantified_vars))
        for op_symbol, op_func in [('|', z3.Or), ('&', z3.And)]:
            idx, op = _find_top_level_operator(expr, [op_symbol])
            if idx != -1: return op_func(self._parse_hakgal_formula_to_z3_expr(expr[:idx], quantified_vars), self._parse_hakgal_formula_to_z3_expr(expr[idx + len(op):], quantified_vars))
        if expr.startswith('-'): return z3.Not(self._parse_hakgal_formula_to_z3_expr(expr[1:], quantified_vars))
        all_match = re.match(r"all\s+([\w]+)\s+\((.*)\)$", expr, re.DOTALL)
        if all_match:
            var_name, body = all_match.groups()
            z3_var = z3.Int(var_name)
            return z3.ForAll([z3_var], self._parse_hakgal_formula_to_z3_expr(body, quantified_vars | {var_name}))
        if expr.startswith('(') and expr.endswith(')'): return self._parse_hakgal_formula_to_z3_expr(expr[1:-1], quantified_vars)
        match = re.match(r"([A-ZÄÖÜ][\w]*)\((.*?)\)", expr)
        if match:
            pred_name, args_str = match.group(1), match.group(2)
            args = [a.strip() for a in args_str.split(',') if a.strip()] if args_str else []
            z3_args = [z3.Int(arg) if arg in quantified_vars else z3.Int(arg) for arg in args]
            z3_func = z3.Function(pred_name, *([z3.IntSort()] * len(z3_args)), z3.BoolSort())
            return z3_func(*z3_args)
        raise ValueError(f"Konnte Formelteil nicht parsen: '{expr}'")
    
    def validate_syntax(self, formula: str) -> tuple[bool, str]:
        success, tree, msg = self.parser.parse(formula)
        if not success:
            return (False, f"Parser: {msg}")
        
        try: 
            self._parse_hakgal_formula_to_z3_expr(formula)
            return (True, f"✅ Syntax OK (Lark + Z3)")
        except (ValueError, z3.Z3Exception) as e: 
            return (False, f"Z3-Konvertierung fehlgeschlagen: {str(e)}")
    
    def prove(self, assumptions: list, goal: str) -> tuple[bool, str]:
        solver = z3.Tactic('smt').solver(); solver.set(model=True)
        try:
            for fact_str in assumptions: solver.add(self._parse_hakgal_formula_to_z3_expr(fact_str))
            goal_expr = self._parse_hakgal_formula_to_z3_expr(goal); solver.add(z3.Not(goal_expr))
        except (ValueError, z3.Z3Exception) as e: return (False, f"Fehler beim Parsen: {e}")
        check_result = solver.check()
        if check_result == z3.unsat: return (True, "Z3 hat das Ziel bewiesen (Problem ist unerfüllbar).")
        elif check_result == z3.sat: return (False, f"Z3 konnte das Ziel nicht beweisen (Gegenmodell existiert):\n{solver.model()}")
        else: return (False, f"Z3 konnte das Ziel nicht beweisen (unbekannt, Grund: {solver.reason_unknown()}).")

#==============================================================================
# 3. Shell-Manager
#==============================================================================
class ShellManager:
    def __init__(self):
        self.system = platform.system()
        self.shell = self._detect_shell()
        print(f"🖥️ System: {self.system}, Shell: {self.shell}")
        
    def _detect_shell(self) -> str:
        if self.system == "Windows":
            try:
                result = subprocess.run(["wsl.exe", "--version"], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    return "wsl"
            except: pass
            
            git_bash_paths = [
                r"C:\Program Files\Git\bin\bash.exe",
                r"C:\Program Files (x86)\Git\bin\bash.exe"
            ]
            for path in git_bash_paths:
                if os.path.exists(path):
                    return "git-bash"
            
            return "powershell"
        else:
            return "bash"
    
    def execute(self, command: str, timeout: int = 30) -> tuple[bool, str, str]:
        try:
            if self.shell == "wsl":
                proc = subprocess.run(["wsl.exe", "bash", "-c", command], capture_output=True, text=True, timeout=timeout)
            elif self.shell == "git-bash":
                proc = subprocess.run([r"C:\Program Files\Git\bin\bash.exe", "-c", command], capture_output=True, text=True, timeout=timeout)
            elif self.shell == "powershell":
                proc = subprocess.run(["powershell", "-Command", command], capture_output=True, text=True, timeout=timeout)
            else:
                proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout)
            return (proc.returncode == 0, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired:
            return (False, "", f"Timeout: Befehl überschritt {timeout} Sekunden")
        except Exception as e:
            return (False, "", f"Fehler: {str(e)}")
    
    def analyze_system_facts(self) -> list[str]:
        facts = []
        facts.append(f"LäuftAuf({self.system}).")
        facts.append(f"VerwendetShell({self.shell}).")
        python_version = platform.python_version()
        facts.append(f"PythonVersion({python_version.replace('.', '_')}).")
        return facts

#==============================================================================
# 4. MANAGER- UND KERN-KLASSEN
#==============================================================================
class EnsembleManager:
    def __init__(self):
        self.providers = []
        self._initialize_providers()
        self.system_prompt_logicalize = """You are a precise logic translator. Your task is to translate the user's question into a single, first-order logic formula that strictly adheres to the HAK/GAL syntax.

**HAK/GAL SYNTAX RULES:**
1.  **Structure:** Every formula must be a predicate with arguments in parentheses, like `Predicate(Argument)`.
2.  **Predicates:** Must be in `PascalCase`. Examples: `IstUrgent`, `HatHoheBetriebskosten`.
3.  **Entities/Constants:** Must be in `PascalCase`. Examples: `UserManagement`, `Ticket123`.
4.  **Variables:** Must be lowercase, like `x`, `y`.
5.  **Quantifiers:** Universal quantification must be written as `all x (...)`.
6.  **Operators:** Use `&` for AND, `|` for OR, `->` for IMPLIES, `-` for NOT.
7.  **Termination:** Every formula MUST end with a period `.`.
8.  **Idempotency Rule:** If the user's input is ALREADY a valid HAK/GAL formula, return it EXACTLY as is, without any changes or explanation.
9.  **NO DEVIATION:** Do not invent new predicates. Do not merge words. Respond ONLY with the formula.

**Example Translation:**
- User: "is ticket 123 urgent?" -> `IstUrgent(Ticket123).`
- User: "SollteRefactoredWerden(UserManagement)." -> `SollteRefactoredWerden(UserManagement).`
- User: "are all legacy systems critical?" -> `all x (IstLegacySystem(x) -> IstKritisch(x)).`

**Translate the following sentence:**
"""
        self.fact_extraction_prompt = """You are a precise logic extractor. Your task is to extract all facts and rules from the provided text and format them as a Python list of strings. Each string must be a valid HAK/GAL first-order logic formula.

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
        if key := os.getenv("DEEPSEEK_API_KEY"): self.providers.append(DeepSeekProvider(api_key=key)); print("   ✅ DeepSeek Provider geladen.")
        else: print("   ℹ️  DeepSeek Provider: DEEPSEEK_API_KEY nicht gefunden.")
        if key := os.getenv("MISTRAL_API_KEY"): self.providers.append(MistralProvider(api_key=key)); print("   ✅ Mistral Provider geladen.")
        else: print("   ℹ️  Mistral Provider: MISTRAL_API_KEY nicht gefunden.")
        if not GEMINI_AVAILABLE: print("   ℹ️  Gemini Provider: 'google-generativeai' Bibliothek nicht gefunden.")
        elif key := os.getenv("GEMINI_API_KEY"):
            try: self.providers.append(GeminiProvider(api_key=key)); print("   ✅ Gemini Provider geladen.")
            except Exception as e: print(f"   ❌ Fehler beim Laden des Gemini Providers: {e}")
        else: print("   ℹ️  Gemini Provider: GEMINI_API_KEY nicht gefunden.")
        if not self.providers: print("   ⚠️ Keine LLM-Provider aktiv. LLM-Funktionen sind deaktiviert.")
    
    def logicalize(self, sentence: str) -> str | None:
        if not self.providers: return None
        try: 
            if isinstance(self.providers[0], GeminiProvider):
                return self.providers[0].query(sentence, self.system_prompt_logicalize, 0)
            else:
                return self.providers[0].query(sentence, self.system_prompt_logicalize, 0)
        except Exception as e: 
            print(f"   [Warnung] Fehler bei Logik-Übersetzung: {e}"); return None
    
    def extract_facts_with_ensemble(self, context: str) -> list[str]:
        if not self.providers: return []
        threads = []
        results = [None] * len(self.providers)
        def worker(provider, index):
            try:
                prompt = self.fact_extraction_prompt.format(context=context)
                raw_output = provider.query(prompt, "", 0.1)
                match = re.search(r'\[.*\]', raw_output, re.DOTALL)
                if match:
                    try:
                        fact_list = eval(match.group(0))
                        if isinstance(fact_list, list): results[index] = fact_list
                    except: pass
            except Exception as e: print(f"   [Warnung] Fehler bei {provider.model_name}: {e}")
        for i, provider in enumerate(self.providers):
            thread = threading.Thread(target=worker, args=(provider, i)); threads.append(thread); thread.start()
        for thread in threads: thread.join()
        mistral_result = None
        other_results = []
        for i, provider in enumerate(self.providers):
            if isinstance(provider, MistralProvider):
                mistral_result = results[i]
            else:
                other_results.append(results[i])
        if mistral_result:
            print("   [Veto-Ensemble] ✅ Entscheidung von Mistral (Chef) wird verwendet.")
            return mistral_result
        print("   [Veto-Ensemble] ⚠️ Mistral hat keine Fakten geliefert. Fallback auf die restlichen Provider...")
        fallback_facts = []
        for res in other_results:
            if res: fallback_facts.extend(res)
        if not fallback_facts:
            return []
        fact_counts = Counter(fallback_facts)
        majority_threshold = len(other_results) // 2 + 1 
        consistent_facts = [fact for fact, count in fact_counts.items() if count >= majority_threshold]
        if consistent_facts:
            print(f"   [Veto-Ensemble] ✅ Konsens unter Fallback-Providern gefunden.")
        return consistent_facts

class WissensbasisManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if not RAG_ENABLED: self.model, self.index, self.chunks, self.doc_paths = None, None, [], {}; print("   ℹ️  RAG-Funktionen: Bibliotheken nicht gefunden."); return
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.doc_paths = {}
    def add_document(self, file_path: str):
        if not RAG_ENABLED: return
        doc_id = os.path.basename(file_path)
        if doc_id in self.doc_paths: print(f"   ℹ️ Dokument '{file_path}' wurde bereits indiziert."); return
        text = ""
        try:
            if file_path.endswith(".pdf"):
                reader = PdfReader(file_path)
                for page in reader.pages: text += page.extract_text() + "\n"
            else:
                with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
        except Exception as e: print(f"   ❌ Fehler beim Lesen der Datei: {e}"); return
        text_chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk.strip()) > 30]
        if not text_chunks: return
        embeddings = self.model.encode(text_chunks, convert_to_tensor=False, show_progress_bar=True)
        self.index.add(np.array(embeddings).astype('float32'))
        chunk_with_meta = [(chunk, doc_id) for chunk in text_chunks]
        self.chunks.extend(chunk_with_meta)
        self.doc_paths[doc_id] = file_path
        print(f"   ✅ {len(text_chunks)} Chunks aus '{doc_id}' erfolgreich indiziert.")
    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> list[str]:
        if not RAG_ENABLED or self.index.ntotal == 0: return []
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        _ , indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        return [self.chunks[i][0] for i in indices[0] if i != -1 and i < len(self.chunks)]

class HAKGAL_Core_FOL:
    def __init__(self):
        self.K = []
        self.prover_cascade = [PatternProver(), Z3Adapter()]
        self.parser_stats = {"total": 0, "success": 0, "failed": 0}
        self.proof_cache = {}

    def add_fact(self, formula_str: str): 
        if formula_str not in self.K:
            self.K.append(formula_str)
            self.proof_cache.clear()
            print(f"   [Cache] Proof-Cache wurde aufgrund neuer Fakten geleert.")
            return True
        return False
    
    def retract_fact(self, fact_to_remove: str) -> bool:
        if fact_to_remove in self.K:
            self.K.remove(fact_to_remove)
            self.proof_cache.clear()
            print(f"   [Cache] Proof-Cache wurde aufgrund entfernter Fakten geleert.")
            return True
        return False

    def check_consistency(self, new_fact: str) -> Tuple[bool, Optional[str]]:
        if new_fact.startswith('-'):
            negated_fact = new_fact[1:]
        else:
            negated_fact = f"-{new_fact}"
        
        if negated_fact.endswith('.'):
            negated_fact = negated_fact[:-1] + '.'

        is_contradictory, reason = self.verify_logical(negated_fact, self.K)
        if is_contradictory:
            return (False, f"Widerspruch gefunden! Der neue Fakt '{new_fact}' widerspricht der Wissensbasis, denn '{reason}'")
        
        return (True, None)

    def verify_logical(self, query_str: str, full_kb: list) -> tuple[int | None, str]:
        cache_key = (tuple(sorted(full_kb)), query_str)
        
        if cache_key in self.proof_cache:
            print("   [Cache] ✅ Treffer im Proof-Cache!")
            success, reason = self.proof_cache[cache_key]
            return (1, reason) if success else (None, reason)

        for prover in self.prover_cascade:
            # print(f"   [Prover] Versuche Beweis mit {prover.name}...") # Deaktiviert für sauberere Ausgabe
            success, reason = prover.prove(assumptions=full_kb, goal=query_str)
            
            if success is not None:
                # print(f"   [Prover] ✅ {prover.name} hat ein Ergebnis geliefert.") # Deaktiviert für sauberere Ausgabe
                if success and isinstance(prover, Z3Adapter):
                    self.proof_cache[cache_key] = (success, reason)
                return (1, reason) if success else (None, reason)
        
        return (None, "Keiner der verfügbaren Prover konnte eine definitive Antwort finden.")

    def update_parser_stats(self, success: bool):
        self.parser_stats["total"] += 1
        if success:
            self.parser_stats["success"] += 1
        else:
            self.parser_stats["failed"] += 1

#==============================================================================
# 5. K-ASSISTANT
#==============================================================================
class KAssistant:
    def __init__(self, kb_filepath="k_assistant.kb"):
        self.kb_filepath = kb_filepath
        self.core = HAKGAL_Core_FOL()
        self.ensemble_manager = EnsembleManager()
        self.wissensbasis_manager = WissensbasisManager()
        self.shell_manager = ShellManager()
        self.parser = HAKGALParser()
        self.potential_new_facts = []
        self.load_kb(kb_filepath)
        self._add_system_facts()
        prover_names = ' -> '.join([p.name for p in self.core.prover_cascade])
        print(f"--- Prover-Kaskade: {prover_names} ---")
        print(f"--- Parser-Modus: {'Lark' if self.parser.parser_available else 'Regex-Fallback'} ---")
    
    def _normalize_and_correct_syntax(self, formula: str) -> str:
        original_formula = formula
        
        # PRIORITÄT 1: Bindestrich-Probleme lösen
        bindestrich_map = {
            'RAG-Pipeline': 'RAGPipeline',
            'AI-System': 'AISystem', 
            'Machine-Learning': 'MachineLearning',
            'Real-Time-System': 'RealTimeSystem',
            'E-Mail-Server': 'EmailServer',
            'Multi-Agent': 'MultiAgent',
            'Deep-Learning': 'DeepLearning',
            'Neural-Network': 'NeuralNetwork'
        }
        
        corrected = formula
        for old_name, new_name in bindestrich_map.items():
            corrected = corrected.replace(old_name, new_name)
        
        # Generelle Bindestrich-Entfernung in Entity-Namen
        import re
        corrected = re.sub(r'\b([A-Z][a-zA-Z]*)-([A-Z][a-zA-Z]*)\b', r'\1\2', corrected)
        
        synonym_map = {
            r'IstTechnischesLegacy(System)?': 'IstLegacy',
            r'SollteRefactoringInBetrachtGezo[h|e]genWerden': 'SollteRefactoredWerden',
            r'SollteIdentifiziertUndRefactoredWerden': 'SollteRefactoredWerden',
            r'IstBasierendAufCobolMainframe': 'IstCobolMainframe',
            r'BasiertAufCobolMainframe': 'IstCobolMainframe',
            r'BasiertAufModernerJavaMicroservice': 'IstJavaMicroservice',
            r'IstBasierendAufJavaMicroservice': 'IstJavaMicroservice'
        }
        
        for pattern, canonical in synonym_map.items():
            corrected = re.sub(pattern, canonical, corrected)

        def normalize_entities(text):
            entities = re.findall(r'\b([A-Z][a-zA-Z0-9_]*)\b', text)
            for entity in entities:
                normalized = re.sub(r'(System|Tool|Architektur)

    def _add_system_facts(self):
        system_facts = self.shell_manager.analyze_system_facts()
        for fact in system_facts:
            if fact not in self.core.K:
                self.core.K.append(fact)
        print(f"   ✅ {len(system_facts)} Systemfakten automatisch hinzugefügt.")
    
    def test_parser(self, formula: str):
        print(f"\n> Parser-Test für: '{formula}'")
        success, tree, msg = self.parser.parse(formula)
        self.core.update_parser_stats(success)
        
        if success:
            print(f"✅ Parse erfolgreich: {msg}")
            if tree and self.parser.parser_available:
                predicates = self.parser.extract_predicates(tree)
                if predicates:
                    print(f"   Gefundene Prädikate: {', '.join(predicates)}")
        else:
            print(f"❌ Parse fehlgeschlagen: {msg}")
    
    def execute_shell(self, command: str):
        print(f"\n> Führe Shell-Befehl aus: '{command}'")
        success, stdout, stderr = self.shell_manager.execute(command)
        
        if success:
            print("✅ Befehl erfolgreich ausgeführt.")
            if stdout: print(f"📋 Ausgabe:\n{stdout}")
        else:
            print("❌ Befehl fehlgeschlagen.")
            if stderr: print(f"🚨 Fehler:\n{stderr}")
        
        if stdout and self.ensemble_manager.providers:
            print("🧠 Analysiere Ausgabe für mögliche Fakten...")
            facts = self.ensemble_manager.extract_facts_with_ensemble(stdout)
            if facts:
                print(f"   [Shell-Analyse] {len(facts)} potenzielle Fakten gefunden.")
                for fact in facts:
                    corrected_fact = self._normalize_and_correct_syntax(fact)
                    is_valid, _, _ = self.parser.parse(corrected_fact)
                    if is_valid:
                        print(f"      ✅ Temporärer Fakt: {corrected_fact}")
    
    def _ask_or_explain(self, q: str, explain: bool, is_raw: bool):
        print(f"\n> Frage{' zur Erklärung' if explain else ''}{' (roh)' if is_raw else ''}: '{q}'")
        self.potential_new_facts = []
        logical_form, temp_assumptions = q, []

        if is_raw:
            is_valid, _, error_msg = self.parser.parse(q)
            self.core.update_parser_stats(is_valid)
            if not is_valid: print(f"   ❌ FEHLER: Ungültige Syntax. {error_msg}"); return
        else:
            if RAG_ENABLED and self.wissensbasis_manager.index.ntotal > 0:
                print("🧠 RAG-Pipeline wird für Kontext angereichert...")
                relevant_chunks = self.wissensbasis_manager.retrieve_relevant_chunks(q)
                if relevant_chunks:
                    context = "\n\n".join(relevant_chunks)
                    print(f"   [RAG] Relevanter Kontext gefunden:\n---\n{context}\n---")
                    print("   [RAG] Extrahiere Fakten mit dem Veto-Ensemble...")
                    extracted_facts = self.ensemble_manager.extract_facts_with_ensemble(context)
                    if extracted_facts:
                        print("   [RAG] Fakten für den Beweis:")
                        for fact in extracted_facts:
                            corrected_fact = self._normalize_and_correct_syntax(fact)
                            is_valid, _, error_msg = self.parser.parse(corrected_fact)
                            self.core.update_parser_stats(is_valid)
                            if is_valid:
                                print(f"      ✅ {corrected_fact}")
                                temp_assumptions.append(corrected_fact)
                                if corrected_fact not in self.core.K and corrected_fact not in self.potential_new_facts:
                                    self.potential_new_facts.append(corrected_fact)
                            else:
                                print(f"      ❌ {fact.strip()} (Parser-Fehler: {error_msg}, wird ignoriert)")
                    else:
                        print("   [RAG] Keine Fakten gefunden.")
                else:
                    print("   [RAG] Keine relevanten Chunks für die Anfrage gefunden.")
            
            print("🔮 Übersetze Anfrage in Logik...")
            logical_form_raw = self.ensemble_manager.logicalize(q)
            if not logical_form_raw: print("   ❌ Orakel konnte keine Formel erzeugen."); return
            
            logical_form = self._normalize_and_correct_syntax(logical_form_raw)
            
            print(f"   -> '{logical_form}'")
            is_valid, _, error_msg = self.parser.parse(logical_form)
            self.core.update_parser_stats(is_valid)
            if not is_valid: print(f"   ❌ FEHLER: LLM hat ungültige Syntax generiert. {error_msg}"); return
        
        print(f"🛡️  Übergebe an Prover-Kaskade...")
        r, reason = self.core.verify_logical(logical_form, self.core.K + temp_assumptions)
        success = (r == 1)
        
        print("\n--- ERGEBNIS ---")
        if not explain:
            print("✅ Antwort: Ja." if success else "❔ Antwort: Nein/Unbekannt.")
            print(f"   [Begründung] {reason}")
        else:
            success_status_text = "Ja (bewiesen)" if success else "Nein (nicht bewiesen)"
            if not self.ensemble_manager.providers: print("   ❌ Konnte keine Erklärung generieren (keine LLM-Provider)."); return
            
            print("🗣️  Generiere einfache Erklärung...")
            explanation = self.ensemble_manager.providers[0].query(
                f"**Anfrage:** '{q}'\n**Ergebnis:** {success_status_text}\n**Grund:** '{reason}'\n\n**Aufgabe:** Übersetze dies in eine einfache Erklärung.",
                "Du bist ein Logik-Experte, der formale Beweisergebnisse in einfache Sprache übersetzt.", 0.2)
            print(f"\n--- Erklärung ---\n{explanation}\n-------------------\n")

        if self.potential_new_facts:
            print(f"💡 INFO: {len(self.potential_new_facts)} neue Fakten wurden gefunden. Benutze 'learn', um sie permanent zu speichern.")

    def add_raw(self, formula: str):
        print(f"\n> Füge KERNREGEL hinzu: '{formula}'")
        is_valid, _, error_msg = self.parser.parse(formula)
        self.core.update_parser_stats(is_valid)
        if not is_valid: print(f"   ❌ FEHLER: Ungültige Syntax. {error_msg}"); return
        
        is_consistent, reason = self.core.check_consistency(formula)
        if not is_consistent:
            print(f"   🛡️  WARNUNG: {reason}")
            print(f"   -> Fakt '{formula}' wurde NICHT hinzugefügt, um die Konsistenz zu wahren.")
            return

        if self.core.add_fact(formula):
            print("   -> Erfolgreich zur permanenten Wissensbasis hinzugefügt.")
        else:
            print("   -> Fakt ist bereits in der Wissensbasis vorhanden.")
        
    def retract(self, formula_to_retract: str):
        print(f"\n> Entferne KERNREGEL: '{formula_to_retract}'")
        
        normalized_target = self._normalize_and_correct_syntax(formula_to_retract)

        fact_to_remove = None
        for fact_in_kb in self.core.K:
            normalized_kb_fact = self._normalize_and_correct_syntax(fact_in_kb)
            if normalized_kb_fact == normalized_target:
                fact_to_remove = fact_in_kb
                break
        
        if fact_to_remove:
            if self.core.retract_fact(fact_to_remove):
                print(f"   -> Fakt '{fact_to_remove}' erfolgreich entfernt.")
            else:
                print(f"   -> INTERNER FEHLER: Fakt '{fact_to_remove}' gefunden, aber konnte nicht entfernt werden.")
        else:
            print(f"   -> Fakt, der zu '{normalized_target}' normalisiert wird, wurde nicht in der Wissensbasis gefunden.")

    def learn_facts(self):
        if not self.potential_new_facts:
            print("🧠 Nichts Neues zu lernen. Führen Sie zuerst eine 'ask'-Anfrage aus.")
            return
        
        print(f"🧠 Lerne {len(self.potential_new_facts)} neue Fakten...")
        added_count = 0
        for fact in self.potential_new_facts:
            is_consistent, reason = self.core.check_consistency(fact)
            if not is_consistent:
                print(f"   🛡️  WARNUNG: {reason}")
                print(f"   -> Fakt '{fact}' wird NICHT gelernt, um die Konsistenz zu wahren.")
                continue

            if self.core.add_fact(fact):
                print(f"   + {fact}")
                added_count += 1
        
        if added_count > 0:
            print(f"✅ {added_count} neue Fakten wurden permanent gespeichert.")
        else:
            print("ℹ️ Alle vorgeschlagenen Fakten waren bereits bekannt oder inkonsistent.")
            
        self.potential_new_facts = []

    def clear_cache(self):
        self.core.proof_cache.clear()
        print("🗑️ Proof-Cache wurde geleert.")

    def ask(self, q: str): self._ask_or_explain(q, explain=False, is_raw=False)
    def explain(self, q: str): self._ask_or_explain(q, explain=True, is_raw=False)
    def ask_raw(self, formula: str): self._ask_or_explain(formula, explain=False, is_raw=True)
    
    def status(self): 
        print(f"\n--- System Status (v22.2) ---")
        prover_names = ' -> '.join([p.name for p in self.core.prover_cascade])
        print(f"  Prover-Kaskade: {prover_names}")
        print(f"  Parser-Modus: {'Lark' if self.parser.parser_available else 'Regex-Fallback'}")
        print(f"  Permanente Fakten: {len(self.core.K)}")
        print(f"  Gecachte Beweise: {len(self.core.proof_cache)}")
        if self.potential_new_facts:
            print(f"  Vorgeschlagene Fakten zum Lernen: {len(self.potential_new_facts)}")
        print("\n  --- Parser Statistiken ---")
        stats = self.core.parser_stats
        if stats["total"] > 0:
            success_rate = (stats["success"] / stats["total"]) * 100
            print(f"  Parse-Versuche: {stats['total']}")
            print(f"  Erfolgreich: {stats['success']} ({success_rate:.1f}%)")
            print(f"  Fehlgeschlagen: {stats['failed']}")
        else:
            print("  (Noch keine Parse-Versuche)")
        print("\n  --- Shell Status ---")
        print(f"  System: {self.shell_manager.system}")
        print(f"  Shell: {self.shell_manager.shell}")
        print("\n  --- RAG Status ---")
        if RAG_ENABLED:
            print(f"  Indizierte Dokumente: {len(self.wissensbasis_manager.doc_paths)}")
            print(f"  Indizierte Chunks: {self.wissensbasis_manager.index.ntotal}")
        else: print("  (RAG deaktiviert)")
        print("\n  --- LLM Status ---")
        print(f"  Verfügbare LLM-Provider: {len(self.ensemble_manager.providers)}")
        for p in self.ensemble_manager.providers: print(f"    - {p.model_name}")
        
    def show(self):
        print("\n--- Permanente Wissensbasis (Kernregeln) ---")
        if not self.core.K: print("   (Leer)")
        else:
            for i, fact in enumerate(self.core.K): print(f"   [{i}] {fact}")
        
        if self.potential_new_facts:
            print("\n--- Vorgeschlagene Fakten zum Lernen (mit 'learn' übernehmen) ---")
            for i, fact in enumerate(self.potential_new_facts):
                print(f"   [{i}] {fact}")

        print("\n--- Indizierte Wissens-Chunks ---")
        if RAG_ENABLED and self.wissensbasis_manager.chunks:
            for i, (chunk, doc_id) in enumerate(self.wissensbasis_manager.chunks): print(f"   [{i} from {doc_id}] {chunk[:80]}...")
        else: print("   (Leer oder RAG deaktiviert)")
        
    def save_kb(self, filepath: str):
        try:
            rag_data = {'chunks': self.wissensbasis_manager.chunks, 'doc_paths': self.wissensbasis_manager.doc_paths} if RAG_ENABLED else {}
            data = {
                'facts': self.core.K, 
                'rag_data': rag_data,
                'parser_stats': self.core.parser_stats,
                'proof_cache': self.core.proof_cache
            }
            with open(filepath, 'wb') as f: pickle.dump(data, f)
            print(f"✅ Wissensbasis erfolgreich in '{filepath}' gespeichert.")
        except Exception as e: print(f"❌ Fehler beim Speichern: {e}")
        
    def load_kb(self, filepath: str):
        if not os.path.exists(filepath): return
        try:
            with open(filepath, 'rb') as f: data = pickle.load(f)
            self.core.K = data.get('facts', [])
            self.core.parser_stats = data.get('parser_stats', {"total": 0, "success": 0, "failed": 0})
            self.core.proof_cache = data.get('proof_cache', {})
            print(f"✅ {len(self.core.K)} Kernregeln und {len(self.core.proof_cache)} gecachte Beweise geladen.")
            if RAG_ENABLED and 'rag_data' in data:
                rag_chunks_with_meta = data['rag_data'].get('chunks', [])
                self.wissensbasis_manager.chunks = rag_chunks_with_meta
                self.wissensbasis_manager.doc_paths = data['rag_data'].get('doc_paths', {})
                if rag_chunks_with_meta:
                    just_chunks = [c[0] for c in rag_chunks_with_meta]
                    embeddings = self.wissensbasis_manager.model.encode(just_chunks, convert_to_tensor=False, show_progress_bar=False)
                    self.wissensbasis_manager.index.add(np.array(embeddings).astype('float32'))
                    print(f"✅ {len(rag_chunks_with_meta)} RAG-Chunks aus Speicher geladen und indiziert.")
        except Exception as e: print(f"❌ Fehler beim Laden: {e}")
        
    def build_kb_from_file(self, filepath: str): self.wissensbasis_manager.add_document(filepath)
    def search(self, query: str):
        if not RAG_ENABLED: print("   ❌ RAG-Funktionen sind deaktiviert."); return
        print(f"\n> Suche nach Kontext für: '{query}'")
        relevant_chunks = self.wissensbasis_manager.retrieve_relevant_chunks(query)
        if not relevant_chunks: print("   [RAG] Keine relevanten Informationen gefunden."); return
        print(f"   [RAG] Relevanter Kontext gefunden:\n---")
        for i, chunk in enumerate(relevant_chunks, 1): print(f"[{i}] {chunk}\n")
    def sources(self):
        if not RAG_ENABLED: print("   ❌ RAG-Funktionen sind deaktiviert."); return
        print("\n📑 Indizierte Wissensquellen:")
        if not self.wissensbasis_manager.doc_paths: print("   (Keine Dokumente geladen)")
        else:
            for doc_id, path in self.wissensbasis_manager.doc_paths.items(): print(f"   - {doc_id} (aus {path})")
    
    def what_is(self, entity: str):
        print(f"\n> Analysiere Wissen über Entität: '{entity}'")
        
        explicit_facts = [fact for fact in self.core.K if f"({entity})" in fact or f",{entity})" in fact or f"({entity}," in fact]
        
        unary_predicates_to_test = [
            "IstLegacy", "IstKritisch", "IstOnline", "IstStabil", 
            "HatHoheBetriebskosten", "SollteRefactoredWerden", "MussÜberwachtWerden"
        ]
        
        derived_properties = []
        print("🧠 Leite Eigenschaften ab...")
        for pred in unary_predicates_to_test:
            positive_goal = f"{pred}({entity})."
            is_positive, _ = self.core.verify_logical(positive_goal, self.core.K)
            if is_positive:
                if positive_goal not in explicit_facts:
                    derived_properties.append(positive_goal)
                continue 

            negative_goal = f"-{pred}({entity})."
            is_negative, _ = self.core.verify_logical(negative_goal, self.core.K)
            if is_negative:
                if negative_goal not in explicit_facts:
                    derived_properties.append(negative_goal)

        print("\n--- Profil für Entität: " + entity + " ---")
        if explicit_facts:
            print("\n  [Explizite Fakten]")
            for fact in explicit_facts:
                print(f"   - {fact}")
        else:
            print("\n  [Explizite Fakten]")
            print("   (Keine direkten Fakten in der Wissensbasis gefunden)")

        if derived_properties:
            print("\n  [Abgeleitete Eigenschaften]")
            for prop in derived_properties:
                print(f"   - {prop}")
        else:
            print("\n  [Abgeleitete Eigenschaften]")
            print("   (Keine Eigenschaften konnten abgeleitet werden)")
        print("------------------------------------")

#==============================================================================
# 6. MAIN LOOP
#==============================================================================
def print_help():
    print("\n--- K-Assistant Hilfe (v22.2) ---")
    print("\n  === Wissensbasis & RAG ===")
    print("  build_kb <pfad>    - Indiziert ein Dokument für RAG (persistent)")
    print("  add_raw <formel>   - Fügt eine KERNREGEL hinzu (mit Konsistenzprüfung)")
    print("  retract <formel>   - Entfernt eine KERNREGEL aus der Wissensbasis")
    print("  learn              - Speichert die zuletzt durch RAG gefundenen Fakten permanent")
    print("  show               - Zeigt die permanente & indizierte Wissensbasis an")
    print("  sources            - Zeigt alle indizierten Wissensquellen an")
    print("\n  === Analyse & Anfragen ===")
    print("  search <anfrage>   - Findet relevanten Text in der Wissensbasis (RAG)")
    print("  ask <frage>        - Beantwortet eine Frage (nutzt automatisch RAG)")
    print("  explain <frage>    - Erklärt eine Antwort (nutzt automatisch RAG)")
    print("  ask_raw <formel>   - Stellt eine rohe logische Frage (ohne RAG)")
    print("  what_is <entität>  - Zeigt alle Fakten und abgeleiteten Eigenschaften einer Entität an")
    print("\n  === Parser & Validierung ===")
    print("  parse <formel>     - Testet den Parser mit einer Formel")
    print("\n  === Shell & System ===")
    print("  shell <befehl>     - Führt einen Shell-Befehl aus")
    print("\n  === System & Steuerung ===")
    print("  status             - Zeigt den Systemstatus und Metriken an")
    print("  clearcache         - Leert den Proof-Cache")
    print("  exit               - Beendet die Anwendung")
    print("---------------------------------\n")

def main_loop():
    try:
        from backup_manager import BackupManager
        backup_mgr = BackupManager()
        backup_mgr.auto_backup("v22.2", "Final `what_is` Bugfix")
    except ImportError:
        print("ℹ️  'backup_manager.py' nicht gefunden, Backup wird übersprungen.")
    
    assistant = KAssistant()
    print_help()
    
    while True:
        try:
            user_input = input("> ").strip()
            if not user_input: continue
            parts = user_input.split(" ", 1)
            command = parts[0].lower()
            args = parts[1].strip('"\'') if len(parts) > 1 else ""
            
            if command == "exit": assistant.save_kb(assistant.kb_filepath); break
            elif command == "help": print_help()
            elif command == "build_kb" and args: assistant.build_kb_from_file(args)
            elif command == "add_raw" and args: assistant.add_raw(args)
            elif command == "retract" and args: assistant.retract(args)
            elif command == "learn" and not args: assistant.learn_facts()
            elif command == "clearcache" and not args: assistant.clear_cache()
            elif command == "ask" and args: assistant.ask(args)
            elif command == "explain" and args: assistant.explain(args)
            elif command == "ask_raw" and args: assistant.ask_raw(args)
            elif command == "status": assistant.status()
            elif command == "show": assistant.show()
            elif command == "prover": print("ℹ️ Der `prover`-Befehl ist veraltet. Die Prover-Kaskade wird automatisch ausgeführt.")
            elif command == "search" and args: assistant.search(args)
            elif command == "sources": assistant.sources()
            elif command == "what_is" and args: assistant.what_is(args)
            elif command == "shell" and args: assistant.execute_shell(args)
            elif command == "parse" and args: assistant.test_parser(args)
            else: print(f"Unbekannter Befehl: '{user_input}'.")
        except (KeyboardInterrupt, EOFError):
            if 'assistant' in locals(): assistant.save_kb(assistant.kb_filepath)
            break
        except Exception as e:
            import traceback
            print(f"\n🚨 Unerwarteter Fehler: {e}"); traceback.print_exc()

if __name__ == "__main__":
    main_loop(), '', entity)
                if normalized != entity and normalized != "":
                    text = text.replace(entity, normalized)
            return text

        corrected = normalize_entities(corrected)
        corrected = corrected.strip()
        corrected = re.sub(r"^[^\w-]+", "", corrected)
        corrected = corrected.replace(':-', '->')
        
        while corrected.startswith('--'):
            corrected = corrected[2:]
            
        corrected = corrected.replace('~', '-')
        
        if re.match(r"^[A-Z][a-zA-Z0-9_]*\.$", corrected):
            corrected = corrected.replace('.', '().')

        if corrected != original_formula:
            print(f"   [Syntax-Korrektur] '{original_formula.strip()}' -> '{corrected.strip()}'")
        
        return corrected

    def _add_system_facts(self):
        system_facts = self.shell_manager.analyze_system_facts()
        for fact in system_facts:
            if fact not in self.core.K:
                self.core.K.append(fact)
        print(f"   ✅ {len(system_facts)} Systemfakten automatisch hinzugefügt.")
    
    def test_parser(self, formula: str):
        print(f"\n> Parser-Test für: '{formula}'")
        success, tree, msg = self.parser.parse(formula)
        self.core.update_parser_stats(success)
        
        if success:
            print(f"✅ Parse erfolgreich: {msg}")
            if tree and self.parser.parser_available:
                predicates = self.parser.extract_predicates(tree)
                if predicates:
                    print(f"   Gefundene Prädikate: {', '.join(predicates)}")
        else:
            print(f"❌ Parse fehlgeschlagen: {msg}")
    
    def execute_shell(self, command: str):
        print(f"\n> Führe Shell-Befehl aus: '{command}'")
        success, stdout, stderr = self.shell_manager.execute(command)
        
        if success:
            print("✅ Befehl erfolgreich ausgeführt.")
            if stdout: print(f"📋 Ausgabe:\n{stdout}")
        else:
            print("❌ Befehl fehlgeschlagen.")
            if stderr: print(f"🚨 Fehler:\n{stderr}")
        
        if stdout and self.ensemble_manager.providers:
            print("🧠 Analysiere Ausgabe für mögliche Fakten...")
            facts = self.ensemble_manager.extract_facts_with_ensemble(stdout)
            if facts:
                print(f"   [Shell-Analyse] {len(facts)} potenzielle Fakten gefunden.")
                for fact in facts:
                    corrected_fact = self._normalize_and_correct_syntax(fact)
                    is_valid, _, _ = self.parser.parse(corrected_fact)
                    if is_valid:
                        print(f"      ✅ Temporärer Fakt: {corrected_fact}")
    
    def _ask_or_explain(self, q: str, explain: bool, is_raw: bool):
        print(f"\n> Frage{' zur Erklärung' if explain else ''}{' (roh)' if is_raw else ''}: '{q}'")
        self.potential_new_facts = []
        logical_form, temp_assumptions = q, []

        if is_raw:
            is_valid, _, error_msg = self.parser.parse(q)
            self.core.update_parser_stats(is_valid)
            if not is_valid: print(f"   ❌ FEHLER: Ungültige Syntax. {error_msg}"); return
        else:
            if RAG_ENABLED and self.wissensbasis_manager.index.ntotal > 0:
                print("🧠 RAG-Pipeline wird für Kontext angereichert...")
                relevant_chunks = self.wissensbasis_manager.retrieve_relevant_chunks(q)
                if relevant_chunks:
                    context = "\n\n".join(relevant_chunks)
                    print(f"   [RAG] Relevanter Kontext gefunden:\n---\n{context}\n---")
                    print("   [RAG] Extrahiere Fakten mit dem Veto-Ensemble...")
                    extracted_facts = self.ensemble_manager.extract_facts_with_ensemble(context)
                    if extracted_facts:
                        print("   [RAG] Fakten für den Beweis:")
                        for fact in extracted_facts:
                            corrected_fact = self._normalize_and_correct_syntax(fact)
                            is_valid, _, error_msg = self.parser.parse(corrected_fact)
                            self.core.update_parser_stats(is_valid)
                            if is_valid:
                                print(f"      ✅ {corrected_fact}")
                                temp_assumptions.append(corrected_fact)
                                if corrected_fact not in self.core.K and corrected_fact not in self.potential_new_facts:
                                    self.potential_new_facts.append(corrected_fact)
                            else:
                                print(f"      ❌ {fact.strip()} (Parser-Fehler: {error_msg}, wird ignoriert)")
                    else:
                        print("   [RAG] Keine Fakten gefunden.")
                else:
                    print("   [RAG] Keine relevanten Chunks für die Anfrage gefunden.")
            
            print("🔮 Übersetze Anfrage in Logik...")
            logical_form_raw = self.ensemble_manager.logicalize(q)
            if not logical_form_raw: print("   ❌ Orakel konnte keine Formel erzeugen."); return
            
            logical_form = self._normalize_and_correct_syntax(logical_form_raw)
            
            print(f"   -> '{logical_form}'")
            is_valid, _, error_msg = self.parser.parse(logical_form)
            self.core.update_parser_stats(is_valid)
            if not is_valid: print(f"   ❌ FEHLER: LLM hat ungültige Syntax generiert. {error_msg}"); return
        
        print(f"🛡️  Übergebe an Prover-Kaskade...")
        r, reason = self.core.verify_logical(logical_form, self.core.K + temp_assumptions)
        success = (r == 1)
        
        print("\n--- ERGEBNIS ---")
        if not explain:
            print("✅ Antwort: Ja." if success else "❔ Antwort: Nein/Unbekannt.")
            print(f"   [Begründung] {reason}")
        else:
            success_status_text = "Ja (bewiesen)" if success else "Nein (nicht bewiesen)"
            if not self.ensemble_manager.providers: print("   ❌ Konnte keine Erklärung generieren (keine LLM-Provider)."); return
            
            print("🗣️  Generiere einfache Erklärung...")
            explanation = self.ensemble_manager.providers[0].query(
                f"**Anfrage:** '{q}'\n**Ergebnis:** {success_status_text}\n**Grund:** '{reason}'\n\n**Aufgabe:** Übersetze dies in eine einfache Erklärung.",
                "Du bist ein Logik-Experte, der formale Beweisergebnisse in einfache Sprache übersetzt.", 0.2)
            print(f"\n--- Erklärung ---\n{explanation}\n-------------------\n")

        if self.potential_new_facts:
            print(f"💡 INFO: {len(self.potential_new_facts)} neue Fakten wurden gefunden. Benutze 'learn', um sie permanent zu speichern.")

    def add_raw(self, formula: str):
        print(f"\n> Füge KERNREGEL hinzu: '{formula}'")
        is_valid, _, error_msg = self.parser.parse(formula)
        self.core.update_parser_stats(is_valid)
        if not is_valid: print(f"   ❌ FEHLER: Ungültige Syntax. {error_msg}"); return
        
        is_consistent, reason = self.core.check_consistency(formula)
        if not is_consistent:
            print(f"   🛡️  WARNUNG: {reason}")
            print(f"   -> Fakt '{formula}' wurde NICHT hinzugefügt, um die Konsistenz zu wahren.")
            return

        if self.core.add_fact(formula):
            print("   -> Erfolgreich zur permanenten Wissensbasis hinzugefügt.")
        else:
            print("   -> Fakt ist bereits in der Wissensbasis vorhanden.")
        
    def retract(self, formula_to_retract: str):
        print(f"\n> Entferne KERNREGEL: '{formula_to_retract}'")
        
        normalized_target = self._normalize_and_correct_syntax(formula_to_retract)

        fact_to_remove = None
        for fact_in_kb in self.core.K:
            normalized_kb_fact = self._normalize_and_correct_syntax(fact_in_kb)
            if normalized_kb_fact == normalized_target:
                fact_to_remove = fact_in_kb
                break
        
        if fact_to_remove:
            if self.core.retract_fact(fact_to_remove):
                print(f"   -> Fakt '{fact_to_remove}' erfolgreich entfernt.")
            else:
                print(f"   -> INTERNER FEHLER: Fakt '{fact_to_remove}' gefunden, aber konnte nicht entfernt werden.")
        else:
            print(f"   -> Fakt, der zu '{normalized_target}' normalisiert wird, wurde nicht in der Wissensbasis gefunden.")

    def learn_facts(self):
        if not self.potential_new_facts:
            print("🧠 Nichts Neues zu lernen. Führen Sie zuerst eine 'ask'-Anfrage aus.")
            return
        
        print(f"🧠 Lerne {len(self.potential_new_facts)} neue Fakten...")
        added_count = 0
        for fact in self.potential_new_facts:
            is_consistent, reason = self.core.check_consistency(fact)
            if not is_consistent:
                print(f"   🛡️  WARNUNG: {reason}")
                print(f"   -> Fakt '{fact}' wird NICHT gelernt, um die Konsistenz zu wahren.")
                continue

            if self.core.add_fact(fact):
                print(f"   + {fact}")
                added_count += 1
        
        if added_count > 0:
            print(f"✅ {added_count} neue Fakten wurden permanent gespeichert.")
        else:
            print("ℹ️ Alle vorgeschlagenen Fakten waren bereits bekannt oder inkonsistent.")
            
        self.potential_new_facts = []

    def clear_cache(self):
        self.core.proof_cache.clear()
        print("🗑️ Proof-Cache wurde geleert.")

    def ask(self, q: str): self._ask_or_explain(q, explain=False, is_raw=False)
    def explain(self, q: str): self._ask_or_explain(q, explain=True, is_raw=False)
    def ask_raw(self, formula: str): self._ask_or_explain(formula, explain=False, is_raw=True)
    
    def status(self): 
        print(f"\n--- System Status (v22.2) ---")
        prover_names = ' -> '.join([p.name for p in self.core.prover_cascade])
        print(f"  Prover-Kaskade: {prover_names}")
        print(f"  Parser-Modus: {'Lark' if self.parser.parser_available else 'Regex-Fallback'}")
        print(f"  Permanente Fakten: {len(self.core.K)}")
        print(f"  Gecachte Beweise: {len(self.core.proof_cache)}")
        if self.potential_new_facts:
            print(f"  Vorgeschlagene Fakten zum Lernen: {len(self.potential_new_facts)}")
        print("\n  --- Parser Statistiken ---")
        stats = self.core.parser_stats
        if stats["total"] > 0:
            success_rate = (stats["success"] / stats["total"]) * 100
            print(f"  Parse-Versuche: {stats['total']}")
            print(f"  Erfolgreich: {stats['success']} ({success_rate:.1f}%)")
            print(f"  Fehlgeschlagen: {stats['failed']}")
        else:
            print("  (Noch keine Parse-Versuche)")
        print("\n  --- Shell Status ---")
        print(f"  System: {self.shell_manager.system}")
        print(f"  Shell: {self.shell_manager.shell}")
        print("\n  --- RAG Status ---")
        if RAG_ENABLED:
            print(f"  Indizierte Dokumente: {len(self.wissensbasis_manager.doc_paths)}")
            print(f"  Indizierte Chunks: {self.wissensbasis_manager.index.ntotal}")
        else: print("  (RAG deaktiviert)")
        print("\n  --- LLM Status ---")
        print(f"  Verfügbare LLM-Provider: {len(self.ensemble_manager.providers)}")
        for p in self.ensemble_manager.providers: print(f"    - {p.model_name}")
        
    def show(self):
        print("\n--- Permanente Wissensbasis (Kernregeln) ---")
        if not self.core.K: print("   (Leer)")
        else:
            for i, fact in enumerate(self.core.K): print(f"   [{i}] {fact}")
        
        if self.potential_new_facts:
            print("\n--- Vorgeschlagene Fakten zum Lernen (mit 'learn' übernehmen) ---")
            for i, fact in enumerate(self.potential_new_facts):
                print(f"   [{i}] {fact}")

        print("\n--- Indizierte Wissens-Chunks ---")
        if RAG_ENABLED and self.wissensbasis_manager.chunks:
            for i, (chunk, doc_id) in enumerate(self.wissensbasis_manager.chunks): print(f"   [{i} from {doc_id}] {chunk[:80]}...")
        else: print("   (Leer oder RAG deaktiviert)")
        
    def save_kb(self, filepath: str):
        try:
            rag_data = {'chunks': self.wissensbasis_manager.chunks, 'doc_paths': self.wissensbasis_manager.doc_paths} if RAG_ENABLED else {}
            data = {
                'facts': self.core.K, 
                'rag_data': rag_data,
                'parser_stats': self.core.parser_stats,
                'proof_cache': self.core.proof_cache
            }
            with open(filepath, 'wb') as f: pickle.dump(data, f)
            print(f"✅ Wissensbasis erfolgreich in '{filepath}' gespeichert.")
        except Exception as e: print(f"❌ Fehler beim Speichern: {e}")
        
    def load_kb(self, filepath: str):
        if not os.path.exists(filepath): return
        try:
            with open(filepath, 'rb') as f: data = pickle.load(f)
            self.core.K = data.get('facts', [])
            self.core.parser_stats = data.get('parser_stats', {"total": 0, "success": 0, "failed": 0})
            self.core.proof_cache = data.get('proof_cache', {})
            print(f"✅ {len(self.core.K)} Kernregeln und {len(self.core.proof_cache)} gecachte Beweise geladen.")
            if RAG_ENABLED and 'rag_data' in data:
                rag_chunks_with_meta = data['rag_data'].get('chunks', [])
                self.wissensbasis_manager.chunks = rag_chunks_with_meta
                self.wissensbasis_manager.doc_paths = data['rag_data'].get('doc_paths', {})
                if rag_chunks_with_meta:
                    just_chunks = [c[0] for c in rag_chunks_with_meta]
                    embeddings = self.wissensbasis_manager.model.encode(just_chunks, convert_to_tensor=False, show_progress_bar=False)
                    self.wissensbasis_manager.index.add(np.array(embeddings).astype('float32'))
                    print(f"✅ {len(rag_chunks_with_meta)} RAG-Chunks aus Speicher geladen und indiziert.")
        except Exception as e: print(f"❌ Fehler beim Laden: {e}")
        
    def build_kb_from_file(self, filepath: str): self.wissensbasis_manager.add_document(filepath)
    def search(self, query: str):
        if not RAG_ENABLED: print("   ❌ RAG-Funktionen sind deaktiviert."); return
        print(f"\n> Suche nach Kontext für: '{query}'")
        relevant_chunks = self.wissensbasis_manager.retrieve_relevant_chunks(query)
        if not relevant_chunks: print("   [RAG] Keine relevanten Informationen gefunden."); return
        print(f"   [RAG] Relevanter Kontext gefunden:\n---")
        for i, chunk in enumerate(relevant_chunks, 1): print(f"[{i}] {chunk}\n")
    def sources(self):
        if not RAG_ENABLED: print("   ❌ RAG-Funktionen sind deaktiviert."); return
        print("\n📑 Indizierte Wissensquellen:")
        if not self.wissensbasis_manager.doc_paths: print("   (Keine Dokumente geladen)")
        else:
            for doc_id, path in self.wissensbasis_manager.doc_paths.items(): print(f"   - {doc_id} (aus {path})")
    
    def what_is(self, entity: str):
        print(f"\n> Analysiere Wissen über Entität: '{entity}'")
        
        explicit_facts = [fact for fact in self.core.K if f"({entity})" in fact or f",{entity})" in fact or f"({entity}," in fact]
        
        unary_predicates_to_test = [
            "IstLegacy", "IstKritisch", "IstOnline", "IstStabil", 
            "HatHoheBetriebskosten", "SollteRefactoredWerden", "MussÜberwachtWerden"
        ]
        
        derived_properties = []
        print("🧠 Leite Eigenschaften ab...")
        for pred in unary_predicates_to_test:
            positive_goal = f"{pred}({entity})."
            is_positive, _ = self.core.verify_logical(positive_goal, self.core.K)
            if is_positive:
                if positive_goal not in explicit_facts:
                    derived_properties.append(positive_goal)
                continue 

            negative_goal = f"-{pred}({entity})."
            is_negative, _ = self.core.verify_logical(negative_goal, self.core.K)
            if is_negative:
                if negative_goal not in explicit_facts:
                    derived_properties.append(negative_goal)

        print("\n--- Profil für Entität: " + entity + " ---")
        if explicit_facts:
            print("\n  [Explizite Fakten]")
            for fact in explicit_facts:
                print(f"   - {fact}")
        else:
            print("\n  [Explizite Fakten]")
            print("   (Keine direkten Fakten in der Wissensbasis gefunden)")

        if derived_properties:
            print("\n  [Abgeleitete Eigenschaften]")
            for prop in derived_properties:
                print(f"   - {prop}")
        else:
            print("\n  [Abgeleitete Eigenschaften]")
            print("   (Keine Eigenschaften konnten abgeleitet werden)")
        print("------------------------------------")

#==============================================================================
# 6. MAIN LOOP
#==============================================================================
def print_help():
    print("\n--- K-Assistant Hilfe (v22.2) ---")
    print("\n  === Wissensbasis & RAG ===")
    print("  build_kb <pfad>    - Indiziert ein Dokument für RAG (persistent)")
    print("  add_raw <formel>   - Fügt eine KERNREGEL hinzu (mit Konsistenzprüfung)")
    print("  retract <formel>   - Entfernt eine KERNREGEL aus der Wissensbasis")
    print("  learn              - Speichert die zuletzt durch RAG gefundenen Fakten permanent")
    print("  show               - Zeigt die permanente & indizierte Wissensbasis an")
    print("  sources            - Zeigt alle indizierten Wissensquellen an")
    print("\n  === Analyse & Anfragen ===")
    print("  search <anfrage>   - Findet relevanten Text in der Wissensbasis (RAG)")
    print("  ask <frage>        - Beantwortet eine Frage (nutzt automatisch RAG)")
    print("  explain <frage>    - Erklärt eine Antwort (nutzt automatisch RAG)")
    print("  ask_raw <formel>   - Stellt eine rohe logische Frage (ohne RAG)")
    print("  what_is <entität>  - Zeigt alle Fakten und abgeleiteten Eigenschaften einer Entität an")
    print("\n  === Parser & Validierung ===")
    print("  parse <formel>     - Testet den Parser mit einer Formel")
    print("\n  === Shell & System ===")
    print("  shell <befehl>     - Führt einen Shell-Befehl aus")
    print("\n  === System & Steuerung ===")
    print("  status             - Zeigt den Systemstatus und Metriken an")
    print("  clearcache         - Leert den Proof-Cache")
    print("  exit               - Beendet die Anwendung")
    print("---------------------------------\n")

def main_loop():
    try:
        from backup_manager import BackupManager
        backup_mgr = BackupManager()
        backup_mgr.auto_backup("v22.2", "Final `what_is` Bugfix")
    except ImportError:
        print("ℹ️  'backup_manager.py' nicht gefunden, Backup wird übersprungen.")
    
    assistant = KAssistant()
    print_help()
    
    while True:
        try:
            user_input = input("> ").strip()
            if not user_input: continue
            parts = user_input.split(" ", 1)
            command = parts[0].lower()
            args = parts[1].strip('"\'') if len(parts) > 1 else ""
            
            if command == "exit": assistant.save_kb(assistant.kb_filepath); break
            elif command == "help": print_help()
            elif command == "build_kb" and args: assistant.build_kb_from_file(args)
            elif command == "add_raw" and args: assistant.add_raw(args)
            elif command == "retract" and args: assistant.retract(args)
            elif command == "learn" and not args: assistant.learn_facts()
            elif command == "clearcache" and not args: assistant.clear_cache()
            elif command == "ask" and args: assistant.ask(args)
            elif command == "explain" and args: assistant.explain(args)
            elif command == "ask_raw" and args: assistant.ask_raw(args)
            elif command == "status": assistant.status()
            elif command == "show": assistant.show()
            elif command == "prover": print("ℹ️ Der `prover`-Befehl ist veraltet. Die Prover-Kaskade wird automatisch ausgeführt.")
            elif command == "search" and args: assistant.search(args)
            elif command == "sources": assistant.sources()
            elif command == "what_is" and args: assistant.what_is(args)
            elif command == "shell" and args: assistant.execute_shell(args)
            elif command == "parse" and args: assistant.test_parser(args)
            else: print(f"Unbekannter Befehl: '{user_input}'.")
        except (KeyboardInterrupt, EOFError):
            if 'assistant' in locals(): assistant.save_kb(assistant.kb_filepath)
            break
        except Exception as e:
            import traceback
            print(f"\n🚨 Unerwarteter Fehler: {e}"); traceback.print_exc()

if __name__ == "__main__":
    main_loop()