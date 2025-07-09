# -*- coding: utf-8 -*-
# k_assistant_main.py - HAK/GAL Suite Backend
# ✅ CLEANED VERSION - Bindestrich-Fix implementiert
# ✅ RAG-Pipeline funktional
# ✅ Timeout-System < 45s
# ✅ Learning Suggestions aktiv

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

# Verwende externe Grammar-Datei
try:
    from .hakgal_grammar import HAKGAL_GRAMMAR
except ImportError:
    print("❌ Verwende interne Grammar-Definition...")
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
        ?term: PREDICATE | VAR
        PREDICATE: /[A-ZÄÖÜ][a-zA-ZÄÖÜäöüß0-9_-]*/
        VAR: /[a-z][a-zA-Z0-9_]*/
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

#==============================================================================
# Parser-Implementierung - MIT BINDESTRICH-FIX
#==============================================================================
class HAKGALParser:
    def __init__(self):
        self.parser_available = LARK_AVAILABLE
        if self.parser_available:
            try:
                self.parser = lark.Lark(HAKGAL_GRAMMAR, parser='lalr', debug=False)
                print("✅ Lark-Parser initialisiert (mit Bindestrich-Support)")
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
        # ✅ ERWEITERTE REGEX MIT BINDESTRICH-SUPPORT
        if not re.match(r'^[A-Za-zÄÖÜäöüß0-9\s\(\),\->&|._-]+$', formula): return (False, None, "Ungültige Zeichen")
        if formula.count('(') != formula.count(')'): return (False, None, "Unbalancierte Klammern")
        return (True, None, "Syntax OK (Regex-Fallback mit Bindestrich-Support)")

    def extract_predicates(self, tree: 'lark.Tree') -> List[str]:
        predicates = [str(node.children[0]) for node in tree.find_data('atom') if node.children and isinstance(node.children[0], lark.Token) and node.children[0].type == 'PREDICATE']
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
        response = self.model.generate_content(full_prompt, generation_config=genai.types.GenerationConfig(temperature=temperature))
        return response.text.strip()

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
        op_map = {'->': z3.Implies, '|': z3.Or, '&': z3.And}
        for op_str, op_func in op_map.items():
            idx, op = _find_top_level_operator(expr, [op_str])
            if idx != -1: return op_func(self._parse_hakgal_formula_to_z3_expr(expr[:idx], quantified_vars), self._parse_hakgal_formula_to_z3_expr(expr[idx + len(op):], quantified_vars))
        if expr.startswith('-'): return z3.Not(self._parse_hakgal_formula_to_z3_expr(expr[1:], quantified_vars))
        if expr.startswith('all '):
            match = re.match(r"all\s+([\w]+)\s+\((.*)\)$", expr, re.DOTALL)
            var_name, body = match.groups()
            z3_var = z3.Int(var_name)
            return z3.ForAll([z3_var], self._parse_hakgal_formula_to_z3_expr(body, quantified_vars | {var_name}))
        if expr.startswith('(') and expr.endswith(')'): return self._parse_hakgal_formula_to_z3_expr(expr[1:-1], quantified_vars)
        # ✅ ERWEITERTE PREDICATE-MATCHING MIT BINDESTRICH-SUPPORT
        match = re.match(r"([A-ZÄÖÜ][\w-]*)\s*\((.*?)\)", expr)
        if match:
            pred_name, args_str = match.groups()
            args = [a.strip() for a in args_str.split(',') if a.strip()]
            z3_args = [z3.Int(arg) if arg in quantified_vars else z3.Int(arg) for arg in args]
            func_sig = (pred_name, len(z3_args))
            if func_sig not in self.func_cache:
                self.func_cache[func_sig] = z3.Function(pred_name, *([z3.IntSort()] * len(z3_args)), z3.BoolSort())
            return self.func_cache[func_sig](*z3_args)
        # ✅ ERWEITERTE NULLARY PREDICATE-MATCHING MIT BINDESTRICH-SUPPORT
        if re.match(r"^[A-ZÄÖÜ][\w-]*$", expr):
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
# 3. Shell-Manager
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
# 4. MANAGER- UND KERN-KLASSEN
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
        2.  **Predicates & Constants:** `PascalCase`. Examples: `IstUrgent`, `UserManagement`, `RAG-Pipeline`.
        3.  **Variables:** `lowercase`. Example: `x`.
        4.  **Operators:** `&` (AND), `|` (OR), `->` (IMPLIES), `-` (NOT).
        5.  **Termination:** Every formula MUST end with a period `.`.
        6.  **BINDESTRICHE:** Bindestriche sind in Prädikaten erlaubt: `RAG-Pipeline`, `AI-System`.
        7.  **NO CONVERSATION:** Do not ask for clarification. Do not explain yourself. Do not add any text before or after the formula. Your response must contain ONLY the formula.
        8.  **VAGUE INPUT RULE:** If the user input is short, vague, or a single word (like "system", "test", "help"), translate it to a generic query about its properties.
            - "system" -> `Eigenschaften(System).`
            - "hakgal" -> `Eigenschaften(HAKGAL).`
            - "test" -> `IstTest().`
        9.  **IDEMPOTENCY:** If the input is already a valid formula, return it UNCHANGED. `IstKritisch(X).` -> `IstKritisch(X).`

        **Translate the following sentence into a single HAK/GAL formula and nothing else:**
        """
        self.fact_extraction_prompt = """
        You are a precise logic extractor. Your task is to extract all facts and rules from the provided text and format them as a Python list of strings. Each string must be a valid HAK/GAL first-order logic formula.

        **HAK/GAL SYNTAX RULES (MUST BE FOLLOWED EXACTLY):**
        1.  **Structure:** `Predicate(Argument).` or `all x (Rule(x)).`
        2.  **Predicates & Constants:** `PascalCase`. (e.g., `IstLegacy`, `UserManagement`, `RAG-Pipeline`)
        3.  **Variables:** Lowercase. (e.g., `x`)
        4.  **Quantifiers:** Rules with variables MUST use `all x (...)`.
        5.  **Operators:** `&` (AND), `|` (OR), `->` (IMPLIES), `-` (NOT).
        6.  **Termination:** Every formula MUST end with a period `.`.
        7.  **BINDESTRICHE:** Bindestriche sind in Prädikaten erlaubt und erwünscht.
        8.  **Output Format:** A single Python list of strings, and nothing else.

        **Example Extraction:**
        - Text: "The billing system is a legacy system. All legacy systems are critical."
        - Output: `["IstLegacySystem(BillingSystem).", "all x (IstLegacySystem(x) -> IstKritisch(x))."]`

        - Text: "The RAG-Pipeline is working perfectly."
        - Output: `["Funktioniert(RAG-Pipeline)."]`

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
        self.provers: List[BaseProver] = [PatternProver(), Z3Adapter()]
        self.parser_stats = {"total": 0, "success": 0, "failed": 0}
        self.proof_cache = ProofCache()

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
        negated_fact = f"-{new_fact}" if not new_fact.startswith('-') else new_fact[1:]
        is_contradictory, reason = self.verify_logical(negated_fact, self.K)
        if is_contradictory:
            return (False, f"Widerspruch! Neuer Fakt '{new_fact}' widerspricht KB ({reason})")
        return (True, None)

    def verify_logical(self, query_str: str, full_kb: list) -> tuple[Optional[bool], str]:
        cache_key = (tuple(sorted(full_kb)), query_str)
        if cached_result := self.proof_cache.get(query_str, cache_key):
            print("   [Cache] ✅ Treffer im Proof-Cache!")
            return cached_result[0], cached_result[1]

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.provers)) as executor:
            futures = {executor.submit(p.prove, full_kb, query_str): p for p in self.provers}
            for future in concurrent.futures.as_completed(futures):
                prover = futures[future]
                try:
                    success, reason = future.result()
                    if success is not None:
                        if success: self.proof_cache.put(query_str, cache_key, success, reason)
                        return success, reason
                except Exception as e: print(f"   [Prover] ❌ {prover.name} Fehler: {e}")
        
        return (None, "Kein Prover konnte eine definitive Antwort finden.")

    def update_parser_stats(self, success: bool):
        self.parser_stats["total"] += 1
        if success: self.parser_stats["success"] += 1
        else: self.parser_stats["failed"] += 1

#==============================================================================
# 5. K-ASSISTANT - MIT VERBESSERTER BINDESTRICH-NORMALISIERUNG
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
        prover_names = ', '.join([p.name for p in self.core.provers])
        print(f"--- Prover-Kaskade (parallel): {prover_names} ---")
        print(f"--- Parser-Modus: {'Lark' if self.parser.parser_available else 'Regex-Fallback'} ---")
    
    def _normalize_and_correct_syntax(self, formula: str) -> str:
        """✅ ERWEITERTE BINDESTRICH-NORMALISIERUNG"""
        original_formula = formula
        corrected = formula
        
        # ✅ PHASE 1: BINDESTRICH-NORMALISIERUNG
        # Für häufige bindestrich-haltige Begriffe die korrekte Schreibweise sicherstellen
        bindestrich_map = {
            'RAG-Pipeline': 'RAGPipeline',  # Bindestrich entfernen
            'AI-System': 'AISystem',
            'Machine-Learning': 'MachineLearning',
            'Real-Time': 'RealTime',
            'E-Mail': 'EMail',
            'Multi-Agent': 'MultiAgent',
            'Deep-Learning': 'DeepLearning',
            'Neural-Network': 'NeuralNetwork',
            'HAK-GAL': 'HAKGAL',
            'Self-Healing': 'SelfHealing',
            'Auto-Scaling': 'AutoScaling',
            'Load-Balancer': 'LoadBalancer',
            'Event-Driven': 'EventDriven',
            'Micro-Service': 'MicroService',
        }
        
        # Ersetze bekannte Bindestrich-Begriffe
        for old_term, new_term in bindestrich_map.items():
            corrected = corrected.replace(old_term, new_term)
        
        # ✅ PHASE 2: ALLGEMEINE BINDESTRICH-BEHANDLUNG
        # Generelle Bindestrich-Entfernung in PascalCase Prädikaten (außer wenn explizit gewünscht)
        # Hier belassen wir Bindestriche für den Parser, da die Grammar sie unterstützt
        
        # ✅ PHASE 3: SYNONYM-MAPPING
        synonym_map = {
            r'IstTechnischesLegacy(System)?': 'IstLegacy',
            r'SollteRefactoringInBetrachtGezo[h|e]genWerden': 'SollteRefactoredWerden',
            r'SollteIdentifiziertUndRefactoredWerden': 'SollteRefactoredWerden',
            r'IstBasierendAufCobolMainframe': 'IstCobolMainframe',
            r'BasiertAufCobolMainframe': 'IstCobolMainframe',
            r'BasiertAufModernerJavaMicroservice': 'IstJavaMicroservice',
            r'IstBasierendAufJavaMicroservice': 'IstJavaMicroservice',
            r'LiegenAktuelleBetriebskostenDatenVor': 'HatHoheBetriebskosten',
            r'HatGeringeBetriebskosten': 'HatNiedrigeBetriebskosten',
            r'VerbindetMit': 'KombiniertMit',
            r'SpeichertWissen': 'SpeichertFaktenInWissensbasis',
            r'HatDatei': 'EnthältDatei',
            r'HatOrdner': 'EnthältOrdner',
            r'FrontendDateienKopiert': 'HatFrontendDateienErfolgreichKopiert',
            r'KonfigurationsdateienErstellt': 'AlleKonfigurationsdateienErstellt',
            r'IstBereitFürPhase': 'BereitFürPhase',
            r'Sollte([A-ZÄÖÜ]\w*)': r'Soll\1',
        }
        
        for pattern, canonical in synonym_map.items():
            corrected = re.sub(pattern, canonical, corrected)
            
        # ✅ PHASE 4: ALLGEMEINE KORREKTUREN
        corrected = corrected.strip().replace(':-', '->').replace('~', '-')
        while corrected.startswith('--'): corrected = corrected[2:]
        if re.match(r"^[A-Z][a-zA-Z0-9_-]*\.$", corrected): corrected = corrected.replace('.', '().')
        
        if corrected != original_formula: 
            print(f"   [Bindestrich-Fix] '{original_formula.strip()}' -> '{corrected.strip()}'")
        return corrected

    def _add_system_facts(self):
        system_facts = self.shell_manager.analyze_system_facts()
        for fact in system_facts:
            if fact not in self.core.K: self.core.K.append(fact)
        print(f"   ✅ {len(system_facts)} Systemfakten hinzugefügt.")
    
    def _ask_or_explain(self, q: str, explain: bool, is_raw: bool):
        print(f"\n> {'Erklärung für' if explain else 'Frage'}{' (roh)' if is_raw else ''}: '{q}'")
        self.potential_new_facts = []
        temp_assumptions = []
        logical_form = q
        
        if not is_raw:
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
            
            if not logical_form_raw or not isinstance(logical_form_raw, str):
                print(f"   ❌ FEHLER: LLM konnte keine Formel erzeugen. Antwort war: '{logical_form_raw}'")
                return

            if isinstance(logical_form_raw, dict): 
                print(f"   [Orakel] Klärungsbedarf: {logical_form_raw}"); return

            logical_form = self._normalize_and_correct_syntax(logical_form_raw)
        
        print(f"   -> Logische Form: '{logical_form}'")
        is_valid, _, msg = self.parser.parse(logical_form)
        self.core.update_parser_stats(is_valid)
        if not is_valid: 
            print(f"   ❌ FEHLER: Ungültige Syntax. {msg}")
            return
        
        print(f"🛡️  Übergebe an Prover-Kaskade...")
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
        is_valid, _, msg = self.parser.parse(formula)
        self.core.update_parser_stats(is_valid)
        if not is_valid: print(f"   ❌ FEHLER: Ungültige Syntax. {msg}"); return
        is_consistent, reason = self.core.check_consistency(formula)
        if not is_consistent: print(f"   🛡️  WARNUNG: {reason}"); return
        if self.core.add_fact(formula): print("   -> Erfolgreich hinzugefügt.")
        else: print("   -> Fakt bereits vorhanden.")
        
    def retract(self, formula_to_retract: str):
        print(f"\n> Entferne KERNREGEL: '{formula_to_retract}'")
        normalized_target = self._normalize_and_correct_syntax(formula_to_retract)
        fact_to_remove = next((f for f in self.core.K if self._normalize_and_correct_syntax(f) == normalized_target), None)
        if fact_to_remove and self.core.retract_fact(fact_to_remove): print(f"   -> '{fact_to_remove}' entfernt.")
        else: print(f"   -> Fakt nicht gefunden.")

    def learn_facts(self):
        if not self.potential_new_facts: print("🧠 Nichts Neues zu lernen."); return
        print(f"🧠 Lerne {len(self.potential_new_facts)} neue Fakten...")
        added_count = sum(1 for fact in self.potential_new_facts if self.core.check_consistency(fact)[0] and self.core.add_fact(fact))
        if added_count > 0: print(f"✅ {added_count} neue Fakten gespeichert.")
        else: print("ℹ️ Alle Fakten waren bereits bekannt oder inkonsistent.")
        self.potential_new_facts = []

    def clear_cache(self): self.core.proof_cache.clear(); self.ensemble_manager.prompt_cache.clear()

    def status(self): 
        print(f"\n--- System Status ---")
        pc, pmc = self.core.proof_cache, self.ensemble_manager.prompt_cache
        stats = self.core.parser_stats
        success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  Prover: {', '.join([p.name for p in self.core.provers])} (parallel)")
        print(f"  Parser: {'Lark' if self.parser.parser_available else 'Regex'} | Versuche: {stats['total']} | Erfolg: {success_rate:.1f}%")
        print(f"  Wissen: {len(self.core.K)} Kernfakten | {len(self.potential_new_facts)} lernbare Fakten")
        print(f"  Caches: Beweise={pc.size} (Rate {pc.hit_rate:.1f}%) | Prompts={pmc.size} (Rate {pmc.hit_rate:.1f}%)")
        if RAG_ENABLED: print(f"  RAG: {self.wissensbasis_manager.index.ntotal} Chunks aus {len(self.wissensbasis_manager.doc_paths)} Docs")
        
    def show(self) -> Dict[str, Any]:
        permanent_knowledge = sorted(self.core.K)
        learnable_facts = sorted(self.potential_new_facts)
        
        rag_chunks_summary = []
        if RAG_ENABLED and self.wissensbasis_manager.chunks:
            for i, chunk_info in enumerate(self.wissensbasis_manager.chunks):
                rag_chunks_summary.append({
                    "id": i,
                    "source": chunk_info.get('source', 'Unbekannt'),
                    "text_preview": chunk_info.get('text', '')[:80] + "..."
                })

        return {
            "permanent_knowledge": permanent_knowledge,
            "learnable_facts": learnable_facts,
            "rag_chunks": rag_chunks_summary,
            "rag_stats": {
                "doc_count": len(self.wissensbasis_manager.doc_paths) if RAG_ENABLED else 0,
                "chunk_count": self.wissensbasis_manager.index.ntotal if RAG_ENABLED and self.wissensbasis_manager.index else 0,
            }
        }
        
    def save_kb(self, filepath: str):
        try:
            rag_data = {'chunks': self.wissensbasis_manager.chunks, 'doc_paths': self.wissensbasis_manager.doc_paths} if RAG_ENABLED else {}
            data = {'facts': self.core.K, 'rag_data': rag_data, 'parser_stats': self.core.parser_stats, 'proof_cache': self.core.proof_cache.cache}
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
        print(f"\n> Analysiere Wissen über Entität: '{entity}'")
        explicit_facts = [fact for fact in self.core.K if f"({entity})" in fact or f",{entity})" in fact or f"({entity}," in fact]
        unary_predicates_to_test = ["IstLegacy", "IstKritisch", "IstOnline", "IstStabil", "HatHoheBetriebskosten", "SollteRefactoredWerden"]
        derived_properties = []
        print("🧠 Leite Eigenschaften ab...")
        for pred in unary_predicates_to_test:
            positive_goal = f"{pred}({entity})."
            if self.core.verify_logical(positive_goal, self.core.K)[0] and positive_goal not in explicit_facts:
                derived_properties.append(positive_goal)
                continue
            negative_goal = f"-{pred}({entity})."
            if self.core.verify_logical(negative_goal, self.core.K)[0] and negative_goal not in explicit_facts:
                derived_properties.append(negative_goal)

        print("\n" + f"--- Profil für: {entity} ---".center(60, "-"))
        print("\n  [Explizite Fakten]")
        if explicit_facts: [print(f"   - {f}") for f in sorted(explicit_facts)]
        else: print("   (Keine)")
        print("\n  [Abgeleitete Eigenschaften]")
        if derived_properties: [print(f"   - {p}") for p in sorted(derived_properties)]
        else: print("   (Keine)")
        print("-" * 60)

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
        success, tree, msg = self.parser.parse(formula)
        self.core.update_parser_stats(success)
        if success:
            print(f"✅ Parse erfolgreich: {msg}")
            if tree: print(f"   Gefundene Prädikate: {', '.join(self.parser.extract_predicates(tree))}")
        else: print(f"❌ Parse fehlgeschlagen: {msg}")

#==============================================================================
# 6. MAIN LOOP
#==============================================================================
def print_help():
    print("\n" + " HAK/GAL Suite - K-Assistant ".center(60, "-"))
    print("  build_kb <pfad>  - Indiziert Dokument für RAG")
    print("  add_raw <formel> - Fügt KERNREGEL hinzu")
    print("  retract <formel> - Entfernt KERNREGEL")
    print("  learn            - Speichert gefundene Fakten")
    print("  show             - Zeigt Wissensbasis an")
    print("  sources          - Zeigt Wissensquellen an")
    print("  search <anfrage> - Findet Text in der KB (RAG)")
    print("  ask <frage>      - Beantwortet Frage (mit RAG)")
    print("  explain <frage>  - Erklärt eine Antwort")
    print("  ask_raw <formel> - Stellt rohe logische Frage")
    print("  what_is <entity> - Zeigt Profil einer Entität an")
    print("  status           - Zeigt Systemstatus und Metriken")
    print("  shell <befehl>   - Führt Shell-Befehl aus")
    print("  parse <formel>   - Testet Parser mit Formel")
    print("  clearcache       - Leert alle Caches")
    print("  exit             - Beendet und speichert die KB")
    print("-" * 60)
    print("  ✅ BINDESTRICH-SUPPORT: RAG-Pipeline, AI-System, etc.")
    print("-" * 60 + "\n")

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
            "add_raw": lambda a, args: a.add_raw(args), "retract": lambda a, args: a.retract(args),
            "learn": lambda a, args: a.learn_facts(), "clearcache": lambda a, args: a.clear_cache(),
            "ask": lambda a, args: a.ask(args), "explain": lambda a, args: a.explain(args),
            "ask_raw": lambda a, args: a.ask_raw(args), "status": lambda a, args: a.status(),
            "show": lambda a, args: show_in_console(a),
            "search": lambda a, args: a.search(args),
            "sources": lambda a, args: a.sources(), "what_is": lambda a, args: a.what_is(args),
            "shell": lambda a, args: a.execute_shell(args), "parse": lambda a, args: a.test_parser(args),
        }
        while True:
            try:
                user_input = input("k-assistant> ").strip()
                if not user_input: continue
                parts = user_input.split(" ", 1)
                command, args = parts[0].lower(), parts[1].strip('"\'') if len(parts) > 1 else ""
                
                if command in command_map:
                    if command in ["exit", "help", "learn", "clearcache", "status", "show", "sources"] and args:
                        print(f"Befehl '{command}' erwartet keine Argumente.")
                    elif command not in ["exit", "help", "learn", "clearcache", "status", "show", "sources"] and not args:
                         print(f"Befehl '{command}' benötigt ein Argument.")
                    else:
                        command_map[command](assistant, args)
                        if command == "exit": break
                else: print(f"Unbekannter Befehl: '{command}'.")
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
