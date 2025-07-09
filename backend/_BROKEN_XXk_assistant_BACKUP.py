# -*- coding: utf-8 -*-
# k_assistant.py - Merged and Corrected Version
# Combines features from v41 and v22.2, fixing critical bugs.

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
    # Verwende pypdf, da es die neuere, empfohlene Bibliothek ist (aus v22.2)
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

# ==============================================================================
# GRAMMAR DEFINITION (aus v41)
# ==============================================================================

HAKGAL_GRAMMAR = r"""
    ?start: formula

    formula: expression "."

    ?expression: quantified_formula
               | implication

    ?implication: disjunction ( "->" implication )?

    ?disjunction: conjunction ( "|" disjunction )?

    ?conjunction: negation ( "&" conjunction )?

    ?negation: "-" atom_expression
             | atom_expression

    ?atom_expression: atom
                    | "(" expression ")"

    quantified_formula: "all" VAR "(" expression ")"

    atom: PREDICATE ("(" [arg_list] ")")?

    arg_list: term ("," term)*

    ?term: PREDICATE | VAR

    PREDICATE: /[A-ZÄÖÜ][a-zA-ZÄÖÜäöüß0-9_-]*/
    VAR: /[a-z][a-zA-Z0-9_]*/

    %import common.WS
    %ignore WS
"""

#==============================================================================
# KORREKTUR: Implementierung der fehlenden Cache-Klassen
# Diese Klassen wurden in v41 verwendet, aber nie definiert, was den NameError verursachte.
#==============================================================================
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
    def size(self) -> int:
        return len(self.cache)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

class ProofCache(BaseCache):
    def get(self, query_str: str, key: Tuple) -> Optional[Tuple[bool, str, float]]:
        return super().get(key)

    def put(self, query_str: str, key: Tuple, success: bool, reason: str):
        value = (success, reason, time.time())
        super().put(key, value)

class PromptCache(BaseCache):
    def get(self, prompt: str) -> Optional[str]:
        return super().get(prompt)

    def put(self, prompt: str, response: str):
        super().put(prompt, response)


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
# Parser-Implementierung (aus v41)
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

    def prove(self, assumptions: list, goal: str) -> tuple[Optional[bool], str]:
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
            # Deklariere die Funktion einmalig, um Konsistenz zu gewährleisten
            arg_sorts = [z3.IntSort()] * len(z3_args)
            z3_func = z3.Function(pred_name, *arg_sorts, z3.BoolSort())
            return z3_func(*z3_args)
        raise ValueError(f"Konnte Formelteil nicht parsen: '{expr}'")

    def validate_syntax(self, formula: str) -> tuple[bool, str]:
        success, _, msg = self.parser.parse(formula)
        if not success:
            return (False, f"Parser: {msg}")

        try:
            self._parse_hakgal_formula_to_z3_expr(formula)
            return (True, f"✅ Syntax OK (Lark + Z3)")
        except (ValueError, z3.Z3Exception) as e:
            return (False, f"Z3-Konvertierung fehlgeschlagen: {str(e)}")

    def prove(self, assumptions: list, goal: str) -> tuple[Optional[bool], str]:
        solver = z3.Tactic('smt').solver(); solver.set(model=True)
        try:
            for fact_str in assumptions: solver.add(self._parse_hakgal_formula_to_z3_expr(fact_str))
            goal_expr = self._parse_hakgal_formula_to_z3_expr(goal); solver.add(z3.Not(goal_expr))
        except (ValueError, z3.Z3Exception) as e: return (None, f"Fehler beim Parsen: {e}")
        check_result = solver.check()
        if check_result == z3.unsat: return (True, "Z3 hat das Ziel bewiesen (Problem ist unerfüllbar).")
        elif check_result == z3.sat: return (False, f"Z3 konnte das Ziel nicht beweisen (Gegenmodell existiert):\n{solver.model()}")
        else: return (None, f"Z3 konnte das Ziel nicht beweisen (unbekannt, Grund: {solver.reason_unknown()}).")

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
                git_bash_path = r"C:\Program Files\Git\bin\bash.exe" # Vereinfacht
                proc = subprocess.run([git_bash_path, "-c", command], capture_output=True, text=True, timeout=timeout)
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
        self.providers: List[BaseLLMProvider] = []
        self._initialize_providers()
        self.prompt_cache = PromptCache()
        self.system_prompt_logicalize = """
        You are a precise logic translator. Your task is to translate the user's question into a single,
        first-order logic formula that strictly adheres to the HAK/GAL syntax.

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
        - User: "what is HAK-GAL?" -> `Eigenschaften(HAK_GAL_Suite).`

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
        if api_key := os.getenv("DEEPSEEK_API_KEY"):
            self.providers.append(DeepSeekProvider(api_key=api_key))
            print("   ✅ DeepSeek Provider geladen.")
        if api_key := os.getenv("MISTRAL_API_KEY"):
            self.providers.append(MistralProvider(api_key=api_key))
            print("   ✅ Mistral Provider geladen.")
        if GEMINI_AVAILABLE and (api_key := os.getenv("GEMINI_API_KEY")):
            try:
                self.providers.append(GeminiProvider(api_key=api_key))
                print("   ✅ Gemini Provider geladen.")
            except Exception as e:
                print(f"   ❌ Fehler beim Laden des Gemini Providers: {e}")
        if not self.providers:
            print("   ⚠️ Keine LLM-Provider aktiv. LLM-Funktionen sind deaktiviert.")

    def logicalize(self, sentence: str) -> str | dict | None:
        if not self.providers: return None
        full_prompt = f"{self.system_prompt_logicalize}\n\n{sentence}"
        
        cached_response = self.prompt_cache.get(full_prompt)
        if cached_response:
            print("   [Cache] ✅ Treffer im Prompt-Cache!")
            try: return json.loads(cached_response)
            except json.JSONDecodeError: return cached_response

        print("   [Cache] ❌ Kein Treffer im Prompt-Cache, frage LLM an...")
        try:
            response_text = self.providers[0].query(sentence, self.system_prompt_logicalize, 0)
            self.prompt_cache.put(full_prompt, response_text)
            try: return json.loads(response_text)
            except json.JSONDecodeError: return response_text
        except Exception as e:
            print(f"   [Warnung] Fehler bei Logik-Übersetzung: {e}")
            return None

    # Übernommene "Mistral Veto"-Logik aus v22.2, da sie robuster ist
    def extract_facts_with_ensemble(self, context: str) -> list[str]:
        if not self.providers: return []
        threads = []
        results: List[Optional[List[str]]] = [None] * len(self.providers)
        def worker(provider: BaseLLMProvider, index: int):
            try:
                prompt = self.fact_extraction_prompt.format(context=context)
                raw_output = provider.query(prompt, "", 0.1)
                match = re.search(r'\[.*\]', raw_output, re.DOTALL)
                if match:
                    try:
                        fact_list = eval(match.group(0))
                        if isinstance(fact_list, list): results[index] = list(dict.fromkeys(fact_list))
                    except: pass
            except Exception as e: print(f"   [Warnung] Fehler bei {provider.model_name}: {e}")
        
        for i, provider in enumerate(self.providers):
            thread = threading.Thread(target=worker, args=(provider, i)); threads.append(thread); thread.start()
        for thread in threads: thread.join()
        
        mistral_result = None
        other_results = []
        for i, provider in enumerate(self.providers):
            if isinstance(provider, MistralProvider) and results[i]:
                mistral_result = results[i]
            elif results[i]:
                other_results.append(results[i])

        if mistral_result:
            print("   [Veto-Ensemble] ✅ Entscheidung von Mistral (Chef) wird verwendet.")
            return mistral_result

        if not other_results:
            return []
            
        print("   [Veto-Ensemble] ⚠️ Mistral hat keine Fakten geliefert. Fallback auf die restlichen Provider...")
        fallback_facts = [fact for res in other_results for fact in res]
        if not fallback_facts:
            return []
            
        fact_counts = Counter(fallback_facts)
        majority_threshold = len(other_results) // 2 + 1 if len(other_results) > 1 else 1
        consistent_facts = [fact for fact, count in fact_counts.items() if count >= majority_threshold]
        
        if consistent_facts:
            print(f"   [Veto-Ensemble] Konsens unter Fallback-Providern für {len(consistent_facts)} Fakten gefunden.")
        return consistent_facts

class WissensbasisManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if not RAG_ENABLED:
            self.model, self.index, self.chunks, self.doc_paths = None, None, [], {}
            print("   ℹ️  RAG-Funktionen: Bibliotheken nicht gefunden.")
            return
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks: List[Dict[str, str]] = [] # Verwende Diktionär für mehr Klarheit (aus v41)
        self.doc_paths: Dict[str, str] = {}
        
    def add_document(self, file_path: str):
        if not RAG_ENABLED: return
        doc_id = os.path.basename(file_path)
        if doc_id in self.doc_paths:
            print(f"   ℹ️ Dokument '{file_path}' wurde bereits indiziert.")
            return
        try:
            # Verbessertes PDF-Lesen mit pypdf (aus v22.2)
            if file_path.endswith(".pdf"):
                reader = PdfReader(file_path)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
        except Exception as e:
            print(f"   ❌ Fehler beim Lesen der Datei '{file_path}': {e}")
            return
        
        text_chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk.strip()) > 30]
        if not text_chunks:
            print(f"   ℹ️ Keine sinnvollen Chunks in '{file_path}' gefunden.")
            return
        
        embeddings = self.model.encode(text_chunks, convert_to_tensor=False, show_progress_bar=True)
        self.index.add(np.array(embeddings).astype('float32'))
        
        for chunk in text_chunks:
            self.chunks.append({'text': chunk, 'source': doc_id})
        
        self.doc_paths[doc_id] = file_path
        print(f"   ✅ {len(text_chunks)} Chunks aus '{doc_id}' erfolgreich indiziert.")
        
    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        if not RAG_ENABLED or self.index.ntotal == 0: return []
        query_embedding = self.model.encode([query])
        _, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        return [self.chunks[i] for i in indices[0] if i != -1 and i < len(self.chunks)]

class HAKGAL_Core_FOL:
    def __init__(self):
        self.K: List[str] = []
        self.parser = HAKGALParser()
        self.provers: List[BaseProver] = [PatternProver(), Z3Adapter()]
        self.parser_stats = {"total": 0, "success": 0, "failed": 0}
        self.proof_cache = ProofCache()

    def add_fact(self, formula_str: str):
        if formula_str not in self.K:
            self.K.append(formula_str)
            self.proof_cache.clear() # Cache leeren, da sich die Wissensbasis geändert hat
            return True
        return False

    def retract_fact(self, fact_to_remove: str) -> bool:
        if fact_to_remove in self.K:
            self.K.remove(fact_to_remove)
            self.proof_cache.clear()
            return True
        return False

    def check_consistency(self, new_fact: str) -> Tuple[bool, Optional[str]]:
        if new_fact.startswith('-'):
            negated_fact = new_fact[1:]
        else:
            negated_fact = f"-{new_fact}"

        is_contradictory, reason = self.verify_logical(negated_fact, self.K)
        if is_contradictory:
            return (False, f"Widerspruch gefunden! Der neue Fakt '{new_fact}' widerspricht der Wissensbasis, denn '{reason}'")
        
        return (True, None)

    # Beibehaltung der parallelen Ausführung aus v41, da dies performanter ist
    def verify_logical(self, query_str: str, full_kb: list) -> tuple[Optional[bool], str]:
        cache_key = (tuple(sorted(full_kb)), query_str)
        
        cached_result = self.proof_cache.get(query_str, cache_key)
        if cached_result:
            print("   [Cache] ✅ Treffer im Proof-Cache!")
            success, reason, _ = cached_result
            return success, reason

        print("   [Cache] ❌ Kein Treffer im Proof-Cache, starte Prover...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.provers)) as executor:
            futures = {executor.submit(p.prove, full_kb, query_str): p for p in self.provers}
            
            for future in concurrent.futures.as_completed(futures):
                prover = futures[future]
                try:
                    success, reason = future.result()
                    # Wenn ein Prover ein definitives Ergebnis (True/False) liefert, nehmen wir es sofort
                    if success is not None:
                        # Nur erfolgreiche Beweise cachen
                        if success:
                            self.proof_cache.put(query_str, cache_key, success, reason)
                        return success, reason
                except Exception as e:
                    print(f"   [Prover] ❌ {prover.name} hat einen Fehler erzeugt: {e}")
        
        return (None, "Keiner der verfügbaren Prover konnte eine definitive Antwort finden.")

    def update_parser_stats(self, success: bool):
        self.parser_stats["total"] += 1
        if success: self.parser_stats["success"] += 1
        else: self.parser_stats["failed"] += 1

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
        self.potential_new_facts: List[str] = []
        self.load_kb(kb_filepath)
        self._add_system_facts()
        prover_names = ' -> '.join([p.name for p in self.core.provers])
        print(f"--- Prover-Kaskade (parallel): {prover_names} ---")
        print(f"--- Parser-Modus: {'Lark' if self.parser.parser_available else 'Regex-Fallback'} ---")
    
    # Kombinierte und erweiterte Normalisierungsregeln aus beiden Versionen
    def _normalize_and_correct_syntax(self, formula: str) -> str:
        original_formula = formula
        corrected = formula
        
        # Spezifische Ersetzungen
        corrected = re.sub(r'\bHAK-GAL\b', 'HAK_GAL_Suite', corrected, flags=re.IGNORECASE)
        corrected = re.sub(r'\b([A-Z][a-zA-Z]*)-([A-Z][a-zA-Z]*)\b', r'\1\2', corrected) # Bindestriche
        
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

        # Allgemeine Korrekturen
        corrected = corrected.strip()
        corrected = re.sub(r"^[^\w-]+", "", corrected) # Führende Sonderzeichen entfernen
        corrected = corrected.replace(':-', '->')
        corrected = corrected.replace('~', '-')
        while corrected.startswith('--'): corrected = corrected[2:]
        if re.match(r"^[A-Z][a-zA-Z0-9_]*\.$", corrected): corrected = corrected.replace('.', '().')
        corrected = re.sub(r'([A-Za-z0-9_]+)_py', r'\1', corrected)
        
        if corrected != original_formula:
            print(f"   [Syntax-Korrektur] '{original_formula.strip()}' -> '{corrected.strip()}'")
        
        return corrected

    def _add_system_facts(self):
        system_facts = self.shell_manager.analyze_system_facts()
        for fact in system_facts:
            if fact not in self.core.K:
                self.core.K.append(fact)
        print(f"   ✅ {len(system_facts)} Systemfakten automatisch hinzugefügt.")
    
    def _ask_or_explain(self, q: str, explain: bool, is_raw: bool):
        print(f"\n> {'Erklärung für' if explain else 'Frage'}{' (roh)' if is_raw else ''}: '{q}'")
        self.potential_new_facts = []
        temp_assumptions = []
        
        if is_raw:
            logical_form = q
        else:
            # RAG-Anreicherung
            if RAG_ENABLED and self.wissensbasis_manager.index.ntotal > 0:
                print("🧠 RAG-Pipeline wird für Kontext angereichert...")
                relevant_chunks_info = self.wissensbasis_manager.retrieve_relevant_chunks(q)
                if relevant_chunks_info:
                    context = "\n\n".join([chunk['text'] for chunk in relevant_chunks_info])
                    print(f"   [RAG] Relevanter Kontext gefunden.")
                    print("   [RAG] Extrahiere Fakten mit dem Ensemble...")
                    extracted_facts = self.ensemble_manager.extract_facts_with_ensemble(context)
                    if extracted_facts:
                        print("   [RAG] Fakten für den Beweis:")
                        for fact in extracted_facts:
                            corrected_fact = self._normalize_and_correct_syntax(fact)
                            is_valid, _, _ = self.parser.parse(corrected_fact)
                            if is_valid:
                                temp_assumptions.append(corrected_fact)
                                if corrected_fact not in self.core.K and corrected_fact not in self.potential_new_facts:
                                    self.potential_new_facts.append(corrected_fact)
                        print(f"   [RAG] {len(temp_assumptions)} temporäre Fakten hinzugefügt.")
                    else: print("   [RAG] Keine Fakten gefunden.")
                else: print("   [RAG] Keine relevanten Chunks für die Anfrage gefunden.")

            # Übersetzung und Validierung
            print("🔮 Übersetze Anfrage in Logik...")
            logical_form_raw = self.ensemble_manager.logicalize(q)
            if not logical_form_raw: 
                print("   ❌ Orakel konnte keine Formel erzeugen.")
                return
            
            # Hier könnte eine Erweiterung für Klärungsfragen stehen, falls `logicalize` ein dict zurückgibt
            if isinstance(logical_form_raw, dict):
                print(f"   [Orakel] Klärungsbedarf: {logical_form_raw}")
                return

            logical_form = self._normalize_and_correct_syntax(logical_form_raw)
            print(f"   -> '{logical_form}'")
            is_valid, _, msg = self.parser.parse(logical_form)
            self.core.update_parser_stats(is_valid)
            if not is_valid:
                print(f"   ❌ FEHLER: LLM hat ungültige Syntax generiert. {msg}")
                return
        
        print(f"🛡️  Übergebe an Prover-Kaskade...")
        full_kb = self.core.K + temp_assumptions
        success, reason = self.core.verify_logical(logical_form, full_kb)
        
        print("\n--- ERGEBNIS ---")
        if not explain:
            print("✅ Antwort: Ja." if success else "❔ Antwort: Nein/Unbekannt.")
            print(f"   [Begründung] {reason}")
        else:
            success_status_text = "Ja (bewiesen)" if success else "Nein (nicht bewiesen)"
            if not self.ensemble_manager.providers: 
                print("   ❌ Konnte keine Erklärung generieren (keine LLM-Provider).")
                return
            
            print("🗣️  Generiere einfache Erklärung...")
            explanation_prompt = f"**Anfrage:** '{q}'\n**Ergebnis:** {success_status_text}\n**Grund:** '{reason}'\n\n**Aufgabe:** Übersetze dies in eine einfache Erklärung."
            explanation = self.ensemble_manager.providers[0].query(explanation_prompt, "Du bist ein Logik-Experte, der formale Beweisergebnisse in einfache Sprache übersetzt.", 0.2)
            print(f"\n--- Erklärung ---\n{explanation}\n-------------------\n")

        if self.potential_new_facts:
            print(f"💡 INFO: {len(self.potential_new_facts)} neue Fakten wurden gefunden. Benutze 'learn', um sie permanent zu speichern.")

    def add_raw(self, formula: str):
        print(f"\n> Füge KERNREGEL hinzu: '{formula}'")
        is_valid, _, msg = self.parser.parse(formula)
        self.core.update_parser_stats(is_valid)
        if not is_valid:
            print(f"   ❌ FEHLER: Ungültige Syntax. {msg}")
            return
        
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
            if self._normalize_and_correct_syntax(fact_in_kb) == normalized_target:
                fact_to_remove = fact_in_kb
                break
        
        if fact_to_remove and self.core.retract_fact(fact_to_remove):
            print(f"   -> Fakt '{fact_to_remove}' erfolgreich entfernt.")
        else:
            print(f"   -> Fakt, der zu '{normalized_target}' normalisiert, wurde nicht gefunden oder konnte nicht entfernt werden.")

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
                print(f"   -> Fakt '{fact}' wird NICHT gelernt.")
                continue

            if self.core.add_fact(fact):
                print(f"   + {fact}")
                added_count += 1
        
        if added_count > 0: print(f"✅ {added_count} neue Fakten wurden permanent gespeichert.")
        else: print("ℹ️ Alle vorgeschlagenen Fakten waren bereits bekannt oder inkonsistent.")
        self.potential_new_facts = []

    def clear_cache(self):
        self.core.proof_cache.clear()
        self.ensemble_manager.prompt_cache.clear()

    def status(self): 
        print(f"\n--- System Status (Merged Version) ---")
        prover_names = ' -> '.join([p.name for p in self.core.provers])
        print(f"  Prover-Kaskade: {prover_names} (parallel ausgeführt)")
        print(f"  Parser-Modus: {'Lark' if self.parser.parser_available else 'Regex-Fallback'}")
        print(f"  Permanente Fakten: {len(self.core.K)}")
        pc = self.core.proof_cache
        pmc = self.ensemble_manager.prompt_cache
        print(f"  Gecachte Beweise: {pc.size} (Hits: {pc.hits}, Misses: {pc.misses}, Rate: {pc.hit_rate:.1f}%)")
        print(f"  Gecachte Prompts: {pmc.size} (Hits: {pmc.hits}, Misses: {pmc.misses}, Rate: {pmc.hit_rate:.1f}%)")
        if self.potential_new_facts: print(f"  Vorgeschlagene Fakten zum Lernen: {len(self.potential_new_facts)}")
        
        print("\n  --- Parser Statistiken ---")
        stats = self.core.parser_stats
        if stats["total"] > 0:
            success_rate = (stats["success"] / stats["total"]) * 100
            print(f"  Parse-Versuche: {stats['total']}, Erfolgreich: {stats['success']} ({success_rate:.1f}%), Fehlgeschlagen: {stats['failed']}")
        else: print("  (Noch keine Parse-Versuche)")
        
        print("\n  --- System & RAG & LLMs ---")
        print(f"  Shell: {self.shell_manager.shell} auf {self.shell_manager.system}")
        if RAG_ENABLED: print(f"  RAG: {len(self.wissensbasis_manager.doc_paths)} Dokumente, {self.wissensbasis_manager.index.ntotal} Chunks")
        else: print("  RAG: Deaktiviert")
        print(f"  LLMs: {len(self.ensemble_manager.providers)} Provider verfügbar ({', '.join([p.model_name for p in self.ensemble_manager.providers])})")
        
    def show(self):
        print("\n--- Permanente Wissensbasis (Kernregeln) ---")
        if not self.core.K: print("   (Leer)")
        else:
            for i, fact in enumerate(self.core.K): print(f"   [{i}] {fact}")
        
        if self.potential_new_facts:
            print("\n--- Vorgeschlagene Fakten zum Lernen (mit 'learn' übernehmen) ---")
            for i, fact in enumerate(self.potential_new_facts): print(f"   [{i}] {fact}")

        print("\n--- Indizierte Wissens-Chunks ---")
        if RAG_ENABLED and self.wissensbasis_manager.chunks:
            for i, chunk_info in enumerate(self.wissensbasis_manager.chunks): 
                print(f"   [{i} from {chunk_info['source']}] {chunk_info['text'][:80]}...")
        else: print("   (Leer oder RAG deaktiviert)")
        
    def save_kb(self, filepath: str):
        try:
            rag_data = {'chunks': self.wissensbasis_manager.chunks, 'doc_paths': self.wissensbasis_manager.doc_paths} if RAG_ENABLED else {}
            data = {
                'facts': self.core.K, 
                'rag_data': rag_data,
                'parser_stats': self.core.parser_stats,
                'proof_cache': self.core.proof_cache.cache
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
            self.core.proof_cache.cache = data.get('proof_cache', {})
            print(f"✅ {len(self.core.K)} Kernregeln und {len(self.core.proof_cache.cache)} gecachte Beweise geladen.")
            if RAG_ENABLED and 'rag_data' in data and data['rag_data']:
                rag_chunks_with_meta = data['rag_data'].get('chunks', [])
                self.wissensbasis_manager.chunks = rag_chunks_with_meta
                self.wissensbasis_manager.doc_paths = data['rag_data'].get('doc_paths', {})
                if rag_chunks_with_meta:
                    just_chunks = [c['text'] for c in rag_chunks_with_meta]
                    embeddings = self.wissensbasis_manager.model.encode(just_chunks, convert_to_tensor=False, show_progress_bar=False)
                    self.wissensbasis_manager.index.add(np.array(embeddings).astype('float32'))
                    print(f"✅ {len(rag_chunks_with_meta)} RAG-Chunks aus Speicher geladen und indiziert.")
        except Exception as e: print(f"❌ Fehler beim Laden der KB: {e}")
        
    def what_is(self, entity: str):
        print(f"\n> Analysiere Wissen über Entität: '{entity}'")
        
        # Direkte, in der KB gespeicherte Fakten
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
                if positive_goal not in explicit_facts: # Nur anzeigen, wenn nicht schon explizit da
                    derived_properties.append(positive_goal)
                continue # Wenn positiv bewiesen, nicht mehr nach negativ suchen

            negative_goal = f"-{pred}({entity})."
            is_negative, _ = self.core.verify_logical(negative_goal, self.core.K)
            if is_negative:
                if negative_goal not in explicit_facts:
                    derived_properties.append(negative_goal)

        print("\n" + f"--- Profil für Entität: {entity} ---".center(60, "-"))
        if explicit_facts:
            print("\n  [Explizite Fakten]")
            for fact in sorted(explicit_facts): print(f"   - {fact}")
        else:
            print("\n  [Explizite Fakten]\n   (Keine)")

        if derived_properties:
            print("\n  [Abgeleitete Eigenschaften]")
            for prop in sorted(derived_properties): print(f"   - {prop}")
        else:
            print("\n  [Abgeleitete Eigenschaften]\n   (Keine)")
        print("-" * 60)

    # Beibehaltung aller Befehle
    def test_parser(self, formula: str): self.parser.parse(formula)
    def execute_shell(self, command: str): self.shell_manager.execute(command)
    def ask(self, q: str): self._ask_or_explain(q, explain=False, is_raw=False)
    def explain(self, q: str): self._ask_or_explain(q, explain=True, is_raw=False)
    def ask_raw(self, formula: str): self._ask_or_explain(formula, explain=False, is_raw=True)
    def build_kb_from_file(self, filepath: str): self.wissensbasis_manager.add_document(filepath)
    def search(self, query: str):
        if not RAG_ENABLED: print("   ❌ RAG-Funktionen sind deaktiviert."); return
        print(f"\n> Suche nach Kontext für: '{query}'")
        relevant_chunks_info = self.wissensbasis_manager.retrieve_relevant_chunks(query)
        if not relevant_chunks_info: print("   [RAG] Keine relevanten Informationen gefunden."); return
        print(f"   [RAG] Relevanter Kontext gefunden:\n---")
        for i, chunk_info in enumerate(relevant_chunks_info, 1): 
            print(f"[{i} from {chunk_info['source']}] {chunk_info['text']}\n")
    def sources(self):
        if not RAG_ENABLED: print("   ❌ RAG-Funktionen sind deaktiviert."); return
        print("\n📑 Indizierte Wissensquellen:")
        if not self.wissensbasis_manager.doc_paths: print("   (Keine Dokumente geladen)")
        else:
            for doc_id, path in self.wissensbasis_manager.doc_paths.items(): print(f"   - {doc_id} (aus {path})")

#==============================================================================
# 6. MAIN LOOP
#==============================================================================
def print_help():
    print("\n" + " K-Assistant Hilfe ".center(60, "-"))
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
    print("\n  === System & Steuerung ===")
    print("  status             - Zeigt den Systemstatus und Metriken an")
    print("  shell <befehl>     - Führt einen Shell-Befehl aus")
    print("  parse <formel>     - Testet den Parser mit einer Formel")
    print("  clearcache         - Leert den Proof- und Prompt-Cache")
    print("  exit               - Beendet die Anwendung und speichert die KB")
    print("-" * 60 + "\n")

def main_loop():
    # Optional: Backup-Manager-Integration
    try:
        from backup_manager import BackupManager
        backup_mgr = BackupManager()
        backup_mgr.auto_backup("merged_v41_v22.2", "Functional merge and bugfix release")
    except ImportError:
        print("ℹ️  'backup_manager.py' nicht gefunden, Backup wird übersprungen.")
    
    assistant = KAssistant()
    print_help()
    
    while True:
        try:
            user_input = input("k-assistant> ").strip()
            if not user_input: continue
            parts = user_input.split(" ", 1)
            command = parts[0].lower()
            args = parts[1].strip('"\'') if len(parts) > 1 else ""
            
            command_map = {
                "exit": lambda: assistant.save_kb(assistant.kb_filepath),
                "help": print_help,
                "build_kb": lambda: assistant.build_kb_from_file(args) if args else print("Fehler: Pfad fehlt."),
                "add_raw": lambda: assistant.add_raw(args) if args else print("Fehler: Formel fehlt."),
                "retract": lambda: assistant.retract(args) if args else print("Fehler: Formel fehlt."),
                "learn": assistant.learn_facts,
                "clearcache": assistant.clear_cache,
                "ask": lambda: assistant.ask(args) if args else print("Fehler: Frage fehlt."),
                "explain": lambda: assistant.explain(args) if args else print("Fehler: Frage fehlt."),
                "ask_raw": lambda: assistant.ask_raw(args) if args else print("Fehler: Formel fehlt."),
                "status": assistant.status,
                "show": assistant.show,
                "search": lambda: assistant.search(args) if args else print("Fehler: Suchanfrage fehlt."),
                "sources": assistant.sources,
                "what_is": lambda: assistant.what_is(args) if args else print("Fehler: Entität fehlt."),
                "shell": lambda: assistant.execute_shell(args) if args else print("Fehler: Befehl fehlt."),
                "parse": lambda: assistant.test_parser(args) if args else print("Fehler: Formel fehlt."),
            }
            
            if command in command_map:
                command_map[command]()
                if command == "exit":
                    break
            else:
                print(f"Unbekannter Befehl: '{command}'. Tippen Sie 'help' für eine Liste der Befehle.")

        except (KeyboardInterrupt, EOFError):
            print("\nBeende... Speichere Wissensbasis.")
            if 'assistant' in locals():
                assistant.save_kb(assistant.kb_filepath)
            break
        except Exception as e:
            import traceback
            print(f"\n🚨 Unerwarteter Fehler: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main_loop()