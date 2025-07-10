# backend/ingestion_pipeline.py
import os
import re
from typing import List, Dict, Any, Literal

# Optionale, aber empfohlene Abhängigkeiten
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# Diese Klassen werden aus dem Hauptmodul importiert.
# Wir definieren hier Platzhalter, um Type Hinting zu ermöglichen,
# falls diese Datei eigenständig analysiert wird.
class EnsembleManager: pass
class HAKGALParser: pass
class TemporalFact: pass

@dataclass
class DocumentChunk:
    """Repräsentiert einen Text-Chunk mit Metadaten."""
    text: str
    source_doc: str
    page_num: int
    chunk_type: Literal["title", "narrativetext", "listitem", "table", "unknown"] = "unknown"
    classification: Literal["rule", "definition", "context", "noise"] = "context"

class DocumentIngestionPipeline:
    def __init__(self, ensemble_manager: EnsembleManager, sentence_model: Optional[SentenceTransformer]):
        self.ensemble = ensemble_manager
        self.parser = HAKGALParser()
        self.sentence_model = sentence_model
        
        # Prüfe, ob die notwendigen Bibliotheken für die volle Funktionalität vorhanden sind
        if not self.sentence_model or not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("⚠️ WARNUNG: Sentence-Transformers nicht verfügbar. Semantische Deduplizierung ist deaktiviert.")
            self.sentence_model = None
        
        print("✅ DocumentIngestionPipeline initialisiert.")

    def process_document(self, filepath: str) -> Dict[str, Any]:
        """
        Hauptmethode, die den gesamten Ingestion-Prozess für eine Datei steuert.
        """
        print(f"📄 Starte Verarbeitung von: {filepath}")
        
        # 1. Dokument laden und in Chunks zerlegen
        chunks = self._load_and_chunk_pdf(filepath)
        if not chunks:
            return {"status": "error", "message": "Dokument konnte nicht gelesen oder zerlegt werden."}
        print(f"   -> 1. Dokument in {len(chunks)} Chunks zerlegt.")

        # 2. Chunks klassifizieren (Regel, Definition, etc.)
        self._classify_chunks(chunks)
        print(f"   -> 2. Chunks klassifiziert.")

        # 3. Fakten/Regeln aus relevanten Chunks extrahieren
        extracted_formulas = self._extract_formulas(chunks)
        print(f"   -> 3. {len(extracted_formulas)} potenzielle Formeln extrahiert.")

        # 3.5 Semantische Deduplizierung
        unique_formulas = self._deduplicate_and_filter(extracted_formulas)

        # 4. Extrahierte Formeln validieren und strukturieren
        review_queue = self._validate_and_prepare_for_review(unique_formulas)
        print(f"   -> 4. {len(review_queue)} Formeln für die Überprüfung vorbereitet.")
        
        return {
            "status": "success",
            "source_doc": os.path.basename(filepath),
            "rag_chunks": [chunk.text for chunk in chunks],
            "review_queue": review_queue
        }

    def _load_and_chunk_pdf(self, filepath: str) -> List[DocumentChunk]:
        """Verwendet 'unstructured', um Layout-Elemente zu erkennen, mit Fallback auf PyPDF."""
        chunks = []
        doc_basename = os.path.basename(filepath)
        print(f"   -> Lade Dokument mit {'unstructured' if UNSTRUCTURED_AVAILABLE else 'PyPDF Fallback'}...")

        if UNSTRUCTURED_AVAILABLE:
            try:
                elements = partition_pdf(filename=filepath, strategy="hi_res")
                for el in elements:
                    chunk_type = el.category.lower() if hasattr(el, 'category') else 'unknown'
                    if chunk_type in ["title", "narrativetext", "listitem"]:
                         chunks.append(DocumentChunk(
                            text=str(el),
                            source_doc=doc_basename,
                            page_num=el.metadata.page_number or 0,
                            chunk_type=chunk_type
                        ))
                if chunks: return chunks
                print(f"   [Warnung] 'unstructured' fand keine textuellen Elemente, wechsle zu PyPDF Fallback.")
            except Exception as e:
                print(f"   [Fehler] 'unstructured' fehlgeschlagen ({e}), wechsle zu PyPDF Fallback.")
        
        if not PYPDF_AVAILABLE:
            print("❌ FEHLER: Weder 'unstructured' noch 'pypdf' sind verfügbar. PDF-Verarbeitung nicht möglich.")
            return []
            
        try:
            reader = PdfReader(filepath)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text: continue
                paragraphs = text.split('\n\n')
                for p in paragraphs:
                    if len(p.strip()) > 30:
                        chunks.append(DocumentChunk(
                            text=p.strip(),
                            source_doc=doc_basename,
                            page_num=i + 1
                        ))
        except Exception as e:
            print(f"   [Fehler] Beim Lesen des PDFs mit PyPDF: {e}")
            return []
        return chunks

    def _classify_chunks(self, chunks: List[DocumentChunk]):
        """Verwendet Heuristiken, um den Typ jedes Chunks zu bestimmen."""
        for chunk in chunks:
            text = chunk.text.lower()
            if chunk.chunk_type == "title" or len(text.split()) < 10:
                chunk.classification = "noise"
            elif any(keyword in text for keyword in ["muss", "sollte", "ist erforderlich", "darf nicht", "verpflichtet"]):
                chunk.classification = "rule"
            elif re.match(r".* ist definiert als .*|.* bedeutet .*", text):
                chunk.classification = "definition"

    def _extract_formulas(self, chunks: List[DocumentChunk]) -> List[str]:
        """Ruft die LLM-Faktenextraktion nur für relevante Chunks auf."""
        relevant_chunks = [c.text for c in chunks if c.classification in ["rule", "definition"]]
        if not relevant_chunks:
            return []
            
        print(f"      -> Sende {len(relevant_chunks)} relevante Chunks zur Faktenextraktion an LLM...")
        context_block = "\n---\n".join(relevant_chunks)
        
        return self.ensemble.extract_facts_with_ensemble(context_block)

    def _deduplicate_and_filter(self, formulas: List[str], similarity_threshold=0.95) -> List[str]:
        """Entfernt semantisch redundante Formeln, wenn Sentence-Transformers verfügbar ist."""
        if not self.sentence_model or len(formulas) < 2:
            return formulas

        print(f"   -> Führe semantische Deduplizierung für {len(formulas)} Formeln durch...")
        try:
            embeddings = self.sentence_model.encode(formulas, convert_to_tensor=True)
            clusters = util.community_detection(embeddings, min_community_size=1, threshold=similarity_threshold)
            
            unique_formulas = [formulas[cluster[0]] for cluster in clusters]
                
            print(f"      -> Reduziert auf {len(unique_formulas)} einzigartige Formeln.")
            return unique_formulas
        except Exception as e:
            print(f"   [Fehler] Bei der Deduplizierung: {e}")
            return formulas # Im Fehlerfall alle Formeln zurückgeben

    def _validate_and_prepare_for_review(self, formulas: List[str]) -> List[Dict[str, Any]]:
        """Prüft die Syntax und bereitet die Daten für die Review-Queue vor."""
        review_queue = []
        for formula in formulas:
            is_valid, _, msg = self.parser.parse(formula)
            review_queue.append({
                "formula": formula,
                "is_valid_syntax": is_valid,
                "validation_message": msg,
                "suggested_action": "accept" if is_valid else "edit",
                "status": "pending_review"
            })
        return review_queue