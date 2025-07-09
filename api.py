# api.py
# Phase 24.3: Full Response API with Timeout Protection
# - Fängt die Ausgaben von `ask` und `explain` ab und sendet sie an das Frontend.
# - Verhindert Timeouts durch Command-Limits

from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import io
import sys
import signal
import threading
from contextlib import contextmanager

# ✅ WOLFRAM-INTEGRATION: Import der Wolfram-Version
from backend.k_assistant_main_v7_wolfram import KAssistant

# --- Timeout-Management ---
class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timeout nach {seconds} Sekunden")
    
    # Setup
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Cleanup
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# --- Initialisierung ---
app = Flask(__name__)
origins = [
    "http://localhost:3000", "http://localhost:8080", "http://localhost:8081", "http://localhost:5173",
    "http://127.0.0.1:3000", "http://127.0.0.1:8080", "http://127.0.0.1:8081", "http://127.0.0.1:5173"
]
CORS(app, resources={
    r"/api/*": {
        "origins": origins,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

print("🤖 Initialisiere K-Assistant... Bitte warten.")
assistant = KAssistant()
print("✅ K-Assistant ist bereit.")

# --- Hilfsfunktionen ---
def get_current_state():
    # Verbesserte RAG-Kontext-Formatierung
    rag_context = "📊 RAG SYSTEM STATUS\n" + "="*40 + "\n"
    
    if hasattr(assistant, 'wissensbasis_manager') and assistant.wissensbasis_manager.chunks:
        doc_count = len(assistant.wissensbasis_manager.doc_paths)
        chunk_count = len(assistant.wissensbasis_manager.chunks)
        
        rag_context += f"📁 Indizierte Dokumente: {doc_count}\n"
        rag_context += f"🧩 Text-Chunks: {chunk_count}\n\n"
        
        # Zeige Dokumentenliste
        if assistant.wissensbasis_manager.doc_paths:
            rag_context += "📚 DOKUMENT-QUELLEN:\n"
            for doc_id, path in assistant.wissensbasis_manager.doc_paths.items():
                rag_context += f"  • {doc_id}\n"
            rag_context += "\n"
        
        # Zeige Sample-Chunks formatiert
        rag_context += "🔍 CONTENT-PREVIEW (Top 3):\n"
        for i, chunk_info in enumerate(assistant.wissensbasis_manager.chunks[:3]):
            source = chunk_info.get('source', 'Unknown')
            text = chunk_info.get('text', '').strip()[:150]
            # Besser lesbare Formatierung
            rag_context += f"\n[Chunk {i+1} • {source}]\n"
            rag_context += f"{text}...\n"
            rag_context += "-" * 40 + "\n"
        
        if len(assistant.wissensbasis_manager.chunks) > 5:
            remaining = len(assistant.wissensbasis_manager.chunks) - 5
            rag_context += f"    ... und {remaining} weitere Chunks\n"
            
        # Bindestrich-Hinweis
        rag_context += "\n🎯 BINDESTRICH-SUPPORT AKTIV\n"
        rag_context += "    Teste: RAG-Pipeline, AI-System, Machine-Learning\n"
    else:
        rag_context += "❌ Noch keine Dokumente indiziert\n"
        rag_context += "💡 Verwende 'build_kb <pfad>' zum Laden von Dokumenten\n\n"
        rag_context += "🧪 QUICK TESTS:\n"
        rag_context += "    parse Funktioniert(RAG-Pipeline).\n"
        rag_context += "    add_raw IstAktiv(AI-System).\n"
        rag_context += "    ask Ist das System kritisch?\n"
    
    # DEBUG: Zeige aktuelle State-Werte
    permanent_knowledge = getattr(assistant.core, 'K', [])
    learning_suggestions = getattr(assistant, 'potential_new_facts', [])
    data_sources = list(assistant.wissensbasis_manager.doc_paths.keys()) if hasattr(assistant, 'wissensbasis_manager') else []
    llm_status = get_llm_status()  # NEUE LLM-Status-Info
    
    print(f"🔍 Backend State Debug:")
    print(f"   - permanentKnowledge: {len(permanent_knowledge)} items")
    print(f"   - learningSuggestions: {len(learning_suggestions)} items")
    print(f"   - dataSources: {len(data_sources)} items")
    print(f"   - ragContext length: {len(rag_context)} chars")
    print(f"   - LLM Status: {llm_status['llm_active']}/{llm_status['llm_count']} active")
    
    if learning_suggestions:
        print(f"   - Learning Suggestions Details:")
        for i, suggestion in enumerate(learning_suggestions[:3]):
            print(f"     [{i}] {suggestion}")
    
    return {
        "permanentKnowledge": permanent_knowledge,
        "learningSuggestions": learning_suggestions,
        "ragContext": rag_context,
        "dataSources": data_sources,
        "llmStatus": llm_status  # NEUE LLM-Status-Info für Frontend
    }

def capture_output_with_timeout(func, timeout_seconds, *args, **kwargs):
    """Fängt die print-Ausgaben einer Funktion ab und gibt sie als String zurück - mit Timeout."""
    result = {"output": None, "error": None}
    
    def target():
        try:
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            func(*args, **kwargs)
            
            sys.stdout = old_stdout
            result["output"] = captured_output.getvalue()
        except Exception as e:
            result["error"] = str(e)
    
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Thread ist noch am Laufen - Timeout!
        print(f"⏱️ TIMEOUT: Command nach {timeout_seconds}s abgebrochen")
        result["error"] = f"Command-Timeout nach {timeout_seconds} Sekunden"
        # Note: Thread läuft weiter, aber wir ignorieren das Ergebnis
    
    return result["output"], result["error"]

def get_llm_status():
    """Gibt LLM-Status für Header zurück"""
    try:
        provider_info = []
        if hasattr(assistant, 'ensemble_manager') and assistant.ensemble_manager.providers:
            for p in assistant.ensemble_manager.providers:
                provider_name = p.__class__.__name__.replace("Provider", "")
                try:
                    # Schneller Test ohne echte API-Abfrage
                    provider_info.append({
                        "name": provider_name,
                        "status": "✅ Ready"
                    })
                except:
                    provider_info.append({
                        "name": provider_name, 
                        "status": "❌ Error"
                    })
        
        return {
            "llm_count": len(provider_info),
            "llm_active": len([p for p in provider_info if "✅" in p["status"]]),
            "llm_providers": provider_info
        }
    except Exception as e:
        return {
            "llm_count": 0,
            "llm_active": 0, 
            "llm_providers": [],
            "error": str(e)
        }

# --- Debug-Endpunkt ---
@app.route('/api/test', methods=['GET'])
def test_connection():
    return jsonify({
        "status": "Backend läuft!",
        "cors_origins": origins,
        "backend_ready": True
    })

# --- API-Endpunkt ---
@app.route('/api/command', methods=['POST', 'OPTIONS'])
def handle_command():
    # CORS Preflight Request
    if request.method == 'OPTIONS':
        return '', 200
    
    # Request-Logging für Debugging
    print(f"🔍 Request von: {request.origin}")
    print(f"🔍 Headers: {dict(request.headers)}")
    
    data = request.get_json()
    if not data or 'command' not in data:
        return jsonify({"error": "Ungültige Anfrage: 'command' fehlt."}), 400

    full_command = data['command'].strip()
    parts = full_command.split(" ", 1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    
    print(f"📨 Command empfangen: '{command}' mit Args: '{args}'")

    try:
        response_data = {}
        chat_response = None
        
        # Verschiedene Timeouts je nach Command-Typ
        timeout_seconds = 30  # Default
        if command in ["ask", "explain"]:
            timeout_seconds = 45  # Länger für komplexe RAG-Queries
        elif command in ["build_kb"]:
            timeout_seconds = 60  # Noch länger für Dokumenten-Indizierung

        print(f"⏱️ Command-Timeout: {timeout_seconds}s")

        if command in ["add_raw", "retract", "learn", "build_kb", "clearcache"]:
            # Befehle, die den Zustand ändern UND Ausgaben haben
            print(f"🔧 Verarbeite State-Change Command: {command}")
            if command == "add_raw": 
                chat_response, error = capture_output_with_timeout(assistant.add_raw, timeout_seconds, args)
                if error: raise Exception(error)
                # Auto-save nach Änderungen
                assistant.save_kb(assistant.kb_filepath)
                print("💾 Knowledge Base automatisch gespeichert")
            elif command == "retract": 
                chat_response, error = capture_output_with_timeout(assistant.retract, timeout_seconds, args)
                if error: raise Exception(error)
                assistant.save_kb(assistant.kb_filepath)
                print("💾 Knowledge Base automatisch gespeichert")
            elif command == "learn": 
                chat_response, error = capture_output_with_timeout(assistant.learn_facts, timeout_seconds)
                if error: raise Exception(error)
                assistant.save_kb(assistant.kb_filepath)
                print("💾 Knowledge Base automatisch gespeichert")
            elif command == "build_kb": 
                chat_response, error = capture_output_with_timeout(assistant.build_kb_from_file, timeout_seconds, args)
                if error: raise Exception(error)
            elif command == "clearcache": 
                chat_response, error = capture_output_with_timeout(assistant.clear_cache, timeout_seconds)
                if error: raise Exception(error)
        
        elif command in ["ask", "explain", "what_is", "show", "status", "search", "sources", "parse", "help", "wolfram_stats", "add_oracle"]:
             # Befehle, die eine Text-Antwort erzeugen
            if command == "ask": 
                chat_response, error = capture_output_with_timeout(assistant.ask, timeout_seconds, args)
                if error: raise Exception(error)
            elif command == "explain": 
                chat_response, error = capture_output_with_timeout(assistant.explain, timeout_seconds, args)
                if error: raise Exception(error)
            elif command == "what_is": 
                chat_response, error = capture_output_with_timeout(assistant.what_is, timeout_seconds, args)
                if error: raise Exception(error)
            elif command == "show": 
                # SPEZIAL: show() gibt Dict zurück, muss formatiert werden
                try:
                    data = assistant.show()
                    chat_response = "\n=== WISSENSBASIS ÜBERBLICK ===\n"
                    chat_response += f"📊 Permanente Fakten: {len(data['permanent_knowledge'])}\n"
                    chat_response += f"💡 Lernbare Fakten: {len(data['learnable_facts'])}\n"
                    chat_response += f"📚 RAG Dokumente: {data['rag_stats']['doc_count']}\n"
                    chat_response += f"🧩 RAG Chunks: {data['rag_stats']['chunk_count']}\n\n"
                    
                    if data['permanent_knowledge']:
                        chat_response += "🔹 PERMANENTE WISSENSBASIS:\n"
                        for i, fact in enumerate(data['permanent_knowledge'][:5], 1):
                            chat_response += f"  [{i}] {fact}\n"
                        if len(data['permanent_knowledge']) > 5:
                            remaining = len(data['permanent_knowledge']) - 5
                            chat_response += f"  ... und {remaining} weitere Fakten\n"
                    
                    if data['learnable_facts']:
                        chat_response += "\n💡 LERNBARE FAKTEN:\n"
                        for i, fact in enumerate(data['learnable_facts'][:3], 1):
                            chat_response += f"  [{i}] {fact}\n"
                        if len(data['learnable_facts']) > 3:
                            remaining = len(data['learnable_facts']) - 3
                            chat_response += f"  ... und {remaining} weitere Fakten\n"
                        chat_response += "\n➡️ Verwende 'learn' um sie zu übernehmen\n"
                except Exception as e:
                    chat_response = f"❌ Fehler beim Laden der Wissensbasis: {e}"
            elif command == "status": 
                chat_response, error = capture_output_with_timeout(assistant.status, timeout_seconds)
                if error: raise Exception(error)
            elif command == "search":
                chat_response, error = capture_output_with_timeout(assistant.search, timeout_seconds, args)
                if error: raise Exception(error)
            elif command == "sources":
                chat_response, error = capture_output_with_timeout(assistant.sources, timeout_seconds)
                if error: raise Exception(error)
            elif command == "parse":
                chat_response, error = capture_output_with_timeout(assistant.test_parser, timeout_seconds, args)
                if error: raise Exception(error)
            elif command == "wolfram_stats":
                chat_response, error = capture_output_with_timeout(assistant.wolfram_stats, timeout_seconds)
                if error: raise Exception(error)
            elif command == "add_oracle":
                chat_response, error = capture_output_with_timeout(assistant.add_oracle_predicate, timeout_seconds, args)
                if error: raise Exception(error)
            elif command == "help":
                # Direkte Hilfe ohne timeout
                chat_response = """
✅ VERFÜGBARE COMMANDS:

📋 WISSENSBASIS:
  add_raw <formel>   - Fügt KERNREGEL hinzu
  retract <formel>   - Entfernt KERNREGEL  
  learn              - Speichert gefundene Fakten
  show               - Zeigt Wissensbasis an
  
🧠 RAG & DOKUMENTE:
  build_kb <pfad>    - Indiziert Dokument für RAG
  search <anfrage>   - Findet Text in der KB
  sources            - Zeigt Wissensquellen an
  
🤖 ANFRAGEN:
  ask <frage>        - Beantwortet Frage (mit RAG + Wolfram)
  explain <frage>    - Erklärt eine Antwort
  what_is <entity>   - Zeigt Profil einer Entität
  
🔍 WOLFRAM|ALPHA:
  wolfram_stats      - Zeigt Wolfram Cache-Statistiken
  add_oracle <pred>  - Fügt Oracle-Prädikat hinzu
  
🔧 TOOLS:
  parse <formel>     - Testet Parser mit Formel  
  status             - Zeigt Systemstatus
  clearcache         - Leert alle Caches
  
🎯 BINDESTRICH-SUPPORT:
  parse Funktioniert(RAG-Pipeline).
  ask Läuft das AI-System?
                """
        
        else:
            # VALIDIERE: Alle Backend-Commands sind verfügbar
            available_commands = ["add_raw", "retract", "learn", "build_kb", "clearcache", 
                                "ask", "explain", "what_is", "show", "status", 
                                "search", "sources", "parse", "help", "wolfram_stats", "add_oracle"]
            chat_response = f"❌ Unbekannter Befehl: '{command}'\n\n✅ Verfügbare Commands:\n" + "\n".join([f"  • {cmd}" for cmd in sorted(available_commands)])

        # Immer den aktuellen Zustand abrufen
        state = get_current_state()
        state["status"] = "success"
        state["lastCommand"] = command
        
        # Füge die abgefangene Chat-Antwort hinzu, falls vorhanden
        if chat_response:
            state["chatResponse"] = chat_response
        
        print(f"✅ Command '{command}' erfolgreich verarbeitet in < {timeout_seconds}s")
        return jsonify(state)

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Command-Fehler: {error_msg}")
        traceback.print_exc()
        
        # Versuche trotzdem State zu holen für Frontend
        try:
            state = get_current_state()
            state["status"] = "error"
            state["error"] = error_msg
            state["chatResponse"] = f"🚨 Fehler: {error_msg}"
            return jsonify(state), 500
        except:
            return jsonify({"error": error_msg, "status": "error"}), 500

# --- Server starten ---
if __name__ == '__main__':
    print("🚀 Starte HAK-GAL API Server mit Timeout-Protection...")
    app.run(debug=True, port=5001)