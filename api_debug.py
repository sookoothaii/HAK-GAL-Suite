# api_debug.py - CORS-Fix Version für HAK-GAL Suite
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import io
import sys

# Backend-Import anpassen (falls nötig)
from backend.Xk_assistant_v41_antlr_parser_HHHHXXXXXXX import KAssistant

# --- Initialisierung ---
app = Flask(__name__)

# ERWEITERTE CORS-Konfiguration für bessere Kompatibilität
origins = [
    "http://localhost:3000", 
    "http://localhost:8080", 
    "http://localhost:8081", 
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080", 
    "http://127.0.0.1:8081", 
    "http://127.0.0.1:5173"
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
print(f"🔒 CORS konfiguriert für: {origins}")

# --- Hilfsfunktionen ---
def get_current_state():
    return {
        "permanentKnowledge": assistant.core.K,
        "learningSuggestions": assistant.potential_new_facts,
    }

def capture_output(func, *args, **kwargs):
    """Fängt die print-Ausgaben einer Funktion ab und gibt sie als String zurück."""
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f"Error in function: {e}")
    finally:
        sys.stdout = old_stdout
    
    return captured_output.getvalue()

# --- Debug-Endpunkt ---
@app.route('/api/test', methods=['GET'])
def test_connection():
    return jsonify({
        "status": "Backend läuft!",
        "cors_origins": origins,
        "backend_ready": True
    })

# --- Hauptendpunkt mit verbesserter Fehlerbehandlung ---
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
        print("❌ Fehler: Kein 'command' in Request-Daten")
        return jsonify({"error": "Ungültige Anfrage: 'command' fehlt."}), 400

    full_command = data['command'].strip()
    parts = full_command.split(" ", 1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    
    print(f"📨 Command empfangen: '{command}' mit Args: '{args}'")

    try:
        response_data = {}
        chat_response = None

        if command in ["add_raw", "retract", "learn", "build_kb", "clearcache"]:
            # Befehle, die nur den Zustand ändern
            print(f"🔧 Verarbeite State-Change Command: {command}")
            if command == "add_raw": 
                assistant.add_raw(args)
                chat_response = f"✅ Fakt hinzugefügt: {args}"
            elif command == "retract": 
                assistant.retract(args)
                chat_response = f"✅ Fakt entfernt: {args}"
            elif command == "learn": 
                assistant.learn_facts()
                chat_response = "✅ Alle vorgeschlagenen Fakten gelernt"
            elif command == "build_kb": 
                assistant.build_kb_from_file(args)
                chat_response = f"✅ Knowledge Base von Datei geladen: {args}"
            elif command == "clearcache": 
                assistant.clear_cache()
                chat_response = "✅ Cache geleert"
        
        elif command in ["ask", "explain", "what_is", "show", "status"]:
            # Befehle, die eine Text-Antwort erzeugen
            print(f"💬 Verarbeite Chat Command: {command}")
            if command == "ask": 
                chat_response = capture_output(assistant.ask, args)
            elif command == "explain": 
                chat_response = capture_output(assistant.explain, args)
            elif command == "what_is": 
                chat_response = capture_output(assistant.what_is, args)
            elif command == "show": 
                chat_response = capture_output(assistant.show)
            elif command == "status": 
                chat_response = capture_output(assistant.status)
        
        else:
            print(f"❓ Unbekannter Command: {command}")
            chat_response = f"Unbekannter Befehl: '{command}'"

        # State immer abrufen
        try:
            state = get_current_state()
        except Exception as e:
            print(f"⚠️ Fehler beim Abrufen des States: {e}")
            state = {
                "permanentKnowledge": [],
                "learningSuggestions": []
            }
        
        state["status"] = "success"
        state["lastCommand"] = command
        
        # Chat-Response hinzufügen
        if chat_response:
            # Säubere die Response von potentiellen Problemen
            clean_response = str(chat_response).strip()
            if not clean_response:
                clean_response = f"✅ Command '{command}' ausgeführt (keine Ausgabe)"
            state["chatResponse"] = clean_response
        else:
            state["chatResponse"] = f"✅ Command '{command}' erfolgreich ausgeführt"
        
        print(f"✅ Response generiert: {len(str(state))} Zeichen")
        return jsonify(state)

    except Exception as e:
        error_msg = f"Backend-Fehler: {str(e)}"
        print(f"💥 Exception: {error_msg}")
        traceback.print_exc()
        return jsonify({
            "error": error_msg,
            "status": "error",
            "lastCommand": command if 'command' in locals() else "unknown"
        }), 500

# --- Server starten ---
if __name__ == '__main__':
    print("🚀 Server startet auf Port 5001...")
    print("🌐 Frontend-URLs die funktionieren sollten:")
    for origin in origins:
        print(f"   - {origin}")
    app.run(debug=True, port=5001, host='0.0.0.0')
