# api.py
# Phase 24.2: Full Response API
# - Fängt die Ausgaben von `ask` und `explain` ab und sendet sie an das Frontend.

from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import io
import sys

# Passen Sie diesen Import an den exakten Namen Ihrer K-Assistant-Datei an!
from backend.Xk_assistant_v41_antlr_parser_HHHHXXXXXXX import KAssistant

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
    return {
        "permanentKnowledge": assistant.core.K,
        "learningSuggestions": assistant.potential_new_facts,
    }

def capture_output(func, *args, **kwargs):
    """Fängt die print-Ausgaben einer Funktion ab und gibt sie als String zurück."""
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    func(*args, **kwargs)
    
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

        if command in ["add_raw", "retract", "learn", "build_kb", "clearcache"]:
            # Befehle, die den Zustand ändern UND Ausgaben haben
            print(f"🔧 Verarbeite State-Change Command: {command}")
            if command == "add_raw": 
                chat_response = capture_output(assistant.add_raw, args)
            elif command == "retract": 
                chat_response = capture_output(assistant.retract, args)
            elif command == "learn": 
                chat_response = capture_output(assistant.learn_facts)
            elif command == "build_kb": 
                chat_response = capture_output(assistant.build_kb_from_file, args)
            elif command == "clearcache": 
                chat_response = capture_output(assistant.clear_cache)
        
        elif command in ["ask", "explain", "what_is", "show", "status"]:
             # Befehle, die eine Text-Antwort erzeugen
            if command == "ask": chat_response = capture_output(assistant.ask, args)
            elif command == "explain": chat_response = capture_output(assistant.explain, args)
            elif command == "what_is": chat_response = capture_output(assistant.what_is, args)
            elif command == "show": chat_response = capture_output(assistant.show)
            elif command == "status": chat_response = capture_output(assistant.status)
        
        else:
            chat_response = f"Unbekannter Befehl: '{command}'"

        # Immer den aktuellen Zustand abrufen
        state = get_current_state()
        state["status"] = "success"
        state["lastCommand"] = command
        
        # Füge die abgefangene Chat-Antwort hinzu, falls vorhanden
        if chat_response:
            state["chatResponse"] = chat_response
        
        return jsonify(state)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- Server starten ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)