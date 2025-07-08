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
origins = ["http://localhost:3000", "http://localhost:8080", "http://localhost:8081", "http://localhost:5173"]
CORS(app, resources={r"/api/*": {"origins": origins}})

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

# --- API-Endpunkt ---
@app.route('/api/command', methods=['POST'])
def handle_command():
    data = request.get_json()
    if not data or 'command' not in data:
        return jsonify({"error": "Ungültige Anfrage: 'command' fehlt."}), 400

    full_command = data['command'].strip()
    parts = full_command.split(" ", 1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    try:
        response_data = {}
        chat_response = None

        if command in ["add_raw", "retract", "learn", "build_kb", "clearcache"]:
            # Befehle, die nur den Zustand ändern
            if command == "add_raw": assistant.add_raw(args)
            elif command == "retract": assistant.retract(args)
            elif command == "learn": assistant.learn_facts()
            elif command == "build_kb": assistant.build_kb_from_file(args)
            elif command == "clearcache": assistant.clear_cache()
        
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
    app.run(debug=True, port=5001)