# HAK/GAL Suite - Installationsanleitung

## Voraussetzungen

- Python 3.8 oder höher
- Node.js 16 oder höher
- Git (optional)

## Schritt-für-Schritt Installation

### 1. Dateien kopieren

#### Backend-Dateien:
Kopieren Sie folgende Dateien von:
`D:\MCP Mods\hak_gal_prototype\Backend\`

Nach:
`D:\MCP Mods\HAK_GAL_SUITE\backend\`

- Xk_assistant_v41_antlr_parser_HHHHXXXXXXX.py
- hakgal_grammar.py
- backup_manager.py

#### Frontend-Dateien:
Kopieren Sie ALLE Inhalte (außer .git und node_modules) von:
`D:\MCP Mods\file_flow_assistant\file-flow-assistant\`

Nach:
`D:\MCP Mods\HAK_GAL_SUITE\frontend\`

### 2. Python-Umgebung vorbereiten

Öffnen Sie ein Terminal in `D:\MCP Mods\HAK_GAL_SUITE` und führen Sie aus:

```bash
# Virtuelle Umgebung erstellen
python -m venv venv

# Virtuelle Umgebung aktivieren (Windows)
venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### 3. Frontend vorbereiten

```bash
cd frontend
npm install
```

### 4. API-Datei konfigurieren

Öffnen Sie `D:\MCP Mods\HAK_GAL_SUITE\api.py` und passen Sie Zeile 6 an:

```python
# Ändern Sie HHHHXXXXXXX zu Ihrem tatsächlichen Dateinamen
from backend.Xk_assistant_v41_antlr_parser_HHHHXXXXXXX import KAssistant
```

### 5. System starten

Doppelklicken Sie auf `start.bat` oder starten Sie manuell:

**Terminal 1 - Backend:**
```bash
cd D:\MCP Mods\HAK_GAL_SUITE
python api.py
```

**Terminal 2 - Frontend:**
```bash
cd D:\MCP Mods\HAK_GAL_SUITE\frontend
npm run dev
```

### 6. Zugriff

- Frontend: http://localhost:3000
- API: http://localhost:5001

## Fehlerbehebung

### ModuleNotFoundError
- Stellen Sie sicher, dass alle Backend-Dateien korrekt kopiert wurden
- Überprüfen Sie den Dateinamen in api.py

### npm Fehler
- Löschen Sie node_modules und package-lock.json
- Führen Sie `npm install` erneut aus

### Port bereits belegt
- Ändern Sie Port 5001 in api.py auf einen anderen Port
- Passen Sie die CORS-Einstellung entsprechend an
