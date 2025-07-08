# Contributing to HAK/GAL Suite

Vielen Dank für Ihr Interesse an der Mitarbeit am HAK/GAL Suite Projekt!

## Entwicklungsrichtlinien

### Code-Standards

1. **Python Code**
   - Befolgen Sie PEP 8
   - Verwenden Sie Type Hints wo möglich
   - Dokumentieren Sie alle Funktionen mit Docstrings

2. **TypeScript/React Code**
   - Verwenden Sie TypeScript strict mode
   - Befolgen Sie die ESLint-Konfiguration
   - Komponenten sollten funktional sein (keine Klassenkomponenten)

### Commit-Nachrichten

Verwenden Sie aussagekräftige Commit-Nachrichten:
- `feat:` für neue Features
- `fix:` für Bugfixes
- `docs:` für Dokumentation
- `refactor:` für Code-Refactoring
- `test:` für Tests

### Pull Requests

1. Forken Sie das Repository
2. Erstellen Sie einen Feature-Branch (`git checkout -b feature/AmazingFeature`)
3. Committen Sie Ihre Änderungen (`git commit -m 'feat: Add some AmazingFeature'`)
4. Pushen Sie zum Branch (`git push origin feature/AmazingFeature`)
5. Öffnen Sie einen Pull Request

### Tests

- Schreiben Sie Tests für neue Features
- Stellen Sie sicher, dass alle Tests durchlaufen
- Fügen Sie Integrationstests für kritische Pfade hinzu

## Entwicklungsumgebung einrichten

1. Python-Umgebung:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Frontend-Umgebung:
   ```bash
   cd frontend
   npm install
   ```

## Fragen?

Öffnen Sie ein Issue für Diskussionen oder Fragen.
