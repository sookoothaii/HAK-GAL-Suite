@echo off
setlocal enabledelayedexpansion

REM ==============================================================================
REM HAK-GAL Suite - Web-UI Version (Auto-Fix für Wolfram)
REM ==============================================================================

title HAK-GAL Suite - Web UI Launcher (Fixed)
color 0A

REM Ins richtige Verzeichnis wechseln
cd /d "%~dp0"

cls
echo.
echo ================================================================
echo       HAK-GAL Suite - Web UI Smart Launcher v2.1 FIXED        
echo ================================================================
echo     🔧 AUTO-FIX für Wolfram-Integration aktiviert!           
echo ================================================================
echo.

REM ================================================================
REM AUTO-FIX WOLFRAM INTEGRATION
REM ================================================================
echo 🔍 Prüfe Wolfram-Integration...
python -c "import wolframalpha; print('✅ wolframalpha bereits installiert')" 2>nul
if errorlevel 1 (
    echo.
    echo ⚠️  WOLFRAM-BIBLIOTHEK FEHLT!
    echo.
    echo 📦 Installiere wolframalpha automatisch...
    pip install wolframalpha
    if errorlevel 1 (
        echo.
        echo ❌ FEHLER: Automatische Installation fehlgeschlagen!
        echo.
        echo Bitte manuell ausführen:
        echo   pip install wolframalpha
        echo.
        echo Oder versuchen Sie:
        echo   python -m pip install wolframalpha
        echo.
        pause
        goto SKIP_WOLFRAM_CHECK
    )
    echo ✅ wolframalpha erfolgreich installiert!
)

REM Prüfe ob App ID konfiguriert ist
echo.
echo 🔑 Prüfe Wolfram App ID...
if exist ".env" (
    findstr /C:"WOLFRAM_APP_ID=your_wolfram_app_id_here" ".env" >nul
    if not errorlevel 1 (
        echo.
        echo ⚠️  WOLFRAM APP ID NICHT KONFIGURIERT!
        echo.
        echo Für volle Funktionalität benötigen Sie eine kostenlose App ID:
        echo 1. Besuchen Sie: https://developer.wolframalpha.com/portal/myapps/
        echo 2. Erstellen Sie ein kostenloses Konto
        echo 3. Erhalten Sie Ihre App ID
        echo 4. Fügen Sie sie zur .env Datei hinzu
        echo.
        echo Möchten Sie trotzdem fortfahren? (Wolfram wird deaktiviert sein)
        set /p continue="Fortfahren? (j/n): "
        if /i "!continue!" neq "j" exit /b 0
    ) else (
        echo ✅ Wolfram App ID konfiguriert
    )
) else (
    echo ⚠️  Keine .env Datei gefunden - erstelle aus Vorlage...
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo ✅ .env Datei erstellt
    )
)

:SKIP_WOLFRAM_CHECK

REM ================================================================
REM WEB-INTERFACE STARTEN
REM ================================================================
echo.
echo ================================================================
echo              Starte HAK-GAL Web-Interface
echo ================================================================

REM Backend prüfen
if not exist "api.py" (
    echo ❌ FEHLER: api.py nicht gefunden!
    pause
    exit /b 1
)

REM Flask prüfen
echo.
echo 🔍 Prüfe Flask...
python -c "import flask" 2>nul || (
    echo 📦 Installiere Flask...
    pip install flask flask-cors
)

echo.
echo ================================================================
echo                    Web-Interface Setup
echo ================================================================
echo Backend startet auf:  http://localhost:5001
echo Frontend startet auf: http://localhost:3000
echo.
echo 🚀 Starte Backend mit Wolfram-Integration...
start "HAK-GAL Backend" cmd /k "cd /d "%CD%" && echo HAK-GAL Backend (mit Wolfram) startet... && python api.py"

echo ⏳ Warte 8 Sekunden auf Backend-Start...
timeout /t 8 /nobreak >nul

REM Teste ob Backend läuft
echo.
echo 🔍 Teste Backend-Verbindung...
curl -s http://localhost:5001/api/test >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Backend noch nicht bereit, warte weitere 5 Sekunden...
    timeout /t 5 /nobreak >nul
)

echo.
echo 🚀 Starte Frontend...
if exist "frontend" (
    cd frontend
    
    REM Node.js prüfen
    node --version >nul 2>&1
    if errorlevel 1 (
        echo.
        echo ❌ FEHLER: Node.js nicht gefunden!
        echo Installieren Sie Node.js von https://nodejs.org/
        echo.
        echo Das Backend läuft bereits auf http://localhost:5001
        echo Sie können es mit einem API-Client testen.
        pause
        cd ..
        exit /b 1
    )
    
    REM npm dependencies prüfen
    if not exist "node_modules" (
        echo 📦 Installiere Frontend-Dependencies...
        npm install
    )
    
    echo.
    echo Frontend startet...
    start "HAK-GAL Frontend" cmd /k "echo HAK-GAL Frontend startet... && npm run dev"
    cd ..
    
    echo.
    echo ================================================================
    echo           ✅ WEB-INTERFACE ERFOLGREICH GESTARTET!
    echo ================================================================
    echo.
    echo 🌐 Frontend-URL: http://localhost:3000
    echo 🔧 Backend-URL:  http://localhost:5001
    echo.
    echo 🎯 WOLFRAM-STATUS:
    python -c "import wolframalpha; print('   ✅ Wolfram-Bibliothek: AKTIV')" 2>nul || echo    ❌ Wolfram-Bibliothek: FEHLT
    
    REM Prüfe ob Wolfram App ID konfiguriert ist
    if exist ".env" (
        findstr /C:"WOLFRAM_APP_ID=your_wolfram_app_id_here" ".env" >nul
        if errorlevel 1 (
            echo    ✅ Wolfram App ID: KONFIGURIERT
            echo    📊 Loading-Status sollte jetzt "YES" zeigen!
        ) else (
            echo    ⚠️  Wolfram App ID: NICHT KONFIGURIERT
            echo    📊 Loading-Status zeigt "NO"
        )
    )
    
    echo.
    echo 📚 VERFÜGBARE FEATURES:
    echo    • Interaktive HAK-GAL Abfragen
    echo    • Wolfram Alpha Integration (wenn konfiguriert)
    echo    • Wissensbasis-Management  
    echo    • Echtzeit-Reasoning
    echo    • RAG-System für Dokumente
    echo.
    echo 🧪 WOLFRAM-TEST-BEFEHLE:
    echo    • wolfram_stats
    echo    • ask was ist die hauptstadt von deutschland
    echo    • ask_raw HauptstadtVon(Frankreich).
    echo.
    
    REM Browser öffnen
    timeout /t 3 /nobreak
    echo.
    set /p open_browser="🌐 Browser automatisch öffnen? (j/n): "
    if /i "!open_browser!"=="j" (
        start http://localhost:3000
    )
    
) else (
    echo ❌ FEHLER: Frontend-Verzeichnis nicht gefunden!
    echo.
    echo Das Backend läuft trotzdem auf http://localhost:5001
    echo Sie können es mit einem API-Client verwenden.
    pause
)

echo.
echo ================================================================
echo System läuft. Drücken Sie eine Taste zum Beenden...
echo ================================================================
pause

REM Beende die gestarteten Prozesse
echo.
echo 🛑 Beende Prozesse...
taskkill /FI "WINDOWTITLE eq HAK-GAL Backend*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq HAK-GAL Frontend*" /F >nul 2>&1

echo ✅ Fertig.
exit /b 0