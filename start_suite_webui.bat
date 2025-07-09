@echo off
setlocal enabledelayedexpansion

REM ==============================================================================
REM HAK-GAL Suite - Web-UI Version (Import-Fixed)
REM ==============================================================================

title HAK-GAL Suite - Web UI Launcher
color 0A

REM Ins richtige Verzeichnis wechseln (wo die .bat liegt)
cd /d "%~dp0"

cls
echo.
echo ================================================================
echo          HAK-GAL Suite - Web UI Smart Launcher v2.0           
echo ================================================================
echo     Hybrid AI Framework + Wolfram Alpha Integration          
echo     Import-Probleme behoben - Vollständig einsatzbereit      
echo     Neue Features: test_wolfram, wolfram_stats, erweiterte API
echo ================================================================
echo.
echo Aktuelles Verzeichnis: %CD%
echo.

REM Variablen definieren
set "WOLFRAM_SYSTEM=backend\k_assistant_main_v7_wolfram.py"
set "STANDARD_SYSTEM=backend\k_assistant_main.py"
set "WEB_API=api.py"

:MAIN_MENU
echo Verfuegbare Optionen:
echo.
echo    [1] Web-Interface starten (Empfohlen - localhost:3000)
echo    [2] Wolfram-System starten (Konsole)
echo    [3] Standard-System starten (Konsole)
echo    [4] Automatisches Setup
echo    [5] Tests ausfuehren (mit Wolfram-Tests)
echo    [6] Demo starten
echo    [7] System-Status
echo    [8] Wolfram-Integration testen
echo    [9] Dokumentation
echo    [0] Beenden
echo.
set /p choice="Waehlen Sie eine Option (1-9 oder 0): "

if "%choice%"=="1" goto START_WEB
if "%choice%"=="2" goto START_WOLFRAM
if "%choice%"=="3" goto START_STANDARD  
if "%choice%"=="4" goto RUN_SETUP
if "%choice%"=="5" goto RUN_TESTS
if "%choice%"=="6" goto RUN_DEMO
if "%choice%"=="7" goto CHECK_STATUS
if "%choice%"=="8" goto TEST_WOLFRAM_INTEGRATION
if "%choice%"=="9" goto OPEN_DOCS
if "%choice%"=="0" goto EXIT

echo FEHLER: Ungueltige Auswahl. Bitte waehlen Sie 1-9 oder 0 zum Beenden.
timeout /t 2 /nobreak >nul
goto MAIN_MENU

:START_WEB
cls
echo ================================================================
echo              Starte HAK-GAL Web-Interface
echo ================================================================

REM Backend-Verfügbarkeit prüfen
if not exist "%WEB_API%" (
    echo FEHLER: Web-API nicht gefunden!
    echo Datei fehlt: %CD%\%WEB_API%
    pause
    goto MAIN_MENU
)

REM .env Datei prüfen
if not exist ".env" (
    echo WARNUNG: .env Datei nicht gefunden!
    if exist ".env.example" (
        echo Erstelle .env aus Vorlage...
        copy ".env.example" ".env" >nul
        echo .env Datei erstellt
    ) else (
        echo Nutze Standard-Konfiguration...
    )
)

REM Python-Abhängigkeiten prüfen
echo Pruefe Python-Abhängigkeiten...
python -c "import flask" 2>nul || (
    echo Flask nicht gefunden. Installiere...
    pip install flask flask-cors
)

echo.
echo ================================================================
echo                    Web-Interface Setup
echo ================================================================
echo Backend startet auf:  http://localhost:5001
echo Frontend startet auf: http://localhost:3000
echo.
echo Starte Backend...
start "HAK-GAL Backend" cmd /k "cd /d "%CD%" && echo Starte HAK-GAL Backend... && python %WEB_API%"

echo Warte 8 Sekunden auf Backend-Start...
timeout /t 8 /nobreak >nul

echo Starte Frontend...
if exist "frontend" (
    echo Frontend-Verzeichnis gefunden
    cd frontend
    
    REM Node.js prüfen
    node --version >nul 2>&1
    if errorlevel 1 (
        echo FEHLER: Node.js nicht gefunden!
        echo Installieren Sie Node.js von https://nodejs.org/
        echo.
        echo Oder verwenden Sie Option 2 fuer Konsolen-Version
        pause
        cd ..
        goto MAIN_MENU
    )
    
    REM npm dependencies prüfen
    if not exist "node_modules" (
        echo Installiere Frontend-Dependencies...
        npm install
    )
    
    echo.
    echo Frontend startet...
    start "HAK-GAL Frontend" cmd /k "echo HAK-GAL Frontend startet... && npm run dev"
    cd ..
    
    echo.
    echo ================================================================
    echo                  WEB-INTERFACE GESTARTET!
    echo ================================================================
    echo.
    echo Frontend-URL: http://localhost:3000
    echo Backend-URL:  http://localhost:5001
    echo.
    echo Oeffnen Sie http://localhost:3000 in Ihrem Browser
    echo.
    echo Features:
    echo - Interaktive HAK-GAL Abfragen
    echo - Wolfram Alpha Integration
    echo - Wissensbasis-Management
    echo - Echtzeit-Reasoning
    echo.
    
    REM Browser automatisch öffnen
    set /p open_browser="Browser automatisch oeffnen? (j/n): "
    if /i "!open_browser!"=="j" (
        start http://localhost:3000
    )
    
) else (
    echo FEHLER: Frontend-Verzeichnis nicht gefunden!
    echo Verzeichnis: %CD%\frontend
    echo.
    echo Alternativen:
    echo - Verwenden Sie Option 2 fuer Konsolen-Version
    echo - Fuehren Sie Setup aus (Option 4)
    pause
    goto MAIN_MENU
)

echo.
echo Web-Interface läuft...
echo Druecken Sie eine Taste zum Zurueckkehren zum Menue
pause
goto MAIN_MENU

:START_WOLFRAM
cls
echo ================================================================
echo          Starte HAK-GAL mit Wolfram Integration (Konsole)
echo ================================================================

if not exist "%WOLFRAM_SYSTEM%" (
    echo FEHLER: Wolfram-System nicht gefunden!
    echo Datei fehlt: %CD%\%WOLFRAM_SYSTEM%
    echo.
    echo Fuehren Sie zuerst das Setup aus (Option 4)
    pause
    goto MAIN_MENU
)

echo Pruefe .env Datei...
if not exist ".env" (
    echo Erstelle .env aus Vorlage...
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo .env Datei erstellt
    ) else (
        echo FEHLER: Keine .env.example gefunden!
        pause
        goto MAIN_MENU
    )
)

echo Pruefe Wolfram-Bibliothek...
python -c "import wolframalpha; print('wolframalpha bereits installiert')" 2>nul
if errorlevel 1 (
    echo Installiere wolframalpha...
    pip install wolframalpha
    if errorlevel 1 (
        echo FEHLER: Installation fehlgeschlagen!
        pause
        goto MAIN_MENU
    )
)

echo.
echo Verfuegbare Features:
echo - Wolfram Alpha Oracle fuer Realwelt-Wissen
echo - Mathematische Berechnungen (Integral, Ableitung)
echo - Geografische Daten (Hauptstädte, Bevölkerung, Flächen)
echo - Wetter- und Währungsinformationen
echo - Intelligente Oracle-Erkennung
echo - Deutsch-Englisch Übersetzung für Anfragen
echo - Neue Befehle: wolfram_stats, test_wolfram, add_oracle
echo.
echo Starte Konsolen-System...
python "%WOLFRAM_SYSTEM%"
goto POST_EXECUTION

:START_STANDARD
cls
echo ================================================================
echo             Starte HAK-GAL Standard-System (Konsole)
echo ================================================================

if not exist "%STANDARD_SYSTEM%" (
    echo FEHLER: Standard-System nicht gefunden!
    echo Datei fehlt: %CD%\%STANDARD_SYSTEM%
    pause
    goto MAIN_MENU
)

echo Starte Standard-System...
python "%STANDARD_SYSTEM%"
goto POST_EXECUTION

:RUN_SETUP
cls
echo ================================================================
echo                 Automatisches Setup
echo ================================================================
echo Verzeichnis: %CD%
echo.

echo Pruefe Python...
python --version 2>nul
if errorlevel 1 (
    echo FEHLER: Python nicht gefunden!
    echo Installieren Sie Python 3.8+ und fuegen Sie es zum PATH hinzu
    pause
    goto MAIN_MENU
)
echo Python verfuegbar

echo.
echo Pruefe Node.js (fuer Frontend)...
node --version 2>nul
if errorlevel 1 (
    echo WARNUNG: Node.js nicht gefunden!
    echo Fuer Web-Interface benoetigt. Download: https://nodejs.org/
) else (
    echo Node.js verfuegbar
)

echo.
echo Pruefe Python-Abhängigkeiten...
set "MISSING="

python -c "import z3" 2>nul || set "MISSING=!MISSING! z3-solver"
python -c "import lark" 2>nul || set "MISSING=!MISSING! lark"
python -c "import openai" 2>nul || set "MISSING=!MISSING! openai"
python -c "from dotenv import load_dotenv" 2>nul || set "MISSING=!MISSING! python-dotenv"
python -c "import wolframalpha" 2>nul || set "MISSING=!MISSING! wolframalpha"
python -c "import flask" 2>nul || set "MISSING=!MISSING! flask flask-cors"

if not "!MISSING!"=="" (
    echo Fehlende Pakete:!MISSING!
    echo.
    set /p install="Installieren? (j/n): "
    if /i "!install!"=="j" (
        echo Installiere Python-Pakete...
        pip install!MISSING!
        if errorlevel 1 (
            echo FEHLER: Installation fehlgeschlagen!
            pause
            goto MAIN_MENU
        )
        echo Python-Pakete installiert
    )
) else (
    echo Alle Python-Abhängigkeiten verfuegbar
)

echo.
echo Pruefe Frontend-Dependencies...
if exist "frontend" (
    cd frontend
    if not exist "node_modules" (
        echo Installiere Frontend-Dependencies...
        npm install
        if errorlevel 1 (
            echo WARNUNG: Frontend-Installation fehlgeschlagen
        ) else (
            echo Frontend-Dependencies installiert
        )
    ) else (
        echo Frontend-Dependencies bereits vorhanden
    )
    cd ..
) else (
    echo WARNUNG: Frontend-Verzeichnis nicht gefunden
)

echo.
echo Pruefe .env Konfiguration...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo .env Datei erstellt
    ) else (
        echo ERSTELLE .env.example...
        echo # HAK-GAL Suite Konfiguration > .env.example
        echo. >> .env.example
        echo # LLM Provider API Keys >> .env.example
        echo DEEPSEEK_API_KEY=your_deepseek_api_key_here >> .env.example
        echo MISTRAL_API_KEY=your_mistral_api_key_here >> .env.example
        echo GEMINI_API_KEY=your_gemini_api_key_here >> .env.example
        echo. >> .env.example
        echo # Wolfram Alpha Integration >> .env.example
        echo WOLFRAM_APP_ID=your_wolfram_app_id_here >> .env.example
        echo WOLFRAM_CACHE_TIMEOUT=3600 >> .env.example
        echo WOLFRAM_DEBUG=false >> .env.example
        
        copy ".env.example" ".env" >nul
        echo .env Datei erstellt
    )
)

echo.
echo ================================================================
echo                    API-Konfiguration
echo ================================================================
echo.
echo Wolfram App ID Setup:
echo 1. Besuchen Sie: https://developer.wolframalpha.com/portal/myapps/
echo 2. Erstellen Sie kostenloses Konto
echo 3. Erhalten Sie App ID (2000 kostenlose Anfragen/Monat)
echo.
set /p app_id="Geben Sie Ihre Wolfram App ID ein (oder Enter zum Ueberspringen): "
if not "!app_id!"=="" (
    echo Aktualisiere .env Datei...
    powershell -Command "(Get-Content .env) -replace 'WOLFRAM_APP_ID=your_wolfram_app_id_here', 'WOLFRAM_APP_ID=!app_id!' | Set-Content .env" 2>nul || (
        echo WOLFRAM_APP_ID=!app_id! >> .env
    )
    echo Wolfram App ID gespeichert
)

echo.
echo ================================================================
echo                   Setup abgeschlossen!
echo ================================================================
echo.
echo Empfohlen: Starten Sie das Web-Interface (Option 1)
echo Alternativ: Konsolen-Version (Option 2)
echo.
pause
goto MAIN_MENU

:TEST_WOLFRAM_INTEGRATION
cls
echo ================================================================
echo             Interaktive Wolfram-Integration Tests
echo ================================================================

if not exist "%WOLFRAM_SYSTEM%" (
    echo FEHLER: Wolfram-System nicht gefunden!
    pause
    goto MAIN_MENU
)

echo Teste Wolfram-Integration...
echo.
echo [1] Schnell-Test (Hauptstadt Deutschland)
echo [2] Umfassender Test (mehrere Kategorien)
echo [3] Interaktive Tests
echo [4] Zurück
echo.
set /p wtest="Wählen Sie (1-4): "

if "%wtest%"=="1" (
    echo Teste: HauptstadtVon(Deutschland)...
    python -c "import sys; sys.path.insert(0, 'backend'); from k_assistant_main_v7_wolfram import KAssistant; k = KAssistant(); k.test_wolfram('HauptstadtVon(Deutschland).')"
) else if "%wtest%"=="2" (
    echo Umfassende Tests...
    python -c "import sys; sys.path.insert(0, 'backend'); from k_assistant_main_v7_wolfram import KAssistant; k = KAssistant(); print('=== GEOGRAFIE ==='); k.test_wolfram('HauptstadtVon(Deutschland).'); k.test_wolfram('Bevölkerung(Berlin).'); print('\n=== MATHEMATIK ==='); k.test_wolfram('Integral(x^2).'); print('\n=== STATUS ==='); k.wolfram_stats()"
) else if "%wtest%"=="3" (
    echo Starte interaktives Test-System...
    echo.
    echo Verfügbare Test-Befehle:
    echo   test_wolfram HauptstadtVon(Deutschland).
    echo   test_wolfram Bevölkerung(Berlin).
    echo   test_wolfram WetterIn(München).
    echo   wolfram_stats
    echo   add_oracle NeuesPrädikat
    echo   status
    echo   exit
    echo.
    python "%WOLFRAM_SYSTEM%"
) else if "%wtest%"=="4" (
    goto MAIN_MENU
) else (
    echo Ungültige Auswahl
    timeout /t 2 /nobreak >nul
    goto TEST_WOLFRAM_INTEGRATION
)

echo.
pause
goto MAIN_MENU

:RUN_TESTS
cls
echo ================================================================
echo                 System-Tests (Erweitert)
echo ================================================================

if exist "test_wolfram_integration.py" (
    echo Fuehre offizielle Test-Suite durch...
    python "test_wolfram_integration.py"
) else (
    echo Fuehre Basis-Tests durch...
    echo.
    echo Python-Version:
    python --version
    
    echo.
    echo Teste HAK-GAL Import (behoben)...
    python -c "import sys; sys.path.insert(0, 'backend'); from k_assistant_main_v7_wolfram import KAssistant; print('✅ Wolfram-System importierbar')" 2>nul || (
        echo ❌ Wolfram-System Import fehlgeschlagen
    )
    
    echo.
    echo Teste Wolfram-Integration...
    python -c "import sys; sys.path.insert(0, 'backend'); from k_assistant_main_v7_wolfram import KAssistant; k = KAssistant(); prover = next((p for p in k.core.provers if p.name == 'Wolfram|Alpha Orakel'), None); print('✅ Wolfram-Prover geladen') if prover else print('⚠️ Wolfram-Prover nicht geladen')" 2>nul
    
    echo.
    echo Teste Web-API...
    python -c "import api; print('✅ Web-API importierbar')" 2>nul || (
        echo ❌ Web-API Import fehlgeschlagen
    )
    
    echo.
    echo Teste Frontend (falls vorhanden)...
    if exist "frontend\package.json" (
        echo ✅ Frontend-Konfiguration gefunden
    ) else (
        echo ❌ Frontend-Konfiguration fehlt
    )
)

echo.
pause
goto MAIN_MENU

:RUN_DEMO
cls
echo ================================================================
echo                  Interaktive Demo
echo ================================================================

echo Waehlen Sie Demo-Modus:
echo.
echo [1] Web-Interface Demo (empfohlen)
echo [2] Konsolen-Demo
echo [3] Zurueck
echo.
set /p demo_choice="Waehlen Sie (1-3): "

if "%demo_choice%"=="1" (
    echo Starte Web-Interface fuer Demo...
    goto START_WEB
) else if "%demo_choice%"=="2" (
    echo Starte Konsolen-Demo...
    if exist "%WOLFRAM_SYSTEM%" (
        echo.
        echo Demo-Befehle (Erweitert):
        echo   ask was ist die hauptstadt von deutschland
        echo   ask_raw HauptstadtVon(Deutschland).
        echo   test_wolfram HauptstadtVon(Deutschland).
        echo   test_wolfram Bevölkerung(Berlin).
        echo   wolfram_stats
        echo   add_oracle TemperaturIn
        echo   status
        echo   exit
        echo.
        python "%WOLFRAM_SYSTEM%"
    ) else (
        echo FEHLER: System nicht gefunden
        pause
    )
) else if "%demo_choice%"=="3" (
    goto MAIN_MENU
) else (
    echo Ungueltige Auswahl
    timeout /t 2 /nobreak >nul
    goto RUN_DEMO
)

pause
goto MAIN_MENU

:CHECK_STATUS
cls
echo ================================================================
echo                   System-Status
echo ================================================================
echo Verzeichnis: %CD%
echo.

echo Python-Installation:
python --version 2>nul && echo ✅ Python OK || echo ❌ Python FEHLER

echo.
echo Node.js (fuer Frontend):
node --version 2>nul && echo ✅ Node.js OK || echo ❌ Node.js FEHLER

echo.
echo Python-Abhängigkeiten:
python -c "import z3; print('✅ z3-solver OK')" 2>nul || echo ❌ z3-solver FEHLER
python -c "import lark; print('✅ lark OK')" 2>nul || echo ❌ lark FEHLER
python -c "import openai; print('✅ openai OK')" 2>nul || echo ❌ openai FEHLER
python -c "from dotenv import load_dotenv; print('✅ python-dotenv OK')" 2>nul || echo ❌ python-dotenv FEHLER
python -c "import wolframalpha; print('✅ wolframalpha OK')" 2>nul || echo ❌ wolframalpha FEHLER
python -c "import flask; print('✅ flask OK')" 2>nul || echo ❌ flask FEHLER

echo.
echo Systemdateien:
if exist "%WOLFRAM_SYSTEM%" (echo ✅ Wolfram-System OK) else (echo ❌ Wolfram-System FEHLER)
if exist "%STANDARD_SYSTEM%" (echo ✅ Standard-System OK) else (echo ❌ Standard-System FEHLER)
if exist "%WEB_API%" (echo ✅ Web-API OK) else (echo ❌ Web-API FEHLER)

echo.
echo Frontend:
if exist "frontend" (echo ✅ Frontend-Verzeichnis OK) else (echo ❌ Frontend-Verzeichnis FEHLER)
if exist "frontend\package.json" (echo ✅ Frontend-Konfiguration OK) else (echo ❌ Frontend-Konfiguration FEHLER)
if exist "frontend\node_modules" (echo ✅ Frontend-Dependencies OK) else (echo ❌ Frontend-Dependencies FEHLER)

echo.
echo Konfiguration:
if exist ".env" (echo ✅ .env OK) else (echo ❌ .env FEHLER)
if exist ".env.example" (echo ✅ .env.example OK) else (echo ❌ .env.example FEHLER)

echo.
echo Import-Fix Status:
python -c "import sys; sys.path.insert(0, 'backend'); from k_assistant_main_v7_wolfram import KAssistant; print('✅ Import-Problem behoben')" 2>nul || echo ❌ Import-Problem besteht

echo.
pause
goto MAIN_MENU

:OPEN_DOCS
cls
echo ================================================================
echo                   Dokumentation
echo ================================================================

echo Verfuegbare Dokumentation:
echo.
echo [1] WOLFRAM_INSTALLATION.md
echo [2] WOLFRAM_FEATURES.md  
echo [3] README.md
echo [4] Wolfram Developer Portal (online)
echo [5] Zurueck
echo.
set /p doc="Waehlen Sie (1-5): "

if "%doc%"=="1" start notepad "WOLFRAM_INSTALLATION.md" 2>nul
if "%doc%"=="2" start notepad "WOLFRAM_FEATURES.md" 2>nul
if "%doc%"=="3" start notepad "README.md" 2>nul
if "%doc%"=="4" start https://developer.wolframalpha.com/portal/myapps/
if "%doc%"=="5" goto MAIN_MENU

if not "%doc%"=="5" (
    echo Dokumentation geoeffnet
    timeout /t 2 /nobreak >nul
)
goto MAIN_MENU

:POST_EXECUTION
echo.
echo System beendet
echo [1] Neu starten  [2] Web-Interface  [3] Hauptmenue  [4] Beenden
set /p post="Waehlen Sie (1-4): "

if "%post%"=="1" goto START_WOLFRAM
if "%post%"=="2" goto START_WEB
if "%post%"=="3" goto MAIN_MENU
if "%post%"=="4" goto EXIT
goto MAIN_MENU

:EXIT
cls
echo.
echo ================================================================
echo        Vielen Dank fuer die Nutzung der HAK-GAL Suite v2.0!    
echo ================================================================
echo          Web-Interface: http://localhost:3000                   
echo          Wolfram Alpha Integration - Hybrid AI Reasoning        
echo          Import-Probleme behoben - Vollständig einsatzbereit   
echo          Neue Features: test_wolfram, wolfram_stats, Oracle-API 
echo ================================================================
echo.
timeout /t 3 /nobreak >nul
exit /b 0
