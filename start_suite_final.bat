@echo off
setlocal enabledelayedexpansion

REM ==============================================================================
REM HAK-GAL Suite - Path-Fixed Version
REM ==============================================================================

title HAK-GAL Suite
color 0A

REM Ins richtige Verzeichnis wechseln (wo die .bat liegt)
cd /d "%~dp0"

cls
echo.
echo ================================================================
echo                HAK-GAL Suite - Smart Launcher                
echo ================================================================
echo     Hybrid AI Framework + Wolfram Alpha Integration          
echo     Automatisches Setup und intelligente System-Auswahl      
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
echo    [1] Wolfram-System starten (Empfohlen)
echo    [2] Standard-System starten
echo    [3] Web-Interface starten
echo    [4] Automatisches Setup
echo    [5] Tests ausfuehren
echo    [6] Demo starten
echo    [7] System-Status
echo    [8] Dokumentation
echo    [9] Beenden
echo.
set /p choice="Waehlen Sie eine Option (1-9): "

if "%choice%"=="1" goto START_WOLFRAM
if "%choice%"=="2" goto START_STANDARD  
if "%choice%"=="3" goto START_WEB
if "%choice%"=="4" goto RUN_SETUP
if "%choice%"=="5" goto RUN_TESTS
if "%choice%"=="6" goto RUN_DEMO
if "%choice%"=="7" goto CHECK_STATUS
if "%choice%"=="8" goto OPEN_DOCS
if "%choice%"=="9" goto EXIT

echo FEHLER: Ungueltige Auswahl. Bitte waehlen Sie 1-9.
timeout /t 2 /nobreak >nul
goto MAIN_MENU

:START_WOLFRAM
cls
echo ================================================================
echo          Starte HAK-GAL mit Wolfram Integration
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
        echo Verzeichnis: %CD%
        echo Suche nach .env.example...
        dir .env* /b
        pause
        goto MAIN_MENU
    )
) else (
    echo .env Datei bereits vorhanden
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
    echo wolframalpha erfolgreich installiert
) else (
    echo wolframalpha bereits verfuegbar
)

echo.
echo Verfuegbare Features:
echo - Wolfram Alpha Oracle fuer Realwelt-Wissen
echo - Mathematische Berechnungen
echo - Geografische Daten
echo - Wetter und Realzeit-Informationen
echo - Intelligente Oracle-Erkennung
echo.
echo Starte System...
python "%WOLFRAM_SYSTEM%"
goto POST_EXECUTION

:START_STANDARD
cls
echo ================================================================
echo             Starte HAK-GAL Standard-System
echo ================================================================

if not exist "%STANDARD_SYSTEM%" (
    echo FEHLER: Standard-System nicht gefunden!
    echo Datei fehlt: %CD%\%STANDARD_SYSTEM%
    pause
    goto MAIN_MENU
)

echo.
echo Verfuegbare Features:
echo - Z3 SMT Solver fuer formale Logik
echo - Pattern Matching
echo - RAG-basierte Wissensbasis
echo - Multi-LLM Ensemble
echo.
echo Starte System...
python "%STANDARD_SYSTEM%"
goto POST_EXECUTION

:START_WEB
cls
echo ================================================================
echo              Starte HAK-GAL Web-Interface
echo ================================================================

echo Backend Port: 5001
echo Frontend Port: 3000
echo.

echo Starte Backend...
start "HAK-GAL Backend" cmd /k "cd /d "%CD%" && python %WEB_API%"

echo Warte 5 Sekunden...
timeout /t 5 /nobreak >nul

echo Starte Frontend...
if exist "frontend" (
    cd frontend
    start "HAK-GAL Frontend" cmd /k "npm run dev"
    cd ..
    echo Frontend gestartet
) else (
    echo WARNUNG: Frontend-Verzeichnis nicht gefunden
)

echo.
echo Web-Interface gestartet!
echo Backend:  http://localhost:5001
echo Frontend: http://localhost:3000
echo.
pause
goto MAIN_MENU

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
echo Pruefe Kern-Abhaengigkeiten...
set "MISSING="

python -c "import z3" 2>nul || set "MISSING=!MISSING! z3-solver"
python -c "import lark" 2>nul || set "MISSING=!MISSING! lark"
python -c "import openai" 2>nul || set "MISSING=!MISSING! openai"
python -c "from dotenv import load_dotenv" 2>nul || set "MISSING=!MISSING! python-dotenv"
python -c "import wolframalpha" 2>nul || set "MISSING=!MISSING! wolframalpha"

if not "!MISSING!"=="" (
    echo Fehlende Pakete:!MISSING!
    echo.
    set /p install="Installieren? (j/n): "
    if /i "!install!"=="j" (
        echo Installiere Pakete...
        pip install!MISSING!
        if errorlevel 1 (
            echo FEHLER: Installation fehlgeschlagen!
            pause
            goto MAIN_MENU
        )
        echo Pakete installiert
    )
) else (
    echo Alle Abhaengigkeiten verfuegbar
)

echo.
echo Pruefe .env Konfiguration...
echo Suche .env.example in: %CD%
dir .env* /b 2>nul

if not exist ".env" (
    if exist ".env.example" (
        echo Kopiere .env.example zu .env...
        copy ".env.example" ".env" >nul
        if exist ".env" (
            echo .env Datei erfolgreich erstellt
        ) else (
            echo FEHLER: Kopieren fehlgeschlagen
            pause
            goto MAIN_MENU
        )
    ) else (
        echo ERSTELLE .env.example da sie fehlt...
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
        
        echo .env.example erstellt, kopiere zu .env...
        copy ".env.example" ".env" >nul
        echo .env Datei erstellt
    )
) else (
    echo .env Datei bereits vorhanden
)

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
        echo Fallback: Manuelle Aktualisierung...
        echo WOLFRAM_APP_ID=!app_id! >> .env
    )
    echo Wolfram App ID gespeichert
)

echo.
echo Setup abgeschlossen!
echo Sie koennen jetzt Option 1 verwenden
pause
goto MAIN_MENU

:RUN_TESTS
cls
echo ================================================================
echo                    Tests ausfuehren
echo ================================================================

if exist "test_wolfram_integration.py" (
    echo Fuehre Test-Suite durch...
    python "test_wolfram_integration.py"
) else (
    echo Test-Script nicht gefunden - fuehre Basis-Tests durch...
    echo.
    echo Python-Version:
    python --version
    
    echo.
    echo Teste Importe...
    python -c "import sys; sys.path.insert(0, 'backend'); from k_assistant_main_v7_wolfram import KAssistant; print('Wolfram-System OK')" 2>nul || (
        python -c "import sys; sys.path.insert(0, 'backend'); from k_assistant_main import KAssistant; print('Standard-System OK')" 2>nul || (
            echo FEHLER: Kein HAK-GAL System importierbar
        )
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

if exist "demo_wolfram_integration.py" (
    echo Starte Demo...
    python "demo_wolfram_integration.py"
) else (
    echo Demo-Script nicht gefunden - starte manuellen Test...
    echo.
    echo Probieren Sie diese Befehle:
    echo   ask was ist die hauptstadt von deutschland
    echo   ask_raw HauptstadtVon(Deutschland, x).
    echo   status
    echo   exit
    echo.
    
    if exist "%WOLFRAM_SYSTEM%" (
        python "%WOLFRAM_SYSTEM%"
    ) else if exist "%STANDARD_SYSTEM%" (
        python "%STANDARD_SYSTEM%"
    ) else (
        echo FEHLER: Kein System gefunden
        pause
        goto MAIN_MENU
    )
)

echo.
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
python --version 2>nul && echo Python OK || echo Python FEHLER

echo.
echo Kern-Abhaengigkeiten:
python -c "import z3; print('z3-solver OK')" 2>nul || echo z3-solver FEHLER
python -c "import lark; print('lark OK')" 2>nul || echo lark FEHLER
python -c "import openai; print('openai OK')" 2>nul || echo openai FEHLER
python -c "from dotenv import load_dotenv; print('python-dotenv OK')" 2>nul || echo python-dotenv FEHLER

echo.
echo Wolfram-Integration:
python -c "import wolframalpha; print('wolframalpha OK')" 2>nul || echo wolframalpha FEHLER

echo.
echo Systemdateien:
if exist "%WOLFRAM_SYSTEM%" (echo Wolfram-System OK) else (echo Wolfram-System FEHLER)
if exist "%STANDARD_SYSTEM%" (echo Standard-System OK) else (echo Standard-System FEHLER)
if exist "%WEB_API%" (echo Web-API OK) else (echo Web-API FEHLER)

echo.
echo Konfiguration:
if exist ".env" (echo .env OK) else (echo .env FEHLER)
if exist ".env.example" (echo .env.example OK) else (echo .env.example FEHLER)

echo.
echo Verfuegbare Dateien:
dir *.py /b 2>nul
dir *.md /b 2>nul

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
echo [1] Neu starten  [2] Status  [3] Hauptmenue  [4] Beenden
set /p post="Waehlen Sie (1-4): "

if "%post%"=="1" goto START_WOLFRAM
if "%post%"=="2" goto CHECK_STATUS
if "%post%"=="3" goto MAIN_MENU
if "%post%"=="4" goto EXIT
goto MAIN_MENU

:EXIT
cls
echo.
echo ================================================================
echo        Vielen Dank fuer die Nutzung der HAK-GAL Suite!        
echo ================================================================
echo          Wolfram Alpha Integration - Hybrid AI Reasoning          
echo ================================================================
echo.
timeout /t 3 /nobreak >nul
exit /b 0
