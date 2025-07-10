@echo off
echo ================================================================
echo         🔧 QUICK-FIX: Wolfram Integration aktivieren
echo ================================================================
echo.

REM Ins HAK-GAL Verzeichnis wechseln
cd /d "%~dp0"

echo 📦 Installiere wolframalpha Bibliothek...
pip install wolframalpha

if errorlevel 1 (
    echo.
    echo ❌ Installation fehlgeschlagen!
    echo.
    echo Versuche Alternative:
    python -m pip install wolframalpha
    
    if errorlevel 1 (
        echo.
        echo ❌ Auch die Alternative fehlgeschlagen!
        echo.
        echo Mögliche Lösungen:
        echo 1. Als Administrator ausführen
        echo 2. pip aktualisieren: python -m pip install --upgrade pip
        echo 3. Mit --user flag: pip install --user wolframalpha
        pause
        exit /b 1
    )
)

echo.
echo ✅ wolframalpha erfolgreich installiert!
echo.

REM Teste die Installation
echo 🧪 Teste Installation...
python -c "import wolframalpha; print('✅ Import erfolgreich!')"

if errorlevel 1 (
    echo ❌ Import-Test fehlgeschlagen!
    pause
    exit /b 1
)

echo.
echo ================================================================
echo            ✅ WOLFRAM-FIX ERFOLGREICH!
echo ================================================================
echo.
echo Was passiert jetzt:
echo.
echo 1. Starten Sie HAK-GAL neu (start_suite_webui.bat)
echo 2. Sie sollten jetzt "Loading: YES" sehen
echo 3. Wolfram-Befehle sind verfügbar:
echo    • wolfram_stats
echo    • ask was ist die hauptstadt von deutschland
echo    • ask_raw HauptstadtVon(Deutschland).
echo.
echo ================================================================
echo.
pause