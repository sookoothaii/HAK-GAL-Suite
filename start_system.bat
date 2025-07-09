@echo off
:: HAK-GAL Suite - Wissenschaftlich validierter Launcher
:: IMPORT-PROBLEM BEHOBEN - Sollte jetzt funktionieren

title HAK-GAL Suite - Fixed Import Launcher
color 0A

:: Ins richtige Verzeichnis wechseln (kritisch!)
cd /d "%~dp0"

echo ================================================================
echo   HAK-GAL Suite - IMPORT-PROBLEM BEHOBEN                      
echo ================================================================
echo   Aktuelles Verzeichnis: %CD%
echo   Wolfram-System wird gestartet...
echo ================================================================
echo.

:: Direkt das Wolfram-System starten
if exist "backend\k_assistant_main_v7_wolfram.py" (
    echo Starte Wolfram-integriertes HAK-GAL System...
    python backend\k_assistant_main_v7_wolfram.py
) else (
    echo FEHLER: Wolfram-System nicht gefunden!
    echo Pfad: %CD%\backend\k_assistant_main_v7_wolfram.py
    pause
)

echo.
echo System beendet. Druecken Sie eine Taste...
pause > nul
