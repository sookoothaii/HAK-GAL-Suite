@echo off
title HAK-GAL Suite - Wolfram Web-Interface
color 0A
cd /d "%~dp0"

echo ================================================================
echo     HAK-GAL Suite - Wolfram Web-Interface Starter
echo ================================================================
echo     Backend mit Wolfram-Integration wird gestartet...
echo ================================================================

:: Backend-Prozess beenden falls läuft
taskkill /F /IM python.exe /FI "WINDOWTITLE eq HAK-GAL Backend" >nul 2>&1

echo [1/3] Starte Backend mit Wolfram-Integration...
start "HAK-GAL Backend" cmd /k "cd /d "%CD%" && echo === HAK-GAL Backend mit Wolfram-Integration === && python api.py"

echo [2/3] Warte auf Backend-Start...
timeout /t 8 /nobreak >nul

echo [3/3] Starte Frontend...
if exist "frontend" (
    cd frontend
    start "HAK-GAL Frontend" cmd /k "echo === HAK-GAL Frontend === && npm run dev"
    cd ..
) else (
    echo FEHLER: Frontend-Verzeichnis nicht gefunden!
    pause
    exit /b 1
)

echo.
echo ================================================================
echo         WOLFRAM-INTEGRATION ERFOLGREICH GESTARTET!
echo ================================================================
echo.
echo  Backend (Wolfram):  http://localhost:5001
echo  Frontend:           http://localhost:3000
echo.
echo  Jetzt verfügbar im Web-Interface:
echo  - wolfram_stats     (Wolfram Cache-Statistiken)
echo  - add_oracle        (Oracle-Prädikate hinzufügen)
echo  - ask mit Wolfram   (Realwelt-Wissen)
echo.

set /p open_browser="Browser automatisch öffnen? (j/n): "
if /i "%open_browser%"=="j" (
    start http://localhost:3000
)

echo.
echo System läuft... Drücken Sie eine Taste zum Beenden.
pause

:: Cleanup
taskkill /F /IM node.exe /FI "WINDOWTITLE eq HAK-GAL Frontend" >nul 2>&1
taskkill /F /IM python.exe /FI "WINDOWTITLE eq HAK-GAL Backend" >nul 2>&1
