@echo off
echo =====================================
echo HAK-GAL Frontend Recovery Script
echo =====================================
echo.
echo [INFO] Repariere Frontend nach depseek Schaden...
echo [INFO] 1. vite.config.ts repariert
echo [INFO] 2. package.json wiederhergestellt  
echo [INFO] 3. Installiere Abhängigkeiten...
echo.

cd /d "%~dp0frontend"

echo [1/3] Lösche defekte node_modules...
if exist node_modules rmdir /s /q node_modules
if exist package-lock.json del package-lock.json

echo [2/3] Installiere alle Dependencies neu...
npm install

echo [3/3] Versuche Frontend zu starten...
echo.
echo =====================================
echo Recovery abgeschlossen!
echo =====================================
echo.
echo Falls weitere Probleme auftreten:
echo 1. npm run build (test build)
echo 2. npm run dev (start development)
echo.
pause
