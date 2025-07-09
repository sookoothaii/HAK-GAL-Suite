@echo off
setlocal enabledelayedexpansion

REM ==============================================================================
REM HAK-GAL Suite - Git Cleanup Manager v1.0
REM ==============================================================================

title HAK-GAL Suite - Git Cleanup Manager
color 0E

REM Ins richtige Verzeichnis wechseln (wo die .bat liegt)
cd /d "%~dp0"

cls
echo.
echo ================================================================
echo          HAK-GAL Suite - Git Cleanup Manager v1.0
echo ================================================================
echo     Bereitet Ihr Projekt für Git/GitHub vor
echo     Entfernt temporäre Dateien, Caches und Builds
echo ================================================================
echo.
echo Projekt-Verzeichnis: %CD%
echo GitHub Repository: https://github.com/sookoothaii/HAK-GAL-Suite
echo.

REM Git-Status prüfen
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ Git ist nicht installiert oder nicht im PATH
    echo.
)

:MAIN_MENU
echo Git-Cleanup Optionen:
echo.
echo    [1] Schnell-Cleanup (Empfohlen für Git)
echo    [2] Vollständiger Cleanup (Alles entfernen)
echo    [3] Entwickler-Cleanup (Behält Caches für lokale Entwicklung)
echo    [4] Nur temporäre Dateien
echo    [5] Nur Node.js/Frontend Cleanup  
echo    [6] Nur Python Cleanup
echo    [7] Git-Status anzeigen
echo    [8] .gitignore prüfen/reparieren
echo    [9] Dateien auflisten (Was würde gelöscht)
echo    [0] Beenden
echo.
set /p choice="Wählen Sie eine Option (0-9): "

if "%choice%"=="1" goto QUICK_CLEANUP
if "%choice%"=="2" goto FULL_CLEANUP
if "%choice%"=="3" goto DEV_CLEANUP
if "%choice%"=="4" goto TEMP_CLEANUP
if "%choice%"=="5" goto NODE_CLEANUP
if "%choice%"=="6" goto PYTHON_CLEANUP
if "%choice%"=="7" goto GIT_STATUS
if "%choice%"=="8" goto CHECK_GITIGNORE
if "%choice%"=="9" goto LIST_FILES
if "%choice%"=="0" goto EXIT

echo FEHLER: Ungültige Auswahl. Bitte wählen Sie 0-9.
timeout /t 2 /nobreak >nul
goto MAIN_MENU

:QUICK_CLEANUP
cls
echo ================================================================
echo              Schnell-Cleanup für Git/GitHub
echo ================================================================
echo.
echo Dies entfernt die häufigsten temporären Dateien für Git-Commits
echo.
set /p confirm="Möchten Sie fortfahren? (j/N): "
if /i not "%confirm%"=="j" goto MAIN_MENU

echo.
echo 🧹 Beginne Schnell-Cleanup...
echo.

REM Python Cleanup
call :CLEANUP_PYTHON_BASIC

REM Node.js Cleanup  
call :CLEANUP_NODE_BASIC

REM Temporäre Dateien
call :CLEANUP_TEMP_BASIC

REM Log-Dateien
call :CLEANUP_LOGS

echo.
echo ✅ Schnell-Cleanup abgeschlossen!
echo 📁 Projekt ist bereit für Git-Commit
pause
goto MAIN_MENU

:FULL_CLEANUP
cls
echo ================================================================
echo                 Vollständiger Cleanup
echo ================================================================
echo.
echo ⚠️ WARNUNG: Dies entfernt ALLE temporären Dateien und Caches!
echo           Sie müssen danach npm install und pip install ausführen.
echo.
set /p confirm="Sind Sie sicher? (j/N): "
if /i not "%confirm%"=="j" goto MAIN_MENU

echo.
echo 🧹 Beginne vollständigen Cleanup...
echo.

call :CLEANUP_PYTHON_FULL
call :CLEANUP_NODE_FULL
call :CLEANUP_TEMP_FULL
call :CLEANUP_LOGS
call :CLEANUP_BUILD
call :CLEANUP_CACHE

echo.
echo ✅ Vollständiger Cleanup abgeschlossen!
echo 📋 Nächste Schritte:
echo    1. npm install (im frontend Ordner)
echo    2. pip install -r requirements.txt
pause
goto MAIN_MENU

:DEV_CLEANUP
cls
echo ================================================================
echo              Entwickler-Cleanup
echo ================================================================
echo.
echo Entfernt nur störende Dateien, behält nützliche Caches
echo.

call :CLEANUP_TEMP_BASIC
call :CLEANUP_LOGS

echo.
echo ✅ Entwickler-Cleanup abgeschlossen!
pause
goto MAIN_MENU

:TEMP_CLEANUP
cls
echo ================================================================
echo               Temporäre Dateien Cleanup
echo ================================================================

call :CLEANUP_TEMP_FULL

echo ✅ Temporäre Dateien entfernt!
pause
goto MAIN_MENU

:NODE_CLEANUP
cls
echo ================================================================
echo                Node.js/Frontend Cleanup
echo ================================================================

call :CLEANUP_NODE_FULL

echo ✅ Node.js Cleanup abgeschlossen!
pause
goto MAIN_MENU

:PYTHON_CLEANUP
cls
echo ================================================================
echo                  Python Cleanup
echo ================================================================

call :CLEANUP_PYTHON_FULL

echo ✅ Python Cleanup abgeschlossen!
pause
goto MAIN_MENU

:GIT_STATUS
cls
echo ================================================================
echo                    Git-Status
echo ================================================================

git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Git ist nicht verfügbar
    pause
    goto MAIN_MENU
)

echo Git Repository Status:
echo.
git status

echo.
echo Git Remote:
git remote -v

echo.
echo Letzte Commits:
git log --oneline -5

echo.
pause
goto MAIN_MENU

:CHECK_GITIGNORE
cls
echo ================================================================
echo                .gitignore Prüfung
echo ================================================================

if not exist ".gitignore" (
    echo ❌ .gitignore Datei nicht gefunden!
    echo.
    set /p create="Möchten Sie eine .gitignore erstellen? (j/N): "
    if /i "!create!"=="j" (
        call :CREATE_GITIGNORE
        echo ✅ .gitignore erstellt!
    )
) else (
    echo ✅ .gitignore gefunden
    echo.
    echo Prüfe wichtige Einträge...
    
    findstr /i "node_modules" .gitignore >nul || echo ⚠️ node_modules/ fehlt in .gitignore
    findstr /i "__pycache__" .gitignore >nul || echo ⚠️ __pycache__/ fehlt in .gitignore  
    findstr /i ".env" .gitignore >nul || echo ⚠️ .env fehlt in .gitignore
    findstr /i "*.log" .gitignore >nul || echo ⚠️ *.log fehlt in .gitignore
    
    echo.
    echo Aktueller Inhalt:
    type .gitignore
)

echo.
pause
goto MAIN_MENU

:LIST_FILES
cls
echo ================================================================
echo            Dateien die gelöscht würden
echo ================================================================
echo.
echo 📁 Python Cache-Dateien:
if exist "__pycache__" echo    __pycache__/
for /r %%i in (*.pyc *.pyo *.pyd) do echo    %%i

echo.
echo 📁 Node.js Dateien:
if exist "node_modules" echo    node_modules/
if exist "frontend\node_modules" echo    frontend\node_modules/
if exist "frontend\.next" echo    frontend\.next/
if exist "frontend\dist" echo    frontend\dist/

echo.
echo 📁 Temporäre Dateien:
for /r %%i in (*.tmp *.temp *.log) do echo    %%i

echo.
echo 📁 Build-Artefakte:
if exist "build" echo    build/
if exist "dist" echo    dist/

echo.
pause
goto MAIN_MENU

REM ===== CLEANUP FUNKTIONEN =====

:CLEANUP_PYTHON_BASIC
echo 🐍 Python Basis-Cleanup...
if exist "__pycache__" (
    rmdir /s /q "__pycache__" >nul 2>&1
    echo    ✅ __pycache__ entfernt
)
for /r %%i in (*.pyc) do (
    del "%%i" >nul 2>&1
    echo    ✅ %%i entfernt
)
exit /b

:CLEANUP_PYTHON_FULL
echo 🐍 Python Vollständig-Cleanup...
if exist "__pycache__" rmdir /s /q "__pycache__" >nul 2>&1
for /r %%i in (*.pyc *.pyo *.pyd) do del "%%i" >nul 2>&1
if exist ".pytest_cache" rmdir /s /q ".pytest_cache" >nul 2>&1
if exist "*.egg-info" rmdir /s /q "*.egg-info" >nul 2>&1
if exist "venv" (
    echo    ⚠️ Virtual Environment 'venv' gefunden - nicht entfernt
)
if exist "env" (
    echo    ⚠️ Virtual Environment 'env' gefunden - nicht entfernt  
)
echo    ✅ Python-Dateien bereinigt
exit /b

:CLEANUP_NODE_BASIC
echo 📦 Node.js Basis-Cleanup...
if exist "frontend\.next" (
    rmdir /s /q "frontend\.next" >nul 2>&1
    echo    ✅ .next entfernt
)
if exist "frontend\dist" (
    rmdir /s /q "frontend\dist" >nul 2>&1
    echo    ✅ dist entfernt
)
exit /b

:CLEANUP_NODE_FULL
echo 📦 Node.js Vollständig-Cleanup...
if exist "node_modules" (
    rmdir /s /q "node_modules" >nul 2>&1
    echo    ✅ node_modules entfernt
)
if exist "frontend\node_modules" (
    rmdir /s /q "frontend\node_modules" >nul 2>&1
    echo    ✅ frontend\node_modules entfernt
)
if exist "frontend\.next" rmdir /s /q "frontend\.next" >nul 2>&1
if exist "frontend\dist" rmdir /s /q "frontend\dist" >nul 2>&1
if exist "frontend\.cache" rmdir /s /q "frontend\.cache" >nul 2>&1
for %%f in (package-lock.json yarn.lock) do (
    if exist "%%f" (
        echo    ℹ️ %%f gefunden - behalten für Reproduzierbarkeit
    )
)
echo    ✅ Node.js-Dateien bereinigt
exit /b

:CLEANUP_TEMP_BASIC
echo 🗑️ Temporäre Dateien Basis-Cleanup...
for /r %%i in (*.tmp *.temp) do (
    del "%%i" >nul 2>&1
    echo    ✅ %%i entfernt
)
exit /b

:CLEANUP_TEMP_FULL
echo 🗑️ Temporäre Dateien Vollständig-Cleanup...
for /r %%i in (*.tmp *.temp *.bak *~ .DS_Store Thumbs.db) do del "%%i" >nul 2>&1
echo    ✅ Temporäre Dateien entfernt
exit /b

:CLEANUP_LOGS
echo 📄 Log-Dateien Cleanup...
for /r %%i in (*.log npm-debug.log* yarn-debug.log* yarn-error.log*) do (
    del "%%i" >nul 2>&1
    echo    ✅ %%i entfernt
)
exit /b

:CLEANUP_BUILD
echo 🔨 Build-Artefakte Cleanup...
if exist "build" rmdir /s /q "build" >nul 2>&1
if exist "dist" rmdir /s /q "dist" >nul 2>&1
echo    ✅ Build-Artefakte entfernt
exit /b

:CLEANUP_CACHE
echo 💾 Cache-Cleanup...
if exist ".cache" rmdir /s /q ".cache" >nul 2>&1
if exist ".parcel-cache" rmdir /s /q ".parcel-cache" >nul 2>&1
if exist ".nyc_output" rmdir /s /q ".nyc_output" >nul 2>&1
echo    ✅ Caches entfernt
exit /b

:CREATE_GITIGNORE
echo # HAK-GAL Suite - .gitignore > .gitignore
echo. >> .gitignore
echo # Python >> .gitignore
echo __pycache__/ >> .gitignore
echo *.py[cod] >> .gitignore
echo *$py.class >> .gitignore
echo *.so >> .gitignore
echo .Python >> .gitignore
echo venv/ >> .gitignore
echo env/ >> .gitignore
echo ENV/ >> .gitignore
echo .env >> .gitignore
echo *.egg-info/ >> .gitignore
echo .pytest_cache/ >> .gitignore
echo. >> .gitignore
echo # Node.js >> .gitignore
echo node_modules/ >> .gitignore
echo npm-debug.log* >> .gitignore
echo yarn-debug.log* >> .gitignore
echo yarn-error.log* >> .gitignore
echo .pnpm-debug.log* >> .gitignore
echo .next/ >> .gitignore
echo dist/ >> .gitignore
echo build/ >> .gitignore
echo. >> .gitignore
echo # IDE >> .gitignore
echo .vscode/ >> .gitignore
echo .idea/ >> .gitignore
echo *.swp >> .gitignore
echo *.swo >> .gitignore
echo *~ >> .gitignore
echo. >> .gitignore
echo # OS >> .gitignore
echo .DS_Store >> .gitignore
echo Thumbs.db >> .gitignore
echo. >> .gitignore
echo # Logs >> .gitignore
echo *.log >> .gitignore
echo. >> .gitignore
echo # Project specific >> .gitignore
echo *.db >> .gitignore
echo *.sqlite >> .gitignore
echo knowledge_base.pkl >> .gitignore
echo proof_cache.pkl >> .gitignore
echo learning_suggestions.pkl >> .gitignore
echo *.kb >> .gitignore
exit /b

:EXIT
cls
echo.
echo ================================================================
echo        HAK-GAL Suite ist bereit für Git/GitHub! 🚀
echo ================================================================
echo          Repository: https://github.com/sookoothaii/HAK-GAL-Suite
echo          Empfohlene nächste Schritte:
echo          1. git add .
echo          2. git commit -m "Clean project for GitHub"
echo          3. git push
echo ================================================================
echo.
timeout /t 3 /nobreak >nul
exit /b 0
