@echo off
setlocal enabledelayedexpansion

REM ==============================================================================
REM HAK-GAL Suite - Intelligentes Backup System v1.0
REM ==============================================================================

title HAK-GAL Suite - Backup Manager
color 0B

REM Ins richtige Verzeichnis wechseln (wo die .bat liegt)
cd /d "%~dp0"

cls
echo.
echo ================================================================
echo            HAK-GAL Suite - Backup Manager v1.0
echo ================================================================
echo     Intelligentes Backup-System für Ihr Projekt
echo     Sichert Code, Konfiguration und Wissensbasis
echo ================================================================
echo.
echo Projekt-Verzeichnis: %CD%
echo.

REM Variablen definieren
set "PROJECT_NAME=HAK_GAL_SUITE"
set "SOURCE_DIR=%CD%"
set "BACKUP_BASE_DIR=D:\Backups\%PROJECT_NAME%"
set "TIMESTAMP=%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=!TIMESTAMP: =0!"

REM Backup-Verzeichnis erstellen
if not exist "%BACKUP_BASE_DIR%" (
    mkdir "%BACKUP_BASE_DIR%"
    echo ✅ Backup-Verzeichnis erstellt: %BACKUP_BASE_DIR%
)

:MAIN_MENU
echo Backup-Optionen:
echo.
echo    [1] Vollständiges Backup (Alles inklusive node_modules)
echo    [2] Entwickler-Backup (Ohne node_modules, temp-Dateien)
echo    [3] Code-Only Backup (Nur Python + Frontend Code)
echo    [4] Wissensbasis-Backup (Nur .kb, .env, Daten)
echo    [5] Komprimiertes Backup (mit 7zip)
echo    [6] Backup-Status anzeigen
echo    [7] Alte Backups verwalten
echo    [8] Backup wiederherstellen
echo    [9] Beenden
echo.
set /p choice="Wählen Sie eine Option (1-9): "

if "%choice%"=="1" goto FULL_BACKUP
if "%choice%"=="2" goto DEV_BACKUP
if "%choice%"=="3" goto CODE_BACKUP
if "%choice%"=="4" goto DATA_BACKUP
if "%choice%"=="5" goto COMPRESSED_BACKUP
if "%choice%"=="6" goto SHOW_STATUS
if "%choice%"=="7" goto MANAGE_BACKUPS
if "%choice%"=="8" goto RESTORE_BACKUP
if "%choice%"=="9" goto EXIT

echo FEHLER: Ungültige Auswahl. Bitte wählen Sie 1-9.
timeout /t 2 /nobreak >nul
goto MAIN_MENU

:FULL_BACKUP
cls
echo ================================================================
echo              Vollständiges Backup wird erstellt
echo ================================================================

set "BACKUP_DIR=%BACKUP_BASE_DIR%\FULL_%TIMESTAMP%"
mkdir "%BACKUP_DIR%"

echo Kopiere alle Dateien...
echo Quelle: %SOURCE_DIR%
echo Ziel:   %BACKUP_DIR%
echo.

robocopy "%SOURCE_DIR%" "%BACKUP_DIR%" /E /R:3 /W:10 /MT:8 /ETA /TEE

if %errorlevel% leq 3 (
    echo.
    echo ✅ Vollständiges Backup erfolgreich erstellt!
    echo 📁 Speicherort: %BACKUP_DIR%
    call :GET_FOLDER_SIZE "%BACKUP_DIR%"
) else (
    echo ❌ Backup fehlgeschlagen! (Fehlercode: %errorlevel%)
)

echo.
pause
goto MAIN_MENU

:DEV_BACKUP
cls
echo ================================================================
echo              Entwickler-Backup wird erstellt
echo ================================================================

set "BACKUP_DIR=%BACKUP_BASE_DIR%\DEV_%TIMESTAMP%"
mkdir "%BACKUP_DIR%"

echo Kopiere Entwicklungsdateien (ohne temporäre Dateien)...
echo Quelle: %SOURCE_DIR%
echo Ziel:   %BACKUP_DIR%
echo.

REM Erweiterte Ausschlüsse für Entwickler-Backup
robocopy "%SOURCE_DIR%" "%BACKUP_DIR%" /E /R:3 /W:10 /MT:8 /ETA /TEE ^
    /XD node_modules __pycache__ .git .next dist build coverage .nyc_output .cache .parcel-cache ^
    /XF *.log *.tmp *.temp npm-debug.log* yarn-debug.log* yarn-error.log* *.pid *.seed *.pid.lock ^
        *.pyc *.pyo *.pyd .Python *.so *.egg *.egg-info

if %errorlevel% leq 3 (
    echo.
    echo ✅ Entwickler-Backup erfolgreich erstellt!
    echo 📁 Speicherort: %BACKUP_DIR%
    call :GET_FOLDER_SIZE "%BACKUP_DIR%"
) else (
    echo ❌ Backup fehlgeschlagen! (Fehlercode: %errorlevel%)
)

echo.
pause
goto MAIN_MENU

:CODE_BACKUP
cls
echo ================================================================
echo               Code-Only Backup wird erstellt
echo ================================================================

set "BACKUP_DIR=%BACKUP_BASE_DIR%\CODE_%TIMESTAMP%"
mkdir "%BACKUP_DIR%"

echo Kopiere nur Code-Dateien...
echo.

REM Python Backend
if exist "backend" (
    robocopy "backend" "%BACKUP_DIR%\backend" *.py *.txt *.md /S /R:3 /W:10
    echo ✅ Backend Python-Code kopiert
)

REM Frontend
if exist "frontend" (
    robocopy "frontend" "%BACKUP_DIR%\frontend" *.js *.jsx *.ts *.tsx *.json *.css *.scss *.html *.md /S /R:3 /W:10 ^
        /XD node_modules .next dist build
    echo ✅ Frontend-Code kopiert
)

REM Root-Dateien
robocopy "%SOURCE_DIR%" "%BACKUP_DIR%" *.py *.js *.json *.md *.txt *.bat *.sh /R:3 /W:10
echo ✅ Root-Dateien kopiert

REM Konfigurationsdateien
if exist ".env.example" copy ".env.example" "%BACKUP_DIR%\"
if exist "package.json" copy "package.json" "%BACKUP_DIR%\"
if exist "requirements.txt" copy "requirements.txt" "%BACKUP_DIR%\"

echo.
echo ✅ Code-Only Backup erfolgreich erstellt!
echo 📁 Speicherort: %BACKUP_DIR%
call :GET_FOLDER_SIZE "%BACKUP_DIR%"

echo.
pause
goto MAIN_MENU

:DATA_BACKUP
cls
echo ================================================================
echo            Wissensbasis-Backup wird erstellt
echo ================================================================

set "BACKUP_DIR=%BACKUP_BASE_DIR%\DATA_%TIMESTAMP%"
mkdir "%BACKUP_DIR%"

echo Kopiere Wissensbasis und Konfiguration...
echo.

REM Wissensbasis-Dateien
robocopy "%SOURCE_DIR%" "%BACKUP_DIR%" *.kb *.pkl *.db *.sqlite /S /R:3 /W:10
echo ✅ Wissensbasis-Dateien kopiert

REM Konfiguration (aber NICHT die echte .env mit Keys!)
if exist ".env.example" copy ".env.example" "%BACKUP_DIR%\"
echo ✅ Konfigurationsvorlagen kopiert

REM Dokumentation
robocopy "%SOURCE_DIR%" "%BACKUP_DIR%" *.md *.txt /S /R:3 /W:10
echo ✅ Dokumentation kopiert

REM RAG-Dokumente falls vorhanden
if exist "documents" (
    robocopy "documents" "%BACKUP_DIR%\documents" /E /R:3 /W:10
    echo ✅ RAG-Dokumente kopiert
)

echo.
echo ✅ Wissensbasis-Backup erfolgreich erstellt!
echo 📁 Speicherort: %BACKUP_DIR%
call :GET_FOLDER_SIZE "%BACKUP_DIR%"

echo.
echo ⚠️ WICHTIG: Ihre .env Datei mit API-Keys wurde aus Sicherheitsgründen NICHT kopiert!
echo           Bewahren Sie diese separat und sicher auf.
echo.
pause
goto MAIN_MENU

:COMPRESSED_BACKUP
cls
echo ================================================================
echo            Komprimiertes Backup wird erstellt
echo ================================================================

REM Prüfe ob 7zip verfügbar ist
where 7z >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 7zip nicht gefunden!
    echo    Installieren Sie 7zip oder nutzen Sie ein anderes Backup
    pause
    goto MAIN_MENU
)

set "BACKUP_FILE=%BACKUP_BASE_DIR%\%PROJECT_NAME%_COMPRESSED_%TIMESTAMP%.7z"

echo Erstelle komprimiertes Backup mit 7zip...
echo Quelle: %SOURCE_DIR%
echo Ziel:   %BACKUP_FILE%
echo.

REM Temporäre Excludes-Datei erstellen
echo node_modules\ > "%TEMP%\backup_excludes.txt"
echo __pycache__\ >> "%TEMP%\backup_excludes.txt"
echo .git\ >> "%TEMP%\backup_excludes.txt"
echo *.log >> "%TEMP%\backup_excludes.txt"
echo *.tmp >> "%TEMP%\backup_excludes.txt"
echo .env >> "%TEMP%\backup_excludes.txt"

7z a -t7z "%BACKUP_FILE%" "%SOURCE_DIR%\*" -mx=7 -xr@"%TEMP%\backup_excludes.txt"

del "%TEMP%\backup_excludes.txt"

if %errorlevel% equ 0 (
    echo.
    echo ✅ Komprimiertes Backup erfolgreich erstellt!
    echo 📁 Speicherort: %BACKUP_FILE%
    call :GET_FILE_SIZE "%BACKUP_FILE%"
) else (
    echo ❌ Komprimierung fehlgeschlagen!
)

echo.
pause
goto MAIN_MENU

:SHOW_STATUS
cls
echo ================================================================
echo                    Backup-Status
echo ================================================================
echo Backup-Verzeichnis: %BACKUP_BASE_DIR%
echo.

if not exist "%BACKUP_BASE_DIR%" (
    echo ℹ️ Noch keine Backups vorhanden
    echo.
    pause
    goto MAIN_MENU
)

echo Verfügbare Backups:
echo.
echo Typ      Datum/Zeit           Größe
echo ----------------------------------------

for /f "delims=" %%i in ('dir /b /ad "%BACKUP_BASE_DIR%\*" 2^>nul') do (
    set "backup_name=%%i"
    set "backup_type=!backup_name:~0,4!"
    set "backup_date=!backup_name:~5,8!"
    set "backup_time=!backup_name:~14,6!"
    
    REM Formatiere Datum und Zeit
    set "formatted_date=!backup_date:~0,4!-!backup_date:~4,2!-!backup_date:~6,2!"
    set "formatted_time=!backup_time:~0,2!:!backup_time:~2,2!:!backup_time:~4,2!"
    
    call :GET_FOLDER_SIZE_COMPACT "%BACKUP_BASE_DIR%\%%i"
    echo !backup_type!     !formatted_date! !formatted_time!   !folder_size_compact!
)

REM Zeige auch komprimierte Backups
for /f "delims=" %%i in ('dir /b "%BACKUP_BASE_DIR%\*.7z" 2^>nul') do (
    echo COMP     %%i
)

echo.
pause
goto MAIN_MENU

:MANAGE_BACKUPS
cls
echo ================================================================
echo                 Backup-Verwaltung
echo ================================================================

echo [1] Alte Backups löschen (älter als 30 Tage)
echo [2] Bestimmtes Backup löschen
echo [3] Backup-Größen analysieren
echo [4] Zurück zum Hauptmenü
echo.
set /p mgmt_choice="Wählen Sie (1-4): "

if "%mgmt_choice%"=="1" goto DELETE_OLD_BACKUPS
if "%mgmt_choice%"=="2" goto DELETE_SPECIFIC_BACKUP
if "%mgmt_choice%"=="3" goto ANALYZE_SIZES
if "%mgmt_choice%"=="4" goto MAIN_MENU

echo Ungültige Auswahl
timeout /t 2 /nobreak >nul
goto MANAGE_BACKUPS

:DELETE_OLD_BACKUPS
echo Lösche Backups älter als 30 Tage...
forfiles /p "%BACKUP_BASE_DIR%" /m *.* /d -30 /c "cmd /c if @isdir==TRUE rmdir /s /q @path & echo Gelöscht: @file"
echo.
echo ✅ Alte Backups bereinigt
pause
goto MANAGE_BACKUPS

:RESTORE_BACKUP
cls
echo ================================================================
echo                 Backup wiederherstellen
echo ================================================================
echo.
echo ⚠️ WARNUNG: Dies überschreibt das aktuelle Projekt!
echo           Erstellen Sie vorher ein Backup des aktuellen Zustands!
echo.
set /p confirm="Möchten Sie fortfahren? (j/N): "
if /i not "%confirm%"=="j" goto MAIN_MENU

echo.
echo Verfügbare Backups:
echo.
dir /b /ad "%BACKUP_BASE_DIR%\*" 2>nul

echo.
set /p backup_name="Geben Sie den Backup-Namen ein: "
set "restore_path=%BACKUP_BASE_DIR%\%backup_name%"

if not exist "%restore_path%" (
    echo ❌ Backup nicht gefunden!
    pause
    goto MAIN_MENU
)

echo.
echo Stelle Backup wieder her...
echo Von: %restore_path%
echo Nach: %SOURCE_DIR%
echo.

robocopy "%restore_path%" "%SOURCE_DIR%" /E /R:3 /W:10 /MT:8

if %errorlevel% leq 3 (
    echo ✅ Backup erfolgreich wiederhergestellt!
) else (
    echo ❌ Wiederherstellung fehlgeschlagen!
)

pause
goto MAIN_MENU

:GET_FOLDER_SIZE
set "folder_path=%~1"
set "folder_size=0"
for /f "tokens=3" %%a in ('dir "%folder_path%" /s /-c ^| find "Datei(en)"') do set "folder_size=%%a"
set "folder_size=!folder_size: =!"
echo 📊 Backup-Größe: !folder_size! Bytes
exit /b

:GET_FOLDER_SIZE_COMPACT
set "folder_path=%~1"
for /f "tokens=3" %%a in ('dir "%folder_path%" /s /-c ^| find "Datei(en)"') do (
    set "size_bytes=%%a"
    set "size_bytes=!size_bytes: =!"
    if !size_bytes! gtr 1073741824 (
        set /a "size_gb=!size_bytes!/1073741824"
        set "folder_size_compact=!size_gb! GB"
    ) else if !size_bytes! gtr 1048576 (
        set /a "size_mb=!size_bytes!/1048576"
        set "folder_size_compact=!size_mb! MB"
    ) else (
        set /a "size_kb=!size_bytes!/1024"
        set "folder_size_compact=!size_kb! KB"
    )
)
exit /b

:GET_FILE_SIZE
set "file_path=%~1"
for %%F in ("%file_path%") do set "file_size=%%~zF"
echo 📊 Komprimierte Größe: !file_size! Bytes
exit /b

:ANALYZE_SIZES
cls
echo ================================================================
echo                 Backup-Größen Analyse
echo ================================================================
echo.

set "total_size=0"
for /f "delims=" %%i in ('dir /b /ad "%BACKUP_BASE_DIR%\*" 2^>nul') do (
    echo Analysiere: %%i
    call :GET_FOLDER_SIZE_COMPACT "%BACKUP_BASE_DIR%\%%i"
    echo   Größe: !folder_size_compact!
    echo.
)

echo Empfehlungen:
echo - DEV-Backups sind meist ausreichend für Entwicklung
echo - CODE-Backups für reine Code-Sicherung
echo - COMPRESSED-Backups sparen Speicherplatz
echo - Löschen Sie regelmäßig alte Backups

pause
goto MANAGE_BACKUPS

:EXIT
cls
echo.
echo ================================================================
echo     Vielen Dank für die Nutzung des HAK-GAL Backup Systems!
echo ================================================================
echo          Ihre Daten sind sicher gesichert
echo          Backup-Verzeichnis: %BACKUP_BASE_DIR%
echo ================================================================
echo.
timeout /t 3 /nobreak >nul
exit /b 0
