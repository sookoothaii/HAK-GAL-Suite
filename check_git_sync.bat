@echo off
echo ======================================================================
echo                  GIT-GITHUB SYNCHRONISATIONS-CHECK
echo ======================================================================
echo.

cd /d "D:\MCP Mods\HAK_GAL_SUITE"

echo 📊 GIT STATUS:
echo ----------------------------------------------------------------------
git status --short
if errorlevel 1 (
    echo ❌ Git ist nicht installiert oder kein Git-Repository!
    pause
    exit /b 1
)

echo.
echo 🌿 BRANCH INFORMATIONEN:
echo ----------------------------------------------------------------------
git branch --show-current

echo.
echo 🌐 REMOTE REPOSITORY:
echo ----------------------------------------------------------------------
git remote -v

echo.
echo 📡 VERGLEICH MIT GITHUB:
echo ----------------------------------------------------------------------
git fetch --dry-run
git status -uno

echo.
echo 📝 LETZTER COMMIT:
echo ----------------------------------------------------------------------
echo Lokal:
git log -1 --oneline

echo.
echo 🚫 NICHT COMMITTETE ÄNDERUNGEN:
echo ----------------------------------------------------------------------
git status --porcelain

echo.
echo 📁 IGNORIERTE DATEIEN (Auszug):
echo ----------------------------------------------------------------------
git ls-files --others --ignored --exclude-standard | findstr /V "node_modules" | findstr /V "__pycache__"

echo.
echo ======================================================================
echo                         CHECK ABGESCHLOSSEN
echo ======================================================================
echo.
echo 💡 BEFEHLE ZUM SYNCHRONISIEREN:
echo    git add .                    (Alle Änderungen hinzufügen)
echo    git commit -m "Nachricht"    (Commit erstellen)
echo    git push                     (Zu GitHub hochladen)
echo    git pull                     (Von GitHub herunterladen)
echo.
pause
