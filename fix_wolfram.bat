@echo off
echo ========================================
echo 🔧 HAK-GAL WOLFRAM FIX
echo ========================================
echo.

echo 1. Installing wolframalpha package...
pip install wolframalpha
if errorlevel 1 (
    echo ❌ Installation failed! 
    echo    Try: python -m pip install wolframalpha
    pause
    exit /b 1
)

echo.
echo ✅ wolframalpha installed successfully!
echo.

echo 2. Verifying installation...
python -c "import wolframalpha; print('✅ Import successful!')"
if errorlevel 1 (
    echo ❌ Import test failed!
    pause
    exit /b 1
)

echo.
echo 3. Checking Wolfram integration status...
cd /d "%~dp0"
python wolfram_check.py

echo.
echo ========================================
echo ✅ WOLFRAM FIX COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. Restart your HAK-GAL backend
echo 2. You should see "Loading: YES" 
echo 3. Wolfram commands will be available
echo.
pause
