@echo off
echo ================================================================
echo           FUNKTIONALE CONSTRAINTS FÜR HAK-GAL
echo ================================================================
echo.
echo Diese Befehle fügen wichtige Logik-Regeln hinzu, damit das
echo System versteht, dass bestimmte Eigenschaften eindeutig sind.
echo.
echo Kopieren Sie diese Befehle in HAK-GAL:
echo ================================================================
echo.
echo add_raw all x all y all z ((Einwohner(x, y) ^& Einwohner(x, z)) -^> (y = z)).
echo add_raw all x all y all z ((Hauptstadt(x, y) ^& Hauptstadt(x, z)) -^> (y = z)).
echo add_raw all x all y all z ((Bevölkerung(x, y) ^& Bevölkerung(x, z)) -^> (y = z)).
echo add_raw all x all y all z ((Fläche(x, y) ^& Fläche(x, z)) -^> (y = z)).
echo.
echo ================================================================
echo Nach dem Hinzufügen dieser Regeln wird das System korrekt
echo erkennen, dass Rom nicht 283000 Einwohner haben kann, wenn
echo es bereits 2873000 Einwohner hat!
echo ================================================================
pause
