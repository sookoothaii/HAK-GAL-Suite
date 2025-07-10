#!/usr/bin/env python3
"""
Fix für funktionale Constraints in HAK-GAL
Stellt sicher, dass bestimmte Relationen als Funktionen behandelt werden
"""

# Diese Regel sollte zur Wissensbasis hinzugefügt werden:
FUNCTIONAL_CONSTRAINTS = [
    # Eine Stadt kann nur EINE Einwohnerzahl haben
    "all x all y all z ((Einwohner(x, y) & Einwohner(x, z)) -> (y = z)).",
    
    # Ein Land hat nur EINE Hauptstadt
    "all x all y all z ((Hauptstadt(x, y) & Hauptstadt(x, z)) -> (y = z)).",
    
    # Eine Stadt liegt nur in EINEM Land
    "all x all y all z ((LiegtIn(x, y) & LiegtIn(x, z)) -> (y = z)).",
    
    # Ein Objekt hat nur EINE Fläche
    "all x all y all z ((Fläche(x, y) & Fläche(x, z)) -> (y = z)).",
    
    # Bevölkerung ist auch funktional
    "all x all y all z ((Bevölkerung(x, y) & Bevölkerung(x, z)) -> (y = z))."
]

print("=" * 60)
print("FUNKTIONALE CONSTRAINTS FÜR HAK-GAL")
print("=" * 60)
print("\nDiese Regeln stellen sicher, dass:")
print("- Eine Stadt nur EINE Einwohnerzahl haben kann")
print("- Ein Land nur EINE Hauptstadt hat")
print("- Etc.\n")

print("FÜGEN SIE DIESE REGELN ZUR WISSENSBASIS HINZU:")
print("-" * 60)
for rule in FUNCTIONAL_CONSTRAINTS:
    print(f"add_raw {rule}")
print("-" * 60)

print("\nODER automatisch mit diesem Befehl:")
print("python fix_functional_constraints.py | python -c \"import sys; [print(line.strip()) for line in sys.stdin if line.startswith('add_raw')]\" > add_constraints.bat")

# Test-Beispiel
print("\n\nTEST-SZENARIO:")
print("-" * 60)
print("1. add_raw Einwohner(Rom, 2873000).")
print("2. add_raw all x all y all z ((Einwohner(x, y) & Einwohner(x, z)) -> (y = z)).")
print("3. ask Einwohner(Rom, 283000).  # Sollte jetzt FALSCH sein!")
print("4. explain Einwohner(Rom, 283000).  # Sollte Widerspruch zeigen!")
