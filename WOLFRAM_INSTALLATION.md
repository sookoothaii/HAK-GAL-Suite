# Wolfram|Alpha Integration - Installationsanleitung

Diese Anleitung führt Sie durch die vollständige Installation und Konfiguration der neuen Wolfram|Alpha Integration für das HAK-GAL System.

## 🎯 Überblick

Die Wolfram-Integration erweitert das HAK-GAL System um:
- **Realwelt-Wissensabfragen** (Hauptstädte, Bevölkerung, Wetter)
- **Mathematische Berechnungen** (Integration, Differentiation, Gleichungen)
- **Intelligente Oracle-Erkennung** mit dem ComplexityAnalyzer
- **Adaptives Portfolio-Management** für optimale Prover-Auswahl
- **Caching-System** für bessere Performance

## 📋 Voraussetzungen

- Python 3.8 oder höher
- HAK-GAL Suite (vorhandene Installation)
- Internetverbindung für Wolfram|Alpha API

## 🚀 Schritt-für-Schritt Installation

### Schritt 1: Abhängigkeiten installieren

```bash
# Wolfram|Alpha Bibliothek installieren
pip install wolframalpha

# Optional: Alle Abhängigkeiten aktualisieren
pip install -r requirements.txt
```

### Schritt 2: Wolfram|Alpha App ID erhalten

1. Besuchen Sie: https://developer.wolframalpha.com/portal/myapps/
2. Erstellen Sie ein kostenloses Konto (falls noch nicht vorhanden)
3. Klicken Sie auf "Get an AppID"
4. Wählen Sie "Personal Use" für private Projekte
5. Notieren Sie sich die generierte App ID

**Kostenlose Limits:**
- 2000 Anfragen pro Monat
- Für Bildungs-/Forschungszwecke oft höhere Limits verfügbar

### Schritt 3: Konfiguration

1. **Kopieren Sie die .env.example Datei:**
   ```bash
   cp .env.example .env
   ```

2. **Bearbeiten Sie die .env Datei:**
   ```bash
   # Erforderlich für Wolfram-Integration
   WOLFRAM_APP_ID=XXXX-XXXXXXXXXX  # Ihre App ID hier einfügen
   
   # Optional: Cache-Konfiguration
   WOLFRAM_CACHE_TIMEOUT=3600      # 1 Stunde (Standard)
   WOLFRAM_DEBUG=false             # Debug-Modus (Standard: aus)
   ```

### Schritt 4: Installation testen

Führen Sie das Test-Script aus:

```bash
python test_wolfram_integration.py
```

**Erwartete Ausgabe bei erfolgreicher Installation:**
```
🚀 HAK-GAL Wolfram-Integration Test Suite
==================================================
✅ Umgebungssetup: BESTANDEN
✅ WolframProver: BESTANDEN  
✅ ComplexityAnalyzer: BESTANDEN
✅ ProverPortfolioManager: BESTANDEN
✅ End-to-End Integration: BESTANDEN

🎉 ALLE TESTS BESTANDEN! Wolfram-Integration ist einsatzbereit.
```

### Schritt 5: System starten

Starten Sie das erweiterte System:

```bash
# Mit der neuen Wolfram-Integration
python backend/k_assistant_main_v7_wolfram.py

# Oder verwenden Sie das alte System als Fallback
python backend/k_assistant_main.py
```

## 🧪 Erste Tests

Probieren Sie diese Beispiel-Anfragen aus:

### Geografische Abfragen
```
k-assistant> ask was ist die hauptstadt von deutschland
k-assistant> ask_raw HauptstadtVon(Deutschland, x).
```

### Mathematische Berechnungen  
```
k-assistant> ask was ist das integral von x^2
k-assistant> ask_raw Integral(x^2, x).
```

### Wetter und Realzeit-Daten
```
k-assistant> ask wie ist das wetter in berlin
k-assistant> ask_raw WetterIn(Berlin, x).
```

### System-Status überprüfen
```
k-assistant> status
k-assistant> wolfram_stats
```

## 🔧 Erweiterte Konfiguration

### Cache-Optimierung

Für häufige Anfragen können Sie den Cache optimieren:

```bash
# In der .env Datei:
WOLFRAM_CACHE_TIMEOUT=7200  # 2 Stunden für längeres Caching
WOLFRAM_DEBUG=true          # Für detaillierte Logs
```

### Oracle-Prädikate erweitern

Fügen Sie neue Oracle-Prädikate zur Erkennung hinzu:

```
k-assistant> add_oracle MeinNeuesPrädikat
```

### Performance-Monitoring

Überwachen Sie die Portfolio-Performance:

```
k-assistant> status  # Zeigt Portfolio-Statistiken
```

## 📊 Archon-Prime Architektur

Die neue Integration implementiert die **Archon-Prime** Architektur:

```
Anfrage → ComplexityAnalyzer → ProverPortfolioManager → Optimaler Prover
   ↓              ↓                       ↓                    ↓
Analyse → Oracle-Erkennung → Prover-Auswahl → Wolfram/Z3/Pattern
```

**Intelligente Features:**
- **Oracle-Erkennung**: Automatische Erkennung von Wissensabfragen
- **Portfolio-Management**: Adaptive Prover-Auswahl basierend auf Performance
- **Caching**: Intelligentes Caching für bessere Response-Zeiten
- **Performance-Tracking**: Kontinuierliche Optimierung der Prover-Reihenfolge

## 🐛 Problembehandlung

### Problem: "WolframProver deaktiviert"
**Lösung:** 
- Überprüfen Sie WOLFRAM_APP_ID in der .env Datei
- Stellen Sie sicher, dass die App ID korrekt ist
- Prüfen Sie Ihre Internetverbindung

### Problem: "wolframalpha nicht gefunden"
**Lösung:**
```bash
pip install wolframalpha
```

### Problem: "Timeout" bei Wolfram-Anfragen
**Lösung:**
- Überprüfen Sie Ihre Internetverbindung
- Erhöhen Sie den Cache-Timeout
- Kontaktieren Sie Wolfram für API-Limits

### Problem: Cache-Issues
**Lösung:**
```
k-assistant> clearcache  # Leert alle Caches
```

## 📈 Performance-Optimierung

### 1. Cache-Tuning
```bash
# Für häufige Anfragen
WOLFRAM_CACHE_TIMEOUT=86400  # 24 Stunden

# Für Entwicklung
WOLFRAM_CACHE_TIMEOUT=300    # 5 Minuten
```

### 2. Portfolio-Optimierung
Das System lernt automatisch, welcher Prover für welche Art von Anfragen am besten geeignet ist.

### 3. API-Limit Management
- Überwachen Sie Ihr Wolfram|Alpha Dashboard
- Nutzen Sie Caching für wiederkehrende Anfragen
- Verwenden Sie spezifische Prädikate für bessere Erkennung

## 🔄 Backup und Migration

### Backup vor Installation
Das ursprüngliche System wurde automatisch gesichert in:
```
HAK_GAL_SUITE_BACKUP_20250709_BEFORE_WOLFRAM/
```

### Rollback bei Problemen
```bash
# Originales System verwenden
python backend/k_assistant_main.py

# Oder Backup wiederherstellen
cp HAK_GAL_SUITE_BACKUP_20250709_BEFORE_WOLFRAM/k_assistant_main_BACKUP.py backend/k_assistant_main.py
```

## 📚 Weitere Ressourcen

- **Wolfram|Alpha Developer Portal:** https://developer.wolframalpha.com/
- **HAK-GAL Dokumentation:** `docs/` Verzeichnis
- **API Limits überwachen:** https://developer.wolframalpha.com/portal/myapps/
- **Support:** GitHub Issues oder direkte Kontaktaufnahme

## 🎉 Nächste Schritte

Nach erfolgreicher Installation können Sie:

1. **Eigene Oracle-Prädikate definieren** für spezifische Wissensdomänen
2. **Cache-Strategien optimieren** für Ihre Anwendungsfälle  
3. **Portfolio-Performance überwachen** und anpassen
4. **RAG-System integrieren** für lokale Wissensdokumente
5. **Erweiterte Queries entwickeln** mit Multi-Step Reasoning

Willkommen in der neuen Ära der hybriden KI mit HAK-GAL + Wolfram|Alpha! 🚀
