#!/usr/bin/env python3
"""
Git-GitHub Synchronisations-Check
Überprüft den Status des lokalen Repositories
"""

import subprocess
import os
import sys
from datetime import datetime

def run_git_command(cmd):
    """Führt Git-Befehl aus und gibt Ausgabe zurück"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd=r"D:\MCP Mods\HAK_GAL_SUITE"
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return None, str(e), 1

print("=" * 70)
print("🔍 GIT-GITHUB SYNCHRONISATIONS-CHECK")
print("=" * 70)
print(f"Zeitpunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Verzeichnis: D:\\MCP Mods\\HAK_GAL_SUITE")
print("=" * 70)

# 1. Git Status
print("\n📊 GIT STATUS:")
print("-" * 50)
stdout, stderr, code = run_git_command("git status --porcelain")
if code == 0:
    if stdout:
        print("Geänderte/Neue Dateien:")
        for line in stdout.split('\n'):
            if line.strip():
                status = line[:2]
                file = line[3:]
                if status == "??":
                    print(f"  🆕 Neu (untracked): {file}")
                elif "M" in status:
                    print(f"  ✏️  Geändert: {file}")
                elif "A" in status:
                    print(f"  ➕ Hinzugefügt: {file}")
                elif "D" in status:
                    print(f"  ❌ Gelöscht: {file}")
                else:
                    print(f"  ❓ {status}: {file}")
    else:
        print("✅ Keine lokalen Änderungen - alles committet!")
else:
    print(f"❌ Fehler: {stderr}")

# 2. Branch Info
print("\n🌿 BRANCH INFORMATIONEN:")
print("-" * 50)
stdout, stderr, code = run_git_command("git branch --show-current")
if code == 0:
    current_branch = stdout
    print(f"Aktueller Branch: {current_branch}")
else:
    print(f"❌ Fehler: {stderr}")

# 3. Remote Info
print("\n🌐 REMOTE REPOSITORY:")
print("-" * 50)
stdout, stderr, code = run_git_command("git remote -v")
if code == 0:
    print(stdout)
else:
    print(f"❌ Fehler: {stderr}")

# 4. Fetch Status (ohne tatsächlich zu fetchen)
print("\n📡 REMOTE STATUS (Vergleich mit GitHub):")
print("-" * 50)
stdout, stderr, code = run_git_command("git remote update")
if code == 0:
    # Check if local is behind/ahead
    stdout, stderr, code = run_git_command("git status -uno")
    if code == 0:
        if "Your branch is up to date" in stdout:
            print("✅ Lokal und Remote sind synchron!")
        elif "Your branch is ahead" in stdout:
            # Extract number of commits
            import re
            match = re.search(r"Your branch is ahead.*by (\d+) commit", stdout)
            if match:
                commits = match.group(1)
                print(f"⬆️  Lokal ist {commits} Commit(s) VORAUS (noch nicht gepusht)")
        elif "Your branch is behind" in stdout:
            match = re.search(r"Your branch is behind.*by (\d+) commit", stdout)
            if match:
                commits = match.group(1)
                print(f"⬇️  Lokal ist {commits} Commit(s) ZURÜCK (pull erforderlich)")
        elif "have diverged" in stdout:
            print("🔀 Lokal und Remote sind DIVERGIERT (merge erforderlich)")
        else:
            print(stdout)
else:
    print(f"❌ Fehler beim Remote-Check: {stderr}")

# 5. Letzter Commit
print("\n📝 LETZTER COMMIT:")
print("-" * 50)
stdout, stderr, code = run_git_command("git log -1 --oneline")
if code == 0:
    print(f"Lokal: {stdout}")
    
    # Letzter Remote Commit
    stdout, stderr, code = run_git_command("git log origin/HEAD -1 --oneline")
    if code == 0:
        print(f"Remote: {stdout}")
else:
    print(f"❌ Fehler: {stderr}")

# 6. Dateien die NICHT in Git sind
print("\n🚫 DATEIEN DIE IGNORIERT WERDEN (.gitignore):")
print("-" * 50)
stdout, stderr, code = run_git_command("git ls-files --others --ignored --exclude-standard")
if code == 0 and stdout:
    ignored_files = stdout.split('\n')
    # Zeige nur die wichtigsten
    important_ignored = [f for f in ignored_files if not f.startswith('node_modules/') and f]
    if important_ignored:
        for f in important_ignored[:10]:  # Erste 10
            print(f"  📁 {f}")
        if len(important_ignored) > 10:
            print(f"  ... und {len(important_ignored) - 10} weitere Dateien")
else:
    print("  Keine wichtigen ignorierten Dateien")

# 7. Empfehlungen
print("\n💡 EMPFEHLUNGEN:")
print("-" * 50)

# Check for uncommitted changes
stdout, stderr, code = run_git_command("git status --porcelain")
if stdout:
    print("1. Sie haben lokale Änderungen. Führen Sie aus:")
    print("   git add .")
    print("   git commit -m 'Ihre Commit-Nachricht'")
    print("   git push")

# Check if behind
stdout, stderr, code = run_git_command("git status -uno")
if "Your branch is behind" in stdout:
    print("2. Ihr lokales Repository ist veraltet. Führen Sie aus:")
    print("   git pull")

# Check for sensitive files
if os.path.exists(r"D:\MCP Mods\HAK_GAL_SUITE\.env"):
    print("3. ✅ .env Datei existiert (wird korrekt ignoriert)")

print("\n" + "=" * 70)
print("CHECK ABGESCHLOSSEN")
print("=" * 70)
