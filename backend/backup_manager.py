# backup_manager.py
# Stellt eine sichere und funktionale Backup-Logik bereit.

import os
import datetime
import shutil
import zlib

class BackupManager:
    def __init__(self, backup_dir="backup", files_to_ignore=None):
        """
        Initialisiert den Backup-Manager.

        Args:
            backup_dir (str): Das Verzeichnis, in dem Backups gespeichert werden.
            files_to_ignore (list, optional): Eine Liste von Dateinamen, die ignoriert werden sollen.
                                              Standardmäßig wird die Wissensbasis-Datei ignoriert.
        """
        self.backup_dir = backup_dir
        if files_to_ignore is None:
            self.files_to_ignore = ["k_assistant.kb"]
        else:
            self.files_to_ignore = files_to_ignore
        
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
            print(f"ℹ️ Backup-Verzeichnis '{self.backup_dir}' wurde erstellt.")

    def _calculate_checksum(self, file_path):
        """Berechnet eine CRC32-Prüfsumme für eine Datei."""
        try:
            with open(file_path, 'rb') as f:
                return zlib.crc32(f.read())
        except IOError:
            return 0

    def auto_backup(self, version: str, description: str):
        """
        Erstellt ein vollständiges Backup des aktuellen Verzeichnisses in einem
        Unterordner mit Zeitstempel.
        """
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            target_folder_name = f"k_assistant_{version}_{timestamp}"
            target_path = os.path.join(self.backup_dir, target_folder_name)
            
            os.makedirs(target_path)

            source_dir = "."
            files = [f for f in os.listdir(source_dir) if os.path.isfile(f)]
            
            copied_files_count = 0
            total_size_bytes = 0
            combined_checksum = 0

            for filename in files:
                if filename not in self.files_to_ignore and filename != os.path.basename(__file__):
                    source_file = os.path.join(source_dir, filename)
                    shutil.copy(source_file, target_path)
                    
                    file_size = os.path.getsize(source_file)
                    total_size_bytes += file_size
                    combined_checksum ^= self._calculate_checksum(source_file)
                    copied_files_count += 1
            
            total_size_kb = total_size_bytes / 1024

            print(f"✅ Backup erstellt: {target_folder_name}")
            print(f"   - Dateien: {copied_files_count}")
            print(f"   - Größe: {total_size_kb:.2f} KB")
            print(f"   - Checksum: {combined_checksum:x}") # hexadezimale Darstellung

        except Exception as e:
            print(f"❌ FEHLER beim Erstellen des Backups: {e}")