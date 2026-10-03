# -*- coding: utf-8 -*-
"""
Standalone CLI script to download XTTS-v2 model weights from HuggingFace.
Run this script after cloning the repository on a new computer:
    python download_model.py
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_manager import ModelManager


def main():
    print("=" * 60)
    print(" VoiceClone AI Studio - XTTS-v2 Modell-Downloader")
    print("=" * 60)

    mgr = ModelManager()

    if mgr.is_model_installed():
        size_mb = mgr.get_model_size_mb()
        print(f"\n[OK] Das XTTS-v2 Modell ist bereits vollständig installiert!")
        print(f"     Speicherort: {mgr.model_dir}")
        print(f"     Dateigröße:  {size_mb:.1f} MB")
        print("\nSie können die Anwendung direkt mit 'python main.py' starten.")
        return 0

    print(f"\nZielverzeichnis: {mgr.model_dir}")
    print("Starte Download der Modellgewichte von HuggingFace (~1.9 GB)...")
    print("Dies kann je nach Internetverbindung einige Minuten dauern.\n")

    import threading
    done_event = threading.Event()
    download_success = [False]
    error_msg = [""]

    def on_progress(status_text: str, percent: int):
        bar_len = 30
        filled = int(bar_len * percent / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {percent:3d}% | {status_text:<50}")
        sys.stdout.flush()

    def on_completion(success: bool, msg: str):
        download_success[0] = success
        error_msg[0] = msg
        done_event.set()

    mgr.start_download(progress_callback=on_progress, completion_callback=on_completion)

    try:
        done_event.wait()
    except KeyboardInterrupt:
        print("\n\n[Abbruch] Download durch Benutzer abgebrochen.")
        return 1

    sys.stdout.write("\n\n")
    if download_success[0]:
        print("[ERFOLG] XTTS-v2 Modell wurde erfolgreich heruntergeladen und validiert!")
        print(f"Dateigröße: {mgr.get_model_size_mb():.1f} MB")
        print("\nSie können die Anwendung jetzt mit 'python main.py' oder 'run.bat' ausführen.")
        return 0
    else:
        print(f"[FEHLER] {error_msg[0]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
