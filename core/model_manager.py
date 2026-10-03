# -*- coding: utf-8 -*-
"""
Model download, verification, and hardware status management.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable
import os
import threading
import torch

from .config import XTTS_MODEL_DIR, HF_XTTS_REPO


class ModelManager:
    """Handles XTTS-v2 model files and GPU telemetry."""

    REQUIRED_FILES = [
        "config.json",
        "vocab.json",
        "model.pth"
    ]

    def __init__(self, model_dir: Path = XTTS_MODEL_DIR):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.is_downloading = False
        self._download_thread: Optional[threading.Thread] = None

    def is_model_installed(self) -> bool:
        """Check if all necessary XTTS v2 model weights and configs are present."""
        for filename in self.REQUIRED_FILES:
            file_path = self.model_dir / filename
            if not file_path.exists() or file_path.stat().st_size < 1024:
                return False
        return True

    def get_model_size_mb(self) -> float:
        """Return total size of model files in MB."""
        total_bytes = 0
        if self.model_dir.exists():
            for p in self.model_dir.glob("*"):
                if p.is_file():
                    total_bytes += p.stat().st_size
        return total_bytes / (1024 * 1024)

    def get_hardware_info(self) -> Dict[str, Any]:
        """Return detailed GPU / CPU hardware status."""
        cuda_avail = torch.cuda.is_available()
        info = {
            "cuda_available": cuda_avail,
            "device_name": "CPU (Keine dedizierte GPU aktiv)",
            "device_type": "cpu",
            "vram_total_mb": 0.0,
            "vram_used_mb": 0.0,
            "vram_free_mb": 0.0,
            "vram_percent": 0.0
        }

        if cuda_avail:
            try:
                device_idx = 0
                props = torch.cuda.get_device_properties(device_idx)
                total = props.total_memory / (1024 * 1024)
                allocated = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)
                used = max(allocated, reserved)
                free = max(0.0, total - used)

                info.update({
                    "device_name": props.name,
                    "device_type": "cuda",
                    "vram_total_mb": round(total, 1),
                    "vram_used_mb": round(used, 1),
                    "vram_free_mb": round(free, 1),
                    "vram_percent": round((used / total) * 100 if total > 0 else 0.0, 1)
                })
            except Exception as err:
                print(f"Error querying GPU info: {err}")

        return info

    def start_download(
        self,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        completion_callback: Optional[Callable[[bool, str], None]] = None
    ):
        """
        Download XTTS-v2 weights in a background thread with progress status.
        """
        if self.is_downloading:
            return

        self.is_downloading = True

        def download_worker():
            from huggingface_hub import snapshot_download

            if progress_callback:
                progress_callback("Download von XTTS-v2 Modell wird initialisiert...", 10)

            try:
                # Target download directory
                target_dir = str(self.model_dir)

                if progress_callback:
                    progress_callback("Lade Modellgewichte von HuggingFace herunter (ca. 1.9 GB)...", 30)

                snapshot_download(
                    repo_id=HF_XTTS_REPO,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )

                if progress_callback:
                    progress_callback("Download abgeschlossen. Validiere Dateien...", 95)

                if self.is_model_installed():
                    self.is_downloading = False
                    if progress_callback:
                        progress_callback("Modell erfolgreich geladen und einsatzbereit!", 100)
                    if completion_callback:
                        completion_callback(True, "Modell erfolgreich heruntergeladen!")
                else:
                    self.is_downloading = False
                    if completion_callback:
                        completion_callback(False, "Modell-Download unvollständig. Einige Dateien fehlen.")
            except Exception as err:
                self.is_downloading = False
                error_msg = f"Download-Fehler: {str(err)}"
                print(error_msg)
                if completion_callback:
                    completion_callback(False, error_msg)

        self._download_thread = threading.Thread(target=download_worker, daemon=True)
        self._download_thread.start()
