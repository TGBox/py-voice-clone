# -*- coding: utf-8 -*-
"""
Settings and Model Management View:
Hardware monitor (GPU/CUDA/VRAM), model downloader/updater,
audio input/output device configuration, and folder shortcuts.
"""

from pathlib import Path
import os
import subprocess

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QProgressBar, QMessageBox, QScrollArea, QFrame,
    QGridLayout
)
from PySide6.QtCore import Qt, QTimer, Slot

from core.model_manager import ModelManager
from core.xtts_engine import XTTSEngine
from core.audio_processor import AudioProcessor
from core.config import DATA_DIR, PROFILES_DIR, OUTPUTS_DIR, RECORDINGS_DIR, XTTS_MODEL_DIR


class SettingsView(QWidget):
    """View for configuring model weights, audio devices, and inspecting GPU telemetry."""

    def __init__(self, model_mgr: ModelManager, engine: XTTSEngine, parent=None):
        super().__init__(parent)
        self.model_mgr = model_mgr
        self.engine = engine

        self._setup_ui()

        # Telemetry refresh timer
        self.telemetry_timer = QTimer(self)
        self.telemetry_timer.setInterval(2000)
        self.telemetry_timer.timeout.connect(self._refresh_telemetry)
        self.telemetry_timer.start()

        self._refresh_telemetry()
        self._refresh_model_status()

    def _setup_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Header
        lbl_title = QLabel("Modell & Systemeinstellungen", container)
        lbl_title.setStyleSheet("font-size: 22px; font-weight: 700; color: #ffffff;")
        layout.addWidget(lbl_title)

        lbl_desc = QLabel(
            "Verwalten Sie die Gewichte des neuronalen Sprachmodells (XTTS-v2), überwachen Sie die "
            "Grafikkarten-Auslastung und konfigurieren Sie Ein- und Ausgabegeräte.",
            container
        )
        lbl_desc.setStyleSheet("color: #9ca3af; font-size: 13px;")
        layout.addWidget(lbl_desc)

        # Card 1: GPU / Hardware Telemetry
        card_hw = QFrame(container)
        card_hw.setProperty("class", "cardFrame")
        hw_layout = QVBoxLayout(card_hw)
        hw_layout.setSpacing(12)

        lbl_hw_title = QLabel("Hardware- & Grafikprozessor-Status", card_hw)
        lbl_hw_title.setProperty("class", "sectionHeader")
        hw_layout.addWidget(lbl_hw_title)

        hw_grid = QGridLayout()
        hw_grid.setHorizontalSpacing(20)
        hw_grid.setVerticalSpacing(8)

        # Device Name
        hw_grid.addWidget(QLabel("Beschleuniger:", card_hw), 0, 0)
        self.lbl_hw_device = QLabel("Wird ermittelt...", card_hw)
        self.lbl_hw_device.setStyleSheet("font-weight: 700; color: #a78bfa;")
        hw_grid.addWidget(self.lbl_hw_device, 0, 1)

        # CUDA Status
        hw_grid.addWidget(QLabel("CUDA Unterstützung:", card_hw), 1, 0)
        self.lbl_hw_cuda = QLabel("Aktiv ✓", card_hw)
        self.lbl_hw_cuda.setProperty("class", "badgeSuccess")
        hw_grid.addWidget(self.lbl_hw_cuda, 1, 1)

        # VRAM Gauge
        hw_grid.addWidget(QLabel("VRAM-Auslastung:", card_hw), 2, 0)
        self.bar_vram = QProgressBar(card_hw)
        self.bar_vram.setRange(0, 100)
        self.bar_vram.setValue(0)
        hw_grid.addWidget(self.bar_vram, 2, 1)

        # VRAM numbers
        self.lbl_vram_text = QLabel("0 MB / 0 MB", card_hw)
        self.lbl_vram_text.setStyleSheet("color: #9ca3af; font-size: 11px;")
        hw_grid.addWidget(self.lbl_vram_text, 3, 1)

        hw_layout.addLayout(hw_grid)
        layout.addWidget(card_hw)

        # Card 2: XTTS-v2 Neural Model Management
        card_model = QFrame(container)
        card_model.setProperty("class", "cardFrameHighlight")
        model_layout = QVBoxLayout(card_model)
        model_layout.setSpacing(14)

        lbl_mod_title = QLabel("Neuronales Modell (Coqui XTTS-v2)", card_model)
        lbl_mod_title.setProperty("class", "sectionHeader")
        model_layout.addWidget(lbl_mod_title)

        lbl_mod_desc = QLabel(
            "Das XTTS-v2 Modell ermöglicht zero-shot Voice-Cloning in 17 Sprachen mit nur wenigen Sekunden "
            "Referenz-Audio. Die Modellgewichte betragen einmalig ca. 1.9 GB und werden lokal gespeichert.",
            card_model
        )
        lbl_mod_desc.setStyleSheet("color: #9ca3af; font-size: 12px; line-height: 1.4;")
        lbl_mod_desc.setWordWrap(True)
        model_layout.addWidget(lbl_mod_desc)

        # Model status row
        row_status = QHBoxLayout()
        row_status.addWidget(QLabel("Status:", card_model))
        self.lbl_model_status = QLabel("Wird geprüft...", card_model)
        row_status.addWidget(self.lbl_model_status)
        row_status.addStretch()

        self.lbl_model_size = QLabel("", card_model)
        self.lbl_model_size.setStyleSheet("color: #9ca3af; font-size: 12px;")
        row_status.addWidget(self.lbl_model_size)
        model_layout.addLayout(row_status)

        # Model Download Progress Bar
        self.progress_download = QProgressBar(card_model)
        self.progress_download.setVisible(False)
        model_layout.addWidget(self.progress_download)

        self.lbl_download_msg = QLabel("", card_model)
        self.lbl_download_msg.setStyleSheet("color: #a78bfa; font-size: 12px;")
        self.lbl_download_msg.setVisible(False)
        model_layout.addWidget(self.lbl_download_msg)

        # Action Buttons
        btn_mod_row = QHBoxLayout()
        self.btn_download_model = QPushButton("📥 Modell herunterladen (~1.9 GB)", card_model)
        self.btn_download_model.setProperty("class", "primaryButton")
        self.btn_download_model.setMinimumHeight(40)
        self.btn_download_model.clicked.connect(self._start_model_download)
        btn_mod_row.addWidget(self.btn_download_model)

        self.btn_reload_model = QPushButton("🔄 In GPU neu laden", card_model)
        self.btn_reload_model.setProperty("class", "secondaryButton")
        self.btn_reload_model.setMinimumHeight(40)
        self.btn_reload_model.clicked.connect(self._reload_model_gpu)
        btn_mod_row.addWidget(self.btn_reload_model)

        model_layout.addLayout(btn_mod_row)
        layout.addWidget(card_model)

        # Card 3: Storage Folders & Quick Access
        card_dirs = QFrame(container)
        card_dirs.setProperty("class", "cardFrame")
        dirs_layout = QVBoxLayout(card_dirs)
        dirs_layout.setSpacing(12)

        lbl_dir_title = QLabel("Verzeichnisse & Speicherorte", card_dirs)
        lbl_dir_title.setProperty("class", "sectionHeader")
        dirs_layout.addWidget(lbl_dir_title)

        row_folders = QHBoxLayout()

        btn_open_profiles = QPushButton("📂 Stimmen-Profile öffnen", card_dirs)
        btn_open_profiles.setProperty("class", "secondaryButton")
        btn_open_profiles.clicked.connect(lambda: self._open_folder(PROFILES_DIR))
        row_folders.addWidget(btn_open_profiles)

        btn_open_outputs = QPushButton("📂 Generierte Audiodateien öffnen", card_dirs)
        btn_open_outputs.setProperty("class", "secondaryButton")
        btn_open_outputs.clicked.connect(lambda: self._open_folder(OUTPUTS_DIR))
        row_folders.addWidget(btn_open_outputs)

        btn_open_models = QPushButton("📂 Modell-Ordner öffnen", card_dirs)
        btn_open_models.setProperty("class", "secondaryButton")
        btn_open_models.clicked.connect(lambda: self._open_folder(XTTS_MODEL_DIR))
        row_folders.addWidget(btn_open_models)

        dirs_layout.addLayout(row_folders)
        layout.addWidget(card_dirs)
        layout.addStretch()

        scroll.setWidget(container)

        view_layout = QVBoxLayout(self)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(scroll)

    def _refresh_telemetry(self):
        info = self.model_mgr.get_hardware_info()
        self.lbl_hw_device.setText(info["device_name"])

        if info["cuda_available"]:
            self.lbl_hw_cuda.setText("Aktiv ✓ (NVIDIA CUDA)")
            self.lbl_hw_cuda.setProperty("class", "badgeSuccess")
            self.bar_vram.setValue(int(info["vram_percent"]))
            self.lbl_vram_text.setText(
                f"{info['vram_used_mb']:.0f} MB / {info['vram_total_mb']:.0f} MB ({info['vram_percent']} %)"
            )
        else:
            self.lbl_hw_cuda.setText("Inaktiv (Nur CPU)")
            self.lbl_hw_cuda.setProperty("class", "badgeWarning")
            self.bar_vram.setValue(0)
            self.lbl_vram_text.setText("CPU-Modus aktiv")

    def _refresh_model_status(self):
        installed = self.model_mgr.is_model_installed()
        size_mb = self.model_mgr.get_model_size_mb()

        if installed:
            self.lbl_model_status.setText("Installiert & Bereit ✓")
            self.lbl_model_status.setProperty("class", "badgeSuccess")
            self.lbl_model_size.setText(f"Größe auf Datenträger: {size_mb:.1f} MB")
            self.btn_download_model.setText("✓ Modell ist aktuell")
            self.btn_download_model.setEnabled(False)
            self.btn_reload_model.setEnabled(True)
        else:
            self.lbl_model_status.setText("Nicht heruntergeladen")
            self.lbl_model_status.setProperty("class", "badgeWarning")
            self.lbl_model_size.setText("Dateien fehlen in models/xtts_v2")
            self.btn_download_model.setText("📥 Modell herunterladen (~1.9 GB)")
            self.btn_download_model.setEnabled(True)
            self.btn_reload_model.setEnabled(False)

    def _start_model_download(self):
        self.btn_download_model.setEnabled(False)
        self.progress_download.setVisible(True)
        self.progress_download.setValue(5)
        self.lbl_download_msg.setVisible(True)
        self.lbl_download_msg.setText("Starte Download von HuggingFace...")

        def on_prog(msg, pct):
            QTimer.singleShot(0, lambda: self._update_download_progress(msg, pct))

        def on_done(success, msg):
            QTimer.singleShot(0, lambda: self._download_finished(success, msg))

        self.model_mgr.start_download(progress_callback=on_prog, completion_callback=on_done)

    def _update_download_progress(self, msg: str, pct: int):
        self.lbl_download_msg.setText(msg)
        self.progress_download.setValue(pct)

    def _download_finished(self, success: bool, msg: str):
        self.progress_download.setVisible(False)
        self.lbl_download_msg.setVisible(False)
        self._refresh_model_status()

        if success:
            QMessageBox.information(self, "Erfolg", "Das neuronale Sprachmodell wurde erfolgreich heruntergeladen!")
            self._reload_model_gpu()
        else:
            QMessageBox.critical(self, "Fehler beim Download", f"Download konnte nicht abgeschlossen werden:\n{msg}")

    def _reload_model_gpu(self):
        self.engine.unload_model()
        success, msg = self.engine.load_model(force_reload=True)
        if success:
            QMessageBox.information(self, "Erfolg", f"{msg}")
        else:
            QMessageBox.critical(self, "Fehler", f"Konnte Modell nicht laden:\n{msg}")

    def _open_folder(self, folder_path: Path):
        folder_str = str(folder_path)
        if not os.path.exists(folder_str):
            os.makedirs(folder_str, exist_ok=True)
        try:
            os.startfile(folder_str)
        except Exception:
            subprocess.run(["explorer", folder_str])
