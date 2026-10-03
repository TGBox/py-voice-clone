# -*- coding: utf-8 -*-
"""
Microphone live recording dialog with device selection,
live VU level metering, recording timer, and audio acceptance.
"""

from pathlib import Path
from typing import Optional
import datetime
import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QProgressBar, QMessageBox, QFrame
)
from PySide6.QtCore import Qt, QTimer, Signal

from core.audio_processor import AudioProcessor, AudioRecorder
from core.config import RECORDINGS_DIR


class MicrophoneRecordingDialog(QDialog):
    """Interactive modal dialog for recording reference speech from microphone."""

    recording_completed = Signal(str, np.ndarray)  # Emits (file_path, audio_array)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mikrofon-Aufnahme - Referenzstimme")
        self.setMinimumWidth(480)
        self.setModal(True)

        self.recorder = AudioRecorder(sample_rate=24000)
        self.recorded_audio: Optional[np.ndarray] = None
        self.elapsed_seconds: float = 0.0

        # Timer for duration update
        self.record_timer = QTimer(self)
        self.record_timer.setInterval(100)
        self.record_timer.timeout.connect(self._on_timer_tick)

        self._setup_ui()
        self._populate_devices()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Header
        lbl_title = QLabel("Stimme über Mikrofon aufnehmen", self)
        lbl_title.setProperty("class", "sectionHeader")
        layout.addWidget(lbl_title)

        lbl_desc = QLabel(
            "Tipp: Sprechen Sie für 5 bis 12 Sekunden in normaler Lautstärke und deutlicher Aussprache.\n"
            "Vermeiden Sie Hintergrundgeräusche.",
            self
        )
        lbl_desc.setStyleSheet("color: #9ca3af; font-size: 12px; line-height: 1.4;")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        # Device Selection
        card_device = QFrame(self)
        card_device.setProperty("class", "cardFrame")
        card_layout = QVBoxLayout(card_device)

        lbl_dev = QLabel("Mikrofon / Eingabegerät:", card_device)
        lbl_dev.setStyleSheet("color: #d1d5db; font-weight: 600;")
        card_layout.addWidget(lbl_dev)

        self.combo_devices = QComboBox(card_device)
        card_layout.addWidget(self.combo_devices)
        layout.addWidget(card_device)

        # Metering & Status Card
        card_meter = QFrame(self)
        card_meter.setProperty("class", "cardFrameHighlight")
        meter_layout = QVBoxLayout(card_meter)

        # Duration label
        self.lbl_timer = QLabel("00:00.0", card_meter)
        self.lbl_timer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_timer.setStyleSheet("font-size: 28px; font-weight: 700; color: #a78bfa; font-family: monospace;")
        meter_layout.addWidget(self.lbl_timer)

        self.lbl_status = QLabel("Bereit zur Aufnahme", card_meter)
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: #9ca3af; font-size: 12px;")
        meter_layout.addWidget(self.lbl_status)

        # VU meter level bar
        self.bar_level = QProgressBar(card_meter)
        self.bar_level.setRange(0, 100)
        self.bar_level.setValue(0)
        self.bar_level.setTextVisible(False)
        self.bar_level.setFixedHeight(12)
        meter_layout.addWidget(self.bar_level)

        layout.addWidget(card_meter)

        # Action Buttons
        btn_layout = QHBoxLayout()

        self.btn_toggle_record = QPushButton("● Aufnahme starten", self)
        self.btn_toggle_record.setProperty("class", "primaryButton")
        self.btn_toggle_record.setMinimumHeight(40)
        self.btn_toggle_record.clicked.connect(self._toggle_recording)
        btn_layout.addWidget(self.btn_toggle_record)

        self.btn_accept = QPushButton("✓ Übernehmen", self)
        self.btn_accept.setProperty("class", "secondaryButton")
        self.btn_accept.setMinimumHeight(40)
        self.btn_accept.setEnabled(False)
        self.btn_accept.clicked.connect(self._accept_recording)
        btn_layout.addWidget(self.btn_accept)

        self.btn_cancel = QPushButton("Abbrechen", self)
        self.btn_cancel.setProperty("class", "secondaryButton")
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.clicked.connect(self._cancel)
        btn_layout.addWidget(self.btn_cancel)

        layout.addLayout(btn_layout)

    def _populate_devices(self):
        devices = AudioProcessor.list_input_devices()
        self.combo_devices.clear()
        if not devices:
            self.combo_devices.addItem("Kein Mikrofon gefunden", None)
            self.btn_toggle_record.setEnabled(False)
            return

        for dev in devices:
            self.combo_devices.addItem(f"{dev['name']} ({dev['channels']} Ch)", dev["index"])

    def _toggle_recording(self):
        if not self.recorder.is_recording:
            # Start recording
            dev_idx = self.combo_devices.currentData()
            self.recorder.device_index = dev_idx
            self.elapsed_seconds = 0.0
            self.recorded_audio = None

            success = self.recorder.start(level_callback=self._update_vu_level)
            if success:
                self.record_timer.start()
                self.btn_toggle_record.setText("⏹ Aufnahme stoppen")
                self.btn_toggle_record.setStyleSheet("background-color: #dc2626; color: white;")
                self.lbl_status.setText("Aufnahme läuft... Bitte sprechen.")
                self.btn_accept.setEnabled(False)
            else:
                QMessageBox.critical(self, "Fehler", "Aufnahme konnte nicht gestartet werden. Prüfen Sie das Mikrofon.")
        else:
            # Stop recording
            self.record_timer.stop()
            self.recorded_audio = self.recorder.stop()
            self.bar_level.setValue(0)
            self.btn_toggle_record.setText("● Neu aufnehmen")
            self.btn_toggle_record.setStyleSheet("")

            if self.recorded_audio is not None and len(self.recorded_audio) > 24000 * 1.5:
                dur = len(self.recorded_audio) / 24000.0
                self.lbl_status.setText(f"Aufnahme fertig: {dur:.1f} Sek. bereit zur Übernahme.")
                self.btn_accept.setEnabled(True)
            else:
                self.lbl_status.setText("Aufnahme zu kurz (mindestens 2 Sekunden erforderlich).")
                self.btn_accept.setEnabled(False)

    def _update_vu_level(self, level: float):
        percent = int(level * 100)
        # Using QTimer/direct safe update
        self.bar_level.setValue(percent)

    def _on_timer_tick(self):
        self.elapsed_seconds += 0.1
        sec = int(self.elapsed_seconds)
        sub = int((self.elapsed_seconds - sec) * 10)
        self.lbl_timer.setText(f"{sec // 60:02d}:{sec % 60:02d}.{sub}")

    def _accept_recording(self):
        if self.recorded_audio is None or len(self.recorded_audio) == 0:
            return

        # Save to recordings directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = RECORDINGS_DIR / f"mic_recording_{timestamp}.wav"
        AudioProcessor.save_audio(file_path, self.recorded_audio, 24000)

        self.recording_completed.emit(str(file_path), self.recorded_audio)
        self.accept()

    def _cancel(self):
        if self.recorder.is_recording:
            self.recorder.stop()
        self.reject()

    def closeEvent(self, event):
        if self.recorder.is_recording:
            self.recorder.stop()
        event.accept()
