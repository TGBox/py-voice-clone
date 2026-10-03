# -*- coding: utf-8 -*-
"""
Full-featured audio player widget with waveform visualization,
timeline scrubbing, volume control, and playback looping.
"""

from pathlib import Path
from typing import Optional
import os
import numpy as np
import soundfile as sf

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
    QSlider, QFileDialog, QMessageBox, QFrame
)
from PySide6.QtCore import Qt, QUrl, Signal, QTime
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from .waveform_view import WaveformWidget
from core.audio_processor import AudioProcessor


class AudioPlayerWidget(QFrame):
    """Integrated audio player with transport controls and interactive waveform."""

    audio_exported = Signal(str)

    def __init__(self, parent=None, show_export: bool = True):
        super().__init__(parent)
        self.setObjectName("cardFrame")
        self.setProperty("class", "cardFrame")

        self.current_audio_path: Optional[str] = None
        self.current_audio_data: Optional[np.ndarray] = None
        self.sample_rate: int = 24000
        self.is_looping: bool = False
        self.duration_ms: int = 0

        # Qt Multimedia Player
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.85)

        self._setup_ui(show_export)
        self._connect_signals()

    def _setup_ui(self, show_export: bool):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Waveform viewer
        self.waveform_widget = WaveformWidget(self, is_interactive=True)
        main_layout.addWidget(self.waveform_widget)

        # Controls row
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        # Play / Pause button
        self.btn_play = QPushButton("▶ Abspielen", self)
        self.btn_play.setProperty("class", "primaryButton")
        self.btn_play.setMinimumWidth(120)
        self.btn_play.setEnabled(False)
        controls_layout.addWidget(self.btn_play)

        # Stop button
        self.btn_stop = QPushButton("⏹ Stop", self)
        self.btn_stop.setProperty("class", "secondaryButton")
        self.btn_stop.setEnabled(False)
        controls_layout.addWidget(self.btn_stop)

        # Loop button
        self.btn_loop = QPushButton("🔁 Schleife", self)
        self.btn_loop.setProperty("class", "secondaryButton")
        self.btn_loop.setCheckable(True)
        controls_layout.addWidget(self.btn_loop)

        # Time readout
        self.lbl_time = QLabel("00:00 / 00:00", self)
        self.lbl_time.setStyleSheet("color: #9ca3af; font-family: monospace; font-size: 12px;")
        controls_layout.addWidget(self.lbl_time)

        controls_layout.addStretch()

        # Volume control
        lbl_vol = QLabel("Lautstärke:", self)
        lbl_vol.setStyleSheet("color: #9ca3af; font-size: 11px;")
        controls_layout.addWidget(lbl_vol)

        self.slider_volume = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(85)
        self.slider_volume.setMaximumWidth(90)
        controls_layout.addWidget(self.slider_volume)

        # Export button
        if show_export:
            self.btn_export = QPushButton("💾 Exportieren...", self)
            self.btn_export.setProperty("class", "secondaryButton")
            self.btn_export.setEnabled(False)
            controls_layout.addWidget(self.btn_export)
        else:
            self.btn_export = None

        main_layout.addLayout(controls_layout)

    def _connect_signals(self):
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_loop.toggled.connect(self._on_loop_toggled)
        self.slider_volume.valueChanged.connect(self._on_volume_changed)
        self.waveform_widget.position_clicked.connect(self._on_waveform_seek)

        if self.btn_export:
            self.btn_export.clicked.connect(self._on_export_clicked)

        # Media player event bindings
        self.media_player.positionChanged.connect(self._on_position_changed)
        self.media_player.durationChanged.connect(self._on_duration_changed)
        self.media_player.playbackStateChanged.connect(self._on_state_changed)

    def load_audio_file(self, file_path: str | Path):
        """Load an audio file into the player and compute waveform peaks."""
        file_path = str(file_path)
        if not os.path.exists(file_path):
            return

        self.stop()
        self.current_audio_path = file_path

        # Extract waveform peaks
        audio_array, sr = AudioProcessor.load_audio(file_path)
        self.current_audio_data = audio_array
        self.sample_rate = sr
        peaks = AudioProcessor.extract_waveform_peaks(audio_array, num_points=160)
        self.waveform_widget.set_peaks(peaks)
        self.waveform_widget.set_playback_position(0.0)

        # Set media source
        self.media_player.setSource(QUrl.fromLocalFile(file_path))
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        if self.btn_export:
            self.btn_export.setEnabled(True)

    def load_audio_data(self, audio_data: np.ndarray, sample_rate: int = 24000, temp_filename: str = "temp_preview.wav"):
        """Save numpy audio to temp file and load into player."""
        from core.config import DATA_DIR
        temp_path = DATA_DIR / temp_filename
        AudioProcessor.save_audio(temp_path, audio_data, sample_rate)
        self.load_audio_file(temp_path)

    def toggle_playback(self):
        """Toggle between playing and pausing."""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def stop(self):
        """Stop playback and reset position."""
        self.media_player.stop()
        self.waveform_widget.set_playback_position(0.0)
        self._update_time_label(0, self.duration_ms)

    def _on_loop_toggled(self, checked: bool):
        self.is_looping = checked

    def _on_volume_changed(self, value: int):
        self.audio_output.setVolume(value / 100.0)

    def _on_waveform_seek(self, norm_pos: float):
        if self.duration_ms > 0:
            target_ms = int(norm_pos * self.duration_ms)
            self.media_player.setPosition(target_ms)

    def _on_position_changed(self, pos_ms: int):
        if self.duration_ms > 0:
            norm = pos_ms / float(self.duration_ms)
            self.waveform_widget.set_playback_position(norm)
            self._update_time_label(pos_ms, self.duration_ms)

    def _on_duration_changed(self, dur_ms: int):
        self.duration_ms = dur_ms
        self._update_time_label(self.media_player.position(), dur_ms)

    def _on_state_changed(self, state: QMediaPlayer.PlaybackState):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play.setText("⏸ Pause")
        else:
            self.btn_play.setText("▶ Abspielen")
            # If reached the end and loop is enabled
            if state == QMediaPlayer.PlaybackState.StoppedState and self.is_looping:
                self.media_player.play()

    def _update_time_label(self, current_ms: int, total_ms: int):
        cur_sec = max(0, current_ms // 1000)
        tot_sec = max(0, total_ms // 1000)
        cur_str = f"{cur_sec // 60:02d}:{cur_sec % 60:02d}"
        tot_str = f"{tot_sec // 60:02d}:{tot_sec % 60:02d}"
        self.lbl_time.setText(f"{cur_str} / {tot_str}")

    def _on_export_clicked(self):
        """Open file dialog to export currently loaded audio."""
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            return

        dest_file, _ = QFileDialog.getSaveFileName(
            self,
            "Audio exportieren",
            os.path.expanduser("~/output_stimme.wav"),
            "WAV Audio (*.wav);;MP3 Audio (*.mp3);;Alle Dateien (*.*)"
        )
        if dest_file:
            try:
                import shutil
                shutil.copy2(self.current_audio_path, dest_file)
                QMessageBox.information(self, "Erfolg", f"Audiodatei erfolgreich exportiert:\n{dest_file}")
                self.audio_exported.emit(dest_file)
            except Exception as err:
                QMessageBox.critical(self, "Fehler", f"Konnte Datei nicht exportieren: {err}")
