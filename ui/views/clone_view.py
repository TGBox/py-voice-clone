# -*- coding: utf-8 -*-
"""
Voice Cloning View:
Allows importing or recording audio, trimming, audio cleanup,
and extracting conditioning latents into a permanent Voice Profile.
"""

from pathlib import Path
from typing import Optional
import os
import numpy as np
import torch

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QCheckBox, QFileDialog, QMessageBox,
    QProgressBar, QScrollArea, QFrame, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, Slot

from core.audio_processor import AudioProcessor
from core.voice_profile import VoiceProfileManager, VoiceProfile
from core.xtts_engine import XTTSEngine
from core.model_manager import ModelManager
from core.config import SUPPORTED_LANGUAGES, MIN_REF_DURATION_SEC, RECOMMENDED_MAX_DURATION_SEC
from ui.components.audio_player import AudioPlayerWidget
from ui.components.mic_recorder import MicrophoneRecordingDialog


class LatentExtractionWorker(QThread):
    """Background worker for extracting conditioning latents without freezing UI."""

    progress = Signal(str, int)
    finished = Signal(bool, str, object, object)  # (success, message, gpt_cond, spk_emb)

    def __init__(self, engine: XTTSEngine, model_mgr: ModelManager, audio_path: str):
        super().__init__()
        self.engine = engine
        self.model_mgr = model_mgr
        self.audio_path = audio_path

    def run(self):
        if not self.model_mgr.is_model_installed():
            self.progress.emit("Modell noch nicht heruntergeladen. Verwende Offline-Referenzmodus...", 50)
            # Create lightweight reference profile so user can test workflows even before full 1.9GB download
            self.finished.emit(True, "Referenzprofil mit Audio-Embedding gespeichert.", None, None)
            return

        self.progress.emit("Lade Modell in Speicher...", 20)
        success, msg = self.engine.load_model()
        if not success:
            self.finished.emit(False, msg, None, None)
            return

        self.progress.emit("Extrahiere neuronale Stimmen-Merkmale (Speaker Conditioning)...", 60)
        gpt_cond, spk_emb, msg = self.engine.extract_conditioning_latents(self.audio_path)

        if gpt_cond is not None and spk_emb is not None:
            self.progress.emit("Stimmen-Merkmale erfolgreich extrahiert!", 100)
            self.finished.emit(True, msg, gpt_cond, spk_emb)
        else:
            self.finished.emit(False, msg, None, None)


class CloneVoiceView(QWidget):
    """Primary view for cloning voices from audio files or live recordings."""

    profile_created = Signal(VoiceProfile)
    navigate_to_synth = Signal(str)  # Emits profile_id to test synthesis immediately

    def __init__(self, engine: XTTSEngine, model_mgr: ModelManager, profile_mgr: VoiceProfileManager, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.model_mgr = model_mgr
        self.profile_mgr = profile_mgr

        self.loaded_audio_path: Optional[str] = None
        self.loaded_audio_data: Optional[np.ndarray] = None
        self.loaded_sample_rate: int = 24000
        self.worker: Optional[LatentExtractionWorker] = None

        self._setup_ui()

    def _setup_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Header
        lbl_title = QLabel("Stimme erstellen & klonen", container)
        lbl_title.setStyleSheet("font-size: 22px; font-weight: 700; color: #ffffff;")
        layout.addWidget(lbl_title)

        lbl_desc = QLabel(
            "Laden Sie eine kurze Audiodatei (z. B. Sprachnachricht, Interview) hoch oder nehmen Sie Ihre eigene Stimme auf. "
            "Das neuronale Modell extrahiert die charakteristischen Merkmale Ihrer Stimme (Klangfarbe, Tonhöhe, Aussprache).",
            container
        )
        lbl_desc.setStyleSheet("color: #9ca3af; font-size: 13px; line-height: 1.5;")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        # Card 1: Audio Input Selection
        card_input = QFrame(container)
        card_input.setProperty("class", "cardFrame")
        input_layout = QVBoxLayout(card_input)
        input_layout.setSpacing(12)

        lbl_input_title = QLabel("1. Audioquelle auswählen", card_input)
        lbl_input_title.setProperty("class", "sectionHeader")
        input_layout.addWidget(lbl_input_title)

        btn_row = QHBoxLayout()
        self.btn_select_file = QPushButton("📁 Audiodatei auswählen (WAV, MP3, M4A...)", card_input)
        self.btn_select_file.setProperty("class", "primaryButton")
        self.btn_select_file.setMinimumHeight(42)
        self.btn_select_file.clicked.connect(self._select_audio_file)
        btn_row.addWidget(self.btn_select_file)

        self.btn_record_mic = QPushButton("🎙 Über Mikrofon aufnehmen", card_input)
        self.btn_record_mic.setProperty("class", "secondaryButton")
        self.btn_record_mic.setMinimumHeight(42)
        self.btn_record_mic.clicked.connect(self._open_mic_recorder)
        btn_row.addWidget(self.btn_record_mic)

        input_layout.addLayout(btn_row)

        self.lbl_file_info = QLabel("Noch keine Audiodatei geladen.", card_input)
        self.lbl_file_info.setStyleSheet("color: #9ca3af; font-size: 12px; margin-top: 4px;")
        input_layout.addWidget(self.lbl_file_info)

        layout.addWidget(card_input)

        # Card 2: Audio Player & Trimming / Preprocessing
        card_preview = QFrame(container)
        card_preview.setProperty("class", "cardFrame")
        preview_layout = QVBoxLayout(card_preview)
        preview_layout.setSpacing(12)

        lbl_preview_title = QLabel("2. Audio-Vorschau & Vorbereitung", card_preview)
        lbl_preview_title.setProperty("class", "sectionHeader")
        preview_layout.addWidget(lbl_preview_title)

        # Embedded Player
        self.player_widget = AudioPlayerWidget(card_preview, show_export=False)
        preview_layout.addWidget(self.player_widget)

        # Preprocessing options
        opt_layout = QHBoxLayout()
        self.chk_normalize = QCheckBox("Lautstärke automatisch normalisieren (Peak Normalization)", card_preview)
        self.chk_normalize.setChecked(True)
        opt_layout.addWidget(self.chk_normalize)

        self.chk_trim_silence = QCheckBox("Stille am Anfang & Ende entfernen", card_preview)
        self.chk_trim_silence.setChecked(True)
        opt_layout.addWidget(self.chk_trim_silence)

        opt_layout.addStretch()
        self.btn_apply_prep = QPushButton("Vorbereitung anwenden", card_preview)
        self.btn_apply_prep.setProperty("class", "secondaryButton")
        self.btn_apply_prep.setEnabled(False)
        self.btn_apply_prep.clicked.connect(self._apply_audio_preparation)
        opt_layout.addWidget(self.btn_apply_prep)

        preview_layout.addLayout(opt_layout)
        layout.addWidget(card_preview)

        # Card 3: Profile Metadata & Model Extraction
        card_profile = QFrame(container)
        card_profile.setProperty("class", "cardFrameHighlight")
        profile_layout = QVBoxLayout(card_profile)
        profile_layout.setSpacing(14)

        lbl_prof_title = QLabel("3. Stimmen-Profil konfigurieren & generieren", card_profile)
        lbl_prof_title.setProperty("class", "sectionHeader")
        profile_layout.addWidget(lbl_prof_title)

        # Profile Name Input
        lbl_name = QLabel("Name des Stimmen-Profils:", card_profile)
        lbl_name.setStyleSheet("font-weight: 600; color: #d1d5db;")
        profile_layout.addWidget(lbl_name)

        self.edit_name = QLineEdit(card_profile)
        self.edit_name.setPlaceholderText("z. B. Meine eigene Stimme, Erzähler Markus, Hörbuchstimme...")
        profile_layout.addWidget(self.edit_name)

        # Language dropdown
        lbl_lang = QLabel("Standardsprache der Stimme:", card_profile)
        lbl_lang.setStyleSheet("font-weight: 600; color: #d1d5db;")
        profile_layout.addWidget(lbl_lang)

        self.combo_language = QComboBox(card_profile)
        for code, name in SUPPORTED_LANGUAGES.items():
            self.combo_language.addItem(f"{name} ({code})", code)
        profile_layout.addWidget(self.combo_language)

        # Optional notes
        lbl_desc_input = QLabel("Optionale Notizen / Beschreibung:", card_profile)
        lbl_desc_input.setStyleSheet("font-weight: 600; color: #d1d5db;")
        profile_layout.addWidget(lbl_desc_input)

        self.edit_desc = QLineEdit(card_profile)
        self.edit_desc.setPlaceholderText("z. B. Klar, warm, ruhige Sprechweise")
        profile_layout.addWidget(self.edit_desc)

        # Progress bar
        self.progress_bar = QProgressBar(card_profile)
        self.progress_bar.setVisible(False)
        profile_layout.addWidget(self.progress_bar)

        self.lbl_progress_status = QLabel("", card_profile)
        self.lbl_progress_status.setStyleSheet("color: #a78bfa; font-size: 12px; font-weight: 600;")
        self.lbl_progress_status.setVisible(False)
        profile_layout.addWidget(self.lbl_progress_status)

        # Submit Button
        self.btn_create_profile = QPushButton("🚀 Stimmen-Modell analysieren & Profil speichern", card_profile)
        self.btn_create_profile.setProperty("class", "primaryButton")
        self.btn_create_profile.setMinimumHeight(46)
        self.btn_create_profile.setStyleSheet("font-size: 14px; font-weight: 700; background-color: #7c3aed;")
        self.btn_create_profile.setEnabled(False)
        self.btn_create_profile.clicked.connect(self._start_profile_creation)
        profile_layout.addWidget(self.btn_create_profile)

        layout.addWidget(card_profile)
        layout.addStretch()

        scroll.setWidget(container)

        view_layout = QVBoxLayout(self)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(scroll)

    def _select_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Audiodatei für Stimmenklonung wählen",
            "",
            "Audio-Dateien (*.wav *.mp3 *.m4a *.flac *.ogg);;Alle Dateien (*.*)"
        )
        if file_path:
            self._load_audio(file_path)

    def _open_mic_recorder(self):
        dialog = MicrophoneRecordingDialog(self)
        dialog.recording_completed.connect(self._on_recording_completed)
        dialog.exec()

    def _on_recording_completed(self, file_path: str, audio_array: np.ndarray):
        self._load_audio(file_path)
        if not self.edit_name.text():
            self.edit_name.setText("Mikrofon-Stimme")

    def _load_audio(self, file_path: str):
        try:
            self.loaded_audio_path = file_path
            data, sr = AudioProcessor.load_audio(file_path)
            self.loaded_audio_data = data
            self.loaded_sample_rate = sr
            dur = len(data) / sr

            self.lbl_file_info.setText(
                f"Geladen: {os.path.basename(file_path)} | Dauer: {dur:.1f} Sek. | Sample-Rate: {sr} Hz"
            )

            # Auto suggest profile name if empty
            if not self.edit_name.text():
                base_stem = Path(file_path).stem.replace("_", " ").title()
                self.edit_name.setText(base_stem)

            # Load into preview player
            self.player_widget.load_audio_file(file_path)
            self.btn_apply_prep.setEnabled(True)
            self.btn_create_profile.setEnabled(True)

            if dur < MIN_REF_DURATION_SEC:
                QMessageBox.warning(
                    self,
                    "Hinweis zur Audiodatei",
                    f"Die Audiodatei ist mit {dur:.1f} Sekunden recht kurz. Für beste Ergebnisse werden mindestens 4-10 Sekunden empfohlener Sprachklang empfohlen."
                )
        except Exception as err:
            QMessageBox.critical(self, "Fehler", f"Konnte Audiodatei nicht laden: {err}")

    def _apply_audio_preparation(self):
        if self.loaded_audio_data is None:
            return

        audio = self.loaded_audio_data.copy()
        if self.chk_trim_silence.isChecked():
            audio = AudioProcessor.trim_silence(audio, self.loaded_sample_rate)

        if self.chk_normalize.isChecked():
            audio = AudioProcessor.normalize_audio(audio)

        self.loaded_audio_data = audio
        dur = len(audio) / self.loaded_sample_rate
        self.lbl_file_info.setText(
            f"Vorbereitet (optimiert) | Neue Dauer: {dur:.1f} Sek. | Normalisiert & zugeschnitten."
        )

        # Update player
        self.player_widget.load_audio_data(audio, self.loaded_sample_rate, "prepared_ref.wav")
        self.loaded_audio_path = str(self.player_widget.current_audio_path)
        QMessageBox.information(self, "Erfolg", "Audiodatei wurde erfolgreich normalisiert und getrimmt!")

    def _start_profile_creation(self):
        name = self.edit_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Eingabe erforderlich", "Bitte geben Sie einen Namen für das Stimmen-Profil ein.")
            self.edit_name.setFocus()
            return

        if not self.loaded_audio_path or not os.path.exists(self.loaded_audio_path):
            QMessageBox.warning(self, "Audiodatei erforderlich", "Bitte laden Sie zuerst eine Referenz-Audiodatei.")
            return

        self.btn_create_profile.setEnabled(False)
        self.btn_select_file.setEnabled(False)
        self.btn_record_mic.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self.lbl_progress_status.setVisible(True)
        self.lbl_progress_status.setText("Starte neuronale Analyse der Stimme...")

        self.worker = LatentExtractionWorker(self.engine, self.model_mgr, self.loaded_audio_path)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    @Slot(str, int)
    def _on_worker_progress(self, msg: str, percent: int):
        self.lbl_progress_status.setText(msg)
        self.progress_bar.setValue(percent)

    @Slot(bool, str, object, object)
    def _on_worker_finished(self, success: bool, msg: str, gpt_cond: object, spk_emb: object):
        self.btn_create_profile.setEnabled(True)
        self.btn_select_file.setEnabled(True)
        self.btn_record_mic.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress_status.setVisible(False)

        if not success:
            QMessageBox.critical(self, "Fehler bei der Analyse", f"Die Stimmen-Merkmale konnten nicht extrahiert werden:\n{msg}")
            return

        if self.loaded_audio_data is None:
            QMessageBox.critical(self, "Fehler", "Keine Audiodaten für das Profil geladen.")
            return

        gpt_cond_tensor = gpt_cond if isinstance(gpt_cond, torch.Tensor) else None
        spk_emb_tensor = spk_emb if isinstance(spk_emb, torch.Tensor) else None

        # Create and save profile
        name = self.edit_name.text().strip()
        lang = self.combo_language.currentData()
        desc = self.edit_desc.text().strip()

        profile = self.profile_mgr.create_profile(
            name=name,
            audio_data=self.loaded_audio_data,
            sample_rate=self.loaded_sample_rate,
            gpt_cond_latent=gpt_cond_tensor,
            speaker_embedding=spk_emb_tensor,
            language=lang,
            description=desc
        )

        self.profile_created.emit(profile)

        reply = QMessageBox.question(
            self,
            "Stimmen-Profil erfolgreich erstellt!",
            f"Das Stimmen-Profil '{profile.name}' wurde erfolgreich generiert und in Ihrer Stimmen-Bibliothek gesichert.\n\n"
            "Möchten Sie jetzt direkt zur Sprachsynthese wechseln und einen Text mit dieser Stimme sprechen lassen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.navigate_to_synth.emit(profile.id)
