# -*- coding: utf-8 -*-
"""
Voice Cloning View:
Allows importing single or multiple audio files, recording via microphone,
managing a collection of reference clips, audio cleanup, and extracting
pooled conditioning latents into a high-fidelity Voice Profile.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import os
import numpy as np
import torch

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QCheckBox, QFileDialog, QMessageBox,
    QProgressBar, QScrollArea, QFrame, QSizePolicy
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
    """Background worker for extracting conditioning latents from one or multiple audio clips."""

    progress = Signal(str, int)
    finished = Signal(bool, str, object, object)  # (success, message, gpt_cond, spk_emb)

    def __init__(self, engine: XTTSEngine, model_mgr: ModelManager, audio_paths: List[str]):
        super().__init__()
        self.engine = engine
        self.model_mgr = model_mgr
        self.audio_paths = audio_paths

    def run(self):
        if not self.model_mgr.is_model_installed():
            self.progress.emit("Modell noch nicht heruntergeladen. Verwende Offline-Referenzmodus...", 50)
            self.finished.emit(True, "Referenzprofil mit Audio-Embedding gespeichert.", None, None)
            return

        self.progress.emit("Lade neuronales XTTS-v2 Modell in Speicher...", 20)
        success, msg = self.engine.load_model()
        if not success:
            self.finished.emit(False, msg, None, None)
            return

        self.progress.emit(
            f"Extrahiere neuronale Stimmen-Merkmale aus {len(self.audio_paths)} Audio-Clip(s)...", 60
        )
        gpt_cond, spk_emb, msg = self.engine.extract_conditioning_latents(self.audio_paths)

        if gpt_cond is not None and spk_emb is not None:
            self.progress.emit("Stimmen-Merkmale erfolgreich extrahiert!", 100)
            self.finished.emit(True, msg, gpt_cond, spk_emb)
        else:
            self.finished.emit(False, msg, None, None)


class ClipRowWidget(QFrame):
    """Row widget representing a single audio clip in the multi-sample list."""

    play_clicked = Signal(str)
    remove_clicked = Signal(int)

    def __init__(self, index: int, clip_info: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.index = index
        self.clip_info = clip_info

        self.setStyleSheet("""
            QFrame {
                background-color: #1a1e2a;
                border: 1px solid #2d3348;
                border-radius: 8px;
                padding: 4px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(10)

        # Icon and Name
        lbl_icon = QLabel("🎵", self)
        layout.addWidget(lbl_icon)

        lbl_name = QLabel(f"Clip {index + 1}: {clip_info['name']}", self)
        lbl_name.setStyleSheet("font-weight: 600; color: #f3f4f6;")
        layout.addWidget(lbl_name, 1)

        # Duration badge
        lbl_dur = QLabel(f"{clip_info['duration']:.1f}s", self)
        lbl_dur.setStyleSheet("""
            background-color: #272042;
            color: #a78bfa;
            font-size: 11px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 6px;
            border: 1px solid #7c3aed;
        """)
        layout.addWidget(lbl_dur)

        # Play Preview button
        btn_play = QPushButton("▶ Anhören", self)
        btn_play.setProperty("class", "secondaryButton")
        btn_play.setFixedHeight(28)
        btn_play.clicked.connect(lambda: self.play_clicked.emit(clip_info["path"]))
        layout.addWidget(btn_play)

        # Remove button
        btn_remove = QPushButton("✕", self)
        btn_remove.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #ef4444;
                font-weight: 700;
                font-size: 14px;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7f1d1d;
                color: #ffffff;
            }
        """)
        btn_remove.setToolTip("Diesen Clip entfernen")
        btn_remove.clicked.connect(lambda: self.remove_clicked.emit(self.index))
        layout.addWidget(btn_remove)


class CloneVoiceView(QWidget):
    """Primary view for cloning voices with multi-sample support."""

    profile_created = Signal(VoiceProfile)
    navigate_to_synth = Signal(str)

    def __init__(self, engine: XTTSEngine, model_mgr: ModelManager, profile_mgr: VoiceProfileManager, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.model_mgr = model_mgr
        self.profile_mgr = profile_mgr

        # List of audio clips: dict(path, name, duration, audio, sr)
        self.audio_clips: List[Dict[str, Any]] = []
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
        lbl_title = QLabel("Stimme erstellen & klonen (Multi-Sample Studio)", container)
        lbl_title.setStyleSheet("font-size: 22px; font-weight: 700; color: #ffffff;")
        layout.addWidget(lbl_title)

        lbl_desc = QLabel(
            "Laden Sie eine oder mehrere verschiedene Sprachdateien hoch oder nehmen Sie mehrere Sätze über Ihr Mikrofon auf. "
            "Das Modell poolt alle Audiobeispiele, wodurch eine wesentlich höhere Sprachähnlichkeit, Phonem-Abdeckung "
            "und natürliche Dynamik erreicht wird.",
            container
        )
        lbl_desc.setStyleSheet("color: #9ca3af; font-size: 13px; line-height: 1.5;")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        # Card 1: Multi-Audio Input Selection & Clip Manager
        card_input = QFrame(container)
        card_input.setProperty("class", "cardFrame")
        input_layout = QVBoxLayout(card_input)
        input_layout.setSpacing(12)

        lbl_input_title = QLabel("1. Referenz-Audiodateien (Mehrere Samples)", card_input)
        lbl_input_title.setProperty("class", "sectionHeader")
        input_layout.addWidget(lbl_input_title)

        # Action Buttons row
        btn_row = QHBoxLayout()
        self.btn_select_files = QPushButton("📁 Audiodateien hinzufügen (Mehrfachauswahl)", card_input)
        self.btn_select_files.setProperty("class", "primaryButton")
        self.btn_select_files.setMinimumHeight(40)
        self.btn_select_files.clicked.connect(self._select_audio_files)
        btn_row.addWidget(self.btn_select_files)

        self.btn_record_mic = QPushButton("🎙 Über Mikrofon aufnehmen", card_input)
        self.btn_record_mic.setProperty("class", "secondaryButton")
        self.btn_record_mic.setMinimumHeight(40)
        self.btn_record_mic.clicked.connect(self._open_mic_recorder)
        btn_row.addWidget(self.btn_record_mic)

        self.btn_clear_all = QPushButton("Alle leeren", card_input)
        self.btn_clear_all.setProperty("class", "secondaryButton")
        self.btn_clear_all.setMinimumHeight(40)
        self.btn_clear_all.setEnabled(False)
        self.btn_clear_all.clicked.connect(self._clear_all_clips)
        btn_row.addWidget(self.btn_clear_all)

        input_layout.addLayout(btn_row)

        # Scrollable Clips Container
        self.clips_scroll = QScrollArea(card_input)
        self.clips_scroll.setWidgetResizable(True)
        self.clips_scroll.setMinimumHeight(140)
        self.clips_scroll.setMaximumHeight(220)
        self.clips_scroll.setStyleSheet("QScrollArea { background-color: #12151e; border: 1px solid #232736; border-radius: 8px; }")

        self.clips_container = QWidget()
        self.clips_layout = QVBoxLayout(self.clips_container)
        self.clips_layout.setContentsMargins(8, 8, 8, 8)
        self.clips_layout.setSpacing(6)
        self.clips_layout.addStretch()
        self.clips_scroll.setWidget(self.clips_container)
        input_layout.addWidget(self.clips_scroll)

        # Status & Quality Badge row
        summary_row = QHBoxLayout()
        self.lbl_clips_count = QLabel("0 Audiodateien geladen | Gesamtdauer: 0.0 Sek.", card_input)
        self.lbl_clips_count.setStyleSheet("color: #d1d5db; font-size: 13px; font-weight: 600;")
        summary_row.addWidget(self.lbl_clips_count)

        summary_row.addStretch()

        self.lbl_quality_badge = QLabel("Keine Audio-Daten", card_input)
        self.lbl_quality_badge.setProperty("class", "badgeWarning")
        summary_row.addWidget(self.lbl_quality_badge)
        input_layout.addLayout(summary_row)

        # Tip box
        card_tip = QFrame(card_input)
        card_tip.setStyleSheet("background-color: #1e1b4b; border: 1px solid #4338ca; border-radius: 8px; padding: 10px;")
        tip_layout = QVBoxLayout(card_tip)
        tip_layout.setContentsMargins(8, 6, 8, 6)
        lbl_tip = QLabel(
            "💡 Tipp für maximale Qualität: Verwenden Sie 3 bis 6 verschiedene Audio-Clips (z. B. unterschiedliche Sätze, "
            "eine Frage und eine ruhige Erzählung). Die empfohlene Gesamtdauer liegt zwischen 15 und 45 Sekunden.",
            card_tip
        )
        lbl_tip.setStyleSheet("color: #c7d2fe; font-size: 12px; line-height: 1.4;")
        lbl_tip.setWordWrap(True)
        tip_layout.addWidget(lbl_tip)
        input_layout.addWidget(card_tip)

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
        self.chk_normalize = QCheckBox("Lautstärke aller Clips automatisch normalisieren (Peak Normalization)", card_preview)
        self.chk_normalize.setChecked(True)
        opt_layout.addWidget(self.chk_normalize)

        self.chk_trim_silence = QCheckBox("Stille am Anfang & Ende entfernen", card_preview)
        self.chk_trim_silence.setChecked(True)
        opt_layout.addWidget(self.chk_trim_silence)

        opt_layout.addStretch()
        self.btn_apply_prep = QPushButton("Vorbereitung auf alle Clips anwenden", card_preview)
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
        self.edit_name.setPlaceholderText("z. B. Meine Stimme (Multi-Sample), Erzähler Markus, Hörbuchstimme...")
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
        self.edit_desc.setPlaceholderText("z. B. 4 Samples kombiniert, ruhige und dynamische Sätze")
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

        self._refresh_clips_ui()

    def _select_audio_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Audiodateien für Stimmenklonung wählen (Mehrfachauswahl)",
            "",
            "Audio-Dateien (*.wav *.mp3 *.m4a *.flac *.ogg);;Alle Dateien (*.*)"
        )
        if file_paths:
            for p in file_paths:
                self._add_audio_file(p)
            self._refresh_clips_ui()

    def _open_mic_recorder(self):
        dialog = MicrophoneRecordingDialog(self)
        dialog.recording_completed.connect(self._on_recording_completed)
        dialog.exec()

    def _on_recording_completed(self, file_path: str, audio_array: np.ndarray):
        self._add_audio_file(file_path, display_name=f"Mikrofon-Aufnahme #{len(self.audio_clips) + 1}")
        self._refresh_clips_ui()

    def _add_audio_file(self, file_path: str, display_name: Optional[str] = None):
        try:
            data, sr = AudioProcessor.load_audio(file_path)
            dur = len(data) / sr if sr > 0 else 0.0
            name = display_name or os.path.basename(file_path)
            self.audio_clips.append({
                "path": file_path,
                "name": name,
                "duration": dur,
                "data": data,
                "sr": sr
            })
            if not self.edit_name.text():
                stem = Path(file_path).stem.replace("_", " ").title()
                self.edit_name.setText(stem)
        except Exception as err:
            QMessageBox.critical(self, "Fehler", f"Konnte Datei '{file_path}' nicht laden:\n{err}")

    def _clear_all_clips(self):
        self.audio_clips.clear()
        self._refresh_clips_ui()

    def _remove_clip_at(self, index: int):
        if 0 <= index < len(self.audio_clips):
            self.audio_clips.pop(index)
            self._refresh_clips_ui()

    def _refresh_clips_ui(self):
        # Clear clip rows
        while self.clips_layout.count() > 0:
            item = self.clips_layout.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.deleteLater()

        total_duration = sum(c["duration"] for c in self.audio_clips)
        count = len(self.audio_clips)

        if count == 0:
            lbl_empty = QLabel("Noch keine Audiodateien hinzugefügt. Fügen Sie oben eine oder mehrere Dateien hinzu.", self.clips_container)
            lbl_empty.setStyleSheet("color: #6b7280; font-style: italic; padding: 12px;")
            self.clips_layout.addWidget(lbl_empty)
            self.btn_clear_all.setEnabled(False)
            self.btn_apply_prep.setEnabled(False)
            self.btn_create_profile.setEnabled(False)
            self.lbl_clips_count.setText("0 Audiodateien geladen | Gesamtdauer: 0.0 Sek.")
            self.lbl_quality_badge.setText("Keine Audio-Daten")
            self.lbl_quality_badge.setProperty("class", "badgeWarning")
            self.btn_create_profile.setText("🚀 Stimmen-Modell analysieren & Profil speichern")
        else:
            self.btn_clear_all.setEnabled(True)
            self.btn_apply_prep.setEnabled(True)
            self.btn_create_profile.setEnabled(True)
            self.btn_create_profile.setText(f"🚀 Stimmen-Modell analysieren & Profil speichern ({count} Clip{'s' if count > 1 else ''})")

            for i, clip in enumerate(self.audio_clips):
                row = ClipRowWidget(i, clip, self.clips_container)
                row.play_clicked.connect(self._preview_clip)
                row.remove_clicked.connect(self._remove_clip_at)
                self.clips_layout.addWidget(row)

            self.lbl_clips_count.setText(
                f"{count} Audiodatei{'en' if count > 1 else ''} geladen | Gesamtdauer: {total_duration:.1f} Sek."
            )

            # Quality Assessment Badge
            if total_duration < 4.0:
                self.lbl_quality_badge.setText("🔴 Zu kurz (mind. 5–10s empfohlen)")
                self.lbl_quality_badge.setProperty("class", "badgeWarning")
            elif total_duration < 12.0:
                self.lbl_quality_badge.setText("🟡 Basis-Qualität (5–12s)")
                self.lbl_quality_badge.setProperty("class", "badgeInfo")
            elif total_duration <= 50.0:
                self.lbl_quality_badge.setText("🟢 Hohe Qualität (Optimal: 15–45s)")
                self.lbl_quality_badge.setProperty("class", "badgeSuccess")
            else:
                self.lbl_quality_badge.setText("🔵 Sehr umfangreich (> 50s)")
                self.lbl_quality_badge.setProperty("class", "badgeSuccess")

        self.clips_layout.addStretch()

        # Update preview player with composite audio or first clip
        if count > 0:
            valid_paths = [c["path"] for c in self.audio_clips if os.path.exists(c["path"])]
            if valid_paths:
                combined_audio, sr = AudioProcessor.combine_audio_clips(valid_paths)
                self.player_widget.load_audio_data(combined_audio, sr, "combined_preview.wav")

    def _preview_clip(self, path: str):
        if os.path.exists(path):
            self.player_widget.load_audio_file(path)
            self.player_widget.toggle_playback()

    def _apply_audio_preparation(self):
        if not self.audio_clips:
            return

        for clip in self.audio_clips:
            audio = clip["data"].copy()
            sr = clip["sr"]
            if self.chk_trim_silence.isChecked():
                audio = AudioProcessor.trim_silence(audio, sr)
            if self.chk_normalize.isChecked():
                audio = AudioProcessor.normalize_audio(audio)
            clip["data"] = audio
            clip["duration"] = len(audio) / sr if sr > 0 else 0.0
            # Save normalized version back to temp / existing path
            AudioProcessor.save_audio(clip["path"], audio, sr)

        self._refresh_clips_ui()
        QMessageBox.information(
            self,
            "Erfolg",
            f"Vorbereitung erfolgreich auf alle {len(self.audio_clips)} Clips angewendet!"
        )

    def _start_profile_creation(self):
        name = self.edit_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Eingabe erforderlich", "Bitte geben Sie einen Namen für das Stimmen-Profil ein.")
            self.edit_name.setFocus()
            return

        valid_paths = [c["path"] for c in self.audio_clips if os.path.exists(c["path"])]
        if not valid_paths:
            QMessageBox.warning(self, "Audiodateien erforderlich", "Bitte fügen Sie mindestens eine gültige Audiodatei hinzu.")
            return

        self.btn_create_profile.setEnabled(False)
        self.btn_select_files.setEnabled(False)
        self.btn_record_mic.setEnabled(False)
        self.btn_clear_all.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self.lbl_progress_status.setVisible(True)
        self.lbl_progress_status.setText(f"Starte neuronale Analyse von {len(valid_paths)} Audio-Clips...")

        self.worker = LatentExtractionWorker(self.engine, self.model_mgr, valid_paths)
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
        self.btn_select_files.setEnabled(True)
        self.btn_record_mic.setEnabled(True)
        self.btn_clear_all.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress_status.setVisible(False)

        if not success:
            QMessageBox.critical(self, "Fehler bei der Analyse", f"Die Stimmen-Merkmale konnten nicht extrahiert werden:\n{msg}")
            return

        valid_paths = [c["path"] for c in self.audio_clips if os.path.exists(c["path"])]
        if not valid_paths:
            QMessageBox.critical(self, "Fehler", "Keine gültigen Audio-Dateien vorhanden.")
            return

        # Combine audio clips for the primary reference sample
        combined_audio, sr = AudioProcessor.combine_audio_clips(valid_paths)

        gpt_cond_tensor = gpt_cond if isinstance(gpt_cond, torch.Tensor) else None
        spk_emb_tensor = spk_emb if isinstance(spk_emb, torch.Tensor) else None

        # Create and save profile with all multi-sample sources
        name = self.edit_name.text().strip()
        lang = self.combo_language.currentData()
        desc = self.edit_desc.text().strip()

        profile = self.profile_mgr.create_profile(
            name=name,
            audio_data=combined_audio,
            sample_rate=sr,
            gpt_cond_latent=gpt_cond_tensor,
            speaker_embedding=spk_emb_tensor,
            language=lang,
            description=desc,
            sample_count=len(valid_paths),
            source_audio_paths=valid_paths
        )

        self.profile_created.emit(profile)

        reply = QMessageBox.question(
            self,
            "Stimmen-Profil erfolgreich erstellt!",
            f"Das Profil '{profile.name}' wurde erfolgreich aus {len(valid_paths)} Audio-Clips generiert "
            "und in Ihrer Stimmen-Bibliothek gesichert.\n\n"
            "Möchten Sie jetzt direkt zur Sprachsynthese wechseln und einen Text mit dieser Stimme sprechen lassen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.navigate_to_synth.emit(profile.id)
