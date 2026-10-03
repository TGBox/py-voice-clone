# -*- coding: utf-8 -*-
"""
Speech Synthesis View (Text-to-Speech):
Allows inputting text, choosing a cloned voice profile, adjusting fine-tuning
parameters (speed, temperature, repetition penalty), and generating audio.
"""

from pathlib import Path
from typing import Optional, List
import os
import datetime
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QSlider, QProgressBar, QMessageBox,
    QScrollArea, QFrame, QMenu
)
from PySide6.QtCore import Qt, QThread, Signal, Slot

from core.voice_profile import VoiceProfileManager, VoiceProfile
from core.xtts_engine import XTTSEngine
from core.model_manager import ModelManager
from core.config import (
    SUPPORTED_LANGUAGES, OUTPUTS_DIR,
    DEFAULT_TEMPERATURE, DEFAULT_REPETITION_PENALTY, DEFAULT_SPEED
)
from core.audio_processor import AudioProcessor
from ui.components.audio_player import AudioPlayerWidget


class SynthesisWorker(QThread):
    """Background worker for neural voice synthesis."""

    progress = Signal(str, int)
    finished = Signal(bool, str, object, int)  # (success, message, audio_array, sample_rate)

    def __init__(
        self,
        engine: XTTSEngine,
        model_mgr: ModelManager,
        profile_mgr: VoiceProfileManager,
        profile: VoiceProfile,
        text: str,
        language: str,
        temperature: float,
        repetition_penalty: float,
        speed: float
    ):
        super().__init__()
        self.engine = engine
        self.model_mgr = model_mgr
        self.profile_mgr = profile_mgr
        self.profile = profile
        self.text = text
        self.language = language
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.speed = speed

    def run(self):
        if not self.model_mgr.is_model_installed():
            # If the user has not yet downloaded the 1.9GB XTTS weights, provide a clean simulated tone
            self.progress.emit("Simulationsmodus (Modellgewichte noch nicht geladen)...", 50)
            sr = 24000
            duration = min(6.0, max(2.0, len(self.text) * 0.06))
            t = np.linspace(0, duration, int(sr * duration), False)
            # Gentle harmonized melodic speech-like tone for UI demonstration
            wav = (0.25 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
            # Apply amplitude envelope
            env = np.ones_like(wav)
            fade = int(sr * 0.1)
            env[:fade] = np.linspace(0, 1, fade)
            env[-fade:] = np.linspace(1, 0, fade)
            wav *= env
            self.finished.emit(
                True,
                "Demo-Audio im Simulationsmodus generiert. (Laden Sie das XTTS-v2 Modell in den Einstellungen herunter, um die echte KI-Stimme zu aktivieren).",
                wav,
                sr
            )
            return

        self.progress.emit("Lade neuronales XTTS-v2 Modell in GPU-VRAM...", 20)
        success, msg = self.engine.load_model()
        if not success:
            self.finished.emit(False, msg, None, 0)
            return

        self.progress.emit("Lade Stimmen-Latents...", 40)
        latents_tuple = self.profile_mgr.load_latents(self.profile, device=self.engine.device)
        gpt_cond, spk_emb = (latents_tuple[0], latents_tuple[1]) if latents_tuple else (None, None)

        self.progress.emit("Synthetisiere Sprache mit RTX 3060...", 65)
        wav, sr, msg = self.engine.synthesize(
            text=self.text,
            language=self.language,
            gpt_cond_latent=gpt_cond,
            speaker_embedding=spk_emb,
            reference_audio_path=self.profile.sample_audio_path if not gpt_cond else None,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            speed=self.speed
        )

        if wav is not None:
            self.progress.emit("Synthese abgeschlossen!", 100)
            self.finished.emit(True, msg, wav, sr)
        else:
            self.finished.emit(False, msg, None, 0)


class SpeechSynthesisView(QWidget):
    """Speech synthesis studio view with profile picker, text prompt and audio generator."""

    def __init__(self, engine: XTTSEngine, model_mgr: ModelManager, profile_mgr: VoiceProfileManager, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.model_mgr = model_mgr
        self.profile_mgr = profile_mgr
        self.worker: Optional[SynthesisWorker] = None
        self.profiles: List[VoiceProfile] = []

        self._setup_ui()
        self.refresh_profiles()

    def _setup_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        # Header
        lbl_title = QLabel("Sprachsynthese (Text-to-Speech)", container)
        lbl_title.setStyleSheet("font-size: 22px; font-weight: 700; color: #ffffff;")
        layout.addWidget(lbl_title)

        lbl_desc = QLabel(
            "Wählen Sie ein zuvor erstelltes Stimmen-Profil aus, tippen Sie den gewünschten Text ein "
            "und lassen Sie ihn sofort mit der geklonten Stimme sprechen.",
            container
        )
        lbl_desc.setStyleSheet("color: #9ca3af; font-size: 13px;")
        layout.addWidget(lbl_desc)

        # Card 1: Voice Profile Selection & Language
        card_profile = QFrame(container)
        card_profile.setProperty("class", "cardFrame")
        cp_layout = QVBoxLayout(card_profile)
        cp_layout.setSpacing(12)

        lbl_sec1 = QLabel("1. Stimme & Sprache auswählen", card_profile)
        lbl_sec1.setProperty("class", "sectionHeader")
        cp_layout.addWidget(lbl_sec1)

        row_pickers = QHBoxLayout()
        row_pickers.setSpacing(16)

        # Profile picker
        vbox_prof = QVBoxLayout()
        lbl_prof = QLabel("Stimmen-Profil:", card_profile)
        lbl_prof.setStyleSheet("color: #d1d5db; font-weight: 600;")
        vbox_prof.addWidget(lbl_prof)

        self.combo_profile = QComboBox(card_profile)
        self.combo_profile.currentIndexChanged.connect(self._on_profile_selected)
        vbox_prof.addWidget(self.combo_profile)
        row_pickers.addLayout(vbox_prof, 2)

        # Language picker
        vbox_lang = QVBoxLayout()
        lbl_lang = QLabel("Zielsprache der Ausgabe:", card_profile)
        lbl_lang.setStyleSheet("color: #d1d5db; font-weight: 600;")
        vbox_lang.addWidget(lbl_lang)

        self.combo_lang = QComboBox(card_profile)
        for code, name in SUPPORTED_LANGUAGES.items():
            self.combo_lang.addItem(f"{name} ({code})", code)
        vbox_lang.addWidget(self.combo_lang)
        row_pickers.addLayout(vbox_lang, 1)

        cp_layout.addLayout(row_pickers)

        # Profile meta badge
        self.lbl_profile_details = QLabel("Kein Profil gewählt.", card_profile)
        self.lbl_profile_details.setStyleSheet("color: #a78bfa; font-size: 12px;")
        cp_layout.addWidget(self.lbl_profile_details)

        layout.addWidget(card_profile)

        # Card 2: Text Editor & Presets
        card_text = QFrame(container)
        card_text.setProperty("class", "cardFrame")
        ct_layout = QVBoxLayout(card_text)
        ct_layout.setSpacing(10)

        header_text_row = QHBoxLayout()
        lbl_sec2 = QLabel("2. Zu sprechender Text", card_text)
        lbl_sec2.setProperty("class", "sectionHeader")
        header_text_row.addWidget(lbl_sec2)

        header_text_row.addStretch()

        self.btn_presets = QPushButton("📋 Beispieltexte...", card_text)
        self.btn_presets.setProperty("class", "secondaryButton")
        self.btn_presets.clicked.connect(self._show_preset_menu)
        header_text_row.addWidget(self.btn_presets)

        ct_layout.addLayout(header_text_row)

        self.text_editor = QTextEdit(card_text)
        self.text_editor.setPlaceholderText(
            "Tippen Sie hier den Text ein, der gesprochen werden soll...\n\n"
            "Beispiel: Hallo! Dies ist meine neu erstellte KI-Stimme. Sie klingt verblüffend ähnlich wie das Original."
        )
        self.text_editor.setMinimumHeight(110)
        self.text_editor.textChanged.connect(self._on_text_changed)
        ct_layout.addWidget(self.text_editor)

        # Text statistics
        self.lbl_text_stats = QLabel("0 Zeichen | Geschätzte Sprechdauer: 0 Sek.", card_text)
        self.lbl_text_stats.setStyleSheet("color: #9ca3af; font-size: 11px;")
        ct_layout.addWidget(self.lbl_text_stats)

        layout.addWidget(card_text)

        # Card 3: Fine-Tuning Controls
        card_params = QFrame(container)
        card_params.setProperty("class", "cardFrame")
        param_layout = QVBoxLayout(card_params)
        param_layout.setSpacing(14)

        lbl_sec3 = QLabel("3. Sprachdynamik & Feineinstellung", card_params)
        lbl_sec3.setProperty("class", "sectionHeader")
        param_layout.addWidget(lbl_sec3)

        sliders_grid = QHBoxLayout()
        sliders_grid.setSpacing(24)

        # Speed slider
        vbox_speed = QVBoxLayout()
        self.lbl_speed_val = QLabel(f"Geschwindigkeit: {DEFAULT_SPEED:.1f}x", card_params)
        self.lbl_speed_val.setStyleSheet("color: #d1d5db; font-size: 12px; font-weight: 600;")
        vbox_speed.addWidget(self.lbl_speed_val)

        self.slider_speed = QSlider(Qt.Orientation.Horizontal, card_params)
        self.slider_speed.setRange(50, 200)
        self.slider_speed.setValue(int(DEFAULT_SPEED * 100))
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        vbox_speed.addWidget(self.slider_speed)
        sliders_grid.addLayout(vbox_speed)

        # Temperature slider
        vbox_temp = QVBoxLayout()
        self.lbl_temp_val = QLabel(f"Temperatur (Expressivität): {DEFAULT_TEMPERATURE:.2f}", card_params)
        self.lbl_temp_val.setStyleSheet("color: #d1d5db; font-size: 12px; font-weight: 600;")
        vbox_temp.addWidget(self.lbl_temp_val)

        self.slider_temp = QSlider(Qt.Orientation.Horizontal, card_params)
        self.slider_temp.setRange(10, 100)
        self.slider_temp.setValue(int(DEFAULT_TEMPERATURE * 100))
        self.slider_temp.valueChanged.connect(self._on_temp_changed)
        vbox_temp.addWidget(self.slider_temp)
        sliders_grid.addLayout(vbox_temp)

        # Repetition penalty slider
        vbox_rep = QVBoxLayout()
        self.lbl_rep_val = QLabel(f"Wiederholungsstrafe: {DEFAULT_REPETITION_PENALTY:.1f}", card_params)
        self.lbl_rep_val.setStyleSheet("color: #d1d5db; font-size: 12px; font-weight: 600;")
        vbox_rep.addWidget(self.lbl_rep_val)

        self.slider_rep = QSlider(Qt.Orientation.Horizontal, card_params)
        self.slider_rep.setRange(10, 50)
        self.slider_rep.setValue(int(DEFAULT_REPETITION_PENALTY * 10))
        self.slider_rep.valueChanged.connect(self._on_rep_changed)
        vbox_rep.addWidget(self.slider_rep)
        sliders_grid.addLayout(vbox_rep)

        param_layout.addLayout(sliders_grid)
        layout.addWidget(card_params)

        # Progress Indicator
        self.progress_bar = QProgressBar(container)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.lbl_synth_status = QLabel("", container)
        self.lbl_synth_status.setStyleSheet("color: #a78bfa; font-weight: 600; font-size: 13px;")
        self.lbl_synth_status.setVisible(False)
        layout.addWidget(self.lbl_synth_status)

        # Big Generate Action Button
        self.btn_synthesize = QPushButton("🔊 Audio generieren (Stimme sprechen lassen)", container)
        self.btn_synthesize.setProperty("class", "primaryButton")
        self.btn_synthesize.setMinimumHeight(48)
        self.btn_synthesize.setStyleSheet("font-size: 15px; font-weight: 700; background-color: #7c3aed;")
        self.btn_synthesize.clicked.connect(self._start_synthesis)
        layout.addWidget(self.btn_synthesize)

        # Card 4: Audio Player & Result Waveform
        card_result = QFrame(container)
        card_result.setProperty("class", "cardFrameHighlight")
        cr_layout = QVBoxLayout(card_result)
        cr_layout.setSpacing(12)

        lbl_res_title = QLabel("Generiertes Audio-Ergebnis", card_result)
        lbl_res_title.setProperty("class", "sectionHeader")
        cr_layout.addWidget(lbl_res_title)

        self.result_player = AudioPlayerWidget(card_result, show_export=True)
        cr_layout.addWidget(self.result_player)

        layout.addWidget(card_result)
        layout.addStretch()

        scroll.setWidget(container)

        view_layout = QVBoxLayout(self)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(scroll)

        # Initial dummy text
        self.text_editor.setText("Hallo! Dies ist meine neu erstellte Stimme. Ich spreche Texte flüssig und natürlich aus.")

    def refresh_profiles(self, select_profile_id: Optional[str] = None):
        """Reload profile list from disk and update combo box."""
        self.profiles = self.profile_mgr.list_profiles()
        self.combo_profile.clear()

        if not self.profiles:
            self.combo_profile.addItem("Keine Profile vorhanden (Bitte zuerst eine Stimme erstellen)", None)
            self.lbl_profile_details.setText("Erstellen Sie unter 'Stimme klonen' Ihr erstes Stimmen-Profil.")
            self.btn_synthesize.setEnabled(False)
            return

        target_idx = 0
        for idx, p in enumerate(self.profiles):
            self.combo_profile.addItem(f"👤 {p.name} ({p.duration_sec:.1f}s)", p.id)
            if select_profile_id and p.id == select_profile_id:
                target_idx = idx

        self.combo_profile.setCurrentIndex(target_idx)
        self.btn_synthesize.setEnabled(True)
        self._on_profile_selected(target_idx)

    def select_profile(self, profile_id: str):
        """Programmatically select a specific profile."""
        self.refresh_profiles(select_profile_id=profile_id)

    def _on_profile_selected(self, index: int):
        profile_id = self.combo_profile.currentData()
        if not profile_id:
            return

        profile = self.profile_mgr.get_profile(profile_id)
        if profile:
            self.lbl_profile_details.setText(
                f"Ausgewählt: {profile.name} | Dauer Referenz: {profile.duration_sec:.1f}s | "
                f"Modell-Latents: {'Vorhanden ✓' if profile.has_latents else 'Referenz-Audio'}"
            )
            # Match language dropdown if found
            for i in range(self.combo_lang.count()):
                if self.combo_lang.itemData(i) == profile.language:
                    self.combo_lang.setCurrentIndex(i)
                    break

    def _on_text_changed(self):
        text = self.text_editor.toPlainText().strip()
        chars = len(text)
        # Approximate: ~15 characters per second of speech
        est_sec = max(1, chars // 15)
        self.lbl_text_stats.setText(f"{chars} Zeichen | Geschätzte Sprechdauer: ~{est_sec} Sekunden")

    def _on_speed_changed(self, val: int):
        speed = val / 100.0
        self.lbl_speed_val.setText(f"Geschwindigkeit: {speed:.2f}x")

    def _on_temp_changed(self, val: int):
        temp = val / 100.0
        self.lbl_temp_val.setText(f"Temperatur (Expressivität): {temp:.2f}")

    def _on_rep_changed(self, val: int):
        rep = val / 10.0
        self.lbl_rep_val.setText(f"Wiederholungsstrafe: {rep:.1f}")

    def _show_preset_menu(self):
        menu = QMenu(self)
        presets = [
            ("Freundliche Begrüßung", "Guten Tag! Ich freue mich sehr, heute mit Ihnen zu sprechen. Wie kann ich Ihnen behilflich sein?"),
            ("Hörbuch-Erzählung", "Es war eine sternenklare Nacht, und der Wind strich leise durch die alten Baumkronen des Waldes."),
            ("Nachrichten-Meldung", "Hier sind die aktuellen Meldungen des Tages: Die Technologie für Sprachmodelle schreitet rasant voran."),
            ("Technischer Erklärtext", "Künstliche neuronale Netze analysieren Audiofrequenzen, um die individuelle Stimmfarbe zu rekonstruieren.")
        ]
        for title, text in presets:
            action = menu.addAction(title)
            action.triggered.connect(lambda checked, t=text: self.text_editor.setText(t))

        menu.exec(self.btn_presets.mapToGlobal(self.btn_presets.rect().bottomLeft()))

    def _start_synthesis(self):
        profile_id = self.combo_profile.currentData()
        if not profile_id:
            QMessageBox.warning(self, "Profil erforderlich", "Bitte wählen Sie zuerst ein Stimmen-Profil aus.")
            return

        profile = self.profile_mgr.get_profile(profile_id)
        if not profile:
            QMessageBox.warning(self, "Fehler", "Ausgewähltes Profil konnte nicht geladen werden.")
            return

        text = self.text_editor.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Text erforderlich", "Bitte geben Sie einen Text ein, der gesprochen werden soll.")
            self.text_editor.setFocus()
            return

        lang = self.combo_lang.currentData()
        speed = self.slider_speed.value() / 100.0
        temp = self.slider_temp.value() / 100.0
        rep = self.slider_rep.value() / 10.0

        self.btn_synthesize.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(15)
        self.lbl_synth_status.setVisible(True)
        self.lbl_synth_status.setText("Initialisiere Sprachsynthese...")

        self.worker = SynthesisWorker(
            self.engine,
            self.model_mgr,
            self.profile_mgr,
            profile,
            text,
            lang,
            temp,
            rep,
            speed
        )
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    @Slot(str, int)
    def _on_worker_progress(self, msg: str, percent: int):
        self.lbl_synth_status.setText(msg)
        self.progress_bar.setValue(percent)

    @Slot(bool, str, object, int)
    def _on_worker_finished(self, success: bool, msg: str, wav: object, sr: int):
        self.btn_synthesize.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_synth_status.setVisible(False)

        if not success or not isinstance(wav, np.ndarray):
            QMessageBox.critical(self, "Fehler bei der Sprachsynthese", f"Die Sprachsynthese ist fehlgeschlagen:\n{msg}")
            return

        # Save to outputs directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"speech_{timestamp}.wav"
        output_path = OUTPUTS_DIR / output_filename
        AudioProcessor.save_audio(output_path, wav, sr)

        # Load into result player
        self.result_player.load_audio_file(output_path)
        self.result_player.toggle_playback()  # Auto play result
