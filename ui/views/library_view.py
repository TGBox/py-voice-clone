# -*- coding: utf-8 -*-
"""
Voice Library View:
Displays all stored voice profiles as interactive cards,
provides audio sample playback, deletion, zip export and import.
"""

from pathlib import Path
from typing import List, Optional
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QMessageBox, QScrollArea, QFrame,
    QGridLayout, QSizePolicy
)
from PySide6.QtCore import Qt, Signal

from core.voice_profile import VoiceProfileManager, VoiceProfile
from core.audio_processor import AudioProcessor
from ui.components.audio_player import AudioPlayerWidget


class ProfileCard(QFrame):
    """Card widget representing an individual voice profile."""

    play_requested = Signal(str)  # Emits audio_path
    use_in_synth = Signal(str)    # Emits profile_id
    delete_requested = Signal(str) # Emits profile_id
    export_requested = Signal(str) # Emits profile_id

    def __init__(self, profile: VoiceProfile, parent=None):
        super().__init__(parent)
        self.profile = profile
        self.setObjectName("cardFrame")
        self.setProperty("class", "cardFrame")
        self.setMinimumHeight(140)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # Top row: Name and badges
        top_row = QHBoxLayout()

        lbl_icon = QLabel("🎙", self)
        lbl_icon.setStyleSheet("font-size: 20px;")
        top_row.addWidget(lbl_icon)

        lbl_name = QLabel(self.profile.name, self)
        lbl_name.setStyleSheet("font-size: 16px; font-weight: 700; color: #ffffff;")
        top_row.addWidget(lbl_name)

        top_row.addStretch()

        # Language badge
        lbl_lang = QLabel(self.profile.language.upper(), self)
        lbl_lang.setProperty("class", "badgeInfo")
        top_row.addWidget(lbl_lang)

        # Duration badge
        lbl_dur = QLabel(f"{self.profile.duration_sec:.1f}s", self)
        lbl_dur.setProperty("class", "badgeSuccess")
        top_row.addWidget(lbl_dur)

        layout.addLayout(top_row)

        # Description / details
        desc_text = self.profile.description or "Keine zusätzliche Beschreibung."
        lbl_desc = QLabel(f"Notizen: {desc_text}", self)
        lbl_desc.setStyleSheet("color: #9ca3af; font-size: 12px;")
        layout.addWidget(lbl_desc)

        # Actions row
        act_row = QHBoxLayout()
        act_row.setSpacing(8)

        btn_play = QPushButton("▶ Vorschau", self)
        btn_play.setProperty("class", "secondaryButton")
        btn_play.clicked.connect(lambda: self.play_requested.emit(str(self.profile.sample_audio_path)))
        act_row.addWidget(btn_play)

        btn_use = QPushButton("In Synthese nutzen →", self)
        btn_use.setProperty("class", "primaryButton")
        btn_use.clicked.connect(lambda: self.use_in_synth.emit(self.profile.id))
        act_row.addWidget(btn_use)

        act_row.addStretch()

        btn_export = QPushButton("Exportieren (.zip)", self)
        btn_export.setProperty("class", "secondaryButton")
        btn_export.clicked.connect(lambda: self.export_requested.emit(self.profile.id))
        act_row.addWidget(btn_export)

        btn_del = QPushButton("Löschen", self)
        btn_del.setProperty("class", "dangerButton")
        btn_del.clicked.connect(lambda: self.delete_requested.emit(self.profile.id))
        act_row.addWidget(btn_del)

        layout.addLayout(act_row)


class VoiceLibraryView(QWidget):
    """View displaying and managing the library of saved voice profiles."""

    navigate_to_synth = Signal(str)
    navigate_to_clone = Signal()

    def __init__(self, profile_mgr: VoiceProfileManager, parent=None):
        super().__init__(parent)
        self.profile_mgr = profile_mgr
        self._setup_ui()
        self.refresh_library()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        # Header bar
        header_row = QHBoxLayout()

        lbl_title = QLabel("Stimmen-Bibliothek", self)
        lbl_title.setStyleSheet("font-size: 22px; font-weight: 700; color: #ffffff;")
        header_row.addWidget(lbl_title)

        header_row.addStretch()

        # Action buttons in header
        btn_import = QPushButton("📥 Profil importieren (.zip)", self)
        btn_import.setProperty("class", "secondaryButton")
        btn_import.clicked.connect(self._import_profile_zip)
        header_row.addWidget(btn_import)

        btn_new_clone = QPushButton("+ Neue Stimme erstellen", self)
        btn_new_clone.setProperty("class", "primaryButton")
        btn_new_clone.clicked.connect(self.navigate_to_clone.emit)
        header_row.addWidget(btn_new_clone)

        main_layout.addLayout(header_row)

        # Search / filter bar
        search_row = QHBoxLayout()
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("🔍 Stimmen nach Namen filtern...")
        self.search_input.textChanged.connect(self._on_search_changed)
        search_row.addWidget(self.search_input)
        main_layout.addLayout(search_row)

        # Embedded preview audio player
        self.preview_player = AudioPlayerWidget(self, show_export=False)
        self.preview_player.setVisible(False)
        main_layout.addWidget(self.preview_player)

        # Scroll area with cards
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(12)
        self.cards_layout.addStretch()

        self.scroll_area.setWidget(self.cards_container)
        main_layout.addWidget(self.scroll_area)

    def refresh_library(self):
        """Reload all profiles and re-render cards."""
        # Clear existing cards
        while self.cards_layout.count() > 0:
            item = self.cards_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        filter_text = self.search_input.text().strip().lower()
        profiles = self.profile_mgr.list_profiles()

        matching_profiles = [
            p for p in profiles if not filter_text or filter_text in p.name.lower() or filter_text in p.description.lower()
        ]

        if not matching_profiles:
            empty_frame = QFrame(self.cards_container)
            empty_frame.setProperty("class", "cardFrame")
            ef_layout = QVBoxLayout(empty_frame)
            ef_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ef_layout.setContentsMargins(40, 40, 40, 40)

            lbl_empty_icon = QLabel("🎙️", empty_frame)
            lbl_empty_icon.setStyleSheet("font-size: 36px;")
            lbl_empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ef_layout.addWidget(lbl_empty_icon)

            lbl_empty = QLabel("Noch keine Stimmen-Profile vorhanden.", empty_frame)
            lbl_empty.setStyleSheet("font-size: 16px; font-weight: 600; color: #ffffff;")
            lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ef_layout.addWidget(lbl_empty)

            lbl_empty_sub = QLabel("Klicken Sie auf 'Neue Stimme erstellen', um Ihr erstes Voice-Modell zu generieren.", empty_frame)
            lbl_empty_sub.setStyleSheet("color: #9ca3af; font-size: 13px; margin-top: 4px;")
            lbl_empty_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ef_layout.addWidget(lbl_empty_sub)

            self.cards_layout.addWidget(empty_frame)
        else:
            for p in matching_profiles:
                card = ProfileCard(p, self.cards_container)
                card.play_requested.connect(self._on_play_sample)
                card.use_in_synth.connect(self.navigate_to_synth.emit)
                card.delete_requested.connect(self._on_delete_profile)
                card.export_requested.connect(self._on_export_profile)
                self.cards_layout.addWidget(card)

        self.cards_layout.addStretch()

    def _on_search_changed(self):
        self.refresh_library()

    def _on_play_sample(self, audio_path: str):
        if os.path.exists(audio_path):
            self.preview_player.setVisible(True)
            self.preview_player.load_audio_file(audio_path)
            self.preview_player.toggle_playback()

    def _on_delete_profile(self, profile_id: str):
        profile = self.profile_mgr.get_profile(profile_id)
        if not profile:
            return

        reply = QMessageBox.question(
            self,
            "Profil löschen",
            f"Möchten Sie das Stimmen-Profil '{profile.name}' wirklich unwiderruflich löschen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.profile_mgr.delete_profile(profile_id)
            self.refresh_library()

    def _on_export_profile(self, profile_id: str):
        profile = self.profile_mgr.get_profile(profile_id)
        if not profile:
            return

        dest_file, _ = QFileDialog.getSaveFileName(
            self,
            "Stimmen-Profil exportieren",
            f"{profile.name}_voice_profile.zip",
            "Zip Archiv (*.zip)"
        )
        if dest_file:
            success = self.profile_mgr.export_profile_zip(profile_id, dest_file)
            if success:
                QMessageBox.information(self, "Export erfolgreich", f"Profil wurde exportiert nach:\n{dest_file}")
            else:
                QMessageBox.critical(self, "Fehler", "Export fehlgeschlagen.")

    def _import_profile_zip(self):
        zip_path, _ = QFileDialog.getOpenFileName(
            self,
            "Stimmen-Profil importieren",
            "",
            "Zip Archiv (*.zip)"
        )
        if zip_path:
            profile = self.profile_mgr.import_profile_zip(zip_path)
            if profile:
                QMessageBox.information(self, "Import erfolgreich", f"Stimmen-Profil '{profile.name}' importiert!")
                self.refresh_library()
            else:
                QMessageBox.critical(self, "Fehler", "Das ZIP-Archiv enthält kein gültiges Stimmen-Profil.")
