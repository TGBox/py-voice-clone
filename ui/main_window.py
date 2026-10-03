# -*- coding: utf-8 -*-
"""
Main Application Window for VoiceClone AI Studio.
Includes sidebar navigation, view switching, fullscreen/windowed toggle (F11),
real-time status updates, and view orchestration.
"""

from typing import Optional
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QLabel, QStackedWidget, QStatusBar, QFrame, QButtonGroup,
    QMenuBar, QMenu, QMessageBox
)
from PySide6.QtCore import Qt, QKeyCombination, QTimer
from PySide6.QtGui import QKeySequence, QShortcut, QAction

from core.voice_profile import VoiceProfileManager
from core.xtts_engine import XTTSEngine
from core.model_manager import ModelManager
from ui.theme import DARK_THEME_QSS
from ui.views.clone_view import CloneVoiceView
from ui.views.synth_view import SpeechSynthesisView
from ui.views.library_view import VoiceLibraryView
from ui.views.settings_view import SettingsView


class MainWindow(QMainWindow):
    """Primary application window with sidebar, header, stacked views and fullscreen toggle."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoiceClone AI Studio - Neuronale Stimmen-Klonung & TTS")
        self.setMinimumSize(1150, 760)
        self.resize(1320, 860)

        # Core backend managers
        self.profile_mgr = VoiceProfileManager()
        self.model_mgr = ModelManager()
        self.engine = XTTSEngine()

        # Apply dark theme
        self.setStyleSheet(DARK_THEME_QSS)

        self._setup_menu_and_shortcuts()
        self._setup_ui()
        self._connect_signals()

        # Initial view selection
        self.btn_nav_clone.setChecked(True)
        self._on_nav_clicked(0)

        # GPU quick status refresh timer
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(3000)
        self.status_timer.timeout.connect(self._update_quick_status)
        self.status_timer.start()
        self._update_quick_status()

        # Check model installation shortly after window startup
        QTimer.singleShot(700, self._check_initial_model_status)

    def _setup_menu_and_shortcuts(self):
        # Menu bar
        menu_bar = self.menuBar()

        # Menu: Ansicht (View)
        menu_view = menu_bar.addMenu("Ansicht")

        # Fullscreen action
        self.action_fullscreen = QAction("Vollbildmodus umschalten (F11)", self)
        self.action_fullscreen.setShortcut(QKeySequence("F11"))
        self.action_fullscreen.triggered.connect(self.toggle_fullscreen)
        menu_view.addAction(self.action_fullscreen)

        # Quick F11 shortcut
        self.shortcut_f11 = QShortcut(QKeySequence(Qt.Key.Key_F11), self)
        self.shortcut_f11.activated.connect(self.toggle_fullscreen)

        # Escape shortcut to exit fullscreen if currently in fullscreen
        self.shortcut_esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.shortcut_esc.activated.connect(self._on_escape_pressed)

    def _setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Left Sidebar Navigation
        sidebar_frame = QFrame(central_widget)
        sidebar_frame.setObjectName("sidebarFrame")
        sidebar_frame.setFixedWidth(240)
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(12, 16, 12, 16)
        sidebar_layout.setSpacing(6)

        # App Brand & Logo
        lbl_brand = QLabel("🎙 VoiceClone AI", sidebar_frame)
        lbl_brand.setObjectName("appTitleLabel")
        sidebar_layout.addWidget(lbl_brand)

        lbl_sub = QLabel("XTTS-v2 Neural Studio", sidebar_frame)
        lbl_sub.setObjectName("appSubtitleLabel")
        sidebar_layout.addWidget(lbl_sub)

        # Navigation buttons group
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        self.btn_nav_clone = QPushButton("🎙 Stimme klonen", sidebar_frame)
        self.btn_nav_clone.setProperty("class", "navButton")
        self.btn_nav_clone.setCheckable(True)
        self.nav_group.addButton(self.btn_nav_clone, 0)
        sidebar_layout.addWidget(self.btn_nav_clone)

        self.btn_nav_synth = QPushButton("🔊 Sprachsynthese (TTS)", sidebar_frame)
        self.btn_nav_synth.setProperty("class", "navButton")
        self.btn_nav_synth.setCheckable(True)
        self.nav_group.addButton(self.btn_nav_synth, 1)
        sidebar_layout.addWidget(self.btn_nav_synth)

        self.btn_nav_library = QPushButton("📚 Stimmen-Bibliothek", sidebar_frame)
        self.btn_nav_library.setProperty("class", "navButton")
        self.btn_nav_library.setCheckable(True)
        self.nav_group.addButton(self.btn_nav_library, 2)
        sidebar_layout.addWidget(self.btn_nav_library)

        self.btn_nav_settings = QPushButton("⚙ Modell & Einstellungen", sidebar_frame)
        self.btn_nav_settings.setProperty("class", "navButton")
        self.btn_nav_settings.setCheckable(True)
        self.nav_group.addButton(self.btn_nav_settings, 3)
        sidebar_layout.addWidget(self.btn_nav_settings)

        sidebar_layout.addStretch()

        # Sidebar footer info
        card_foot = QFrame(sidebar_frame)
        card_foot.setStyleSheet("background-color: #1a1e2b; border-radius: 8px; padding: 10px;")
        foot_layout = QVBoxLayout(card_foot)
        foot_layout.setContentsMargins(8, 8, 8, 8)
        foot_layout.setSpacing(4)

        lbl_engine_info = QLabel("Engine: Coqui XTTS-v2", card_foot)
        lbl_engine_info.setStyleSheet("color: #a78bfa; font-size: 11px; font-weight: 600;")
        foot_layout.addWidget(lbl_engine_info)

        lbl_ver = QLabel("PySide6 Desktop Edition", card_foot)
        lbl_ver.setStyleSheet("color: #6b7280; font-size: 10px;")
        foot_layout.addWidget(lbl_ver)

        sidebar_layout.addWidget(card_foot)
        root_layout.addWidget(sidebar_frame)

        # 2. Main Content Area (Header + Stacked Views)
        content_frame = QWidget(central_widget)
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Top Header Bar
        header_frame = QFrame(content_frame)
        header_frame.setStyleSheet("background-color: #141720; border-bottom: 1px solid #232736;")
        header_frame.setFixedHeight(54)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(24, 0, 24, 0)

        self.lbl_view_title = QLabel("Stimme erstellen & klonen", header_frame)
        self.lbl_view_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #ffffff;")
        header_layout.addWidget(self.lbl_view_title)

        header_layout.addStretch()

        # Hardware status indicator badge
        self.lbl_gpu_quick = QLabel("NVIDIA RTX 3060", header_frame)
        self.lbl_gpu_quick.setProperty("class", "badgeSuccess")
        header_layout.addWidget(self.lbl_gpu_quick)

        # Fullscreen Toggle Button (Windowed / Fullscreen mode)
        self.btn_fullscreen_toggle = QPushButton("⛶ Vollbild", header_frame)
        self.btn_fullscreen_toggle.setProperty("class", "secondaryButton")
        self.btn_fullscreen_toggle.setToolTip("Zwischen Vollbild und Fenster wechseln (F11)")
        self.btn_fullscreen_toggle.clicked.connect(self.toggle_fullscreen)
        header_layout.addWidget(self.btn_fullscreen_toggle)

        content_layout.addWidget(header_frame)

        # Stacked Views Container
        self.stacked_widget = QStackedWidget(content_frame)

        # Initialize the views
        self.view_clone = CloneVoiceView(self.engine, self.model_mgr, self.profile_mgr, self.stacked_widget)
        self.view_synth = SpeechSynthesisView(self.engine, self.model_mgr, self.profile_mgr, self.stacked_widget)
        self.view_library = VoiceLibraryView(self.profile_mgr, self.stacked_widget)
        self.view_settings = SettingsView(self.model_mgr, self.engine, self.stacked_widget)

        self.stacked_widget.addWidget(self.view_clone)      # Index 0
        self.stacked_widget.addWidget(self.view_synth)      # Index 1
        self.stacked_widget.addWidget(self.view_library)    # Index 2
        self.stacked_widget.addWidget(self.view_settings)   # Index 3

        content_layout.addWidget(self.stacked_widget)
        root_layout.addWidget(content_frame)

        # 3. Status Bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Bereit | F11 für Vollbildmodus")

    def _connect_signals(self):
        # Nav buttons
        self.btn_nav_clone.clicked.connect(lambda: self._on_nav_clicked(0))
        self.btn_nav_synth.clicked.connect(lambda: self._on_nav_clicked(1))
        self.btn_nav_library.clicked.connect(lambda: self._on_nav_clicked(2))
        self.btn_nav_settings.clicked.connect(lambda: self._on_nav_clicked(3))

        # Inter-view signals
        self.view_clone.profile_created.connect(self._on_profile_created)
        self.view_clone.navigate_to_synth.connect(self._navigate_to_synth_with_profile)

        self.view_library.navigate_to_synth.connect(self._navigate_to_synth_with_profile)
        self.view_library.navigate_to_clone.connect(lambda: self._on_nav_clicked(0))

    def _on_nav_clicked(self, index: int):
        self.stacked_widget.setCurrentIndex(index)

        titles = [
            "Stimme erstellen & klonen",
            "Sprachsynthese (Text-to-Speech)",
            "Stimmen-Bibliothek",
            "Modell & Systemeinstellungen"
        ]
        if index < len(titles):
            self.lbl_view_title.setText(titles[index])

        # Synchronize nav button states
        buttons = [self.btn_nav_clone, self.btn_nav_synth, self.btn_nav_library, self.btn_nav_settings]
        for i, btn in enumerate(buttons):
            btn.setChecked(i == index)

        # Refresh library when entering it
        if index == 2:
            self.view_library.refresh_library()
        elif index == 1:
            self.view_synth.refresh_profiles()

    def _on_profile_created(self, profile):
        self.view_library.refresh_library()
        self.view_synth.refresh_profiles(select_profile_id=profile.id)
        self.status_bar.showMessage(f"Stimmen-Profil '{profile.name}' erfolgreich gespeichert!", 5000)

    def _navigate_to_synth_with_profile(self, profile_id: str):
        self._on_nav_clicked(1)
        self.view_synth.select_profile(profile_id)

    def toggle_fullscreen(self):
        """Toggle between Fullscreen mode and Windowed mode."""
        if self.isFullScreen():
            self.showNormal()
            self.btn_fullscreen_toggle.setText("⛶ Vollbild")
            self.action_fullscreen.setText("Vollbildmodus aktivieren (F11)")
            self.status_bar.showMessage("Fenstermodus aktiv (F11 für Vollbild)", 3000)
        else:
            self.showFullScreen()
            self.btn_fullscreen_toggle.setText("🗗 Fenster")
            self.action_fullscreen.setText("Fenstermodus wiederherstellen (F11)")
            self.status_bar.showMessage("Vollbildmodus aktiv (F11 oder Esc zum Beenden)", 3000)

    def _on_escape_pressed(self):
        """Exit fullscreen on Esc key if currently fullscreen."""
        if self.isFullScreen():
            self.showNormal()
            self.btn_fullscreen_toggle.setText("⛶ Vollbild")
            self.action_fullscreen.setText("Vollbildmodus aktivieren (F11)")

    def _update_quick_status(self):
        info = self.model_mgr.get_hardware_info()
        if info["cuda_available"]:
            self.lbl_gpu_quick.setText(f"🟢 {info['device_name']} ({info['vram_used_mb']:.0f} / {info['vram_total_mb']:.0f} MB)")
            self.lbl_gpu_quick.setProperty("class", "badgeSuccess")
        else:
            self.lbl_gpu_quick.setText("🟡 CPU Modus")
            self.lbl_gpu_quick.setProperty("class", "badgeWarning")

    def _check_initial_model_status(self):
        """Prompt user on launch if neural weights are not yet downloaded."""
        if not self.model_mgr.is_model_installed():
            reply = QMessageBox.question(
                self,
                "XTTS-v2 Modell erforderlich",
                "Das neuronale Basismodell (Coqui XTTS-v2, ca. 1.9 GB) ist auf diesem Computer noch nicht installiert.\n\n"
                "Möchten Sie jetzt zu den Einstellungen wechseln und das Modell automatisch von HuggingFace herunterladen?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._on_nav_clicked(3)  # Switch to Settings view
                self.view_settings._start_model_download()
