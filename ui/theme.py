# -*- coding: utf-8 -*-
"""
Modern Dark Theme stylesheet and styling tokens for PySide6 GUI.
"""

DARK_THEME_QSS = """
/* Global Window & Font Settings */
QWidget {
    background-color: #0f1117;
    color: #f3f4f6;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif;
    font-size: 13px;
    selection-background-color: #7c3aed;
    selection-color: #ffffff;
}

/* Main Window */
QMainWindow {
    background-color: #0f1117;
}

/* Sidebar Navigation */
QFrame#sidebarFrame {
    background-color: #141720;
    border-right: 1px solid #232736;
}

QLabel#appTitleLabel {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    padding: 12px 8px 4px 8px;
}

QLabel#appSubtitleLabel {
    font-size: 11px;
    color: #8b5cf6;
    font-weight: 600;
    padding-left: 8px;
    margin-bottom: 12px;
}

/* Sidebar Navigation Buttons */
QPushButton.navButton {
    background-color: transparent;
    color: #9ca3af;
    text-align: left;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid transparent;
    margin: 2px 8px;
}

QPushButton.navButton:hover {
    background-color: #1e2230;
    color: #f3f4f6;
}

QPushButton.navButton:checked, QPushButton.navButton[active="true"] {
    background-color: #272042;
    color: #a78bfa;
    border: 1px solid #7c3aed;
}

/* Cards & Content Panels */
QFrame.cardFrame {
    background-color: #181c26;
    border: 1px solid #272d3f;
    border-radius: 12px;
    padding: 16px;
}

QFrame.cardFrameHighlight {
    background-color: #1a1e2d;
    border: 1px solid #3b3a58;
    border-radius: 12px;
    padding: 16px;
}

/* Headers */
QLabel.sectionHeader {
    font-size: 16px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
}

QLabel.sectionSubtext {
    font-size: 12px;
    color: #9ca3af;
    margin-bottom: 12px;
}

/* Buttons */
QPushButton.primaryButton {
    background-color: #7c3aed;
    color: #ffffff;
    font-size: 13px;
    font-weight: 600;
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
}

QPushButton.primaryButton:hover {
    background-color: #8b5cf6;
}

QPushButton.primaryButton:pressed {
    background-color: #6d28d9;
}

QPushButton.primaryButton:disabled {
    background-color: #374151;
    color: #6b7280;
}

QPushButton.secondaryButton {
    background-color: #232736;
    color: #e5e7eb;
    font-size: 13px;
    font-weight: 600;
    padding: 9px 18px;
    border-radius: 8px;
    border: 1px solid #374151;
}

QPushButton.secondaryButton:hover {
    background-color: #2d3346;
    border-color: #4b5563;
}

QPushButton.secondaryButton:pressed {
    background-color: #1a1e2a;
}

QPushButton.dangerButton {
    background-color: #991b1b;
    color: #fee2e2;
    font-weight: 600;
    padding: 8px 14px;
    border-radius: 8px;
    border: 1px solid #b91c1c;
}

QPushButton.dangerButton:hover {
    background-color: #dc2626;
}

/* Input Fields & Text Edits */
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #11141c;
    color: #f3f4f6;
    border: 1px solid #2d3348;
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 13px;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #7c3aed;
    background-color: #141722;
}

/* ComboBox */
QComboBox {
    background-color: #11141c;
    color: #f3f4f6;
    border: 1px solid #2d3348;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    min-height: 20px;
}

QComboBox:hover {
    border-color: #4b5563;
}

QComboBox:focus {
    border: 1px solid #7c3aed;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 25px;
    border-left-width: 0px;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}

QComboBox QAbstractItemView {
    background-color: #181c26;
    color: #f3f4f6;
    border: 1px solid #2d3348;
    selection-background-color: #7c3aed;
    selection-color: #ffffff;
    border-radius: 6px;
    outline: none;
    padding: 4px;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 6px;
    background: #232736;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #7c3aed;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #7c3aed;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #a78bfa;
}

/* Progress Bar */
QProgressBar {
    background-color: #181c26;
    border: 1px solid #272d3f;
    border-radius: 6px;
    text-align: center;
    color: #f3f4f6;
    font-size: 11px;
    font-weight: 600;
    height: 16px;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #06b6d4);
    border-radius: 5px;
}

/* ScrollBars */
QScrollBar:vertical {
    background: #11141c;
    width: 10px;
    border-radius: 5px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #2d3348;
    border-radius: 5px;
    min-height: 25px;
}

QScrollBar::handle:vertical:hover {
    background: #4b5563;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #11141c;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #2d3348;
    border-radius: 5px;
    min-width: 25px;
}

/* Status Bar */
QStatusBar {
    background-color: #11141c;
    border-top: 1px solid #232736;
    color: #9ca3af;
    font-size: 12px;
    padding: 4px 10px;
}

/* Tooltip */
QToolTip {
    background-color: #1f2433;
    color: #ffffff;
    border: 1px solid #3b4259;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
}

/* Badges & Tags */
QLabel.badgeSuccess {
    background-color: #064e3b;
    color: #34d399;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid #059669;
}

QLabel.badgeWarning {
    background-color: #451a03;
    color: #fbbf24;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid #b45309;
}

QLabel.badgeInfo {
    background-color: #1e1b4b;
    color: #a78bfa;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid #6366f1;
}
"""
