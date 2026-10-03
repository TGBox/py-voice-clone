# -*- coding: utf-8 -*-
"""
Interactive audio waveform visualization widget.
Supports playback progress indication, hover, and selection/trim region.
"""

from typing import List, Optional
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QLinearGradient


class WaveformWidget(QWidget):
    """
    Renders audio peak data as a smooth, stylized waveform.
    Allows seeking by clicking or dragging.
    """

    position_clicked = Signal(float)  # Emits normalized position [0.0, 1.0]
    range_changed = Signal(float, float)  # Emits start and end [0.0, 1.0]

    def __init__(self, parent=None, is_interactive: bool = True, show_range: bool = False):
        super().__init__(parent)
        self.is_interactive = is_interactive
        self.show_range = show_range

        self.peaks: List[float] = []
        self.playback_position: float = 0.0  # [0.0, 1.0]
        self.range_start: float = 0.0
        self.range_end: float = 1.0

        self.setMinimumHeight(60)
        self.setCursor(Qt.CursorShape.PointingHandCursor if is_interactive else Qt.CursorShape.ArrowCursor)

    def set_peaks(self, peaks: List[float]):
        """Update waveform peak data."""
        self.peaks = peaks
        self.update()

    def set_playback_position(self, pos: float):
        """Update the current playhead position [0.0, 1.0]."""
        self.playback_position = max(0.0, min(1.0, pos))
        self.update()

    def set_range(self, start: float, end: float):
        """Set trimming range boundaries [0.0, 1.0]."""
        self.range_start = max(0.0, min(1.0, start))
        self.range_end = max(0.0, min(1.0, end))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        mid_y = height / 2.0

        # Background rounded rect
        bg_rect = QRectF(0, 0, width, height)
        painter.fillRect(bg_rect, QColor("#12151e"))
        painter.setPen(QPen(QColor("#242938"), 1))
        painter.drawRoundedRect(bg_rect, 8, 8)

        if not self.peaks:
            # Draw subtle placeholder line
            painter.setPen(QPen(QColor("#2d3348"), 2, Qt.PenStyle.DashLine))
            painter.drawLine(10, int(mid_y), width - 10, int(mid_y))
            return

        num_bars = len(self.peaks)
        bar_width = max(2.0, (width - 16) / max(1, num_bars) - 1.5)
        step = (width - 16) / max(1, num_bars)

        # Highlight selection range background if enabled
        if self.show_range and (self.range_start > 0 or self.range_end < 1.0):
            sel_x1 = 8 + self.range_start * (width - 16)
            sel_x2 = 8 + self.range_end * (width - 16)
            sel_rect = QRectF(sel_x1, 2, max(4.0, sel_x2 - sel_x1), height - 4)
            painter.fillRect(sel_rect, QColor(124, 58, 237, 40))
            painter.setPen(QPen(QColor(124, 58, 237, 120), 1))
            painter.drawRect(sel_rect)

        # Draw waveform bars
        for i, peak in enumerate(self.peaks):
            x = 8 + i * step
            bar_height = max(3.0, peak * (height - 16))
            y = mid_y - bar_height / 2.0

            # Determine color based on playhead
            norm_x = (x - 8) / max(1.0, width - 16)
            if norm_x <= self.playback_position:
                # Played portion: vibrant cyan/purple gradient
                bar_color = QColor("#06b6d4")
            else:
                # Unplayed portion: deeper purple/slate
                bar_color = QColor("#4c3e72")

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(bar_color))
            painter.drawRoundedRect(QRectF(x, y, bar_width, bar_height), 1.5, 1.5)

        # Draw playhead indicator
        playhead_x = 8 + self.playback_position * (width - 16)
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(int(playhead_x), 4, int(playhead_x), height - 4)

        # Small glowing head indicator
        painter.setBrush(QBrush(QColor("#06b6d4")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(playhead_x, mid_y), 4, 4)

    def mousePressEvent(self, event):
        if self.is_interactive and event.button() == Qt.MouseButton.LeftButton:
            self._handle_mouse(event.position().x())

    def mouseMoveEvent(self, event):
        if self.is_interactive and (event.buttons() & Qt.MouseButton.LeftButton):
            self._handle_mouse(event.position().x())

    def _handle_mouse(self, mouse_x: float):
        width = self.width() - 16
        if width <= 0:
            return
        rel = max(0.0, min(1.0, (mouse_x - 8) / width))
        self.playback_position = rel
        self.update()
        self.position_clicked.emit(rel)
