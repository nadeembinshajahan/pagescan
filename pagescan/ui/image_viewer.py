"""
Image viewer widget with zoom, pan, and overlay controls.
"""

from typing import Optional, List, Tuple
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QSlider, QPushButton, QButtonGroup, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QPointF, QRectF, QSize
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QBrush,
    QMouseEvent, QWheelEvent, QPaintEvent, QResizeEvent
)


class ImageCanvas(QWidget):
    """Canvas widget for drawing image with overlays."""

    corner_moved = pyqtSignal(np.ndarray)  # Emits 4x2 array of corner positions
    curve_point_moved = pyqtSignal(list)  # Emits list of curve control points

    def __init__(self, parent=None):
        super().__init__(parent)

        self.original_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.display_pixmap: Optional[QPixmap] = None

        self._scale = 1.0
        self._offset = QPointF(0, 0)

        # Interaction mode
        self._mode = 'view'  # 'view', 'corners', 'curve'

        # Corner points for perspective correction (in image coordinates)
        self._corners: Optional[np.ndarray] = None
        self._corner_handle_radius = 15
        self._dragging_corner: int = -1

        # Curve points for dewarping
        self._curve_points: List[Tuple[int, int]] = []
        self._dragging_curve_point: int = -1

        # Comparison mode
        self._comparison_mode = 'side_by_side'  # 'side_by_side', 'overlay', 'slider'
        self._show_processed = True
        self._slider_position = 0.5

        # Mouse tracking
        self._last_mouse_pos: Optional[QPoint] = None
        self._panning = False

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, image: np.ndarray):
        """Set the original image."""
        self.original_image = image
        self.processed_image = None
        self._update_display()

    def set_processed_image(self, image: np.ndarray):
        """Set the processed image for comparison."""
        self.processed_image = image
        self._update_display()

    def set_mode(self, mode: str):
        """Set interaction mode."""
        self._mode = mode
        self.setCursor(Qt.CursorShape.CrossCursor if mode != 'view' else Qt.CursorShape.ArrowCursor)
        self.update()

    def set_corners(self, corners: np.ndarray):
        """Set corner points for perspective overlay."""
        self._corners = corners.copy() if corners is not None else None
        self.update()

    def clear_overlays(self):
        """Clear all overlays."""
        self._corners = None
        self._curve_points = []
        self.update()

    def set_comparison_mode(self, mode: str):
        """Set comparison display mode."""
        self._comparison_mode = mode
        self._update_display()

    def toggle_processed(self):
        """Toggle between original and processed view."""
        self._show_processed = not self._show_processed
        self._update_display()

    def _update_display(self):
        """Update the display pixmap."""
        if self.original_image is None:
            self.display_pixmap = None
            self.update()
            return

        # Choose which image to display
        if self._comparison_mode == 'side_by_side' and self.processed_image is not None:
            # Create side by side view
            image = self._create_side_by_side()
        elif self._show_processed and self.processed_image is not None:
            image = self.processed_image
        else:
            image = self.original_image

        self.display_pixmap = self._numpy_to_pixmap(image)
        self.update()

    def _create_side_by_side(self) -> np.ndarray:
        """Create side-by-side comparison image."""
        if self.processed_image is None:
            return self.original_image

        h1, w1 = self.original_image.shape[:2]
        h2, w2 = self.processed_image.shape[:2]

        # Make same height
        max_h = max(h1, h2)
        scale1 = max_h / h1
        scale2 = max_h / h2

        import cv2
        if scale1 != 1.0:
            img1 = cv2.resize(self.original_image, (int(w1 * scale1), max_h))
        else:
            img1 = self.original_image

        if scale2 != 1.0:
            img2 = cv2.resize(self.processed_image, (int(w2 * scale2), max_h))
        else:
            img2 = self.processed_image

        # Add separator
        separator = np.ones((max_h, 5, 3), dtype=np.uint8) * 128

        return np.hstack([img1, separator, img2])

    def _numpy_to_pixmap(self, image: np.ndarray) -> QPixmap:
        """Convert numpy array to QPixmap."""
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            bytes_per_line = 3 * w
            q_image = QImage(image.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            bytes_per_line = w
            q_image = QImage(image.tobytes(), w, h, bytes_per_line, QImage.Format.Format_Grayscale8)

        return QPixmap.fromImage(q_image)

    def paintEvent(self, event: QPaintEvent):
        """Paint the canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if self.display_pixmap is None:
            # Draw placeholder text
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No image loaded')
            return

        # Calculate display rect
        scaled_size = self.display_pixmap.size() * self._scale
        x = (self.width() - scaled_size.width()) / 2 + self._offset.x()
        y = (self.height() - scaled_size.height()) / 2 + self._offset.y()

        # Draw image
        target_rect = QRectF(x, y, scaled_size.width(), scaled_size.height())
        painter.drawPixmap(target_rect.toRect(), self.display_pixmap)

        # Draw slider comparison line
        if self._comparison_mode == 'slider' and self.processed_image is not None:
            self._draw_slider_comparison(painter, target_rect)

        # Draw corner overlay
        if self._mode == 'corners' and self._corners is not None:
            self._draw_corners(painter, target_rect)

        # Draw curve overlay
        if self._mode == 'curve' and self._curve_points:
            self._draw_curve_points(painter, target_rect)

    def _draw_corners(self, painter: QPainter, image_rect: QRectF):
        """Draw corner handles and lines."""
        if self._corners is None or self.original_image is None:
            return

        h, w = self.original_image.shape[:2]

        # Convert image coordinates to widget coordinates
        def to_widget(point):
            x = image_rect.x() + (point[0] / w) * image_rect.width()
            y = image_rect.y() + (point[1] / h) * image_rect.height()
            return QPointF(x, y)

        # Draw quadrilateral
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)

        points = [to_widget(self._corners[i]) for i in range(4)]
        for i in range(4):
            painter.drawLine(points[i], points[(i + 1) % 4])

        # Draw corner handles
        colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255), QColor(255, 255, 0)]
        labels = ['TL', 'TR', 'BR', 'BL']

        for i, (point, color, label) in enumerate(zip(points, colors, labels)):
            # Circle
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(point, self._corner_handle_radius, self._corner_handle_radius)

            # Label
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(int(point.x()) - 8, int(point.y()) + 5, label)

    def _draw_curve_points(self, painter: QPainter, image_rect: QRectF):
        """Draw curve control points."""
        if not self._curve_points or self.original_image is None:
            return

        h, w = self.original_image.shape[:2]

        def to_widget(point):
            x = image_rect.x() + (point[0] / w) * image_rect.width()
            y = image_rect.y() + (point[1] / h) * image_rect.height()
            return QPointF(x, y)

        # Draw line through points
        if len(self._curve_points) >= 2:
            pen = QPen(QColor(255, 128, 0), 2)
            painter.setPen(pen)

            points = [to_widget(p) for p in self._curve_points]
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])

        # Draw point handles
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.setBrush(QBrush(QColor(255, 128, 0)))

        for point in self._curve_points:
            widget_point = to_widget(point)
            painter.drawEllipse(widget_point, 8, 8)

    def _draw_slider_comparison(self, painter: QPainter, image_rect: QRectF):
        """Draw slider comparison overlay."""
        slider_x = image_rect.x() + self._slider_position * image_rect.width()

        # Draw vertical line
        pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(pen)
        painter.drawLine(
            int(slider_x), int(image_rect.y()),
            int(slider_x), int(image_rect.y() + image_rect.height())
        )

        # Draw handle
        handle_y = image_rect.y() + image_rect.height() / 2
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(QPointF(slider_x, handle_y), 10, 10)

    def _get_image_rect(self) -> QRectF:
        """Get the rectangle where image is displayed."""
        if self.display_pixmap is None:
            return QRectF()

        scaled_size = self.display_pixmap.size() * self._scale
        x = (self.width() - scaled_size.width()) / 2 + self._offset.x()
        y = (self.height() - scaled_size.height()) / 2 + self._offset.y()

        return QRectF(x, y, scaled_size.width(), scaled_size.height())

    def _widget_to_image(self, pos: QPoint) -> Optional[Tuple[int, int]]:
        """Convert widget coordinates to image coordinates."""
        if self.original_image is None:
            return None

        h, w = self.original_image.shape[:2]
        rect = self._get_image_rect()

        if not rect.contains(QPointF(pos)):
            # Allow some margin for corner handles
            margin = self._corner_handle_radius
            expanded_rect = rect.adjusted(-margin, -margin, margin, margin)
            if not expanded_rect.contains(QPointF(pos)):
                return None

        x = int((pos.x() - rect.x()) / rect.width() * w)
        y = int((pos.y() - rect.y()) / rect.height() * h)

        return (max(0, min(w - 1, x)), max(0, min(h - 1, y)))

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse_pos = event.pos()

            if self._mode == 'corners' and self._corners is not None:
                # Check if clicking on a corner handle
                rect = self._get_image_rect()
                h, w = self.original_image.shape[:2] if self.original_image is not None else (1, 1)

                for i, corner in enumerate(self._corners):
                    widget_x = rect.x() + (corner[0] / w) * rect.width()
                    widget_y = rect.y() + (corner[1] / h) * rect.height()

                    if (abs(event.pos().x() - widget_x) < self._corner_handle_radius and
                        abs(event.pos().y() - widget_y) < self._corner_handle_radius):
                        self._dragging_corner = i
                        return

            elif self._mode == 'curve':
                # Check if clicking on a curve point
                rect = self._get_image_rect()
                h, w = self.original_image.shape[:2] if self.original_image is not None else (1, 1)

                for i, point in enumerate(self._curve_points):
                    widget_x = rect.x() + (point[0] / w) * rect.width()
                    widget_y = rect.y() + (point[1] / h) * rect.height()

                    if (abs(event.pos().x() - widget_x) < 12 and
                        abs(event.pos().y() - widget_y) < 12):
                        self._dragging_curve_point = i
                        return

                # Add new curve point
                img_pos = self._widget_to_image(event.pos())
                if img_pos:
                    self._curve_points.append(img_pos)
                    self._curve_points.sort(key=lambda p: p[0])
                    self.curve_point_moved.emit(self._curve_points)
                    self.update()
                    return

            # Start panning
            self._panning = True

        elif event.button() == Qt.MouseButton.RightButton:
            # Remove curve point if in curve mode
            if self._mode == 'curve' and self._curve_points:
                rect = self._get_image_rect()
                h, w = self.original_image.shape[:2] if self.original_image is not None else (1, 1)

                for i, point in enumerate(self._curve_points):
                    widget_x = rect.x() + (point[0] / w) * rect.width()
                    widget_y = rect.y() + (point[1] / h) * rect.height()

                    if (abs(event.pos().x() - widget_x) < 12 and
                        abs(event.pos().y() - widget_y) < 12):
                        del self._curve_points[i]
                        self.curve_point_moved.emit(self._curve_points)
                        self.update()
                        return

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        if self._dragging_corner >= 0 and self._corners is not None:
            # Move corner
            img_pos = self._widget_to_image(event.pos())
            if img_pos:
                self._corners[self._dragging_corner] = [img_pos[0], img_pos[1]]
                self.corner_moved.emit(self._corners)
                self.update()

        elif self._dragging_curve_point >= 0:
            # Move curve point
            img_pos = self._widget_to_image(event.pos())
            if img_pos:
                self._curve_points[self._dragging_curve_point] = img_pos
                self._curve_points.sort(key=lambda p: p[0])
                self.curve_point_moved.emit(self._curve_points)
                self.update()

        elif self._panning and self._last_mouse_pos is not None:
            # Pan view
            delta = event.pos() - self._last_mouse_pos
            self._offset += QPointF(delta)
            self._last_mouse_pos = event.pos()
            self.update()

        # Update cursor based on position
        if self._mode == 'corners' and self._corners is not None:
            rect = self._get_image_rect()
            if self.original_image is not None:
                h, w = self.original_image.shape[:2]
                for corner in self._corners:
                    widget_x = rect.x() + (corner[0] / w) * rect.width()
                    widget_y = rect.y() + (corner[1] / h) * rect.height()

                    if (abs(event.pos().x() - widget_x) < self._corner_handle_radius and
                        abs(event.pos().y() - widget_y) < self._corner_handle_radius):
                        self.setCursor(Qt.CursorShape.SizeAllCursor)
                        return

        self.setCursor(Qt.CursorShape.ArrowCursor if self._mode == 'view' else Qt.CursorShape.CrossCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        self._panning = False
        self._dragging_corner = -1
        self._dragging_curve_point = -1
        self._last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        delta = event.angleDelta().y()

        # Calculate zoom center
        mouse_pos = event.position()

        # Zoom factor
        factor = 1.1 if delta > 0 else 0.9
        new_scale = self._scale * factor

        # Limit scale
        new_scale = max(0.1, min(10.0, new_scale))

        if new_scale != self._scale:
            # Adjust offset to zoom toward mouse position
            old_scale = self._scale
            self._scale = new_scale

            # Calculate the point under the mouse in image space, then adjust offset
            # to keep that point under the mouse
            scale_change = new_scale / old_scale
            mouse_offset = QPointF(mouse_pos.x() - self.width() / 2,
                                   mouse_pos.y() - self.height() / 2)
            self._offset = (self._offset - mouse_offset) * scale_change + mouse_offset

            self.update()

    def zoom_in(self):
        """Zoom in."""
        self._scale = min(10.0, self._scale * 1.2)
        self.update()

    def zoom_out(self):
        """Zoom out."""
        self._scale = max(0.1, self._scale / 1.2)
        self.update()

    def fit_to_window(self):
        """Fit image to window."""
        if self.display_pixmap is None:
            return

        # Calculate scale to fit
        scale_x = self.width() / self.display_pixmap.width()
        scale_y = self.height() / self.display_pixmap.height()
        self._scale = min(scale_x, scale_y) * 0.95

        self._offset = QPointF(0, 0)
        self.update()

    def actual_size(self):
        """Show at actual size (100%)."""
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self.update()


class ImageViewer(QWidget):
    """Image viewer with controls."""

    corner_moved = pyqtSignal(np.ndarray)
    curve_point_moved = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Canvas
        self.canvas = ImageCanvas()
        self.canvas.corner_moved.connect(self.corner_moved.emit)
        self.canvas.curve_point_moved.connect(self.curve_point_moved.emit)
        layout.addWidget(self.canvas, 1)

        # Controls bar
        controls = QFrame()
        controls.setStyleSheet('background-color: rgba(60, 60, 60, 200); border-radius: 5px;')
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(10, 5, 10, 5)

        # Zoom controls
        zoom_out_btn = QPushButton('-')
        zoom_out_btn.setFixedSize(30, 30)
        zoom_out_btn.clicked.connect(self.zoom_out)
        controls_layout.addWidget(zoom_out_btn)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(150)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider)
        controls_layout.addWidget(self.zoom_slider)

        zoom_in_btn = QPushButton('+')
        zoom_in_btn.setFixedSize(30, 30)
        zoom_in_btn.clicked.connect(self.zoom_in)
        controls_layout.addWidget(zoom_in_btn)

        fit_btn = QPushButton('Fit')
        fit_btn.clicked.connect(self.fit_to_window)
        controls_layout.addWidget(fit_btn)

        controls_layout.addStretch()

        # Comparison controls
        controls_layout.addWidget(QLabel('View:'))

        self.view_mode_group = QButtonGroup()

        original_btn = QPushButton('Original')
        original_btn.setCheckable(True)
        original_btn.setChecked(True)
        original_btn.clicked.connect(lambda: self._set_show_processed(False))
        self.view_mode_group.addButton(original_btn)
        controls_layout.addWidget(original_btn)

        processed_btn = QPushButton('Processed')
        processed_btn.setCheckable(True)
        processed_btn.clicked.connect(lambda: self._set_show_processed(True))
        self.view_mode_group.addButton(processed_btn)
        controls_layout.addWidget(processed_btn)

        compare_btn = QPushButton('Compare')
        compare_btn.setCheckable(True)
        compare_btn.clicked.connect(lambda: self._set_comparison_mode('side_by_side'))
        self.view_mode_group.addButton(compare_btn)
        controls_layout.addWidget(compare_btn)

        layout.addWidget(controls)

    def set_image(self, image: np.ndarray):
        """Set the current image."""
        self.canvas.set_image(image)
        self.fit_to_window()

    def set_processed_image(self, image: np.ndarray):
        """Set the processed image."""
        self.canvas.set_processed_image(image)

    def set_corners(self, corners: np.ndarray):
        """Set corner points for overlay."""
        self.canvas.set_corners(corners)

    def set_mode(self, mode: str):
        """Set interaction mode."""
        self.canvas.set_mode(mode)

    def clear_overlays(self):
        """Clear overlays."""
        self.canvas.clear_overlays()

    def zoom_in(self):
        """Zoom in."""
        self.canvas.zoom_in()
        self.zoom_slider.setValue(int(self.canvas._scale * 100))

    def zoom_out(self):
        """Zoom out."""
        self.canvas.zoom_out()
        self.zoom_slider.setValue(int(self.canvas._scale * 100))

    def fit_to_window(self):
        """Fit image to window."""
        self.canvas.fit_to_window()
        self.zoom_slider.setValue(int(self.canvas._scale * 100))

    def _on_zoom_slider(self, value):
        """Handle zoom slider change."""
        self.canvas._scale = value / 100
        self.canvas.update()

    def _set_show_processed(self, show: bool):
        """Set whether to show processed image."""
        self.canvas._show_processed = show
        self.canvas._comparison_mode = 'normal'
        self.canvas._update_display()

    def _set_comparison_mode(self, mode: str):
        """Set comparison mode."""
        self.canvas.set_comparison_mode(mode)

    def toggle_comparison(self):
        """Toggle comparison view."""
        self.canvas.toggle_processed()
