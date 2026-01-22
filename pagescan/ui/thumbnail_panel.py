"""
Thumbnail panel for displaying the image queue.
"""

from pathlib import Path
from typing import List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QAbstractItemView, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QPixmap, QImage, QIcon, QAction
import numpy as np


class ThumbnailLoader(QThread):
    """Background thread for loading thumbnails."""
    thumbnail_loaded = Signal(str, QPixmap)

    def __init__(self, file_paths: List[str], loader, max_size: int = 150):
        super().__init__()
        self.file_paths = file_paths
        self.loader = loader
        self.max_size = max_size
        self._cancelled = False

    def run(self):
        for file_path in self.file_paths:
            if self._cancelled:
                break

            thumbnail = self.loader.load_thumbnail(file_path, self.max_size)
            if thumbnail is not None:
                pixmap = self._numpy_to_pixmap(thumbnail)
                self.thumbnail_loaded.emit(file_path, pixmap)

    def cancel(self):
        self._cancelled = True

    def _numpy_to_pixmap(self, image: np.ndarray) -> QPixmap:
        """Convert numpy array to QPixmap."""
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            bytes_per_line = 3 * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            bytes_per_line = w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)

        return QPixmap.fromImage(q_image.copy())


class ThumbnailItem(QListWidgetItem):
    """Custom list item for thumbnails."""

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

        # Set display name
        self.setText(Path(file_path).name)

        # Set size hint
        self.setSizeHint(QSize(160, 140))

    def set_thumbnail(self, pixmap: QPixmap):
        """Set the thumbnail image."""
        self.setIcon(QIcon(pixmap))


class ThumbnailPanel(QWidget):
    """Panel showing thumbnails of images in the queue."""

    image_selected = pyqtSignal(str)
    images_reordered = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._loader_thread: Optional[ThumbnailLoader] = None
        self._file_paths: List[str] = []

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header
        header = QLabel('Image Queue')
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet('font-weight: bold; padding: 5px;')
        layout.addWidget(header)

        # List widget
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(140, 100))
        self.list_widget.setSpacing(5)
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setMovement(QListWidget.Movement.Free)
        self.list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Connect signals
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self._on_item_clicked)
        self.list_widget.model().rowsMoved.connect(self._on_rows_moved)

        # Context menu
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)

        layout.addWidget(self.list_widget)

        # Info label
        self.info_label = QLabel('No images loaded')
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        self.setMinimumWidth(180)

    def add_images(self, file_paths: List[str], loader):
        """Add images to the panel."""
        # Cancel any existing loader
        if self._loader_thread and self._loader_thread.isRunning():
            self._loader_thread.cancel()
            self._loader_thread.wait()

        # Add items with placeholder
        for file_path in file_paths:
            item = ThumbnailItem(file_path)
            self.list_widget.addItem(item)
            self._file_paths.append(file_path)

        # Update info
        self._update_info()

        # Start thumbnail loading
        self._loader_thread = ThumbnailLoader(file_paths, loader)
        self._loader_thread.thumbnail_loaded.connect(self._on_thumbnail_loaded)
        self._loader_thread.start()

    def _on_thumbnail_loaded(self, file_path: str, pixmap: QPixmap):
        """Handle thumbnail loaded."""
        # Find the item for this file
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if isinstance(item, ThumbnailItem) and item.file_path == file_path:
                item.set_thumbnail(pixmap)
                break

    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click."""
        if isinstance(item, ThumbnailItem):
            self.image_selected.emit(item.file_path)

    def _on_rows_moved(self, parent, start, end, destination, row):
        """Handle rows moved (reordering)."""
        # Rebuild file paths list
        self._file_paths = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if isinstance(item, ThumbnailItem):
                self._file_paths.append(item.file_path)

        self.images_reordered.emit(self._file_paths)

    def _show_context_menu(self, position):
        """Show context menu."""
        item = self.list_widget.itemAt(position)
        if item is None:
            return

        menu = QMenu()

        remove_action = QAction('Remove', self)
        remove_action.triggered.connect(lambda: self._remove_item(item))
        menu.addAction(remove_action)

        remove_all_action = QAction('Remove All', self)
        remove_all_action.triggered.connect(self._remove_all)
        menu.addAction(remove_all_action)

        menu.exec(self.list_widget.mapToGlobal(position))

    def _remove_item(self, item: QListWidgetItem):
        """Remove an item from the list."""
        if isinstance(item, ThumbnailItem):
            self._file_paths.remove(item.file_path)

        row = self.list_widget.row(item)
        self.list_widget.takeItem(row)
        self._update_info()

    def _remove_all(self):
        """Remove all items."""
        self.list_widget.clear()
        self._file_paths.clear()
        self._update_info()

    def _update_info(self):
        """Update the info label."""
        count = self.list_widget.count()
        if count == 0:
            self.info_label.setText('No images loaded')
        elif count == 1:
            self.info_label.setText('1 image')
        else:
            self.info_label.setText(f'{count} images')

    def select_image(self, file_path: str):
        """Select an image in the list."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if isinstance(item, ThumbnailItem) and item.file_path == file_path:
                self.list_widget.setCurrentItem(item)
                break

    def get_file_paths(self) -> List[str]:
        """Get list of file paths in current order."""
        return self._file_paths.copy()

    def clear(self):
        """Clear all items."""
        self._remove_all()
