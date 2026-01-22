"""
Main application window for PageScan.
"""

import os
from pathlib import Path
from typing import Optional, List
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QMenuBar, QMenu, QToolBar, QStatusBar,
    QFileDialog, QMessageBox, QProgressDialog, QApplication
)
from PyQt6.QtCore import Qt, QSettings, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QIcon, QDragEnterEvent, QDropEvent

from .thumbnail_panel import ThumbnailPanel
from .image_viewer import ImageViewer
from .control_panel import ControlPanel
from .export_dialog import ExportDialog
from ..core.image_loader import ImageLoader
from ..core.processor import ImageProcessor, ProcessingSettings
from ..core.exporter import ImageExporter, ExportSettings, BatchProcessor


class ProcessingWorker(QThread):
    """Worker thread for image processing."""
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, file_paths: List[str], processor: ImageProcessor,
                 exporter: ImageExporter, loader: ImageLoader):
        super().__init__()
        self.file_paths = file_paths
        self.processor = processor
        self.exporter = exporter
        self.loader = loader
        self.batch_processor = BatchProcessor()
        self._cancelled = False

    def run(self):
        def progress_callback(current, total, filename, status):
            self.progress.emit(current, total, f"{filename}: {status}")

        self.batch_processor.set_progress_callback(progress_callback)

        try:
            results = self.batch_processor.process_batch(
                self.file_paths, self.processor, self.exporter, self.loader
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._cancelled = True
        self.batch_processor.request_cancel()


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # Initialize core components
        self.image_loader = ImageLoader()
        self.processor = ImageProcessor()
        self.exporter = ImageExporter()

        # Current state
        self.current_image = None
        self.current_file = None
        self.processed_image = None
        self.image_queue: List[str] = []

        # Settings
        self.settings = QSettings('PageScan', 'PageScan')

        # Setup UI
        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_shortcuts()
        self._setup_statusbar()

        # Load settings
        self._load_settings()

        # Enable drag and drop
        self.setAcceptDrops(True)

    def _setup_ui(self):
        """Setup the main UI layout."""
        self.setWindowTitle('PageScan - Book Page Dewarping')
        self.setMinimumSize(1200, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout with splitter
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Thumbnail queue
        self.thumbnail_panel = ThumbnailPanel()
        self.thumbnail_panel.image_selected.connect(self._on_thumbnail_selected)
        self.thumbnail_panel.images_reordered.connect(self._on_images_reordered)
        splitter.addWidget(self.thumbnail_panel)

        # Center - Image viewer
        self.image_viewer = ImageViewer()
        self.image_viewer.corner_moved.connect(self._on_corner_moved)
        self.image_viewer.curve_point_moved.connect(self._on_curve_point_moved)
        splitter.addWidget(self.image_viewer)

        # Right panel - Controls
        self.control_panel = ControlPanel()
        self.control_panel.settings_changed.connect(self._on_settings_changed)
        self.control_panel.process_clicked.connect(self._process_current)
        self.control_panel.process_all_clicked.connect(self._process_all)
        self.control_panel.reset_clicked.connect(self._reset_processing)
        splitter.addWidget(self.control_panel)

        # Set splitter sizes
        splitter.setSizes([200, 700, 300])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

    def _setup_menus(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('&File')

        open_action = QAction('&Open Images...', self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_images)
        file_menu.addAction(open_action)

        open_folder_action = QAction('Open &Folder...', self)
        open_folder_action.setShortcut('Ctrl+Shift+O')
        open_folder_action.triggered.connect(self._open_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        export_action = QAction('&Export Current...', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self._export_current)
        file_menu.addAction(export_action)

        export_all_action = QAction('Export &All...', self)
        export_all_action.setShortcut('Ctrl+Shift+E')
        export_all_action.triggered.connect(self._export_all)
        file_menu.addAction(export_all_action)

        file_menu.addSeparator()

        quit_action = QAction('&Quit', self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = menubar.addMenu('&Edit')

        process_action = QAction('&Process Current', self)
        process_action.setShortcut('Ctrl+P')
        process_action.triggered.connect(self._process_current)
        edit_menu.addAction(process_action)

        process_all_action = QAction('Process &All', self)
        process_all_action.setShortcut('Ctrl+Shift+P')
        process_all_action.triggered.connect(self._process_all)
        edit_menu.addAction(process_all_action)

        edit_menu.addSeparator()

        reset_action = QAction('&Reset', self)
        reset_action.setShortcut('Ctrl+R')
        reset_action.triggered.connect(self._reset_processing)
        edit_menu.addAction(reset_action)

        # View menu
        view_menu = menubar.addMenu('&View')

        zoom_in_action = QAction('Zoom &In', self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(self.image_viewer.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction('Zoom &Out', self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(self.image_viewer.zoom_out)
        view_menu.addAction(zoom_out_action)

        fit_action = QAction('&Fit to Window', self)
        fit_action.setShortcut('Ctrl+0')
        fit_action.triggered.connect(self.image_viewer.fit_to_window)
        view_menu.addAction(fit_action)

        view_menu.addSeparator()

        self.toggle_comparison_action = QAction('Toggle &Comparison', self)
        self.toggle_comparison_action.setShortcut('Space')
        self.toggle_comparison_action.setCheckable(True)
        self.toggle_comparison_action.triggered.connect(self._toggle_comparison)
        view_menu.addAction(self.toggle_comparison_action)

        view_menu.addSeparator()

        self.dark_mode_action = QAction('&Dark Mode', self)
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.triggered.connect(self._toggle_dark_mode)
        view_menu.addAction(self.dark_mode_action)

        # Help menu
        help_menu = menubar.addMenu('&Help')

        about_action = QAction('&About', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        """Setup main toolbar."""
        toolbar = QToolBar('Main Toolbar')
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Open button
        open_btn = QAction('Open', self)
        open_btn.setToolTip('Open images (Ctrl+O)')
        open_btn.triggered.connect(self._open_images)
        toolbar.addAction(open_btn)

        toolbar.addSeparator()

        # Navigation
        prev_btn = QAction('Previous', self)
        prev_btn.setToolTip('Previous image (Left)')
        prev_btn.triggered.connect(self._prev_image)
        toolbar.addAction(prev_btn)

        next_btn = QAction('Next', self)
        next_btn.setToolTip('Next image (Right)')
        next_btn.triggered.connect(self._next_image)
        toolbar.addAction(next_btn)

        toolbar.addSeparator()

        # Processing
        process_btn = QAction('Process', self)
        process_btn.setToolTip('Process current image (Ctrl+P)')
        process_btn.triggered.connect(self._process_current)
        toolbar.addAction(process_btn)

        toolbar.addSeparator()

        # Export
        export_btn = QAction('Export', self)
        export_btn.setToolTip('Export current image (Ctrl+E)')
        export_btn.triggered.connect(self._export_current)
        toolbar.addAction(export_btn)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Navigation shortcuts are handled by actions
        pass

    def _setup_statusbar(self):
        """Setup status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage('Ready')

    def _load_settings(self):
        """Load application settings."""
        # Window geometry
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)

        state = self.settings.value('windowState')
        if state:
            self.restoreState(state)

        # Dark mode
        dark_mode = self.settings.value('darkMode', False, type=bool)
        self.dark_mode_action.setChecked(dark_mode)
        if dark_mode:
            self._apply_dark_mode()

    def _save_settings(self):
        """Save application settings."""
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())
        self.settings.setValue('darkMode', self.dark_mode_action.isChecked())

    def closeEvent(self, event):
        """Handle window close."""
        self._save_settings()
        event.accept()

    # Drag and drop handling
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle drop."""
        urls = event.mimeData().urls()
        file_paths = []

        for url in urls:
            path = url.toLocalFile()
            if os.path.isfile(path) and self.image_loader.is_supported(path):
                file_paths.append(path)
            elif os.path.isdir(path):
                file_paths.extend(self.image_loader.scan_folder(path))

        if file_paths:
            self._add_images(file_paths)

    # File operations
    def _open_images(self):
        """Open image files."""
        extensions = self.image_loader.get_supported_extensions()
        filter_str = "Images ({})".format(' '.join(f'*{ext}' for ext in extensions))

        files, _ = QFileDialog.getOpenFileNames(
            self, 'Open Images', '', filter_str
        )

        if files:
            self._add_images(files)

    def _open_folder(self):
        """Open a folder of images."""
        folder = QFileDialog.getExistingDirectory(self, 'Open Folder')

        if folder:
            files = self.image_loader.scan_folder(folder)
            if files:
                self._add_images(files)
            else:
                QMessageBox.information(
                    self, 'No Images Found',
                    'No supported image files were found in the selected folder.'
                )

    def _add_images(self, file_paths: List[str]):
        """Add images to the queue."""
        self.image_queue.extend(file_paths)
        self.thumbnail_panel.add_images(file_paths, self.image_loader)

        if self.current_file is None and file_paths:
            self._load_image(file_paths[0])

        self.statusbar.showMessage(f'Loaded {len(file_paths)} images')

    def _load_image(self, file_path: str):
        """Load an image for viewing."""
        self.current_file = file_path
        self.current_image = self.image_loader.load_image(file_path)
        self.processed_image = None

        if self.current_image is not None:
            self.image_viewer.set_image(self.current_image)

            # Analyze image and update control panel
            analysis = self.processor.analyze_image(self.current_image)
            self.control_panel.update_analysis(analysis)

            # Update status
            info = self.image_loader.get_image_info(file_path)
            if info:
                self.statusbar.showMessage(
                    f'{Path(file_path).name} - {info["width"]}x{info["height"]}'
                )
        else:
            QMessageBox.warning(
                self, 'Error',
                f'Failed to load image: {file_path}'
            )

    def _on_thumbnail_selected(self, file_path: str):
        """Handle thumbnail selection."""
        self._load_image(file_path)

    def _on_images_reordered(self, file_paths: List[str]):
        """Handle image reordering."""
        self.image_queue = file_paths

    def _prev_image(self):
        """Go to previous image."""
        if not self.image_queue or not self.current_file:
            return

        try:
            idx = self.image_queue.index(self.current_file)
            if idx > 0:
                self._load_image(self.image_queue[idx - 1])
                self.thumbnail_panel.select_image(self.image_queue[idx - 1])
        except ValueError:
            pass

    def _next_image(self):
        """Go to next image."""
        if not self.image_queue or not self.current_file:
            return

        try:
            idx = self.image_queue.index(self.current_file)
            if idx < len(self.image_queue) - 1:
                self._load_image(self.image_queue[idx + 1])
                self.thumbnail_panel.select_image(self.image_queue[idx + 1])
        except ValueError:
            pass

    # Processing
    def _on_settings_changed(self, settings: ProcessingSettings):
        """Handle settings change from control panel."""
        self.processor.settings = settings

    def _on_corner_moved(self, corners):
        """Handle corner adjustment in viewer."""
        self.processor.settings.manual_corners = corners
        self._preview_processing()

    def _on_curve_point_moved(self, points):
        """Handle curve point adjustment."""
        self.processor.settings.dewarp_params.manual_curve_points = points
        self._preview_processing()

    def _preview_processing(self):
        """Preview processing with current settings."""
        if self.current_image is None:
            return

        self.processed_image = self.processor.process(self.current_image)
        self.image_viewer.set_processed_image(self.processed_image)

    def _process_current(self):
        """Process the current image."""
        if self.current_image is None:
            return

        self.statusbar.showMessage('Processing...')
        QApplication.processEvents()

        self.processed_image = self.processor.process(self.current_image)
        self.image_viewer.set_processed_image(self.processed_image)

        self.statusbar.showMessage('Processing complete')

    def _process_all(self):
        """Process all images in queue."""
        if not self.image_queue:
            return

        # Show export dialog first
        dialog = ExportDialog(self, len(self.image_queue))
        if dialog.exec():
            export_settings = dialog.get_settings()
            self.exporter.settings = export_settings

            # Create progress dialog
            progress = QProgressDialog('Processing images...', 'Cancel', 0, len(self.image_queue), self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)

            # Create worker thread
            self.worker = ProcessingWorker(
                self.image_queue, self.processor, self.exporter, self.image_loader
            )

            def on_progress(current, total, status):
                progress.setValue(current)
                progress.setLabelText(status)

            def on_finished(results):
                progress.close()
                self._show_batch_results(results)

            def on_error(error_msg):
                progress.close()
                QMessageBox.critical(self, 'Error', f'Processing failed: {error_msg}')

            self.worker.progress.connect(on_progress)
            self.worker.finished.connect(on_finished)
            self.worker.error.connect(on_error)

            progress.canceled.connect(self.worker.cancel)

            self.worker.start()

    def _show_batch_results(self, results: dict):
        """Show batch processing results."""
        msg = f"Processing complete!\n\n"
        msg += f"Processed: {results['processed']}/{results['total']}\n"
        if results['failed'] > 0:
            msg += f"Failed: {results['failed']}\n"
        if results['cancelled']:
            msg += "\nProcessing was cancelled."

        QMessageBox.information(self, 'Batch Processing Complete', msg)

    def _reset_processing(self):
        """Reset to original image."""
        if self.current_image is not None:
            self.processed_image = None
            self.image_viewer.set_image(self.current_image)
            self.image_viewer.clear_overlays()
            self.control_panel.reset_settings()

    # Export
    def _export_current(self):
        """Export the current processed image."""
        if self.processed_image is None and self.current_image is None:
            QMessageBox.warning(self, 'No Image', 'No image to export.')
            return

        image_to_export = self.processed_image if self.processed_image is not None else self.current_image

        # Get output path
        default_name = Path(self.current_file).stem + '_processed' if self.current_file else 'processed'

        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Export Image', default_name,
            'JPEG (*.jpg);;PNG (*.png);;TIFF (*.tiff)'
        )

        if file_path:
            from ..core.exporter import ExportFormat

            # Determine format from extension
            ext = Path(file_path).suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                self.exporter.settings.format = ExportFormat.JPEG
            elif ext == '.png':
                self.exporter.settings.format = ExportFormat.PNG
            elif ext in ['.tif', '.tiff']:
                self.exporter.settings.format = ExportFormat.TIFF

            self.exporter.settings.output_dir = str(Path(file_path).parent)

            output_path = self.exporter.export_single(
                image_to_export,
                Path(file_path).stem,
                exif_data=self.image_loader.get_exif(self.current_file) if self.current_file else None
            )

            self.statusbar.showMessage(f'Exported to {output_path}')

    def _export_all(self):
        """Export all processed images."""
        if not self.image_queue:
            return

        self._process_all()

    # View
    def _toggle_comparison(self):
        """Toggle before/after comparison view."""
        self.image_viewer.toggle_comparison()

    def _toggle_dark_mode(self):
        """Toggle dark mode."""
        if self.dark_mode_action.isChecked():
            self._apply_dark_mode()
        else:
            self._apply_light_mode()

    def _apply_dark_mode(self):
        """Apply dark mode stylesheet."""
        dark_style = """
            QMainWindow, QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QMenuBar {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QMenuBar::item:selected {
                background-color: #505050;
            }
            QMenu {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QMenu::item:selected {
                background-color: #505050;
            }
            QToolBar {
                background-color: #3c3c3c;
                border: none;
            }
            QToolButton {
                background-color: transparent;
                color: #ffffff;
                padding: 5px;
            }
            QToolButton:hover {
                background-color: #505050;
            }
            QStatusBar {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QSplitter::handle {
                background-color: #505050;
            }
            QScrollArea {
                background-color: #2b2b2b;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #505050;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QGroupBox {
                color: #ffffff;
                border: 1px solid #505050;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QCheckBox, QRadioButton {
                color: #ffffff;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background-color: #505050;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                background-color: #808080;
                border-radius: 8px;
                margin: -5px 0;
            }
            QSlider::handle:horizontal:hover {
                background-color: #a0a0a0;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #505050;
                padding: 3px;
            }
            QPushButton {
                background-color: #505050;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #606060;
            }
            QPushButton:pressed {
                background-color: #404040;
            }
            QLabel {
                color: #ffffff;
            }
            QListWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #505050;
            }
            QListWidget::item:selected {
                background-color: #505050;
            }
        """
        self.setStyleSheet(dark_style)

    def _apply_light_mode(self):
        """Apply light mode (default) stylesheet."""
        self.setStyleSheet('')

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, 'About PageScan',
            '<h2>PageScan</h2>'
            '<p>Version 1.0.0</p>'
            '<p>A desktop application for dewarping curved book pages '
            'and correcting perspective distortion from scanned book images.</p>'
            '<p>Designed for V-cradle book scanning setups.</p>'
        )

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Left:
            self._prev_image()
        elif event.key() == Qt.Key.Key_Right:
            self._next_image()
        else:
            super().keyPressEvent(event)
