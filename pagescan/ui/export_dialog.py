"""
Export dialog for batch export settings.
"""

from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QSpinBox, QLineEdit, QPushButton, QCheckBox,
    QFileDialog, QFormLayout, QDialogButtonBox, QSlider
)
from PyQt6.QtCore import Qt

from ..core.exporter import ExportSettings, ExportFormat


class ExportDialog(QDialog):
    """Dialog for configuring export settings."""

    def __init__(self, parent=None, image_count: int = 1):
        super().__init__(parent)
        self.image_count = image_count
        self._settings = ExportSettings()

        self.setWindowTitle('Export Settings')
        self.setMinimumWidth(450)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Info
        info_label = QLabel(f'Exporting {self.image_count} image(s)')
        info_label.setStyleSheet('font-weight: bold; margin-bottom: 10px;')
        layout.addWidget(info_label)

        # Output folder
        folder_group = QGroupBox('Output Location')
        folder_layout = QHBoxLayout(folder_group)

        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText('Select output folder...')
        folder_layout.addWidget(self.folder_edit)

        browse_btn = QPushButton('Browse...')
        browse_btn.clicked.connect(self._browse_folder)
        folder_layout.addWidget(browse_btn)

        layout.addWidget(folder_group)

        # Format settings
        format_group = QGroupBox('Format')
        format_layout = QFormLayout(format_group)

        self.format_combo = QComboBox()
        self.format_combo.addItems(['JPEG', 'PNG', 'TIFF'])
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        format_layout.addRow('Output Format:', self.format_combo)

        # JPEG quality
        quality_widget = QWidget()
        quality_layout = QHBoxLayout(quality_widget)
        quality_layout.setContentsMargins(0, 0, 0, 0)

        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(95)
        self.quality_slider.valueChanged.connect(self._on_quality_changed)
        quality_layout.addWidget(self.quality_slider)

        self.quality_label = QLabel('95')
        self.quality_label.setFixedWidth(30)
        quality_layout.addWidget(self.quality_label)

        format_layout.addRow('JPEG Quality:', quality_widget)

        # PNG compression
        self.png_compression_spin = QSpinBox()
        self.png_compression_spin.setRange(0, 9)
        self.png_compression_spin.setValue(6)
        self.png_compression_spin.setEnabled(False)
        format_layout.addRow('PNG Compression:', self.png_compression_spin)

        # TIFF compression
        self.tiff_compression_combo = QComboBox()
        self.tiff_compression_combo.addItems(['None', 'LZW', 'Deflate'])
        self.tiff_compression_combo.setCurrentIndex(1)
        self.tiff_compression_combo.setEnabled(False)
        format_layout.addRow('TIFF Compression:', self.tiff_compression_combo)

        layout.addWidget(format_group)

        # Naming convention
        naming_group = QGroupBox('File Naming')
        naming_layout = QFormLayout(naming_group)

        self.prefix_edit = QLineEdit('page_')
        naming_layout.addRow('Prefix:', self.prefix_edit)

        self.start_number_spin = QSpinBox()
        self.start_number_spin.setRange(0, 99999)
        self.start_number_spin.setValue(1)
        naming_layout.addRow('Start Number:', self.start_number_spin)

        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(1, 5)
        self.padding_spin.setValue(3)
        naming_layout.addRow('Zero Padding:', self.padding_spin)

        # Preview
        self.naming_preview = QLabel()
        self._update_naming_preview()
        naming_layout.addRow('Preview:', self.naming_preview)

        self.prefix_edit.textChanged.connect(self._update_naming_preview)
        self.start_number_spin.valueChanged.connect(self._update_naming_preview)
        self.padding_spin.valueChanged.connect(self._update_naming_preview)

        layout.addWidget(naming_group)

        # Resize options
        resize_group = QGroupBox('Resize')
        resize_group.setCheckable(True)
        resize_group.setChecked(False)
        self.resize_group = resize_group

        resize_layout = QFormLayout(resize_group)

        self.max_width_spin = QSpinBox()
        self.max_width_spin.setRange(100, 10000)
        self.max_width_spin.setValue(2000)
        self.max_width_spin.setSpecialValueText('No limit')
        resize_layout.addRow('Max Width:', self.max_width_spin)

        self.max_height_spin = QSpinBox()
        self.max_height_spin.setRange(100, 10000)
        self.max_height_spin.setValue(3000)
        self.max_height_spin.setSpecialValueText('No limit')
        resize_layout.addRow('Max Height:', self.max_height_spin)

        self.resize_method_combo = QComboBox()
        self.resize_method_combo.addItems(['Lanczos (Best)', 'Bilinear', 'Nearest'])
        resize_layout.addRow('Method:', self.resize_method_combo)

        layout.addWidget(resize_group)

        # Options
        options_group = QGroupBox('Options')
        options_layout = QVBoxLayout(options_group)

        self.preserve_exif_check = QCheckBox('Preserve EXIF metadata')
        self.preserve_exif_check.setChecked(True)
        options_layout.addWidget(self.preserve_exif_check)

        layout.addWidget(options_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _browse_folder(self):
        """Browse for output folder."""
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if folder:
            self.folder_edit.setText(folder)

    def _on_format_changed(self, index: int):
        """Handle format change."""
        is_jpeg = (index == 0)
        is_png = (index == 1)
        is_tiff = (index == 2)

        self.quality_slider.setEnabled(is_jpeg)
        self.quality_label.setEnabled(is_jpeg)
        self.png_compression_spin.setEnabled(is_png)
        self.tiff_compression_combo.setEnabled(is_tiff)

        self._update_naming_preview()

    def _on_quality_changed(self, value: int):
        """Handle quality slider change."""
        self.quality_label.setText(str(value))

    def _update_naming_preview(self):
        """Update the naming preview."""
        prefix = self.prefix_edit.text()
        start = self.start_number_spin.value()
        padding = self.padding_spin.value()

        ext_map = {0: '.jpg', 1: '.png', 2: '.tiff'}
        ext = ext_map.get(self.format_combo.currentIndex(), '.jpg')

        preview = f"{prefix}{start:0{padding}d}{ext}"
        if self.image_count > 1:
            last = start + self.image_count - 1
            preview += f" ... {prefix}{last:0{padding}d}{ext}"

        self.naming_preview.setText(preview)

    def get_settings(self) -> ExportSettings:
        """Get the configured export settings."""
        settings = ExportSettings()

        # Format
        format_map = {
            0: ExportFormat.JPEG,
            1: ExportFormat.PNG,
            2: ExportFormat.TIFF
        }
        settings.format = format_map.get(self.format_combo.currentIndex(), ExportFormat.JPEG)

        # Output directory
        settings.output_dir = self.folder_edit.text()

        # Quality settings
        settings.jpeg_quality = self.quality_slider.value()
        settings.png_compression = self.png_compression_spin.value()

        tiff_compression_map = {0: 'none', 1: 'lzw', 2: 'deflate'}
        settings.tiff_compression = tiff_compression_map.get(
            self.tiff_compression_combo.currentIndex(), 'lzw'
        )

        # Naming
        settings.prefix = self.prefix_edit.text()
        settings.start_number = self.start_number_spin.value()
        settings.zero_padding = self.padding_spin.value()

        # Resize
        settings.resize_enabled = self.resize_group.isChecked()
        if settings.resize_enabled:
            settings.max_width = self.max_width_spin.value()
            settings.max_height = self.max_height_spin.value()

            method_map = {0: 'lanczos', 1: 'bilinear', 2: 'nearest'}
            settings.resize_method = method_map.get(
                self.resize_method_combo.currentIndex(), 'lanczos'
            )

        # Options
        settings.preserve_exif = self.preserve_exif_check.isChecked()

        return settings

    def accept(self):
        """Validate and accept."""
        if not self.folder_edit.text():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Error', 'Please select an output folder.')
            return

        # Create folder if it doesn't exist
        folder = Path(self.folder_edit.text())
        if not folder.exists():
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, 'Error', f'Failed to create output folder: {e}')
                return

        super().accept()
