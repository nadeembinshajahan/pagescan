"""
Control panel for processing settings.
"""

from typing import Dict, Any, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QScrollArea, QFrame, QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..core.processor import ProcessingSettings, ProcessingStep
from ..core.dewarper import DewarpParams
from ..core.perspective import PerspectiveParams


class CollapsibleGroup(QGroupBox):
    """A collapsible group box."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._on_toggled)
        self._content_widget: Optional[QWidget] = None

    def set_content(self, widget: QWidget):
        """Set the content widget."""
        self._content_widget = widget
        layout = QVBoxLayout(self)
        layout.addWidget(widget)

    def _on_toggled(self, checked: bool):
        """Handle toggle."""
        if self._content_widget:
            self._content_widget.setVisible(checked)


class ControlPanel(QWidget):
    """Panel with all processing controls."""

    settings_changed = pyqtSignal(ProcessingSettings)
    process_clicked = pyqtSignal()
    process_all_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._settings = ProcessingSettings()
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)

        # Analysis info
        self.analysis_group = QGroupBox('Image Analysis')
        analysis_layout = QVBoxLayout(self.analysis_group)
        self.analysis_label = QLabel('Load an image to see analysis')
        self.analysis_label.setWordWrap(True)
        analysis_layout.addWidget(self.analysis_label)
        layout.addWidget(self.analysis_group)

        # Dewarping controls
        self.dewarp_group = QGroupBox('Dewarping (Curve Correction)')
        self.dewarp_group.setCheckable(True)
        self.dewarp_group.setChecked(True)
        self.dewarp_group.toggled.connect(self._on_dewarp_toggled)

        dewarp_layout = QVBoxLayout(self.dewarp_group)

        # Auto/Manual mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel('Mode:'))
        self.dewarp_mode_combo = QComboBox()
        self.dewarp_mode_combo.addItems(['Auto', 'Manual'])
        self.dewarp_mode_combo.currentIndexChanged.connect(self._on_dewarp_mode_changed)
        mode_layout.addWidget(self.dewarp_mode_combo)
        dewarp_layout.addLayout(mode_layout)

        # Control points
        points_layout = QHBoxLayout()
        points_layout.addWidget(QLabel('Control Points:'))
        self.dewarp_points_spin = QSpinBox()
        self.dewarp_points_spin.setRange(5, 50)
        self.dewarp_points_spin.setValue(20)
        self.dewarp_points_spin.valueChanged.connect(self._on_settings_updated)
        points_layout.addWidget(self.dewarp_points_spin)
        dewarp_layout.addLayout(points_layout)

        # Smoothing
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel('Smoothing:'))
        self.dewarp_smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.dewarp_smooth_slider.setRange(0, 100)
        self.dewarp_smooth_slider.setValue(50)
        self.dewarp_smooth_slider.valueChanged.connect(self._on_settings_updated)
        smooth_layout.addWidget(self.dewarp_smooth_slider)
        dewarp_layout.addLayout(smooth_layout)

        self.edit_curve_btn = QPushButton('Edit Curve Points')
        self.edit_curve_btn.setCheckable(True)
        self.edit_curve_btn.clicked.connect(self._on_edit_curve)
        dewarp_layout.addWidget(self.edit_curve_btn)

        layout.addWidget(self.dewarp_group)

        # Perspective controls
        self.perspective_group = QGroupBox('Perspective Correction')
        self.perspective_group.setCheckable(True)
        self.perspective_group.setChecked(True)
        self.perspective_group.toggled.connect(self._on_perspective_toggled)

        perspective_layout = QVBoxLayout(self.perspective_group)

        # Auto-detect button
        self.detect_corners_btn = QPushButton('Auto-Detect Corners')
        self.detect_corners_btn.clicked.connect(self._on_detect_corners)
        perspective_layout.addWidget(self.detect_corners_btn)

        self.edit_corners_btn = QPushButton('Edit Corners')
        self.edit_corners_btn.setCheckable(True)
        self.edit_corners_btn.clicked.connect(self._on_edit_corners)
        perspective_layout.addWidget(self.edit_corners_btn)

        # Margin
        margin_layout = QHBoxLayout()
        margin_layout.addWidget(QLabel('Margin:'))
        self.perspective_margin_spin = QSpinBox()
        self.perspective_margin_spin.setRange(0, 100)
        self.perspective_margin_spin.setValue(0)
        self.perspective_margin_spin.valueChanged.connect(self._on_settings_updated)
        margin_layout.addWidget(self.perspective_margin_spin)
        perspective_layout.addLayout(margin_layout)

        layout.addWidget(self.perspective_group)

        # Deskew controls
        self.deskew_group = QGroupBox('Deskew (Rotation)')
        self.deskew_group.setCheckable(True)
        self.deskew_group.setChecked(True)
        self.deskew_group.toggled.connect(self._on_deskew_toggled)

        deskew_layout = QVBoxLayout(self.deskew_group)

        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel('Angle:'))
        self.deskew_auto_check = QCheckBox('Auto')
        self.deskew_auto_check.setChecked(True)
        self.deskew_auto_check.toggled.connect(self._on_deskew_auto_toggled)
        angle_layout.addWidget(self.deskew_auto_check)

        self.deskew_angle_spin = QDoubleSpinBox()
        self.deskew_angle_spin.setRange(-45, 45)
        self.deskew_angle_spin.setSingleStep(0.1)
        self.deskew_angle_spin.setValue(0)
        self.deskew_angle_spin.setEnabled(False)
        self.deskew_angle_spin.valueChanged.connect(self._on_settings_updated)
        angle_layout.addWidget(self.deskew_angle_spin)
        deskew_layout.addLayout(angle_layout)

        layout.addWidget(self.deskew_group)

        # Crop controls
        self.crop_group = QGroupBox('Auto Crop')
        self.crop_group.setCheckable(True)
        self.crop_group.setChecked(True)
        self.crop_group.toggled.connect(self._on_crop_toggled)

        crop_layout = QVBoxLayout(self.crop_group)

        crop_margin_layout = QHBoxLayout()
        crop_margin_layout.addWidget(QLabel('Margin:'))
        self.crop_margin_spin = QSpinBox()
        self.crop_margin_spin.setRange(0, 100)
        self.crop_margin_spin.setValue(10)
        self.crop_margin_spin.valueChanged.connect(self._on_settings_updated)
        crop_margin_layout.addWidget(self.crop_margin_spin)
        crop_layout.addLayout(crop_margin_layout)

        layout.addWidget(self.crop_group)

        # White balance controls
        self.wb_group = QGroupBox('White Balance')
        self.wb_group.setCheckable(True)
        self.wb_group.setChecked(False)
        self.wb_group.toggled.connect(self._on_wb_toggled)

        wb_layout = QVBoxLayout(self.wb_group)

        wb_method_layout = QHBoxLayout()
        wb_method_layout.addWidget(QLabel('Method:'))
        self.wb_method_combo = QComboBox()
        self.wb_method_combo.addItems(['Gray World', 'White Patch', 'Manual'])
        self.wb_method_combo.currentIndexChanged.connect(self._on_wb_method_changed)
        wb_method_layout.addWidget(self.wb_method_combo)
        wb_layout.addLayout(wb_method_layout)

        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel('Temperature:'))
        self.wb_temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.wb_temp_slider.setRange(50, 150)
        self.wb_temp_slider.setValue(100)
        self.wb_temp_slider.setEnabled(False)
        self.wb_temp_slider.valueChanged.connect(self._on_settings_updated)
        temp_layout.addWidget(self.wb_temp_slider)
        wb_layout.addLayout(temp_layout)

        layout.addWidget(self.wb_group)

        # Exposure controls
        self.exposure_group = QGroupBox('Exposure & Contrast')
        self.exposure_group.setCheckable(True)
        self.exposure_group.setChecked(False)
        self.exposure_group.toggled.connect(self._on_exposure_toggled)

        exposure_layout = QVBoxLayout(self.exposure_group)

        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel('Exposure:'))
        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setRange(-200, 200)
        self.exposure_slider.setValue(0)
        self.exposure_slider.valueChanged.connect(self._on_settings_updated)
        exp_layout.addWidget(self.exposure_slider)
        self.exposure_label = QLabel('0')
        self.exposure_label.setFixedWidth(40)
        exp_layout.addWidget(self.exposure_label)
        exposure_layout.addLayout(exp_layout)

        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel('Contrast:'))
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(50, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self._on_settings_updated)
        contrast_layout.addWidget(self.contrast_slider)
        self.contrast_label = QLabel('1.0')
        self.contrast_label.setFixedWidth(40)
        contrast_layout.addWidget(self.contrast_label)
        exposure_layout.addLayout(contrast_layout)

        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel('Brightness:'))
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self._on_settings_updated)
        brightness_layout.addWidget(self.brightness_slider)
        self.brightness_label = QLabel('0')
        self.brightness_label.setFixedWidth(40)
        brightness_layout.addWidget(self.brightness_label)
        exposure_layout.addLayout(brightness_layout)

        layout.addWidget(self.exposure_group)

        # Binarization controls
        self.binarize_group = QGroupBox('Binarization (B&W)')
        self.binarize_group.setCheckable(True)
        self.binarize_group.setChecked(False)
        self.binarize_group.toggled.connect(self._on_binarize_toggled)

        binarize_layout = QVBoxLayout(self.binarize_group)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel('Method:'))
        self.binarize_method_combo = QComboBox()
        self.binarize_method_combo.addItems(['Otsu (Auto)', 'Adaptive', 'Sauvola', 'Manual'])
        self.binarize_method_combo.currentIndexChanged.connect(self._on_binarize_method_changed)
        method_layout.addWidget(self.binarize_method_combo)
        binarize_layout.addLayout(method_layout)

        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel('Threshold:'))
        self.binarize_thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.binarize_thresh_slider.setRange(0, 255)
        self.binarize_thresh_slider.setValue(127)
        self.binarize_thresh_slider.setEnabled(False)
        self.binarize_thresh_slider.valueChanged.connect(self._on_settings_updated)
        thresh_layout.addWidget(self.binarize_thresh_slider)
        binarize_layout.addLayout(thresh_layout)

        layout.addWidget(self.binarize_group)

        layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll, 1)

        # Action buttons
        buttons_frame = QFrame()
        buttons_layout = QVBoxLayout(buttons_frame)

        self.process_btn = QPushButton('Process Current')
        self.process_btn.setStyleSheet('font-weight: bold; padding: 10px;')
        self.process_btn.clicked.connect(self.process_clicked.emit)
        buttons_layout.addWidget(self.process_btn)

        self.process_all_btn = QPushButton('Process All')
        self.process_all_btn.clicked.connect(self.process_all_clicked.emit)
        buttons_layout.addWidget(self.process_all_btn)

        self.reset_btn = QPushButton('Reset')
        self.reset_btn.clicked.connect(self._on_reset)
        buttons_layout.addWidget(self.reset_btn)

        main_layout.addWidget(buttons_frame)

        self.setMinimumWidth(280)
        self.setMaximumWidth(350)

    def _get_current_settings(self) -> ProcessingSettings:
        """Get current settings from UI."""
        settings = ProcessingSettings()

        # Dewarp settings
        settings.enable_dewarp = self.dewarp_group.isChecked()
        settings.dewarp_params.num_control_points = self.dewarp_points_spin.value()
        settings.dewarp_params.smoothing_factor = self.dewarp_smooth_slider.value() / 100

        # Perspective settings
        settings.enable_perspective = self.perspective_group.isChecked()
        settings.perspective_params.margin = self.perspective_margin_spin.value()

        # Deskew settings
        settings.enable_deskew = self.deskew_group.isChecked()
        if not self.deskew_auto_check.isChecked():
            settings.deskew_angle = self.deskew_angle_spin.value()

        # Crop settings
        settings.enable_crop = self.crop_group.isChecked()
        settings.crop_margin = self.crop_margin_spin.value()

        # White balance settings
        settings.enable_white_balance = self.wb_group.isChecked()
        wb_methods = ['gray_world', 'white_patch', 'manual']
        settings.wb_method = wb_methods[self.wb_method_combo.currentIndex()]
        settings.wb_temperature = self.wb_temp_slider.value() / 100

        # Exposure settings
        settings.enable_exposure = self.exposure_group.isChecked()
        settings.exposure_adjustment = self.exposure_slider.value() / 100
        settings.contrast = self.contrast_slider.value() / 100
        settings.brightness = self.brightness_slider.value()

        # Binarize settings
        settings.enable_binarize = self.binarize_group.isChecked()
        binarize_methods = ['otsu', 'adaptive', 'sauvola', 'manual']
        settings.binarize_method = binarize_methods[self.binarize_method_combo.currentIndex()]
        settings.binarize_threshold = self.binarize_thresh_slider.value()

        return settings

    def _on_settings_updated(self):
        """Emit settings changed signal."""
        # Update labels
        self.exposure_label.setText(f'{self.exposure_slider.value() / 100:.1f}')
        self.contrast_label.setText(f'{self.contrast_slider.value() / 100:.1f}')
        self.brightness_label.setText(str(self.brightness_slider.value()))

        self._settings = self._get_current_settings()
        self.settings_changed.emit(self._settings)

    def _on_dewarp_toggled(self, checked: bool):
        """Handle dewarp toggle."""
        self._on_settings_updated()

    def _on_dewarp_mode_changed(self, index: int):
        """Handle dewarp mode change."""
        self._on_settings_updated()

    def _on_edit_curve(self, checked: bool):
        """Handle edit curve button."""
        # This would emit a signal to switch viewer to curve edit mode
        pass

    def _on_perspective_toggled(self, checked: bool):
        """Handle perspective toggle."""
        self._on_settings_updated()

    def _on_detect_corners(self):
        """Handle detect corners button."""
        # This would trigger corner detection
        pass

    def _on_edit_corners(self, checked: bool):
        """Handle edit corners button."""
        # This would emit a signal to switch viewer to corner edit mode
        pass

    def _on_deskew_toggled(self, checked: bool):
        """Handle deskew toggle."""
        self._on_settings_updated()

    def _on_deskew_auto_toggled(self, checked: bool):
        """Handle deskew auto toggle."""
        self.deskew_angle_spin.setEnabled(not checked)
        self._on_settings_updated()

    def _on_crop_toggled(self, checked: bool):
        """Handle crop toggle."""
        self._on_settings_updated()

    def _on_wb_toggled(self, checked: bool):
        """Handle white balance toggle."""
        self._on_settings_updated()

    def _on_wb_method_changed(self, index: int):
        """Handle WB method change."""
        self.wb_temp_slider.setEnabled(index == 2)  # Manual mode
        self._on_settings_updated()

    def _on_exposure_toggled(self, checked: bool):
        """Handle exposure toggle."""
        self._on_settings_updated()

    def _on_binarize_toggled(self, checked: bool):
        """Handle binarize toggle."""
        self._on_settings_updated()

    def _on_binarize_method_changed(self, index: int):
        """Handle binarize method change."""
        self.binarize_thresh_slider.setEnabled(index == 3)  # Manual mode
        self._on_settings_updated()

    def _on_reset(self):
        """Handle reset button."""
        self.reset_settings()
        self.reset_clicked.emit()

    def reset_settings(self):
        """Reset all settings to defaults."""
        self.dewarp_group.setChecked(True)
        self.dewarp_points_spin.setValue(20)
        self.dewarp_smooth_slider.setValue(50)
        self.dewarp_mode_combo.setCurrentIndex(0)

        self.perspective_group.setChecked(True)
        self.perspective_margin_spin.setValue(0)

        self.deskew_group.setChecked(True)
        self.deskew_auto_check.setChecked(True)
        self.deskew_angle_spin.setValue(0)

        self.crop_group.setChecked(True)
        self.crop_margin_spin.setValue(10)

        self.wb_group.setChecked(False)
        self.wb_method_combo.setCurrentIndex(0)
        self.wb_temp_slider.setValue(100)

        self.exposure_group.setChecked(False)
        self.exposure_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.brightness_slider.setValue(0)

        self.binarize_group.setChecked(False)
        self.binarize_method_combo.setCurrentIndex(0)
        self.binarize_thresh_slider.setValue(127)

        self._on_settings_updated()

    def update_analysis(self, analysis: Dict[str, Any]):
        """Update the analysis display."""
        text_parts = []

        # Curve analysis
        curve = analysis.get('curve', {})
        if curve.get('detected'):
            severity = curve.get('severity', 'unknown')
            direction = curve.get('curve_direction', 'unknown')
            text_parts.append(f"Curve: {severity} ({direction})")
        else:
            text_parts.append("Curve: Not detected")

        # Perspective analysis
        perspective = analysis.get('perspective', {})
        dist_type = perspective.get('type', 'unknown')
        distortion = perspective.get('distortion', 0)
        if distortion > 0.02:
            text_parts.append(f"Perspective: {dist_type} ({distortion:.1%})")
        else:
            text_parts.append("Perspective: Minimal")

        # Skew
        skew = analysis.get('skew_angle', 0)
        if abs(skew) > 0.5:
            text_parts.append(f"Skew: {skew:.1f}°")
        else:
            text_parts.append("Skew: None")

        # Brightness
        brightness = analysis.get('brightness', {})
        mean_b = brightness.get('mean', 0)
        if mean_b < 100:
            text_parts.append("Brightness: Dark")
        elif mean_b > 200:
            text_parts.append("Brightness: Overexposed")
        else:
            text_parts.append("Brightness: Normal")

        # Recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            text_parts.append("")
            text_parts.append("Recommendations:")
            for rec in recommendations[:3]:  # Show max 3
                text_parts.append(f"  • {rec}")

        self.analysis_label.setText('\n'.join(text_parts))
