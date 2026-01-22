"""
Main entry point for PageScan application.
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from .ui.main_window import MainWindow


def main():
    """Run the PageScan application."""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName('PageScan')
    app.setApplicationVersion('1.0.0')
    app.setOrganizationName('PageScan')

    # Set default style
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
