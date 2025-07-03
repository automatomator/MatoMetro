import sys
import os # Keep os for path operations

from PyQt6.QtWidgets import QApplication

# Import your classes from the new src modules
from src.ui_windows import PartAnalyzerApp # CORRECTED: Changed MainWindow to PartAnalyzerApp

# --- Main Execution Block ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = PartAnalyzerApp()
    main_window.show()
    sys.exit(app.exec())