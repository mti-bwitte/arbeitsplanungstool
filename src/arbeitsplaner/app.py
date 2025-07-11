import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow


app = QApplication([sys.argv])
window = MainWindow()
window.setMinimumSize(600,800)
window.show()
app.exec()
