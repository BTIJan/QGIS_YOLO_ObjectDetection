from qgis.PyQt import QtCore
from qgis.PyQt.QtWidgets import QDockWidget

from qgis.gui import QgsScrollArea  # QGIS-recommended scroll area [web:170]
from .train_dialog import TrainDialog


class TrainDockWidget(QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("YOLO Training")
        self.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )

        # Your big form widget
        self.inner = TrainDialog(self)

        # Make it scrollable
        scroll = QgsScrollArea(self)              # subclass of QScrollArea [web:170]
        scroll.setWidgetResizable(True)           # allow resizing to avoid horizontal scroll [web:161]
        scroll.setVerticalOnly(True)              # QGIS convenience: keep width fitted [web:170]
        scroll.setWidget(self.inner)              # put the form inside the scroll area [web:161]

        # Put scroll area inside the dock
        self.setWidget(scroll)
