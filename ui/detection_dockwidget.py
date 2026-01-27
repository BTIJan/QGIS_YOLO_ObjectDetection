from qgis.PyQt import QtCore
from qgis.PyQt.QtWidgets import QDockWidget
from qgis.gui import QgsScrollArea

from .detection_dialog import DetectionDialog

class DetectionDockWidget(QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Object Detection")
        self.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self.inner = DetectionDialog(self)
        scroll = QgsScrollArea(self)
        scroll.setWidgetResizable(True)  
        scroll.setVerticalOnly(True)     
        scroll.setWidget(self.inner)
        self.setWidget(scroll)
