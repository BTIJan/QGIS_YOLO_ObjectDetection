from qgis.PyQt import QtCore
from qgis.PyQt.QtWidgets import QDockWidget
from qgis.gui import QgsScrollArea
from .tiling_dialog import TilingDialog  

class TilingDockWidget(QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Tiling Tool")
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        self.inner = TilingDialog(self)

        scroll = QgsScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setVerticalOnly(True)      
        scroll.setWidget(self.inner)       
        self.setWidget(scroll)
