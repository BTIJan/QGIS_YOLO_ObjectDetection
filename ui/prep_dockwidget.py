from qgis.PyQt import QtCore
from qgis.PyQt.QtWidgets import QDockWidget
from qgis.gui import QgsScrollArea

from .prep_dialog import PrepDialog


class PrepDockWidget(QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("YOLO Dataset Builder")
        self.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )

        self.inner = PrepDialog(self)

        scroll = QgsScrollArea(self)
        scroll.setWidgetResizable(True)   
        scroll.setVerticalOnly(True)      
        scroll.setWidget(self.inner)    

        self.setWidget(scroll)          
