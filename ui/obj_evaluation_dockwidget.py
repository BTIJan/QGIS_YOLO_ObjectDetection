from qgis.PyQt import QtCore
from qgis.PyQt.QtWidgets import QDockWidget
from qgis.gui import QgsScrollArea

from .obj_evaluation_dialog import ObjEvaluationDialog


class ObjEvaluationDockWidget(QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Evaluation (F1 curve)")
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        self.inner = ObjEvaluationDialog(self)

        scroll = QgsScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.inner)

        self.setWidget(scroll)
