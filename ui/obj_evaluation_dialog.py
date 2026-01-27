import os
from qgis.PyQt import uic, QtWidgets

FORM_CLASS, _ = uic.loadUiType(
    os.path.join(os.path.dirname(__file__), "obj_evaluation_dialog_base.ui")
)


class ObjEvaluationDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super(ObjEvaluationDialog, self).__init__(parent)
        self.setupUi(self)
