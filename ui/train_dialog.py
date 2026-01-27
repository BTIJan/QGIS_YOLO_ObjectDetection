import os
from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import QWidget, QDialogButtonBox

FORM_CLASS, _ = uic.loadUiType(
    os.path.join(os.path.dirname(__file__), "train_dialog_base.ui")
)


class TrainDialog(QWidget, FORM_CLASS):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # Optional: relabel OK button (works if buttonBox exists in UI)
        if hasattr(self, "buttonBox") and self.buttonBox is not None:
            btn = self.buttonBox.button(QDialogButtonBox.Ok)
            if btn is not None:
                btn.setText("Start Training")
