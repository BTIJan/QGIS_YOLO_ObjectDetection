# -*- coding: utf-8 -*-
import os
from qgis.PyQt import uic
from qgis.PyQt import QtWidgets

# This line loads the UI file. It will fail if the .ui file is missing or named incorrectly.
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'prep_dialog_base.ui'))

class PrepDialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        super(PrepDialog, self).__init__(parent)
        self.setupUi(self)
