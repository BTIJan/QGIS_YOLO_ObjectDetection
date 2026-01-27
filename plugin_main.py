import os
from pathlib import Path

from qgis.PyQt.QtWidgets import (
    QAction,
    QMenu,
    QToolButton,
    QProgressBar,
    QFileDialog, 
    QWidget,
    QVBoxLayout, 
    QLabel
)
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt import QtCore
from qgis.core import QgsApplication, Qgis

from .ui.train_dockwidget import TrainDockWidget
from .ui.tiling_dockwidget import TilingDockWidget
from .ui.prep_dockwidget import PrepDockWidget
from .ui.detection_dockwidget import DetectionDockWidget
from .ui.obj_evaluation_dockwidget import ObjEvaluationDockWidget

from .tasks.tiling_task import TilingTask
from .tasks.prep_task import PrepTask
from .tasks.train_task import TrainingTask
from .tasks.detection_task import DetectionTask
from .tasks.object_evaluation_task import ObjEvaluationTask

class PluginMain:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)

        self.dlg_tiling = None
        self.dlg_prep = None
        self.dlg_infer = None
        self.dlg_trainer = None
        self.dlg_classify = None
        
        self.dock_train = None 
        self.dock_tiling = None
        self.dock_prep = None
        self.dock_infer = None
        self.dock_trainer = None
        self.dock_objeval = None
        self.dock_classeval = None
        self.actions = []

        self.progress_widget = None
        self.progress_label = None
        self.progress_bar = None

    # ---------------------------------------------------------
    # GUI init / unload
    # ---------------------------------------------------------
    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, "icon.png")
        menu_name = "&YOLO Toolkit EXPERIMENTAL Objectdetection"

        self.tool_button = QToolButton()
        self.tool_button.setIcon(QIcon(icon_path))
        self.tool_button.setToolTip("YOLO Toolkit EXPERIMENTAL Objectdetection")
        self.tool_button.setPopupMode(QToolButton.InstantPopup)
        self.toolbar_action = self.iface.pluginToolBar().addWidget(self.tool_button)

        parent = self.iface.mainWindow()

        self.root_menu = QMenu(parent)
        detection_menu = QMenu("Object Detection", parent)

        self.act_tiling = QAction("YOLO Tiling Tool", self.iface.mainWindow())
        self.act_tiling.triggered.connect(self.launch_tiling_tool)

        self.act_prep = QAction("YOLO Dataset Builder", self.iface.mainWindow())
        self.act_prep.triggered.connect(self.launch_prep_tool)

        self.act_train = QAction("Train YOLO Model", self.iface.mainWindow())
        self.act_train.triggered.connect(self.launch_train_tool)

        self.act_infer = QAction("Object Detection", self.iface.mainWindow())
        self.act_infer.triggered.connect(self.launch_infer_tool)
        
        self.act_objeval = QAction("Evaluate detections (F1 curve)", self.iface.mainWindow())
        self.act_objeval.triggered.connect(self.launch_obj_evaluationtool)
        
        for act in [self.act_tiling, self.act_prep, self.act_train, self.act_infer, self.act_objeval]:
            detection_menu.addAction(act)
            self.actions.append(act)

        self.root_menu.addMenu(detection_menu)

        self.tool_button.setMenu(self.root_menu)

        for act in self.actions:
            self.iface.addPluginToMenu(menu_name, act)

    def unload(self):
        menu_name = "&YOLO Toolkit"

        if hasattr(self, "toolbar_action"):
            self.iface.removeToolBarIcon(self.toolbar_action)
            self.tool_button.deleteLater()

        for action in self.actions:
            self.iface.removePluginMenu(menu_name, action)

        if self.dock_train is not None:
            self.dock_train.close()
            self.dock_train = None

    # ---------------------------------------------------------
    # Helper: file/folder picker
    # ---------------------------------------------------------
    def _set_text(self, widget, is_folder=False, filter="", save=False):
        if is_folder:
            path = QFileDialog.getExistingDirectory(None, "Select Folder")
        elif save:
            path, _ = QFileDialog.getSaveFileName(None, "Save File", "", filter)
        else:
            path, _ = QFileDialog.getOpenFileName(None, "Select File", "", filter)
        if path:
            widget.setText(path)

    # ---------------------------------------------------------
    # Progress helpers
    # ---------------------------------------------------------
    def _create_progress_widget(self):
        mb = self.iface.messageBar()
        mb.clearWidgets()

        self.progress_widget = QWidget()
        layout = QVBoxLayout(self.progress_widget)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(2)

        self.progress_label = QLabel("Preparing dataset…")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)

        mb.pushWidget(self.progress_widget, Qgis.Info)

    def _destroy_progress_widget(self):
        mb = self.iface.messageBar()
        mb.clearWidgets()
        self.progress_widget = None
        self.progress_label = None
        self.progress_bar = None

    def _on_dataset_progress(self, pct: int):
        if self.progress_bar is None:
            return
        if self.progress_label is not None:
            self.progress_label.setText("Preparing dataset…")
        self.progress_bar.setValue(max(0, min(100, pct // 1)))

    def _on_training_progress(self, pct: int):
        if self.progress_bar is None:
            return
        if self.progress_label is not None:
            self.progress_label.setText("Training model…")
        self.progress_bar.setValue(max(0, min(100, pct // 1)))

    def _on_inference_progress(self, pct):
        if self.progress_bar:
            self.progress_label.setText("Running classification…")
            self.progress_bar.setValue(pct)

    # =================================================================
    # TOOL 1: TILING
    # =================================================================
    def launch_tiling_tool(self):
        if getattr(self, "dock_tiling", None) is None:
            self.dock_tiling = TilingDockWidget(self.iface.mainWindow())

            self.iface.addDockWidget(
                QtCore.Qt.RightDockWidgetArea, self.dock_tiling
            )

            self.dock_tiling.inner.browseShapefile.clicked.connect(
                lambda: self._set_text(
                    self.dock_tiling.inner.shapefile,
                    is_folder=False,
                    filter="Vector files (*.gpkg *.shp *.geojson)",
                )
            )
            self.dock_tiling.inner.browseOutput.clicked.connect(
                lambda: self._set_text(self.dock_tiling.inner.outputFolder, is_folder=True)
            )
            self.dock_tiling.inner.browseRaster.clicked.connect(
                lambda: self._set_text(self.dock_tiling.inner.rasterFolder, is_folder=True)
            )
            self.dock_tiling.inner.buttonBox.accepted.connect(self._run_tiling_logic)
            self.dock_tiling.inner.buttonBox.rejected.connect(self.dock_tiling.hide)

        self.dock_tiling.show()
        self.dock_tiling.raise_()


    def _run_tiling_logic(self):
        shp = self.dock_tiling.inner.shapefile.text().strip()
        out = self.dock_tiling.inner.outputFolder.text().strip()
        folder = self.dock_tiling.inner.rasterFolder.text().strip()
        if not shp or not out or not folder:
            self.iface.messageBar().pushMessage(
                "Error", "Missing inputs.", level=Qgis.Critical
            )
            return

        raster_paths = [str(p) for p in Path(folder).glob("*.[tT][iI][fF]")]

        if not raster_paths:
            self.iface.messageBar().pushMessage(
                "Error", "No .tif files found in folder!", level=Qgis.Warning
            )
            return

        task = TilingTask(
            shapefile_path=shp,
            raster_paths=raster_paths,
            output_dir=out,
            tile_size=self.dock_tiling.inner.tileSize.text(),
            overlap=self.dock_tiling.inner.tileOverlap.text(),
            bg_ratio=self.dock_tiling.inner.backgroundRatioSpinBox.value(),
        )
        QgsApplication.taskManager().addTask(task)
        self.iface.messageBar().pushMessage(
            "Started", f"Tiling {len(raster_paths)} rasters...", level=Qgis.Info
        )


# =================================================================
# TOOL 2: PREP
# =================================================================

    def launch_prep_tool(self):
        if getattr(self, "dock_prep", None) is None:
            self.dock_prep = PrepDockWidget(self.iface.mainWindow())

            self.iface.addDockWidget(
                QtCore.Qt.RightDockWidgetArea, self.dock_prep
            )

            # Use .inner. for all UI widgets
            self.dock_prep.inner.browseTilesButton.clicked.connect(
                lambda: self._set_text(self.dock_prep.inner.tilesDirEdit, is_folder=True)
            )
            self.dock_prep.inner.browseShpButton.clicked.connect(
                lambda: self._set_text(
                    self.dock_prep.inner.shpPathEdit,
                    is_folder=False,
                    filter="Vector files (*.gpkg *.shp *.geojson)",
                )
            )
            self.dock_prep.inner.browseOutButton.clicked.connect(
                lambda: self._set_text(self.dock_prep.inner.outDirEdit, is_folder=True)
            )

            if hasattr(self.dock_prep.inner, "runPrepButton"):
                self.dock_prep.inner.runPrepButton.clicked.connect(self._run_prep_logic)

            if hasattr(self.dock_prep.inner, "buttonBox"):
                self.dock_prep.inner.buttonBox.rejected.connect(self.dock_prep.hide)

        self.dock_prep.show()
        self.dock_prep.raise_()


    def _run_prep_logic(self):
        tiles = self.dock_prep.inner.tilesDirEdit.text().strip()
        shp = self.dock_prep.inner.shpPathEdit.text().strip()
        out = self.dock_prep.inner.outDirEdit.text().strip()

        if not tiles or not shp or not out:
            self.iface.messageBar().pushMessage(
                "Error",
                "Missing required inputs (Tiles, Shapefile, or Output).",
                level=Qgis.Critical,
            )
            return

        task = PrepTask(
            tiles_dir=self.dock_prep.inner.tilesDirEdit.text(),
            shp_path=self.dock_prep.inner.shpPathEdit.text(),
            output_dir=self.dock_prep.inner.outDirEdit.text(),
            val_fraction=float(self.dock_prep.inner.valFractionEdit.text() or 0.15),
        )
        QgsApplication.taskManager().addTask(task)
        self.iface.messageBar().pushMessage(
            "Started", "Dataset prep running in background...", level=Qgis.Info
        )

    # =================================================================
    # TOOL 3: TRAINING
    # =================================================================
    def launch_train_tool(self):
        if self.dock_train is None:
            self.dock_train = TrainDockWidget(self.iface.mainWindow())
            self.dock_train.setWindowTitle("YOLO Training")

            self.iface.addDockWidget(
                QtCore.Qt.RightDockWidgetArea, self.dock_train
            )

            # Use .inner. for all UI controls
            self.dock_train.inner.browseYamlButton.clicked.connect(
                lambda: self._set_text(
                    self.dock_train.inner.yamlPathEdit,
                    is_folder=False,
                    filter="YAML Files (*.yaml *.yml)",
                )
            )
            self.dock_train.inner.browseOutputDirButton.clicked.connect(
                lambda: self._set_text(self.dock_train.inner.outputDirEdit, is_folder=True)
            )
            self.dock_train.inner.browseCustomModelBtn.clicked.connect(
                lambda: self._set_text(
                    self.dock_train.inner.customModelEdit,
                    is_folder=False,
                    filter="YOLO Models (*.pt)",
                )
            )

            self.dock_train.inner.buttonBox.accepted.connect(self._run_train_logic)
            self.dock_train.inner.buttonBox.rejected.connect(self.dock_train.hide)

        self.dock_train.show()
        self.dock_train.raise_()


    def _run_train_logic(self):
        yaml_path = self.dock_train.inner.yamlPathEdit.text().strip()
        out_dir   = self.dock_train.inner.outputDirEdit.text().strip()
        run_name  = self.dock_train.inner.runNameEdit.text().strip()
        custom_weights = self.dock_train.inner.customModelEdit.text().strip()

        if not yaml_path or not out_dir or not run_name:
            self.iface.messageBar().pushMessage(
                "Error",
                "Missing YAML, Output Dir, or Run Name.",
                level=Qgis.Critical,
            )
            return

        def get_float(widget, default):
            try:
                return float(widget.text())
            except Exception:
                return default

        if custom_weights and os.path.exists(custom_weights):
            model_name = custom_weights
            self.iface.messageBar().pushMessage(
                "Training",
                f"Fine-tuning from custom weights: {os.path.basename(model_name)}",
                level=Qgis.Info,
            )
        else:
            size_char = self.dock_train.inner.modelSizeCombo.currentText()
            model_name = f"yolo11{size_char}-obb.pt"
            self.iface.messageBar().pushMessage(
                "Training",
                f"Training from base model: {model_name}",
                level=Qgis.Info,
            )

        hyperparams = {
            "model": model_name,
            "epochs": self.dock_train.inner.epochsSpinBox.value(),
            "batch": self.dock_train.inner.batchSpinBox.value(),
            "imgsz": self.dock_train.inner.imgSizeSpinBox.value(),
            "mosaic": get_float(self.dock_train.inner.mosaicEdit, 0.8),
            "copy_paste": get_float(self.dock_train.inner.copypasteEdit, 0.3),
            "lr0": get_float(self.dock_train.inner.lr0Edit, 0.01),
            "lrf": get_float(self.dock_train.inner.lrfEdit, 0.1),
            "cls": get_float(self.dock_train.inner.clsEdit, 1.2),
            "momentum": get_float(self.dock_train.inner.momentumEdit, 0.937),
            "fliplr": get_float(self.dock_train.inner.fliplrEdit, 0.2),
            "flipud": get_float(self.dock_train.inner.flipudEdit, 0.0),
            "scale": get_float(self.dock_train.inner.scaleEdit, 0.1),
            "box": get_float(self.dock_train.inner.boxEdit, 5.0),
            "cache": self.dock_train.inner.cacheEdit.text() == "True",
            "hsv_h": get_float(self.dock_train.inner.hsv_hEdit, 0.015),
            "hsv_s": get_float(self.dock_train.inner.hsv_sEdit, 0.7),
            "hsv_v": get_float(self.dock_train.inner.hsv_vEdit, 0.4),
            "cutmix": get_float(self.dock_train.inner.cutmixEdit, 0.2),
            "degrees": get_float(self.dock_train.inner.degreesEdit, 2.0),
            "perspective": get_float(self.dock_train.inner.perspectiveEdit, 0.0001),
            "shear": get_float(self.dock_train.inner.shearEdit, 0.4),
            "multi_scale": self.dock_train.inner.multiscaleEdit.text() == "True",
        }

        python_exe = "python"
        script_path = os.path.join(self.plugin_dir, "tasks", "train_script.py")

        self.train_task = TrainingTask(
            python_exe=python_exe,
            train_script_path=script_path,
            data_yaml=yaml_path,
            output_dir=out_dir,
            run_name=run_name,
            iface=self.iface,
            **hyperparams,
        )
        QgsApplication.taskManager().addTask(self.train_task)

    # =================================================================
    # TOOL 4: INFERENCE
    # =================================================================
    def launch_infer_tool(self):
        if getattr(self, "dock_infer", None) is None:
            self.dock_infer = DetectionDockWidget(self.iface.mainWindow())
            self.dock_infer.setWindowTitle("Object Detection")

            self.iface.addDockWidget(
                QtCore.Qt.RightDockWidgetArea, self.dock_infer
            )

            self.dock_infer.inner.browseModelButton.clicked.connect(
                lambda: self._set_text(
                    self.dock_infer.inner.modelPathEdit,
                    is_folder=False,
                    filter="YOLO Models (*.pt)",
                )
            )
            self.dock_infer.inner.browseRasterFolder.clicked.connect(
                lambda: self._set_text(self.dock_infer.inner.rasterFolder, is_folder=True)
            )
            self.dock_infer.inner.browseOutGeoJSON.clicked.connect(
                lambda: self._set_text(
                    self.dock_infer.inner.outGeoJsonEdit,
                    is_folder=False,
                    filter="GeoJSON (*.geojson)",
                    save=True,
                )
            )

            self.dock_infer.inner.runButton.clicked.connect(self._run_infer_logic)

            if hasattr(self.dock_infer.inner, "buttonBox"):
                self.dock_infer.inner.buttonBox.rejected.connect(self.dock_infer.hide)

        self.dock_infer.show()
        self.dock_infer.raise_()


    def _run_infer_logic(self):
        model_path = self.dock_infer.inner.modelPathEdit.text().strip()
        out_geojson = self.dock_infer.inner.outGeoJsonEdit.text().strip()
        folder_path = self.dock_infer.inner.rasterFolder.text().strip()
        slice_size = self.dock_infer.inner.sliceSizeSpin.value()
        min_size = self.dock_infer.inner.minSize.value()
        max_size = self.dock_infer.inner.maxSize.value()

        if not model_path or not out_geojson:
            self.iface.messageBar().pushMessage(
                "Error", "Missing Model or Output path.", level=Qgis.Critical
            )
            return
        if not folder_path or not os.path.isdir(folder_path):
            self.iface.messageBar().pushMessage(
                "Error", "Invalid Raster Folder.", level=Qgis.Critical
            )
            return

        rasters = [str(p) for p in Path(folder_path).glob("*.[tT][iI][fF]")]

        if not rasters:
            self.iface.messageBar().pushMessage(
                "Error",
                "No .tif files found in the selected folder.",
                level=Qgis.Critical,
            )
            return

        task = DetectionTask(
            model_path=model_path,
            rasters=rasters,
            out_geojson=out_geojson,
            slice_h=slice_size,
            slice_w=slice_size,
            overlap=self.dock_infer.inner.overlapDouble.value(),
            conf_th=self.dock_infer.inner.confDouble.value(),
            min_size=min_size,
            max_size=max_size,
            device="cuda",
        )

        QgsApplication.taskManager().addTask(task)
        self.iface.messageBar().pushMessage(
            "Started",
            f"Object detection on {len(rasters)} images...",
            level=Qgis.Info,
        )
    # =================================================================
    # TOOL 5: Objectdetectie externe validatie
    # =================================================================

    def launch_obj_evaluationtool(self):
        if getattr(self, "dockeval", None) is None:
            self.dockeval = ObjEvaluationDockWidget(self.iface.mainWindow())
            self.iface.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dockeval)

            ui = self.dockeval.inner

            # Browse buttons (paths only)
            ui.browsePredictionsButton.clicked.connect(
                lambda: self._set_text(ui.predictionsPathEdit, is_folder=False, filter="GeoJSON (*.geojson *.json)")
            )
            ui.browseGroundTruthButton.clicked.connect(
                lambda: self._set_text(ui.groundTruthPathEdit, is_folder=False, filter="Vector files (*.shp *.gpkg *.geojson *.json)")
            )
            ui.browseAoiButton.clicked.connect(
                lambda: self._set_text(ui.aoiPathEdit, is_folder=False, filter="Vector files (*.shp *.gpkg *.geojson *.json)")
            )
            ui.browseOutputDirButton.clicked.connect(
                lambda: self._set_text(ui.outputDirEdit, is_folder=True)
            )

            # Run
            ui.runButton.clicked.connect(self.runobjevaluationlogic)

            if hasattr(ui, "buttonBox"):
                ui.buttonBox.rejected.connect(self.dockeval.hide)

        self.dockeval.show()
        self.dockeval.raise_()


    def runobjevaluationlogic(self):
        ui = self.dockeval.inner

        pred_path = ui.predictionsPathEdit.text().strip()
        gt_path = ui.groundTruthPathEdit.text().strip()
        aoi_path = ui.aoiPathEdit.text().strip()
        out_dir = ui.outputDirEdit.text().strip()

        if not pred_path or not gt_path or not aoi_path or not out_dir:
            self.iface.messageBar().pushMessage(
                "Error",
                "Missing inputs (predictions, ground truth, AOI, output folder).",
                level=Qgis.Critical,
            )
            return

        iou_thr = ui.iouThresholdSpin.value() if hasattr(ui, "iouThresholdSpin") else 0.3
        
        task = ObjEvaluationTask(
            predictions_path=pred_path,
            ground_truth_path=gt_path,
            aoi_path=aoi_path,
            output_dir=out_dir,
            iou_threshold=iou_thr,
        )
        QgsApplication.taskManager().addTask(task)

        self.iface.messageBar().pushMessage(
            "Started",
            "Evaluation running in background (will write CSV + PNG).",
            level=Qgis.Info,
        )
        self.dockeval.hide()