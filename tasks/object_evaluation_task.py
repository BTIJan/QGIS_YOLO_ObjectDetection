# -*- coding: utf-8 -*-
import csv
from pathlib import Path

import numpy as np

from qgis.core import (
    QgsTask,
    QgsMessageLog,
    Qgis,
    QgsVectorLayer,
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsSpatialIndex,
    QgsFeatureRequest,
    QgsGeometry,
    QgsFeature,
)


class ObjEvaluationTask(QgsTask):
    def __init__(
        self,
        predictions_path,
        ground_truth_path,
        aoi_path,
        output_dir,
        iou_threshold=0.3,
    ):
        super().__init__("YOLO Evaluation (F1 vs confidence)", QgsTask.CanCancel)

        self.predictions_path = str(predictions_path)
        self.ground_truth_path = str(ground_truth_path)
        self.aoi_path = str(aoi_path)
        self.output_dir = Path(output_dir)

        self.iou_threshold = float(iou_threshold)
        self.step_size = 0.02

        self.csv_path = None
        self.png_path = None

    def run(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = self.output_dir / "metrics.csv"
            self.png_path = self.output_dir / "f1_curve.png"

            QgsMessageLog.logMessage("Loading layers...", "Messages", Qgis.Info)

            pred_lyr = QgsVectorLayer(self.predictions_path, "predictions", "ogr")
            gt_lyr = QgsVectorLayer(self.ground_truth_path, "ground_truth", "ogr")
            aoi_lyr = QgsVectorLayer(self.aoi_path, "aoi", "ogr")

            if not pred_lyr.isValid():
                raise ValueError(f"Invalid predictions layer: {self.predictions_path}")
            if not gt_lyr.isValid():
                raise ValueError(f"Invalid ground truth layer: {self.ground_truth_path}")
            if not aoi_lyr.isValid():
                raise ValueError(f"Invalid AOI layer: {self.aoi_path}")

            pred_fields = pred_lyr.fields()
            if pred_fields.indexFromName("confidence") < 0:
                raise ValueError("Predictions layer has no 'confidence' field.")

            if self.isCanceled():
                return False

            # Use predictions CRS as target CRS
            target_crs = pred_lyr.crs()
            QgsMessageLog.logMessage(
                f"Using predictions CRS as target: {target_crs.authid()}", 
                "Messages", 
                Qgis.Info
            )

            proj = QgsProject.instance()
            
            # Create transforms (only if CRS differs)
            tr_gt = None
            tr_aoi = None
            
            if gt_lyr.crs() != target_crs:
                tr_gt = QgsCoordinateTransform(gt_lyr.crs(), target_crs, proj)
                QgsMessageLog.logMessage(
                    f"Will transform GT from {gt_lyr.crs().authid()} to {target_crs.authid()}", 
                    "Messages", 
                    Qgis.Info
                )
            else:
                QgsMessageLog.logMessage("GT CRS matches predictions, no transform needed", "Messages", Qgis.Info)
            
            if aoi_lyr.crs() != target_crs:
                tr_aoi = QgsCoordinateTransform(aoi_lyr.crs(), target_crs, proj)
                QgsMessageLog.logMessage(
                    f"Will transform AOI from {aoi_lyr.crs().authid()} to {target_crs.authid()}", 
                    "Messages", 
                    Qgis.Info
                )
            else:
                QgsMessageLog.logMessage("AOI CRS matches predictions, no transform needed", "Messages", Qgis.Info)

            # AOI union
            QgsMessageLog.logMessage("Building AOI union...", "Messages", Qgis.Info)
            aoi_union = None
            for f in aoi_lyr.getFeatures():
                if self.isCanceled():
                    return False
                g = f.geometry()
                if not g or g.isEmpty():
                    continue
                g = QgsGeometry(g)
                if tr_aoi:
                    g.transform(tr_aoi)
                aoi_union = g if aoi_union is None else aoi_union.combine(g)

            if aoi_union is None or aoi_union.isEmpty():
                raise ValueError("AOI layer has no valid geometry.")

            # Clip GT
            QgsMessageLog.logMessage("Reading + clipping GT...", "Messages", Qgis.Info)
            gt_geoms = {}
            gt_id = 0

            gt_req = QgsFeatureRequest().setFilterRect(aoi_union.boundingBox())
            for f in gt_lyr.getFeatures(gt_req):
                if self.isCanceled():
                    return False
                g = f.geometry()
                if not g or g.isEmpty():
                    continue
                g = QgsGeometry(g)
                if tr_gt:
                    g.transform(tr_gt)
                if not g.intersects(aoi_union):
                    continue
                g_clip = g.intersection(aoi_union)
                if not g_clip or g_clip.isEmpty():
                    continue
                gt_geoms[gt_id] = g_clip
                gt_id += 1

            # Clip predictions (no transform needed - already in target CRS)
            QgsMessageLog.logMessage("Reading + clipping predictions...", "Messages", Qgis.Info)
            pred_geoms = {}
            pred_conf = {}
            pred_id = 0

            pred_req = QgsFeatureRequest().setFilterRect(aoi_union.boundingBox())
            for f in pred_lyr.getFeatures(pred_req):
                if self.isCanceled():
                    return False
                g = f.geometry()
                if not g or g.isEmpty():
                    continue
                try:
                    conf = float(f["confidence"])
                except Exception:
                    conf = 0.0

                g = QgsGeometry(g)
                # No transform needed - predictions are already in target CRS
                
                if not g.intersects(aoi_union):
                    continue
                g_clip = g.intersection(aoi_union)
                if not g_clip or g_clip.isEmpty():
                    continue

                pred_geoms[pred_id] = g_clip
                pred_conf[pred_id] = conf
                pred_id += 1

            QgsMessageLog.logMessage(
                f"After clip -> GT: {len(gt_geoms)} | Preds: {len(pred_geoms)}",
                "Messages",
                Qgis.Info,
            )

            if len(gt_geoms) == 0:
                raise ValueError("No ground-truth geometries after clipping.")
            if len(pred_geoms) == 0:
                raise ValueError("No prediction geometries after clipping.")

            if self.isCanceled():
                return False

            # Spatial index
            QgsMessageLog.logMessage("Building spatial index (GT)...", "Messages", Qgis.Info)
            idx = QgsSpatialIndex()
            for gid, g in gt_geoms.items():
                feat = QgsFeature()
                feat.setId(gid)
                feat.setGeometry(g)
                idx.addFeature(feat)

            # Precompute matches
            QgsMessageLog.logMessage("Precomputing IoU matches...", "Messages", Qgis.Info)
            matches = []  # (pred_id, gt_id, conf, iou)

            for pid, pg in pred_geoms.items():
                if self.isCanceled():
                    return False

                cand_gt_ids = idx.intersects(pg.boundingBox())
                if not cand_gt_ids:
                    continue

                pg_area = pg.area()
                for gid in cand_gt_ids:
                    gg = gt_geoms.get(gid)
                    if gg is None:
                        continue
                    if not pg.intersects(gg):
                        continue

                    inter = pg.intersection(gg)
                    if not inter or inter.isEmpty():
                        continue
                    inter_area = inter.area()

                    union_area = pg_area + gg.area() - inter_area
                    if union_area <= 0:
                        continue

                    iou = inter_area / union_area
                    if iou >= self.iou_threshold:
                        matches.append((pid, gid, pred_conf[pid], iou))

            if not matches:
                QgsMessageLog.logMessage(
                    "No intersections/IoU matches found; writing empty metrics.",
                    "Messages",
                    Qgis.Warning,
                )
                return self._write_outputs_no_matches(len(gt_geoms))

            matches.sort(key=lambda t: (t[2], t[3]), reverse=True)

            thresholds = np.arange(0.0, 1.01, self.step_size)
            metrics_history = []

            QgsMessageLog.logMessage("Calculating F1 curve...", "Messages", Qgis.Info)

            for i, thresh in enumerate(thresholds):
                if self.isCanceled():
                    return False

                active_pred_ids = {pid for pid, c in pred_conf.items() if c >= float(thresh)}

                if not active_pred_ids:
                    metrics_history.append({
                        "confidence": float(thresh),
                        "f1": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "tp": 0,
                        "fp": 0,
                        "fn": int(len(gt_geoms)),
                    })
                    continue

                matched_pred = set()
                matched_gt = set()
                tp = 0

                for pid, gid, conf, iou in matches:
                    if pid not in active_pred_ids:
                        continue
                    if pid in matched_pred or gid in matched_gt:
                        continue
                    matched_pred.add(pid)
                    matched_gt.add(gid)
                    tp += 1

                fp = int(len(active_pred_ids) - tp)
                fn = int(len(gt_geoms) - tp)

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

                metrics_history.append({
                    "confidence": float(thresh),
                    "f1": float(f1),
                    "precision": float(prec),
                    "recall": float(rec),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                })

                self.setProgress(int(100 * (i + 1) / max(1, len(thresholds))))

            self._write_csv(metrics_history)
            QgsMessageLog.logMessage(f"CSV written: {self.csv_path}", "Messages", Qgis.Info)

            self._write_png(metrics_history)
            QgsMessageLog.logMessage(f"PNG written: {self.png_path}", "Messages", Qgis.Info)

            best = max(metrics_history, key=lambda d: d["f1"])
            QgsMessageLog.logMessage(
                f"Evaluation complete. Best F1={best['f1']:.3f} at conf={best['confidence']:.2f}. "
                f"Saved: {self.csv_path.name}, {self.png_path.name}",
                "Messages",
                Qgis.Success,
            )
            return True

        except Exception as e:
            QgsMessageLog.logMessage(f"ObjEvaluationTask failed: {e}", "Messages", Qgis.Critical)
            return False

    def _write_outputs_no_matches(self, n_gt):
        metrics_history = []
        thresholds = np.arange(0.0, 1.01, self.step_size)
        for thresh in thresholds:
            metrics_history.append({
                "confidence": float(thresh),
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": int(n_gt),
            })
        self._write_csv(metrics_history)
        self._write_png(metrics_history)
        return True

    def _write_csv(self, rows):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["confidence", "f1", "precision", "recall", "tp", "fp", "fn"],
            )
            w.writeheader()
            w.writerows(rows)

    def _write_png(self, rows):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        conf = [r["confidence"] for r in rows]
        f1 = [r["f1"] for r in rows]
        prec = [r["precision"] for r in rows]
        rec = [r["recall"] for r in rows]

        best_row = max(rows, key=lambda d: d["f1"])

        fig = plt.figure(figsize=(10, 6))
        plt.plot(conf, f1, label="F1 Score", linewidth=2, marker="o")
        plt.plot(conf, prec, label="Precision", linestyle="--", alpha=0.6)
        plt.plot(conf, rec, label="Recall", linestyle="--", alpha=0.6)

        plt.title(f"F1 Score vs Confidence Threshold (IoU={self.iou_threshold})")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Score")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.annotate(
            f"Max F1: {best_row['f1']:.3f}\n@ Conf: {best_row['confidence']:.2f}",
            xy=(best_row["confidence"], best_row["f1"]),
            xytext=(best_row["confidence"], max(0.0, best_row["f1"] - 0.1)),
            arrowprops=dict(facecolor="black", shrink=0.05),
        )

        fig.savefig(str(self.png_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
