# -*- coding: utf-8 -*-
from pathlib import Path
import random

import rasterio
from shapely.geometry import box as sbox
from shapely.wkb import loads as wkb_loads
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

# QGIS Imports
from qgis.core import (
    QgsTask, QgsMessageLog, Qgis, QgsVectorLayer,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform,
    QgsProject, QgsRectangle
)

try:
    from ..core import get_gsd_cm, resample_image_and_labels, read_enhanced_rgb
    from ..core import geom_to_yolo_aabb, geom_to_yolo_obb
    from ..core import save_split, write_dataset_yaml
except ImportError:
    pass


class PrepTask(QgsTask):
    def __init__(
        self,
        tiles_dir,
        shp_path,
        output_dir,
        val_fraction=0.15,
        enhance_small_objects=True,
        obb=True,
        heading_field=None,
        min_box_px=0.5,
        target_gsd_cm=None,
        keep_5cm_native=False,
    ):
        super().__init__("Preparing YOLO Dataset", QgsTask.CanCancel)

        self.tiles_dir = Path(tiles_dir)
        self.shp_path = shp_path
        self.output_dir = Path(output_dir)

        self.val_fraction = float(val_fraction)
        self.enhance_small_objects = enhance_small_objects
        self.obb = bool(obb)
        self.heading_field = heading_field
        self.min_box_px = float(min_box_px)

        # FIX: allow None/"auto"/"" to mean "auto-compute"
        self.target_gsd_cm = None if target_gsd_cm in (None, "", "auto") else float(target_gsd_cm)

        self.keep_5cm_native = bool(keep_5cm_native)
        self.exception = None

    def run(self):
        try:
            # 1. Find Tiles
            tifs = list(self.tiles_dir.rglob("*.tif")) + list(self.tiles_dir.rglob("*.TIF"))
            tifs = sorted(set(tifs))

            if not tifs:
                QgsMessageLog.logMessage("No .tif tiles found.", "Messages", Qgis.Warning)
                return False

            # 1b. Auto target GSD (pixel-weighted mean)
            if self.target_gsd_cm is None:
                weighted_sum = 0.0
                weight_total = 0.0
                used = 0

                for p in tifs:
                    if self.isCanceled():
                        return False
                    try:
                        with rasterio.open(p) as src:
                            gsd_cm = get_gsd_cm(src, p)
                            if gsd_cm is None:
                                continue
                            w = float(src.width * src.height)  # pixel-count weight
                        weighted_sum += float(gsd_cm) * w
                        weight_total += w
                        used += 1
                    except Exception:
                        pass

                if weight_total <= 0:
                    self.target_gsd_cm = 6.0
                    QgsMessageLog.logMessage(
                        "Could not auto-detect GSD; falling back to 6.0 cm.",
                        "Messages",
                        Qgis.Warning,
                    )
                else:
                    self.target_gsd_cm = weighted_sum / weight_total
                    QgsMessageLog.logMessage(
                        f"Auto target_gsd_cm set to {self.target_gsd_cm:.3f} cm "
                        f"(pixel-weighted mean from {used} tiles).",
                        "Messages",
                        Qgis.Info,
                    )

            # 2. Load Vector Layer (QGIS API)
            layer = QgsVectorLayer(self.shp_path, "Labels", "ogr")
            if not layer.isValid():
                raise ValueError(f"Invalid vector file: {self.shp_path}")

            vector_crs = layer.crs()

            cached_features = []
            for feat in layer.getFeatures():
                if not feat.hasGeometry():
                    continue
                try:
                    s_geom = wkb_loads(bytes(feat.geometry().asWkb()))
                    if s_geom.is_empty:
                        continue

                    heading_val = None
                    if self.heading_field:
                        idx = layer.fields().indexOf(self.heading_field)
                        if idx != -1:
                            heading_val = feat.attributes()[idx]

                    cached_features.append({"geom": s_geom, "heading": heading_val})
                except Exception:
                    pass

            QgsMessageLog.logMessage(
                f"Loaded {len(cached_features)} features from {vector_crs.authid()}.",
                "Messages",
                Qgis.Info,
            )

            # 3. Process Tiles
            labeled_results = []
            total = len(tifs)

            for idx, p in enumerate(tifs, 1):
                if self.isCanceled():
                    return False

                self.setProgress(int(idx * 100 / total))

                res = self._process_tile(p, cached_features, vector_crs)
                if res:
                    labeled_results.append(res)

            if not labeled_results:
                QgsMessageLog.logMessage("No matching labels found. Empty dataset.", "Messages", Qgis.Warning)
                return True

            # 4. Split & Save
            random.seed(42)
            random.shuffle(labeled_results)

            val_count = int(len(labeled_results) * self.val_fraction)
            val_set = labeled_results[:val_count]
            train_set = labeled_results[val_count:]

            save_split(train_set, self.output_dir / "images/train", self.output_dir / "labels/train")
            save_split(val_set, self.output_dir / "images/val", self.output_dir / "labels/val")

            names = {0: "Object"}
            write_dataset_yaml(self.output_dir, names)

            QgsMessageLog.logMessage(
                f"Created dataset with {len(train_set)} train, {len(val_set)} val.",
                "Messages",
                Qgis.Success,
            )
            return True

        except Exception as e:
            self.exception = e
            QgsMessageLog.logMessage(f"Prep Task Failed: {e}", "Messages", Qgis.Critical)
            return False

    def _process_tile(self, tif_path, cached_features, vector_crs):
        try:
            with rasterio.open(tif_path) as src:
                W, H = src.width, src.height
                transform = src.transform

                # Get tile CRS (as QGIS CRS)
                try:
                    qgis_tile_crs = QgsCoordinateReferenceSystem.fromWkt(src.crs.to_wkt())
                except Exception:
                    QgsMessageLog.logMessage(
                        f"Could not parse CRS for {tif_path.name}, using vector CRS",
                        "Messages",
                        Qgis.Warning,
                    )
                    qgis_tile_crs = vector_crs

                tile_bounds = sbox(*src.bounds)

                needs_transform = (
                    vector_crs != qgis_tile_crs
                    and vector_crs.isValid()
                    and qgis_tile_crs.isValid()
                )

                # Transform features to tile CRS if needed
                if not needs_transform:
                    local_features = cached_features
                else:
                    local_features = []
                    try:
                        # Filter using bbox transformed back to vector CRS (cheap)
                        tr_rev = QgsCoordinateTransform(qgis_tile_crs, vector_crs, QgsProject.instance())
                        tile_bbox_vec = tr_rev.transformBoundingBox(QgsRectangle(*src.bounds))
                        tile_bbox_shapely = sbox(
                            tile_bbox_vec.xMinimum(), tile_bbox_vec.yMinimum(),
                            tile_bbox_vec.xMaximum(), tile_bbox_vec.yMaximum(),
                        )

                        transformer = Transformer.from_crs(
                            vector_crs.toWkt(),
                            qgis_tile_crs.toWkt(),
                            always_xy=True,
                        )

                        for item in cached_features:
                            if not item["geom"].intersects(tile_bbox_shapely):
                                continue
                            try:
                                transformed_geom = shapely_transform(transformer.transform, item["geom"])
                                local_features.append({"geom": transformed_geom, "heading": item["heading"]})
                            except Exception:
                                pass

                    except Exception as e:
                        QgsMessageLog.logMessage(
                            f"Transformation failed for {tif_path.name}: {e} (fallback: no transform).",
                            "Messages",
                            Qgis.Warning,
                        )
                        local_features = cached_features

                # Generate YOLO labels
                lines = []
                for item in local_features:
                    geom = item["geom"]
                    if not geom.intersects(tile_bounds):
                        continue

                    heading = item["heading"]
                    class_id = 0

                    if not self.obb:
                        yolo_line = geom_to_yolo_aabb(geom, transform, W, H, class_id, self.min_box_px)
                    else:
                        try:
                            h_val = float(heading) if heading is not None else None
                        except Exception:
                            h_val = None
                        yolo_line = geom_to_yolo_obb(geom, transform, W, H, class_id, h_val, self.min_box_px)

                    if yolo_line:
                        lines.append(yolo_line)

                is_background = "bg" in tif_path.name.lower()
                if not lines and not is_background:
                    return None

                # Read Image
                img = read_enhanced_rgb(src)
                if img is None:
                    return None

                # GSD Normalization
                gsd_cm = get_gsd_cm(src, tif_path)
                scale_factor = 1.0
                if gsd_cm is not None:
                    if not (self.keep_5cm_native and gsd_cm <= self.target_gsd_cm):
                        scale_factor = float(gsd_cm / self.target_gsd_cm)

                if abs(scale_factor - 1.0) > 1e-3:
                    img, lines, newW, newH = resample_image_and_labels(
                        img_uint8=img, lines=lines, W=W, H=H,
                        scale_factor=scale_factor, obb=self.obb
                    )

                return {
                    "img": img,
                    "lines": lines,
                    "tif_path": tif_path,
                    "img_name": tif_path.with_suffix(".jpg").name,
                    "tif_name": tif_path.name,
                    "txt_name": tif_path.with_suffix(".txt").name,
                }

        except Exception as e:
            QgsMessageLog.logMessage(f"Error processing tile {tif_path.name}: {e}", "Messages", Qgis.Warning)
            return None
