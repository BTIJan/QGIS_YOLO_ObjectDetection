# -*- coding: utf-8 -*-
import json
from pathlib import Path
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from shapely.geometry import box, mapping
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from qgis.core import QgsTask, QgsMessageLog, Qgis

from ..core import read_enhanced_rgb


class DetectionTask(QgsTask):
    def __init__(
        self,
        model_path,
        rasters,
        out_geojson,
        slice_h=640,
        slice_w=640,
        overlap=0.3,
        conf_th=0.25,
        device="cuda",
        enhance_small_objects=True,
        min_size=0,
        max_size=0,
    ):
        super().__init__("SAHI Inference", QgsTask.CanCancel)
        self.model_path = str(model_path)
        self.rasters = [Path(p) for p in rasters]
        self.out_geojson = Path(out_geojson)

        self.slice_h = int(slice_h)
        self.slice_w = int(slice_w)
        self.overlap = float(overlap)
        self.conf_th = float(conf_th)
        self.min_size = float(min_size)  # min area in CRS units (m² if CRS in meters)
        self.max_size = float(max_size)  # max area (0 = no max)
        self.device = device
        self.enhance_small_objects = bool(enhance_small_objects)
        self.success = False

        self.chunk_size = 8192
        self.chunk_overlap = 1024  # Large overlap to hide boundary effects

    def run(self):
        try:
            detection_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self.model_path,
                confidence_threshold=self.conf_th,
                device=self.device,
            )

            feats = []
            crs_str = None
            total = len(self.rasters)

            for i, tif in enumerate(self.rasters):
                if self.isCanceled():
                    return False

                QgsMessageLog.logMessage(
                    f"Processing {tif.name}", "Messages", level=Qgis.Info
                )

                with rasterio.open(tif) as src:
                    crs_str = src.crs.to_string() if src.crs else "EPSG:4326"
                    W, H = src.width, src.height
                    transform = src.transform

                    uncompressed_mb = (W * H * src.count) / (1024 * 1024)
                    if uncompressed_mb < 2000:
                        img_uint8 = read_enhanced_rgb(
                            src, self.enhance_small_objects
                        )
                        if img_uint8 is not None:
                            self._run_sahi_on_image(
                                img_uint8,
                                detection_model,
                                feats,
                                transform,
                                tif.name,
                            )

                    else:
                        QgsMessageLog.logMessage(
                            f"Large image ({int(uncompressed_mb)}MB) detected. Using safe chunking.",
                            "Messages",
                            level=Qgis.Info,
                        )

                        step = self.chunk_size - self.chunk_overlap
                        for y_off in range(0, H, step):
                            for x_off in range(0, W, step):
                                if self.isCanceled():
                                    return False

                                window = Window(
                                    x_off,
                                    y_off,
                                    min(self.chunk_size, W - x_off),
                                    min(self.chunk_size, H - y_off),
                                )

                                img_chunk = read_enhanced_rgb(
                                    src,
                                    self.enhance_small_objects,
                                    window=window,
                                )

                                if img_chunk is None:
                                    continue

                                self._run_sahi_on_image(
                                    img_chunk,
                                    detection_model,
                                    feats,
                                    transform,
                                    tif.name,
                                    offset=(x_off, y_off),
                                )

                                del img_chunk

                self.setProgress(int(100 * (i + 1) / max(1, total)))

            # Save to GeoJSON
            self.out_geojson.parent.mkdir(parents=True, exist_ok=True)
            geojson = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": crs_str}},
                "features": feats,
            }

            with open(self.out_geojson, "w", encoding="utf-8") as f:
                json.dump(geojson, f, indent=2)

            QgsMessageLog.logMessage(
                f"SAHI complete: {len(feats)} detections saved.",
                "Messages",
                level=Qgis.Success,
            )
            self.success = True
            return True

        except Exception as e:
            QgsMessageLog.logMessage(
                f"SAHI Task Failed: {e}", "Messages", level=Qgis.Critical
            )
            return False

    def _run_sahi_on_image(self, img, model, feats, transform, source_name, offset=(0, 0)):
        x_off, y_off = offset

        result = get_sliced_prediction(
            image=img,
            detection_model=model,
            slice_height=self.slice_h,
            slice_width=self.slice_w,
            overlap_height_ratio=self.overlap,
            overlap_width_ratio=self.overlap,
            perform_standard_pred=False,
            postprocess_type="NMS",
            postprocess_match_threshold=0.5,
            verbose=0,
        )

        # Pixel center transform
        T1 = transform * Affine.translation(0.5, 0.5)

        for obj in result.object_prediction_list:
            # SAHI / Ultralytics bbox is axis-aligned in pixel coords
            gx_min = obj.bbox.minx + x_off
            gy_min = obj.bbox.miny + y_off
            gx_max = obj.bbox.maxx + x_off
            gy_max = obj.bbox.maxy + y_off

            # Convert to map coordinates
            x1m, y1m = T1 * (gx_min, gy_min)
            x2m, y2m = T1 * (gx_max, gy_max)
            xminm, xmaxm = sorted([x1m, x2m])
            yminm, ymaxm = sorted([y1m, y2m])

            geom = box(xminm, yminm, xmaxm, ymaxm)
            area = geom.area  # m² if CRS units are meters

            # Min/max area filters on axis-aligned AABB
            if self.min_size > 0 and area < self.min_size:
                continue
            if self.max_size > 0 and area > self.max_size:
                continue

            feats.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": {
                        "confidence": float(obj.score.value),
                        "class": int(obj.category.id),
                        "source": source_name,
                    },
                }
            )
