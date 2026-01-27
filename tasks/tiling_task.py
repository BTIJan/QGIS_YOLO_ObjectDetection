import os
import random
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from qgis.core import (
    QgsTask, QgsMessageLog, Qgis, QgsVectorLayer, QgsCoordinateReferenceSystem,
    QgsCoordinateTransform, QgsProject, QgsRectangle
)

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.features import rasterize
    from shapely.geometry import box
    from shapely.wkb import loads as wkb_loads
    from shapely.ops import transform as shapely_transform
    from pyproj import Transformer
except ImportError as e:
    print(f"Critical import error in TilingTask: {e}")


class TilingTask(QgsTask):
    def __init__(self, shapefile_path, raster_paths, output_dir, tile_size, overlap, bg_ratio):
        super().__init__('Tiling', QgsTask.CanCancel)
        self.shapefile_path = shapefile_path
        self.raster_paths = [Path(p) for p in raster_paths]
        self.output_dir = Path(output_dir)
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.bg_ratio = float(bg_ratio)
        self.total = len(self.raster_paths)
        self.processed = 0
        self.exception = None

    def run(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Load Vector Layer
            layer = QgsVectorLayer(self.shapefile_path, "Labels", "ogr")
            if not layer.isValid():
                raise ValueError(f"Could not load shapefile: {self.shapefile_path}")
            
            # 2. Extract Geometries as SHAPELY objects
            shapely_geoms = []
            src_crs = layer.crs()
            
            for feat in layer.getFeatures():
                if feat.hasGeometry():
                    geom = feat.geometry()
                    wkb_bytes = geom.asWkb()
                    if wkb_bytes:
                        try:
                            s_geom = wkb_loads(bytes(wkb_bytes))
                            if not s_geom.is_empty:
                                shapely_geoms.append(s_geom)
                        except Exception as e:
                            pass
            
            QgsMessageLog.logMessage(f"Loaded {len(shapely_geoms)} valid geometries.", 'Messages', Qgis.Info)

            primary_tiles_saved = 0
            background_pool = []

            # 3. Process Rasters
            for i, tif_path in enumerate(self.raster_paths):
                if self.isCanceled(): return False
                
                p_tiles, bg_tiles = self.process_raster(tif_path, shapely_geoms, src_crs)
                
                primary_tiles_saved += p_tiles
                background_pool.extend(bg_tiles)
                
                self.processed = i + 1
                self.setProgress(int(100 * self.processed / self.total))

            # 4. Background Sampling
            num_to_sample = min(int(primary_tiles_saved * self.bg_ratio), len(background_pool))
            if num_to_sample > 0:
                self._save_background_tiles(background_pool, num_to_sample)

            QgsMessageLog.logMessage(f"Finished: {primary_tiles_saved} primary, {num_to_sample} background.", 'Messages', Qgis.Success)
            return True

        except Exception as e:
            self.exception = e
            QgsMessageLog.logMessage(f"Critical Task Error: {e}", 'Messages', Qgis.Critical)
            return False

    def process_raster(self, tif_path, all_shapely_geoms, vector_crs):
        primary_count = 0
        background_candidates = []

        try:
            with rasterio.open(tif_path) as src:
                # Get raster CRS
                raster_crs = src.crs
                raster_bounds = box(*src.bounds)
                
                # Convert to QGIS CRS
                try:
                    r_wkt = raster_crs.to_wkt()
                    qgis_raster_crs = QgsCoordinateReferenceSystem.fromWkt(r_wkt)
                except:
                    QgsMessageLog.logMessage(f"Could not parse raster CRS for {tif_path.name}, using vector CRS", 
                                           'Messages', Qgis.Warning)
                    qgis_raster_crs = vector_crs

                local_geoms = []
                
                # Check if reprojection is needed
                needs_transform = vector_crs != qgis_raster_crs and vector_crs.isValid() and qgis_raster_crs.isValid()
                
                if not needs_transform:
                    # No transform needed: direct intersection test
                    local_geoms = [g for g in all_shapely_geoms if g.intersects(raster_bounds)]
                    QgsMessageLog.logMessage(f"CRS match: {len(local_geoms)} geometries intersect {tif_path.name}", 
                                           'Messages', Qgis.Info)
                else:
                    # Transform needed: reproject geometries to raster CRS
                    QgsMessageLog.logMessage(f"Reprojecting vector from {vector_crs.authid()} to {qgis_raster_crs.authid()}", 
                                           'Messages', Qgis.Info)
                    
                    # Create pyproj transformer for efficient batch transformation
                    try:
                        transformer = Transformer.from_crs(
                            vector_crs.toWkt(), 
                            qgis_raster_crs.toWkt(), 
                            always_xy=True
                        )
                        
                        # Transform raster bounds to vector CRS for initial filter
                        tr_rev = QgsCoordinateTransform(qgis_raster_crs, vector_crs, QgsProject.instance())
                        r_bbox_vec = tr_rev.transformBoundingBox(QgsRectangle(*src.bounds))
                        r_bbox_shapely = box(r_bbox_vec.xMinimum(), r_bbox_vec.yMinimum(), 
                                           r_bbox_vec.xMaximum(), r_bbox_vec.yMaximum())
                        
                        # First filter: only keep geometries that might intersect
                        candidate_geoms = [g for g in all_shapely_geoms if g.intersects(r_bbox_shapely)]
                        
                        # Transform candidates to raster CRS
                        for s_geom in candidate_geoms:
                            try:
                                # Use pyproj for efficient transformation
                                transformed_geom = shapely_transform(transformer.transform, s_geom)
                                if transformed_geom.intersects(raster_bounds):
                                    local_geoms.append(transformed_geom)
                            except Exception as e:
                                QgsMessageLog.logMessage(f"Failed to transform geometry: {e}", 
                                                       'Messages', Qgis.Warning)
                        
                        QgsMessageLog.logMessage(f"Transformed {len(local_geoms)} geometries to raster CRS", 
                                               'Messages', Qgis.Info)
                    
                    except Exception as e:
                        QgsMessageLog.logMessage(f"Transformation setup failed: {e}, using fallback", 
                                               'Messages', Qgis.Warning)
                        # Fallback: assume geometries are in compatible CRS
                        local_geoms = [g for g in all_shapely_geoms if g.intersects(raster_bounds)]

                if not local_geoms:
                    QgsMessageLog.logMessage(f"No geometries intersect {tif_path.name}", 'Messages', Qgis.Info)
                    return 0, []

                # Generate Windows
                step = self.tile_size - self.overlap
                windows = []
                t_idx = 0
                for row in range(0, src.height, step):
                    for col in range(0, src.width, step):
                        win_w = min(self.tile_size, src.width - col)
                        win_h = min(self.tile_size, src.height - row)
                        if win_w > 0 and win_h > 0:
                            windows.append((col, row, win_w, win_h, t_idx))
                            t_idx += 1

            # Worker Function
            def worker(win):
                if self.isCanceled(): return (0, [])
                col, row, win_w, win_h, idx = win
                
                with rasterio.open(tif_path) as src_local:
                    window = Window(col, row, win_w, win_h)
                    tile_transform = src_local.window_transform(window)
                    tile_bounds = src_local.window_bounds(window)
                    tile_box = box(*tile_bounds)

                    hits = [g for g in local_geoms if g.intersects(tile_box)]
                    
                    is_primary = False
                    if hits:
                        shapes = ((g, 1) for g in hits)
                        mask = rasterize(
                            shapes,
                            out_shape=(win_h, win_w),
                            transform=tile_transform,
                            fill=0,
                            all_touched=True,
                            dtype=np.uint8
                        )
                        if np.any(mask):
                            is_primary = True
                    
                    data = src_local.read(window=window)
                    
                    if is_primary:
                        out = self.output_dir / f"{Path(tif_path).stem}_tile_{idx:05d}.tif"
                        with rasterio.open(out, 'w', driver='GTiff',
                                         height=win_h, width=win_w,
                                         count=data.shape[0], dtype=data.dtype,
                                         crs=src_local.crs, transform=tile_transform,
                                         compress='JPEG') as dst:
                            dst.write(data)
                        return (1, [])
                    else:
                        return (0, [(tif_path, window, tile_transform, src_local.crs, data.dtype, Path(tif_path).stem)])

            max_workers = min(os.cpu_count() or 4, 8)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for p, bg in ex.map(worker, windows):
                    primary_count += p
                    if bg: background_candidates.extend(bg)
            
            return primary_count, background_candidates

        except Exception as e:
            QgsMessageLog.logMessage(f"Raster Error {tif_path.name}: {e}", 'Messages', Qgis.Critical)
            return 0, []

    def _save_background_tiles(self, pool, count):
        try:
            selected = random.sample(pool, count)
            for idx, (tif, win, trans, crs, dt, stem) in enumerate(selected):
                if self.isCanceled(): return
                try:
                    with rasterio.open(tif) as src:
                        data = src.read(window=win)
                    out = self.output_dir / f"{stem}_bg_tile_{idx:05d}.tif"
                    with rasterio.open(out, 'w', driver='GTiff', height=data.shape[1], width=data.shape[2],
                                     count=data.shape[0], dtype=dt, crs=crs, transform=trans, compress='JPEG') as dst:
                        dst.write(data)
                except: pass
        except ValueError:
            pass
