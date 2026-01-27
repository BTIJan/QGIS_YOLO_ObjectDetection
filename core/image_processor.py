import rasterio
import numpy as np
import cv2
import re
from pathlib import Path
from qgis.core import QgsMessageLog
def enhance_image_with_clahe(img_uint8: np.ndarray) -> np.ndarray:
    try:
        if img_uint8.shape[0] == 3 and img_uint8.shape[2] != 3:
            img_uint8 = img_uint8.transpose(1, 2, 0)
            
        h, w = img_uint8.shape[:2]
        
        TRAIN_TILE_SIZE = 1024 
        base_grid = 8
        
        grid_h = max(8, int(h / (TRAIN_TILE_SIZE / base_grid)))
        grid_w = max(8, int(w / (TRAIN_TILE_SIZE / base_grid)))
        
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_h, grid_w))
        
        lab[:, :, 0] = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    except Exception as e:
        QgsMessageLog.logMessage(f"[ERROR] CLAHE enhancement failed: {str(e)}", "Messages")
        return img_uint8

    
def read_enhanced_rgb(src, enhance_small_objects: bool = True, window=None):
    try:
        count = src.count
        if count >= 3:
            arr = src.read([1, 2, 3], window=window)
        else:
            a = src.read(1, window=window)
            arr = np.stack([a, a, a], axis=0)

        if arr.dtype == np.uint8:
            img_uint8 = arr.transpose(1, 2, 0)
        else:
            arr = arr.astype(np.float32)
            
            dtype = src.dtypes[0]
            if np.issubdtype(dtype, np.integer):
                type_info = np.iinfo(dtype)
                max_val = type_info.max
            else:
                max_val = np.max(arr) if np.max(arr) > 0 else 1.0

            arr = (arr * (255.0 / max_val)).clip(0, 255)
            img_uint8 = arr.astype("uint8").transpose(1, 2, 0)

        if enhance_small_objects:
            img_uint8 = enhance_image_with_clahe(img_uint8)
            
        return img_uint8

    except Exception as e:
        QgsMessageLog.logMessage(f"[ERROR] Image enhancement failed: {str(e)}", "Messages")
        return None
    
def get_gsd_cm(src: rasterio.io.DatasetReader, tif_path: Path):
    """Detect GSD from raster tags, affine, or filename."""
    tags = src.tags()
    for key in ["GSD", "RES", "Resolution", "PIXEL_SIZE", "PixelSize", "pixel_size"]:
        if key in tags:
            val = str(tags[key]).lower()
            m = re.search(r"([0-9.]+)\s*cm", val)
            if m:
                return float(m.group(1))
            m_m = re.search(r"([0-9.]+)\s*m", val)
            if m_m:
                return float(m_m.group(1))*100
    a = src.transform.a
    e = src.transform.e
    pixel_size_m = (abs(a) + abs(e)) / 2
    if 0 < pixel_size_m < 1:
        return pixel_size_m * 100  # meters to cm

    return None

def resample_image_and_labels(img_uint8: np.ndarray, lines: list[str],
                              W: int, H: int,
                              scale_factor: float,
                              obb: bool) -> tuple[np.ndarray, list[str], int, int]:
   
    if abs(scale_factor - 1.0) < 1e-6:
        return img_uint8, lines, W, H

    newW = max(2, int(round(W * scale_factor)))
    newH = max(2, int(round(H * scale_factor)))
    interp = cv2.INTER_CUBIC if scale_factor > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img_uint8, (newW, newH), interpolation=interp)

    updated = []
    for ln in lines:
        parts = ln.strip().split()
        if not parts:
            continue
        cls = parts[0]
        vals = list(map(float, parts[1:]))

        if obb:
            if len(vals) != 8:
                continue
            denorm = []
            for i in range(0, 8, 2):
                x = vals[i] * W
                y = vals[i+1] * H
                x *= scale_factor
                y *= scale_factor
                denorm.extend([x, y])
            norm = []
            for i in range(0, 8, 2):
                norm.append(denorm[i] / newW)
                norm.append(denorm[i+1] / newH)
            updated.append(" ".join([cls] + [f"{v:.6f}" for v in norm]))
        else:
            if len(vals) != 4:
                continue
            cx = (vals[0] * W) * scale_factor / newW
            cy = (vals[1] * H) * scale_factor / newH
            bw = (vals[2] * W) * scale_factor / newW
            bh = (vals[3] * H) * scale_factor / newH
            updated.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return resized, updated, newW, newH

