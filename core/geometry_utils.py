import numpy as np
from shapely.geometry import Polygon
from qgis.core import QgsMessageLog

def valid_quad_px(pts: list[tuple[float, float]], min_box_px: float) -> bool:
    if len(pts) != 4:
        return False

    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    area = 0.5 * abs(
        x[0]*y[1] + x[1]*y[2] + x[2]*y[3] + x[3]*y[0]
        - y[0]*x[1] - y[1]*x[2] - y[2]*x[3] - y[3]*x[0]
    )
    if area <= 1e-6:
        return False

    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i+1) % 4]
        if np.hypot(x2 - x1, y2 - y1) < min_box_px:
            return False
    return True

def valid_quad_norm(pts: list[tuple[float, float]]) -> bool:
    """
    Ensure normalized points are within image bounds (0–1).
    """
    for x, y in pts:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False
    return True

def geom_to_yolo_aabb(geom, transform, W, H, class_id: int, min_box_px: float):
    try:
        env = geom.envelope
        minx, miny, maxx, maxy = env.bounds
        inv = ~transform
        px_min, py_min = inv * (minx, miny)
        px_max, py_max = inv * (maxx, maxy)

        px_min, px_max = np.clip([px_min, px_max], 0, W)
        py_min, py_max = np.clip([py_min, py_max], 0, H)

        bw = abs(px_max - px_min)
        bh = abs(py_max - py_min)
        if bw < min_box_px or bh < min_box_px:
            return None

        cx, cy = ((px_min + px_max) / 2) / W, ((py_min + py_max) / 2) / H
        bw_n, bh_n = bw / W, bh / H
        cx, cy = np.clip([cx, cy], 0, 1)
        bw_n, bh_n = np.clip([bw_n, bh_n], 0, 1)
        return f"{class_id} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}"
    except Exception:
        return None
    
def geom_to_yolo_obb(geom, transform, W, H, class_id: int, heading_deg: float | None, min_box_px: float) -> str | None:
    try: 
        inv = ~transform        
        if isinstance(geom, Polygon):
            if len(geom.exterior.coords) <= 5: 
                mrr = geom 
            else:
                mrr = geom.minimum_rotated_rectangle
        else:
            mrr = geom.minimum_rotated_rectangle 
        
        if not isinstance(mrr, Polygon):
            mrr = geom.envelope

        xs, ys = mrr.exterior.coords.xy
        pts_world = list(zip(xs, ys))[:4] 
        
        pts = [inv * (x, y) for (x, y) in pts_world]
        

        pts = [(float(np.clip(x, 0, W)), float(np.clip(y, 0, H))) for (x, y) in pts]
        
        if not valid_quad_px(pts, min_box_px):
            return None

        npts = [(x / W, y / H) for (x, y) in pts]
        
        if not valid_quad_norm(npts):
            return None
            
        cx = sum(p[0] for p in npts) / 4.0
        cy = sum(p[1] for p in npts) / 4.0
        ang = [np.arctan2(py - cy, px - cx) for (px, py) in npts]
        order = np.argsort(ang)
        ordered = [npts[i] for i in order][::-1]

        flat = []
        for (x, y) in ordered:
            flat.extend([x, y])
        return " ".join([str(class_id)] + [f"{v:.6f}" for v in flat])

    except Exception as e:
        QgsMessageLog.logMessage(f"[WARN] OBB conversion failed: {str(e)}", "ImageryTiler")
        return None


