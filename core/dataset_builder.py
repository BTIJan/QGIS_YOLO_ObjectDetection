from PIL import Image
import numpy as np
import yaml
from pathlib import Path
from qgis.core import QgsMessageLog

def save_split(dataset, img_dir: Path, lbl_dir: Path):
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for item in dataset:
        img_array = item.get("img") if item is not None else None
        if img_array is None or not isinstance(img_array, np.ndarray):
            QgsMessageLog.logMessage(f"[SAFE SKIP] Invalid image entry skipped: {item}", "Messages")
            continue

        try:
            img_path = img_dir / item["img_name"]
            Image.fromarray(img_array).save(
                img_path,
                format='JPEG',
                quality=93,
                subsampling=2,
                optimize=True,
                progressive=True
            )
            with open(lbl_dir / item["txt_name"], "w", encoding="utf-8") as f:
                f.write("\n".join(item["lines"]))
        except Exception as e:
            QgsMessageLog.logMessage(f"[ERROR] Failed saving {item.get('img_name')}: {str(e)}", "Messages")
            continue

def write_dataset_yaml(ds_root: Path, class_names):
    """
    Write YOLO data.yaml with train/val relative paths and class list.
    """
    data = dict(
        train=(ds_root / "images" / "train").as_posix(),
        val=(ds_root / "images" / "val").as_posix(),
        nc=len(class_names),
        names=class_names,
    )
    ds_root.mkdir(parents=True, exist_ok=True)
    with open(ds_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)