import subprocess
import argparse
from pathlib import Path
import multiprocessing
import logging

class EpochProgressCallback:
    def on_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        total_epochs = trainer.epochs
        
        try:
            metrics = trainer.metrics            
            precision = metrics.get("metrics/precision(B)", 0.0)
            recall = metrics.get("metrics/recall(B)", 0.0)
            map50 = metrics.get("metrics/mAP50(B)", 0.0)
            map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
            
            log_msg = (
                f"EPOCH {epoch}/{total_epochs} | "
                f"Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | "
                f"mAP50: {map50:.4f} | "
                f"mAP50-95: {map50_95:.4f}"
            )
            
            print(log_msg, flush=True)
            
        except Exception as e:
            print(f"EPOCH {epoch}/{total_epochs} completed (Metrics unavailable: {e})", flush=True)

def main():
    def run(cmd):
        print(f"[DEBUG] Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    def str2bool(v):
        return str(v).lower() in ("true", "1", "yes", "on")

    try:
        import os
        import torch
        import ultralytics
        from ultralytics import YOLO

        # Force cache to project directory
        script_dir = Path(__file__).parent.resolve()
        os.environ['ULTRALYTICS_CACHE_DIR'] = str(script_dir)

        amp_weights = script_dir / "yolo11n.pt"
        if not amp_weights.is_file():
            raise FileNotFoundError(f"AMP weights not found at {amp_weights}")
        YOLO.amp_check_weights = str(amp_weights)

        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        print(f"Torch   : {torch.__version__}")
        print(f"Ultra   : {ultralytics.__version__}")

        print("[INFO] Starting YOLO training/tuning…", flush=True)
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available() and system == "Darwin":
            device = "mps"
        else:
            print("[WARNING] No GPU found. Training on CPU.", flush=True)
            device = "cpu"
        print(f"[INFO] Using device: {device.upper()}", flush=True)

        parser = argparse.ArgumentParser()
        parser.add_argument("--data", type=str, required=True)
        parser.add_argument("--model", type=str, required=True)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--imgsz", type=int, default=1024)
        parser.add_argument("--batch", type=int, default=11)
        parser.add_argument("--hsv_h", type=float, default=0.005)
        parser.add_argument("--hsv_s", type=float, default=0.15)
        parser.add_argument("--hsv_v", type=float, default=0.1)
        parser.add_argument("--mosaic", type=float, default=0.4)
        parser.add_argument("--copy_paste", type=float, default=0.0)
        parser.add_argument("--lr0", type=float, default=0.01)
        parser.add_argument("--lrf", type=float, default=0.1)
        parser.add_argument("--cls", type=float, default=0.1)
        parser.add_argument("--momentum", type=float, default=0.937)
        parser.add_argument("--fliplr", type=float, default=0.2)
        parser.add_argument("--flipud", type=float, default=0.0)
        parser.add_argument("--scale", type=float, default=0.1)
        parser.add_argument("--box", type=float, default=5.0)
        parser.add_argument("--cache", type=str2bool, default=True)
        parser.add_argument("--cutmix", type=float, default=0.2)
        parser.add_argument("--degrees", type=float, default=2.5)
        parser.add_argument("--perspective", type=float, default=0.0001)
        parser.add_argument("--shear", type=float, default=0.0001)
        parser.add_argument("--multi_scale", type=str2bool, default=True)
        parser.add_argument("--output", type=str, required=True,
                            help="Output parent directory (Ultralytics project). Will be created if missing.")
        parser.add_argument("--run_name", type=str, required=True,
                            help="Run subfolder name (Ultralytics name).")
        args = parser.parse_args()

        local_model = script_dir / args.model
        if not local_model.is_file():
            raise FileNotFoundError(f"Model not found: {local_model}")

        # Resolve output directory exactly (create parent only; no auto naming)
        out_dir = Path(args.output).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Load the YOLO model ---
        model = YOLO(str(local_model))
        model.add_callback("on_fit_epoch_end", EpochProgressCallback().on_epoch_end)
        model.train(
            data=args.data,
            epochs=args.epochs,
            verbose=False,
            imgsz=args.imgsz,
            batch=args.batch,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            cutmix=args.cutmix,
            degrees=args.degrees,
            perspective=args.perspective,
            shear=args.shear,
            mosaic=args.mosaic,
            copy_paste=args.copy_paste,
            lr0=args.lr0,
            lrf=args.lrf,
            cls=args.cls,
            momentum=args.momentum,
            fliplr=args.fliplr,
            flipud=args.flipud,
            scale=args.scale,
            box=args.box,
            cache=args.cache,
            cos_lr=True,
            augment=True,
            single_cls=True,
            plots=True,
            multi_scale=args.multi_scale,
            device=device,
            project=str(out_dir),
            name=args.run_name,
            workers=0,
            exist_ok=True
        )
        final_dir = out_dir / args.run_name
        print(f"[INFO] Training completed successfully. Outputs: {final_dir}", flush=True)

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()