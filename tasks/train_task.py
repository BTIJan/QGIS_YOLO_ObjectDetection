import subprocess
import os
from qgis.core import QgsTask, QgsMessageLog, Qgis

class TrainingTask(QgsTask):
    def __init__(self, python_exe, train_script_path, data_yaml, epochs, imgsz, batch, model, hsv_h, hsv_s, hsv_v,
                 mosaic, copy_paste, lr0, lrf, cls, momentum, fliplr, flipud, scale, box, cache, iface, cutmix, degrees, perspective, shear, multi_scale,
                 output_dir, run_name):
        super().__init__("YOLO Training Task", QgsTask.CanCancel)
        
        self.python_exe = python_exe
        self.script = train_script_path
        
        self.args_dict = {
            "--data": data_yaml,
            "--model": model,
            "--epochs": str(epochs),
            "--imgsz": str(imgsz),
            "--batch": str(batch),
            "--output": output_dir,
            "--run_name": run_name,
            "--cache": str(cache),
            "--hsv_h": str(hsv_h),
            "--hsv_s": str(hsv_s),
            "--hsv_v": str(hsv_v),
            "--mosaic": str(mosaic),
            "--copy_paste": str(copy_paste),
            "--lr0": str(lr0),
            "--lrf": str(lrf),
            "--cls": str(cls),
            "--momentum": str(momentum),
            "--fliplr": str(fliplr),
            "--flipud": str(flipud),
            "--scale": str(scale),
            "--box": str(box),
            "--cutmix": str(cutmix),
            "--degrees": str(degrees),
            "--perspective": str(perspective),
            "--shear": str(shear),
            "--multi_scale": str(multi_scale),
        }
        
        self.iface = iface
        self.process = None
        self.exception = None

    def run(self):
        try:
            # Build command list
            cmd = [self.python_exe, self.script]
            for k, v in self.args_dict.items():
                cmd.extend([k, v])

            QgsMessageLog.logMessage(f"Executing: {' '.join(cmd)}", 'Messages', Qgis.Info)

            # Startup info to hide console window on Windows
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                startupinfo=startupinfo,
                universal_newlines=True
            )

            # Read output line by line
            while True:
                if self.isCanceled():
                    self.process.terminate()
                    return False
                
                output_line = self.process.stdout.readline()
                if output_line == '' and self.process.poll() is not None:
                    break
                if output_line:
                    clean_line = output_line.strip()
                    QgsMessageLog.logMessage(clean_line, 'Messages', Qgis.Info)

            stderr_output = self.process.stderr.read()
            if self.process.returncode != 0:
                self.exception = f"Training failed (Code {self.process.returncode}):\n{stderr_output}"
                QgsMessageLog.logMessage(self.exception, 'Messages', Qgis.Critical)
                return False
            return True

        except Exception as e:
            self.exception = e
            QgsMessageLog.logMessage(f"Task Exception: {e}", 'Messages', Qgis.Critical)
            return False

    def finished(self, result):
        if result:
            self.iface.messageBar().pushMessage("Messages", "Training Completed Successfully", level=Qgis.Success)
        else:
            msg = str(self.exception) if self.exception else "Training Failed (Check Logs)"
            self.iface.messageBar().pushMessage("Messages", msg, level=Qgis.Critical)

    def cancel(self):
        if self.process:
            self.process.terminate()
        super().cancel()