# -*- coding: utf-8 -*-
def install_requirements():
    missing_packages = []
    
    # 1. Check Geopandas
    try:
        import geopandas
    except ImportError:
        missing_packages.append('geopandas==1.1.1')

    # 2. Check GDAL
    try:
        import osgeo
    except ImportError:
        missing_packages.append('gdal==3.11.5')

    # 3. Check Numpy
    try:
        import numpy
    except ImportError:
        missing_packages.append('numpy==2.3.5')

    # 4. Check Rasterio
    try:
        import rasterio
    except ImportError:
        missing_packages.append('rasterio==1.4.3')

    # 5. Check Shapely
    try:
        import shapely
    except ImportError:
        missing_packages.append('shapely==2.1.2')

    # 6. Check SAHI
    try:
        import sahi
    except ImportError:
        missing_packages.append('sahi==0.11.36')

    # 7. Check Ultralytics
    try:
        import ultralytics
    except ImportError:
        missing_packages.append('ultralytics==8.3.234')

    # 8. Check Torch
    try:
        import torch
    except ImportError:
        missing_packages.append('torch==2.9.1')

    # 9. Check Torchvision
    try:
        import torchvision
    except ImportError:
        missing_packages.append('torchvision==0.24.1')

    # 10. Check Pillow
    try:
        import PIL
    except ImportError:
        missing_packages.append('pillow==12.0.0')

    # 11. Check PyYAML
    try:
        import yaml
    except ImportError:
        missing_packages.append('pyyaml==6.0.3')


    if missing_packages:
        from qgis.PyQt.QtWidgets import QMessageBox
        missing_str = ', '.join(missing_packages)
        
        # Provide platform-specific installation instructions
        install_cmd = f"pip install {' '.join(missing_packages)}"
        
        message = f"""The following required packages are missing: {missing_str}

Installation Instructions:

Use the QGIS Python Console:
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install"] + {repr(missing_packages)})

The YOLO Toolkit will not work without these dependencies."""
        
        try:
            QMessageBox.warning(
                None, "Missing Dependencies - YOLO Toolkit", message
            )
        except Exception:
            print(f"[YOLO Toolkit] Missing dependencies: {missing_str}")
            
        return False
    
    return True

def classFactory(iface):  
    if not install_requirements():
        class DummyPlugin:
            def __init__(self, iface):
                self.iface = iface
            def initGui(self):
                pass
            def unload(self):
                pass
        return DummyPlugin(iface)
    from .plugin_main import PluginMain
    return PluginMain(iface)