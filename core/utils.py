import os, sys

def get_python_executable():
    qgis_dir = os.path.dirname(sys.executable)
    for c in [
        os.path.join(qgis_dir, "python.exe"),
        os.path.join(qgis_dir, "python3.exe"),
        os.path.join(os.path.dirname(qgis_dir), "apps", "Python39", "python.exe"),
        os.path.join(os.path.dirname(qgis_dir), "apps", "Python312", "python.exe"),
    ]:
        if os.path.exists(c):
            return c
    return sys.executable