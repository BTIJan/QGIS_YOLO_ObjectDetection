## Introduction
This QGIS plugin trains and runs Ultralytics YOLO11-OBB object-detection models on aerial and satellite imagery. It provides a guided workflow for tiling large rasters, preparing a YOLO dataset, training a model, running tiled inference (SAHI) on new imagery, and performing external validation against an independent ground-truth dataset.

### Installation 
Install the plugin in QGIS after downloading it as a .zip file. After installation, you might be prompted to install required packages in the python environment of QGIS. The easiest way is to install this using pip in OSGEO shell. 
#### Torch
The plugin is only tested for Cuda version 12.4 to 12.6, make sure your cuda version matches with the torch wheel. Check this using the following commands in OSGEO Shell:
`python -c "import sys, torch; print(sys.version); print(torch.__version__); print('cuda build', torch.version.cuda); print('is_available', torch.cuda.is_available())"`
Install the torch wheel using where 'cu126' stand for Cuda version 12.6: 
`python -m pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
After installing all the other packages the plugin should be visible in QGIS. 

**The plugin is an early version. It has been developed on Windows 11, using QGIS 3.34 and a NVIDIA 4000-series GPU. Please report any issues in this repo**
## Usage manual
### Tiling module
The Tiling module generates YOLO-ready image tiles from large aerial/satellite rasters by cutting only around labeled training objects (plus a controlled amount of empty “background” tiles) to maximize useful training data.
Key settings are the input training polygons and rasters (both in the same CRS), tile size, tile overlap, background ratio (typically 0.1–0.3), and the output folder.

### Dataset preparation
The Preparation module turns the generated tiles and the training shapefile into a ready-to-train YOLO dataset. 
It boosts the contrast of the GeoTIFF tiles and converts them to .jpg images, converts the labeled objects in the shapefile to YOLO text label files, automatically splits everything into training and validation sets, 
and generates the data.yaml configuration file.
In the module, you select the folder containing the tiles, the training shapefile (same as in the tiling module), and an output root folder where the dataset structure (train/val images, labels, and data.yaml) will be created.
You also set the validation fraction

### Training
The Training module uses the YOLO algorithm to train an object-detection model using the YOLO dataset created in the Preparation module. It reads the data.yaml file, loads a selected YOLO model, and starts the training process.
When training finishes, it creates an output folder containing the training results and the resulting model file (.pt). The module settings are grouped into three categories: basic settings, training settings, and augmentation settings. 
See Ultralytics documentation for an elaborate description of each setting

### Detection
The Detection module applies the trained object-detection model to aerial or satellite imagery using the Slicing Aided Hyper Inference (SAHI) approach.
SAHI moves a sliding window systematically across the image and runs detection on each slice. This often improves detection of small objects compared to running object detection on the entire image in a single pass.
Min object size and max object size define the size range of the detections. 

### External validation
During training, validation metrics are computed automatically, but they are based on an internal validation split from the same dataset as the training data, which can make results look more optimistic than truly independent external validation.

This module evaluates detections from the Detection module against an external Ground Truth Dataset (GTD). A detection is counted as a true positive if it overlaps a manually drawn GTD object above an IoU threshold; 
detections without such overlap are false positives, and GTD objects that are not detected are false negatives. To compute reliable performance metrics, 
it’s essential that all objects in the validation area are labeled in the GTD—missing objects lead to incorrect counts and unreliable metrics. The module outputs an F1–confidence plot showing the F1 score at different confidence thresholds.
