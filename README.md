# 2023_InfraRed_Image_Restoration
This repository contains the code used for the manuscript "Image restoration of degraded time-lapse microscopy data mediated by infrared-imaging."

Repository structure:
- Figures: contains the code used to generate the figure of the paper.
- IR2_training_prediction: contains the code used to apply IR2 to microscopy images.
- Sample: example data.

# Installation
Simply download the whole repository.

## Requirements
The code has been tested with Python 3.9 and requires the following packages:
- numpy
- scikit-image
- tifffile
- matplotlib
- scipy
- pandas
- seaborn
- csbdeep

To install CSBDEEP with GPU, please follow the [CSBDEEP documentation.](http://csbdeep.bioimagecomputing.com/doc/install.html)

# Using the repository
To run IR2 on your own data, you can use the scripts in the folder `IR2_training_prediction` in sequential order.
Here we provide a step-by-step guide:

### Data Management
Image data should be tif files accompanied by a txt metadata file.
Data structure examples are provided in the `Sample` subfolders. Images should be arranged as follows:
```
__ sample_folder
 |
 |__ image_channel=ch00.tif
 |__ image_channel=ch01.tif
 |__ image_channel=ch[CCC]_params.txt
```

Where `image_channel=ch[CCC]_params.txt` should contain at least the XYZ dimension of the tif dataset in the following format:
`
2048:ROIWidth
2048:ROIHeight
110:Planes
`
