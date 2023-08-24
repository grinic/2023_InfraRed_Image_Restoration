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

Where `image_ch0.tif`, `image_ch1.tif` are the visible and near infrared light images, respectively, and `image_channel=ch[CCC]_params.txt` contains at least the `XYZ` dimension of the tif dataset in the following format:
`
555:ROIWidth
555:ROIHeight
110:Planes
`

## 01_transfer_and_fuse_patch.py
This script extract patches of defined dimensions and creates a `Dataset` folder containing the images that will be used to train the IR2 network.

Parameters:
- create_Patches: boolean variable, default=True.
- inpaths: list of paths pointing to the raw data to process.
- patchSize: dimension of each patch. Default: (16,64,64).
- N_patches: number of patches to extract. Default: 100000.
- probmeths: probability distribution used to extract patch locations. Default: 'otsu', otherwise 'flat'.
- chOtsu: channel number to use to compute otsu threshold. Default: -1 (the near infrared).
- bias: if probmeths='otsu', defines the fraction of patches centered in bright regions of the sample. Default: 0.75.
- optimizeCoverage: whether to optimize sample coverage. Default: False.
- cThr: fraction of bright pixels to be covered by the patches. Default: 75%.
- nCoverage: number of times the bright pixels have to be covered by the patches. Default: 1.
- thresholdCorrelation: whether to filter out patches that show low correlation (e.g. misaligned or noise). Default: False.
- rThr: threshold of correlation. Default: 0.75.
- localRegister: whether to register the patches after extraction using correlation. Default: True.
- maskFilter: whether to filter out patches using a binary mask. Default: False.
- mask: binary mask. Default: None.

This script will generate a subfolder containing all patches extracted in tif and npz format, as well as the Otsu_mask generated:

```
__ sample_folder
 |
 |__DataSet
   |
   |__ image_channel=[CCC].npz
   |__ Otsu_mask_image_channel=[CCC].tif
   |__ tif_gt
   | |
   | |__ patch_XXX.tif
   |
   |__tif_input
     |
     |__ patch_XXX.tif

```

## 02_train_net_csbdeep.py
Script to train a standard UNet CARE network.

Parameters:
- pathsData: list of paths pointing to the datasets used for training.
- pathModel: path to CARE model.
- modelName
- N_max: set maximum number of patches. Default: None.
- train_batc_size: Default: 8.
- unet_n_depth: number of UNet layers. Default: 2.

## 03_predict_patches.py
Use the trained model to predict the patches for a list of datasets.
Can use multiple models to compare results.

Parameters:
- pathsData
- pathModels
- modelNames

## 04_predict_full.py
Use the trained model to predict the whole tif images for a list of datasets.
Can use multiple models to compare results.

Parameters:
- pathsData
- pathModels
- modelNames

## 05_quantify_patches.py
For each dataset, generates a csv file containing the following information for each patch reconstructed:
- Information content
- structural similarity index (relative to ground truth)
- Mean square error (relative to ground truth)

## 06_plot_quantification_csbdeep.py

## 07_restore_TL.py
