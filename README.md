
# Uncertainty Estimation on Histopathological Slides with Pytorch Lightning

## Setup data
1. Download Camlyon17 slides from <https://camelyon17.grand-challenge.org/Data/>

2. From `CAMELYON17/training` extract the 50 slides with lesion-level annotations contained in `lesion_annotations.zip`

2. Generate tiles by using the preprocessing pipeline from <https://github.com/DBO-DKFZ/wsi_preprocessing> `v0.1` \
Use the config provided under `configs/preprocessing/config.json` and set the correct paths for slides, annotations and output


## Setup system
1. Set the system's path variables for "DATASET_LOCATION" and "EXPERIMENT_LOCATION". One way is to insert the following lines into your `.bashrc` file:
```
export DATASET_LOCATION=YOUR_PATH 
export EXPERIMENT_LOCATION=YOUR_PATH
```

## Setup repo
1. Install Miniconda

2. Create conda environment\
`conda env create -f environment.yml`

3. Run train script\
Example: `python train.py --config configs/final/strong/resnet.yaml` \
The train script by default runs evaluation on the best performing model checkpoint. Predicitions on test splits are stored in 
`{env:EXPERIMENT_LOCATION}/RUN_NAME/VERSION/predictions/`
