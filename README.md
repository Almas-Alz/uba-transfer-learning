# Transfer Learning Using the Global Caravan Dataset for Developing a Local River Streamflow Prediction Model

This repository contains the code and configurations for the paper *"Transfer Learning Using the Global Caravan Dataset for Developing a Local River Streamflow Prediction Model"* by Alzhanov et al.

The code uses the [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) library. This repository provides the specific configuration files for the experiments presented in the study.

## Repository Structure

```
.
├── Uba_local_data/     # Uba local data for training and testing
├── basin_lists/        # Text files with basin IDs for different setups (NC, NCL)
├── configs/            # YML configuration files for each experimental setup
│   ├── lstm-l/
│   ├── lstm-nc/
│   └── lstm-ncl/
├── src/                # Source code
│   ├── run_experiment.py   # Script to run pre-training and fine-tuning
│   └── metrics.py          # Script with hydrological metrics calculations
└── environment.yaml     # Conda environment file
```

## Requirements & Installation

All required packages are listed in the `environment.yml` file. You can create the Conda environment with the following command:

```
conda env create -f environment.yaml
```

## Data Setup

The model training requires two main data sources: the global **Caravan** dataset and the local **Uba River** data. The local data must be placed inside the Caravan dataset directory structure.

1.  **Download the Data:**
    * **Caravan Dataset:** Download from the official Zenodo repository: [https://zenodo.org/records/14673536](https://zenodo.org/records/14673536)
    * **Uba River Data:** The processed local data, used for this study is located inside of this repository.

2.  **Integrate Local Data:**
    * Unzip the Caravan dataset.
    * **Time Series Data:**
        * Create a new folder named `ubakz` inside `.../caravan/time_series/netcdf/`.
        * Place the `ubakz_99999999.nc` file inside this new folder.
    * **Attribute Data:**
        * Create another new folder named `ubakz` inside `.../caravan-v1.0/attributes/`.
        * Place the following three files inside this new attributes folder: `attributes_caravan_ubakz.csv`, `attributes_hydroatlas_ubakz.csv`, and `attributes_other_ubakz.csv`.

3.  **Update Configuration Files:**
    Before running any experiments, you must update the `data_dir` variable in all `.yml` files inside the `/configs` folder to point to the root of the Caravan dataset and also to verify the correct path to the basins list files.


## Running the Experiments

The experiments are run using the script in the `src/` folder.
You must edit the paths at the top of the file to point to the specific configuration files you want to run.
