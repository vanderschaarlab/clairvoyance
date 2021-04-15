# Clairvoyance: a Unified, End-to-End AutoML Pipeline for Medical Time Series

![Clairvoyance Logo](tutorial/figure/clairvoyance_logo.png)

Authors: van der Schaar Lab (www.vanderschaar-lab.com)

This repository contains implementations of Clairvoyance: a unified, end-to-end AutoML 
pipeline for medical time series for the following applications.

- Time-series prediction (one-shot and online)
- Transfer learning
- Individualized time-series treatment effects (ITE) estimation
- Active sensing on time-series data

All API files for those applications can be found in [`/api`](/api) folder. 
All tutorials for those applications can be found in [`/tutorial`](/tutorial) folder.

![Block diagram of Clairvoyance](tutorial/figure/clairvoyance_block.png)

## Installation
There are currently two ways of installing the required dependencies: using Docker or using Conda.

### Note on Requirements
* Clairvoyance has been tested on Ubuntu 20.04, but should be broadly compatible with common Linux systems. 
* The Docker installation method is additionally compatible with Mac and Windows systems that support Docker.
* Hardware requirements depends on the underlying ML models used, but a machine that can handle ML research tasks is recommended.
* For faster computation, CUDA-capable Nvidia card is recommended (follow the CUDA-enabled installation steps below).

### Docker installation
* If you are not familiar with Docker, have a look at the resources:
    * [Introduction to Docker YouTube playlist](https://www.youtube.com/playlist?list=PLhW3qG5bs-L99pQsZ74f-LC-tOEsBp2rK)
    * [Official *Getting Started* guide](https://docs.docker.com/get-started/)
    * [A useful example of the installation process](https://www.celantur.com/blog/run-cuda-in-docker-on-linux/)

1. Install Docker on your system: https://docs.docker.com/get-docker/.
1. **\[Required for CUDA-enabled installation only\]** Install *Nvidia container runtime*: https://github.com/NVIDIA/nvidia-container-runtime/.
    * Assumes Nvidia drivers are correctly installed on your system.
1. Get the latest Clairvoyance Docker image:
    ```bash
    $ docker pull clairvoyancedocker/clv:latest
    ```
1. To run the Docker container as a terminal, execute the below from the Clairvoyance repository root:
    ```bash
    $ docker run -i -t --gpus all --network host -v $(pwd)/datasets/data:/home/clvusr/clairvoyance/datasets/data clairvoyancedocker/clv
    ```
    * Explanation of the `docker run` arguments:
        * `-i -t`: Run a terminal session.
        * `--gpus all`: **\[Required for CUDA-enabled installation only\]**, passes your GPU(s) to the Docker container, otherwise skip this option.
        * `--network host`: Use your machine's network and forward ports. Could alternatively publish ports, e.g. `-p 8888:8888`.
        * `-v $(pwd)/datasets/data:/home/clvusr/clairvoyance/datasets/data`: Share directory/ies with the Docker container as volumes, e.g. data.
        * `clairvoyancedocker/clv`: Specifies Clairvoyance Docker image.
    * If using Windows: 
        * Use PowerShell and first run the command `$pwdwin = $(pwd).Path`. Then use `$pwdwin` instead of `$(pwd)` in the `docker run` command.
        * Due to how Docker networking works on Windows, replace `--network host` with `-p 8888:8888`.
1. Run all following Clairvoyance API commands, jupyter notebooks etc. from within this Docker container.

### Conda installation
Conda installation has been tested on Ubuntu 20.04 only.
1. From the Clairvoyance repo root, execute:
    ```bash
    $ conda env create --name clvenv -f ./environment.yml
    $ conda activate clvenv
    ```
2. Run all following Clairvoyance API commands, jupyter notebooks etc. in the `clvenv` environment.

## Data
Clairvoyance expects your dataset files to be defined as follows:
* Four CSV files (may be compressed), as illustrated below:
    ```
    static_test_data.csv
    static_train_data.csv
    temporal_test_data.csv
    temporal_train_data.csv
    ```
* Static data file content format:
    ```
    id,my_feature,my_other_feature,my_third_feature_etc
    3wOSm2,11.00,4,-1.0
    82HJss,3.40,2,2.1
    iX3fiP,7.01,3,-0.4
    ...
    ```
* Temporal data file content format:
    ```
    id,time,variable,value
    3wOSm2,0.0,my_first_temporal_feature,0.45
    3wOSm2,0.5,my_first_temporal_feature,0.47
    3wOSm2,1.2,my_first_temporal_feature,0.49
    3wOSm2,0.0,my_second_temporal_feature,10.0
    3wOSm2,0.1,my_second_temporal_feature,12.4
    3wOSm2,0.3,my_second_temporal_feature,9.3
    82HJss,0.0,my_first_temporal_feature,0.22
    82HJss,1.0,my_first_temporal_feature,0.44
    ...
    ```
* The `id` column is required in the static data files. The `id,time,variable,value` columns are required in the temporal file. The IDs of samples must match between the static and temporal files.
* Your data files are expected to be under:
    ```
    <clairvoyance_repo_root>/datasets/data/<your_dataset_name>/
    ```
* See tutorials for how to define your dataset(s) in code.
* Clairvoyance examples make reference to some existing datasets, e.g. `mimic`, `ward`. These are confidential datasets (or in case of [MIMIC-III](https://mimic.physionet.org/about/mimic/), it requires a training course and an access request) and are not provided here. Contact [nm736@cam.ac.uk](mailto:nm736@cam.ac.uk) for more details.

### Extract data from MIMIC-III
To use MIMIC-III with Clairvoyance, you need to get access to MIMIC-III and follow the instructions for installing it in a Postgres database: https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/
```bash
$ cd datasets/mimic_data_extraction && python extract_antibiotics_dataset.py
```

## Usage
* To run tutorials:
    * Launch jupyter lab: `$ jupyter-lab`.
        * If using Windows and following the Docker installation method, run `jupyter-lab --ip="0.0.0.0"`.
    * Open jupyter lab in the browser by following the URL with the token.
    * Navigate to `tutorial/` and run a tutorial of your choice.
* To run Clairvoyance API from the command line, execute the appropriate command from within the Docker terminal (see example command below).

## Example: Time-series prediction 
To run the pipeline for training and evaluation on time-series 
prediction framework, simply run `$ python -m api/main_api_prediction.py` or take a look 
at the jupyter notebook `tutorial/tutorial_prediction.ipynb`.

Note that any model architecture can be used as the predictor model such as
RNN, Temporal convolutions, and transformer. The condition for
predictor model is to have fit and predict functions as its subfunctions.

* Stages of the time-series prediction:
    - Import dataset
    - Preprocess data
    - Define the problem (feature, label, etc.)
    - Impute missing components
    - Select the relevant features
    - Train time-series predictive model
    - Estimate the uncertainty of the predictions
    - Interpret the predictions
    - Evaluate the time-series prediction performance on the testing set
    - Visualize the outputs (performance, predictions, uncertainties, and interpretations)

* Command inputs:
    - `data_name`: `mimic`, `ward`, `cf`    
    - `normalization`: `minmax`, `standard`, `None`
    - `one_hot_encoding`: input features that need to be one-hot encoded
    - `problem`: `one-shot` or `online`
    - `max_seq_len`: maximum sequence length after padding
    - `label_name`: the column name for the label(s)
    - `treatment`: the column name for treatments
    - `static_imputation_model`: `mean`, `median`, `mice`, `missforest`, `knn`, `gain`
    - `temporal_imputation_model`: `mean`, `median`, `linear`, `quadratic`, `cubic`, `spline`, `mrnn`, `tgain`   
    - `feature_selection_model`: `greedy-addition`, `greedy-deletion`, `recursive-addition`, `recursive-deletion`, `None`
    - `feature_number`: selected feature number
    - `model_name`: `rnn`, `gru`, `lstm`, `attention`, `tcn`, `transformer`
    - `h_dim`: hidden dimensions
    - `n_layer`: layer number
    - `n_head`: head number (only for transformer model)
    - `batch_size`: number of samples in mini-batch
    - `epochs`: number of epochs
    - `learning_rate`: learning rate
    - `static_mode`: how to utilize static features (`concatenate` or `None`)
    - `time_mode`: how to utilize time information (`concatenate` or `None`)
    - `task`: `classification` or `regression`
    - `uncertainty_model_name`: uncertainty estimation model name (`ensemble`)
    - `interpretation_model_name`: interpretation model name (`tinvase`)
    - `metric_name`: `auc`, `apr`, `mae`, `mse`

* Example command:
    ```bash
    $ cd api
    $ python main_api_prediction.py \
        --data_name cf --normalization minmax --one_hot_encoding admission_type \
        --problem one-shot --max_seq_len 24 --label_name death \
        --static_imputation_model median --temporal_imputation_model median \
        --model_name lstm --h_dim 100 --n_layer 2 --n_head 2 --batch_size 400 \
        --epochs 20 --learning_rate 0.001 \
        --static_mode concatenate --time_mode concatenate \
        --task classification --uncertainty_model_name ensemble \
        --interpretation_model_name tinvase --metric_name auc
    ```

* Outputs:
    - Model prediction
    - Model performance
    - Prediction uncertainty
    - Prediction interpretation
