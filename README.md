# D³: Drunk Driving Detection (CSE 834)

This repository contains code for drunk driver detection using front facing camera feed.

This project is a modified version of the Impaired Driving Dataset (IDD)
codebase from the Toyota Research Institute (TRI), adapted for the **CSE
834 course project**.

------------------------------------------------------------------------

## 🚗 Base Dataset: Impaired Driving Dataset (IDD)

The original dataset and project page can be found here:\
https://toyotaresearchinstitute.github.io/IDD/

------------------------------------------------------------------------

## ⚙️ Installation

Clone the repository and set up the environment:

``` bash
git clone <your-repo-url>
cd <your-repo-name>

git submodule update --init
micromamba create -f environment.dbm.yml
micromamba activate dbm
```

------------------------------------------------------------------------

## 📦 Downloading the Dataset

To download a **sample version** of the dataset:

``` bash
./data/download_processed_dataset.py \
    --output_dir dataset/Vehicle/No-Video \
    --version sample
```

For full experiments:

-   `no-video` (\~16.6 GB) → minimum required to reproduce TRI results\
-   Other versions → see `data/configs/public-dataset.yaml`

------------------------------------------------------------------------

## 📁 Repository Structure

-   **analysis** -- Feature analysis and statistical tests (used in IDD
    paper)\
-   **data** -- Dataset download scripts and configs\
-   **exp** -- Experiment configuration files\
-   **features** -- Feature extraction pipelines\
-   **hail-datasets** -- Data loader implementations\
-   **models** -- Model architectures\
-   **submodules** -- External dependencies\
-   **utils** -- Common utilities\
-   **viz** -- Visualization tools

------------------------------------------------------------------------

## ▶️ Training & Experiments

To train a model:

``` bash
python train.py
```

To run predefined experiments:

``` bash
./run_experiments.py -e exp/example.txt --experiments_per_gpu 2 --num_gpu 2
```

Outputs will be saved in:

    checkpoints/

------------------------------------------------------------------------

## 🔬 Reproducing TRI Results

To reproduce results from the original IDD pipeline:

``` bash
./analysis/feature_test.py

./run_experiments.py \
    -e exp/example.txt \
    --experiments_per_gpu 1 \
    --num_gpu 1

./analysis/itsc25.py checkpoints
```

------------------------------------------------------------------------

## 🧪 Reproducing Our Results (D³)

To reproduce the results reported in this project:

``` bash
bash D3_full.sh
```

------------------------------------------------------------------------

## 📌 Notes

-   This repository is a **modified version** of the original TRI
    codebase.
-   Changes were introduced for the **CSE 834 course project**.
-   The focus is on **drunk driving detection using only front facing video feed

------------------------------------------------------------------------

## 📜 Acknowledgements

This project builds upon the work and dataset provided by the Toyota
Research Institute (TRI) and the Impaired Driving Dataset (IDD) authors.
