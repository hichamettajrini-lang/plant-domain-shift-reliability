# Reliability-Aware Edge AI for Plant Disease Recognition Under Domain Shift

This repository contains the code, article-ready results, and figures associated with the study:

Reliability-Aware Edge AI for Plant Disease Recognition Under Real-World Domain Shift: Calibration, Human-in-the-Loop Gating, and False Autonomous Actuation Risk.

## Overview

This study evaluates the reliability of a lightweight MobileNetV3-Small model for plant disease recognition under external visual domain shift.

The model was trained and fine-tuned on PlantVillage using 27 mapped classes and evaluated on an external PlantDoc test set.

The study focuses on:
- source-domain versus external-domain performance degradation;
- confidence miscalibration under domain shift;
- false confident predictions;
- false autonomous actuation risk;
- temperature scaling as post-hoc calibration;
- human-in-the-loop gating for safer decision support.

## Main results

PlantVillage validation:
Accuracy = 0.9868
Macro-F1 = 0.9840
Mean confidence = 0.9768
ECE = 0.0100
Brier score = 0.0214
NLL = 0.0444

PlantDoc external test:
Accuracy = 0.2288
Macro-F1 = 0.1933
Mean confidence = 0.7457
ECE = 0.5168
Brier score = 1.2092
NLL = 5.6304

Temperature scaling with T = 7.6 reduced PlantDoc test ECE from 0.5168 to 0.0797 and reduced wrong predictions with confidence >= 0.90 from 118 to 0, without changing classification accuracy.

## Repository structure

notebooks/   Jupyter notebooks for dataset preparation, training, evaluation, calibration, and figure generation.
src/         Reusable Python scripts for metrics, calibration, gating, and plotting.
results/     Article-ready CSV result tables.
figures/     Article-ready figures.
docs/        Manuscript-related files.

## Datasets

This study uses PlantVillage as the source-domain dataset and PlantDoc as the external-domain dataset.

The raw datasets are not redistributed in this repository. Users should download them from their original sources and update local paths accordingly.

Processed metadata files, label mappings, evaluation outputs, and article-ready result tables are provided to support reproducibility.

## Model

The main model is MobileNetV3-Small trained on 27 mapped PlantVillage classes. The final evaluation uses the best validation checkpoint obtained during controlled fine-tuning.

## Reproducibility

Install dependencies:

pip install -r requirements.txt

## Citation

If you use this repository, please cite the associated manuscript and the archived Zenodo release when available.

## License

Code is released under the MIT License unless otherwise specified. Dataset licenses remain governed by the original dataset providers.