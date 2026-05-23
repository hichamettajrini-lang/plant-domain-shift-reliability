# Reliability-Aware Edge AI for Plant Disease Recognition Under Domain Shift

Code, article-ready results, and figures associated with the study:

**Reliability-Aware Edge AI for Plant Disease Recognition Under Domain Shift: Calibration, Human-in-the-Loop Gating, and False Autonomous Actuation Risk**

## Overview

This repository supports a reliability-aware evaluation of a lightweight Edge-AI-compatible plant disease recognition model under external visual domain shift.

A MobileNetV3-Small model was trained and fine-tuned on PlantVillage using 27 mapped classes and then evaluated on an external PlantDoc test set. The study does not focus only on classification accuracy. Instead, it investigates whether a model that performs well under controlled source-domain conditions remains reliable when exposed to more heterogeneous external images.

The analysis focuses on:

- source-domain versus external-domain performance degradation;
- confidence miscalibration under domain shift;
- high-confidence incorrect predictions;
- false autonomous actuation risk under confidence-based automation;
- temperature scaling as a lightweight post-hoc calibration method;
- human-in-the-loop gating for safer decision support;
- stratified bootstrap confidence intervals for uncertainty analysis.

## Main experimental setting

| Component | Description |
|---|---|
| Main model | MobileNetV3-Small phase 2 |
| Source-domain training/validation dataset | PlantVillage |
| External evaluation dataset | PlantDoc |
| Number of mapped classes | 27 |
| PlantVillage validation images | 14,747 |
| PlantDoc external test images | 472 |
| Calibration method | Temperature scaling |
| Selected temperature | T = 7.6 |
| Safety analysis | Confidence-threshold false actuation risk and human-in-the-loop gating |
| Uncertainty analysis | 1,000 stratified paired bootstrap resamples |

## Main source-versus-external reliability results

| Evaluation domain | Accuracy | Macro-F1 | Mean confidence | ECE | Brier score | NLL |
|---|---:|---:|---:|---:|---:|---:|
| PlantVillage validation | 0.9868 | 0.9840 | 0.9768 | 0.0100 | 0.0214 | 0.0444 |
| PlantDoc external test | 0.2288 | 0.1933 | 0.7457 | 0.5168 | 1.2092 | 5.6304 |

The model achieved very high performance and near-good calibration on the PlantVillage validation set. However, under external PlantDoc evaluation, accuracy and macro-F1 decreased sharply while confidence remained high. This behaviour indicates severe reliability degradation under domain shift and motivates the analysis of unsafe high-confidence decisions.

## False confident prediction risk

On the PlantDoc external test set, the uncalibrated model retained a large number of incorrect predictions at high confidence:

| Confidence threshold | Retained predictions | Accuracy among retained predictions | False confident predictions |
|---:|---:|---:|---:|
| >= 0.70 | 262 | 0.3053 | 182 |
| >= 0.80 | 228 | 0.2895 | 162 |
| >= 0.90 | 168 | 0.2976 | 118 |
| >= 0.95 | 128 | 0.3594 | 82 |

These values are interpreted as a proxy for false autonomous actuation risk under a hypothetical confidence-triggered decision policy.

## Temperature scaling mitigation

Temperature scaling was fitted on the PlantDoc training split by minimizing negative log-likelihood. The selected scalar temperature was:

```text
T = 7.6