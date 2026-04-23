# Side_Channel_attack_Hybrid-QS

PyTorch framework for profiling side-channel attacks on AES using deep learning, featuring hybrid preprocessing with Signal-to-Noise Ratio (SNR) and Quadrant Scan (QS), and evaluation via Guessing Entropy (GE).

---

# Deep Learning Side-Channel Analysis Framework

## Overview

This repository provides an end-to-end PyTorch framework for profiling side-channel analysis (SCA) on AES using deep learning.

It supports multiple preprocessing strategies:
- Raw traces (baseline)
- SNR-based feature selection
- Hybrid preprocessing (SNR + Quadrant Scan + PCA)

The framework is designed for efficient experimentation, reproducibility, and evaluation using Guessing Entropy (GE).

---

## Features

- ConvTF (Convolutional–Transformer) model for SCA  
- Hybrid preprocessing (SNR + QS + PCA)  
- Automatic caching to avoid recomputation  
- Hybrid training (profiling + attack traces)  
- Guessing Entropy (GE) evaluation and plotting  
- Multiple experiment configurations  

---

## Important: Differences Between Main Scripts

### `main_pytorch.py`
- Uses `dataloader.py`
- Mixes **~80K attack traces** with profiling data for training  
- Properly applies the mixed dataset to training  
- Leaves a larger portion of attack traces for evaluation  
- Implements **true hybrid training**

---

## Preprocessing Selection 


To switch todifferent preprocessing methods, update the cache directory:

- `cache/` → Raw baseline preprocessed
- `cache_wide/` → SNR-based preprocessed
- `cache_hybrid/` → Hybrid preprocessed (SNR + QS)  

 Make sure the dataloader points to the correct directory before running experiments.

---

## Getting Started

### 1. Install Dependencies

## 2. Dataset

 The full dataset is **NOT included** due to its large size.

- Only sample data / small caches are provided  
- The full processed dataset is too large to upload  

 Download the CHES 2025 dataset from : https://pace-tl.gitbook.io/ches-challenge-2025
 and place it at:

`./Dataset/CHES_2025/CHES_Challenge.h5`

---

## 3. Running the Code

### Training

```bash
python main_pytorch.py
```

### Evaluation

```bash
python analyze_pytorch.py
```

---

## Hybrid Preprocessing Pipeline

If no cache is found, preprocessing runs automatically:

- Compute SNR and select Points of Interest (PoIs)  
- Apply Quadrant Scan (QS)  
- Crop windows around PoIs  
- Normalize traces (Z-score)  
- Apply PCA (≤1024 dimensions, 99% variance)  
- Save results into `cache_hybrid/`  

---

## Outputs

- Guessing Entropy (GE) curves  
- NTGE values  
- Saved `.npy` files  
- Plots (`.png`)  

---

## Results

Each experiment folder contains:

- `training_log.csv`  
- Model checkpoints (`.pth`)  
- GE curves (`.npy`, `.png`)  
- Final results (`result_*.npy`)  

---

## Notes

- Only sample data is included  
- Full preprocessing is computationally expensive  
- Cached data is reused automatically if available  
- Dataset paths may need manual adjustment  

---



## Disclaimer

This project is intended for research and educational purposes in evaluating side-channel vulnerabilities.

```bash
pip install -r requirements.txt
