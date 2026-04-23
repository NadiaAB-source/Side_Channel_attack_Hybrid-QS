# Side_Channel_attack_Hybrid-QS

PyTorch framework for profiling side-channel attacks on AES using deep learning, featuring hybrid preprocessing with SNR and Quadrant Scan (QS), and evaluation via Guessing Entropy (GE).

---

# Deep Learning Side-Channel Analysis Framework 

## Overview

This repository provides an end-to-end PyTorch framework for profiling side-channel analysis (SCA) on AES using deep learning.

It supports multiple preprocessing strategies:
- Raw traces (baseline)
- SNR-based feature selection
- Hybrid preprocessing (SNR + Quadrant Scan + PCA)


---

## Features

- ConvTF (Convolutional–Transformer) model for SCA  
- Hybrid preprocessing (SNR + Quadrant Scan + PCA)  
- Automatic caching (avoids recomputation)  
- Optional hybrid training (profiling + attack traces)  
- Guessing Entropy (GE) evaluation and plotting  
- Multiple experiment configurations  

---

## Repository Structure
## Repository Structure


├── src/
│ ├── dataloader.py # Main dataloader (hybrid / updated)
│ ├── dataloader_v0.py # Baseline dataloader
│ ├── preprocessing.py # Hybrid preprocessing (SNR + QS + PCA)
│ ├── net.py # ConvTF model
│ ├── trainer.py # Training loop
│ └── utils.py # Evaluation (GE, AES S-box, etc.)
│
├── cache/ # Sample cache (baseline)
├── cache_wide/ # SNR preprocessing cache
├── cache_hybrid/ # Hybrid preprocessing cache
│
├── src/ # Core source code
│
├── main_pytorch.py # Main script (hybrid / updated pipeline)
├── main_pytorch0.py # Baseline version (older pipeline)
│
├── analyze_pytorch.py # Evaluation script
│
├── HQS_results.ipynb # Visualization notebook
├── Preprocessing_newWay.ipynb # Preprocessing experiments
│
├── README.md
└── requirements.txt

---

## Important: Differences Between Main Scripts

- **`main_pytorch.py`**
  - Uses updated `dataloader.py`
  - Supports hybrid preprocessing and improved pipeline
  - Designed for better performance and flexibility

- **`main_pytorch0.py`**
  - Uses `dataloader_v0.py`
  - Simpler / baseline version
  - Useful for comparison experiments

---

## Preprocessing Selection (Very Important)

The preprocessing type is controlled by the **data path inside the dataloader**.

 To switch between preprocessing methods, change the dataset/cache directory path:

| Preprocessing Type | Folder |
|-------------------|--------|
| Raw (baseline)    | `cache/` |
| SNR-based         | `cache_wide/` |
| Hybrid (SNR + QS) | `cache_hybrid/` |

 Make sure the dataloader points to the correct directory depending on your experiment.

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt


## 2. Dataset

 The full dataset is **NOT included** due to its large size.

- Only sample data / small caches are provided in this repository  
- The full processed dataset is too large to upload  

 You must download the CHES 2025 dataset separately and place it in:
./Dataset/CHES_2025/CHES_Challenge.h5


---

## 3. Running the Code

### Recommended (Hybrid Model)

```bash
python main_pytorch.py









