# Side_Channel_attack_Hybrid-QS
PyTorch framework for profiling side-channel attacks on AES using deep learning, featuring hybrid preprocessing with SNR and Quadrant Scan and evaluation via Guessing Entropy.
# Deep Learning Side-Channel Analysis Framework 

## Overview

This repository provides an end-to-end PyTorch framework for profiling side-channel analysis (SCA) on AES using deep learning. It supports multiple preprocessing strategies, including raw traces, SNR-based selection, and a hybrid approach combining Signal-to-Noise Ratio (SNR) and Quadrant Scan (QS).

The framework is designed for large-scale datasets and focuses on efficient training, reproducibility, and evaluation using Guessing Entropy (GE).

---

## Features

- ConvTF (Convolutional–Transformer) model for SCA
- Hybrid preprocessing (SNR + Quadrant Scan + PCA)
- Automatic caching to avoid repeated preprocessing
- Training with mixed profiling and attack traces
- Guessing Entropy (GE) evaluation and plotting
- Multiple experiment configurations (v0 / v2 / v3)

---

## Repository Structure


├── src/
│ ├── dataloader.py # Dataset loader (handles caching + preprocessing)
│ ├── preprocessing.py # Hybrid preprocessing pipeline (SNR + QS + PCA)
│ ├── net.py # ConvTF model
│ ├── trainer.py # Training loop
│ └── utils.py # Evaluation (GE, AES S-box, etc.)
│
├── cache/ # Sample cache (raw / baseline)
├── cache_wide/ # SNR-based preprocessing cache
├── cache_hybrid/ # Hybrid preprocessing cache (proposed method)
│
├── Result22/ # Example results (baseline)
├── Result33/ # Example results (ConvTF + preprocessing)
│
├── main_pytorch.py # Main training script (baseline)
├── main_pytorch_v2.py # SNR-based version
├── main_pytorch_v3.py # Hybrid preprocessing (recommended)
│
├── analyze_pytorch.py # Final evaluation script
├── HQS_results.ipynb # Visualization notebook
├── Preprocessing_newWay.ipynb # Preprocessing exploration
├── README.md
└── requirements.txt


---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
2. Dataset

The dataset is not included due to its size.

Download the CHES 2025 dataset from the official provider and place it in one of the following locations:

./CHES_Challenge.h5

or

./Dataset/CHES_2025/CHES_Challenge.h5
3. Running the Code
Recommended (Hybrid Method)
python main_pytorch_v3.py
Other Versions

Raw baseline:

python main_pytorch.py

SNR-based preprocessing:

python main_pytorch_v2.py
Preprocessing Pipeline (Hybrid)

If no cached data is found, preprocessing runs automatically:

Compute SNR and select candidate PoIs
Apply Quadrant Scan for structural filtering
Crop windows around PoIs
Normalize traces (Z-score)
Apply PCA (≤1024 dimensions, 99% variance)
Save to cache_hybrid/
Evaluation

To compute final results:

python analyze_pytorch.py
Outputs
Guessing Entropy (GE) curves
NTGE values
Saved .npy result files
Plots (.png)
Results

Each experiment folder contains:

training_log.csv
Model checkpoints (.pth)
GE curves (.npy, .png)
Final results (result_*.npy)
Notes
Only sample data and caches are provided in this repository
Full preprocessing is computationally expensive and dataset-dependent
If cached files exist, preprocessing is skipped automatically
Paths may need adjustment depending on dataset location
Reproducibility
Fixed random seeds
Cached preprocessing
Configurations saved per model
Disclaimer

This project is intended for research and educational purposes in evaluating side-channel vulnerabilities and improving secure system design.


---

If you want, I can also:
- add **badges (PyTorch, GPU, license)** 🚀  
- make it **CHES-paper style README (more academic tone)** 📄  
- or simplify it for recruiters vs researchers 👍
give me everything in mark up lang in one block that i can copy paste

Here you go — everything in one clean Markdown block, ready to copy-paste directly into your README.md 👇

# Deep Learning Side-Channel Analysis Framework (CHES 2025)

## Overview

This repository provides an end-to-end PyTorch framework for profiling side-channel analysis (SCA) on AES using deep learning. It supports multiple preprocessing strategies, including raw traces, SNR-based selection, and a hybrid approach combining Signal-to-Noise Ratio (SNR) and Quadrant Scan (QS).

The framework is designed for large-scale datasets and focuses on efficient training, reproducibility, and evaluation using Guessing Entropy (GE).

---

## Features

- ConvTF (Convolutional–Transformer) model for SCA  
- Hybrid preprocessing (SNR + Quadrant Scan + PCA)  
- Automatic caching to avoid repeated preprocessing  
- Training with mixed profiling and attack traces  
- Guessing Entropy (GE) evaluation and plotting  
- Multiple experiment configurations (v0 / v2 / v3)  

---

## Repository Structure


├── src/
│ ├── dataloader.py # Dataset loader (handles caching + preprocessing)
│ ├── preprocessing.py # Hybrid preprocessing pipeline (SNR + QS + PCA)
│ ├── net.py # ConvTF model
│ ├── trainer.py # Training loop
│ └── utils.py # Evaluation (GE, AES S-box, etc.)
│
├── cache/ # Sample cache (raw / baseline)
├── cache_wide/ # SNR-based preprocessing cache
├── cache_hybrid/ # Hybrid preprocessing cache (proposed method)
│
├── Result22/ # Example results (baseline)
├── Result33/ # Example results (ConvTF + preprocessing)
│
├── main_pytorch.py # Main training script (baseline)
├── main_pytorch_v2.py # SNR-based version
├── main_pytorch_v3.py # Hybrid preprocessing (recommended)
│
├── analyze_pytorch.py # Final evaluation script
├── HQS_results.ipynb # Visualization notebook
├── Preprocessing_newWay.ipynb # Preprocessing exploration
├── README.md
└── requirements.txt


---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
2. Dataset

The dataset is not included due to its size.

Download the CHES 2025 dataset from the official provider and place it in one of the following locations:

./CHES_Challenge.h5

or

./Dataset/CHES_2025/CHES_Challenge.h5
3. Running the Code
Recommended (Hybrid Method)
python main_pytorch_v3.py
Other Versions

Raw baseline:

python main_pytorch.py

SNR-based preprocessing:

python main_pytorch_v2.py
Preprocessing Pipeline (Hybrid)

If no cached data is found, preprocessing runs automatically:

Compute SNR and select candidate PoIs
Apply Quadrant Scan for structural filtering
Crop windows around PoIs
Normalize traces (Z-score)
Apply PCA (≤1024 dimensions, 99% variance)
Save to cache_hybrid/
Evaluation

To compute final results:

python analyze_pytorch.py
Outputs
Guessing Entropy (GE) curves
NTGE values
Saved .npy result files
Plots (.png)
Results

Each experiment folder contains:

training_log.csv
Model checkpoints (.pth)
GE curves (.npy, .png)
Final results (result_*.npy)
## Notes ##
Only sample data and caches are provided in this repository
Full preprocessing is computationally expensive and dataset-dependent
If cached files exist, preprocessing is skipped automatically
Paths may need adjustment depending on dataset location
Reproducibility
Fixed random seeds
Cached preprocessing
Configurations saved per model
Disclaimer

This project is intended for research and educational purposes in evaluating side-channel vulnerabilities and improving secure system design.
