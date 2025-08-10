# GCoDE-Predictor


### Overview

This repository provides the implementation of the **system performance predictor** used in our work:  
**"GCoDE: Efficient Device–Edge Co-Inference for GNNs via Architecture–Mapping Co-Search"** (submitted to *IEEE Transactions on Computers*).


The predictor estimates **co-inference latency** and **on-device energy consumption** for GNN device-edge co-inference . It combines **LUT-based** approach and **GNN-based prediction method** to enable accurate and lightweight system performance awareness.

---

### Repository Structure
```
├── dataset/        # Collected datasets for predictor training/testing
├── LUT/            # Lookup tables of measured GNN operation latency and power
├── Model/          # Predictor model architectures
├── process/        # Data preprocessing and PyG graph construction scripts
├── train/          # Training and testing scripts for the predictor
├── util.py         # Utility functions used by the predictor
```
---

### Requirements
- Python ≥ 3.8  
- PyTorch (recent version)  
- PyG ≥ 2.1  
- numpy  

---

### Usage

#### 1. Prepare Data
Before training the predictor, collect and process:
- **Sample dataset** in `dataset/`
- **LUT data** in `LUT/`

#### 2. Train & Test Predictor
Navigate to the `train/` directory and run:
```
 python training.py --exp_name exp1 --dataset_name latency --batch_size 32 --test_batch_size 16 \
 --epochs 200 --seed 1 --loss mape --device 0 --lr 0.0008 --agg mean --pool sum --layers 3 --norm True
```
- `--exp_name` : Experiment name for logging and results.
- `--dataset_name` : Name of the dataset directory in `dataset/`.
- `--batch_size` / `--test_batch_size` : Training/testing batch sizes.
- `--epochs` : Number of training epochs.
- `--seed` : Random seed for reproducibility.
- `--loss` : Loss function (`mape`, etc.).
- `--device` : Device type, 0:GPU, -1:CPU.
- `--lr` : Learning rate.
- `--agg` : Aggregation function (`mean`, `sum`, etc.).
- `--pool` : Pooling method for graph representation.
- `--layers` : Number of GNN layers in the predictor.
- `--norm` : Whether to apply feature normalization (`True`/`False`).




