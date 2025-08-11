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
- PyTorch ≥ 1.11 
- PyG ≥ 2.1  
- numpy  

---

### Usage

#### 1. Dataset Preparation Guidelines
The dataset preparation process consists of two major stages: **LUT Construction** and **Training Dataset Construction**.

---

##### A. LUT Construction

The Lookup Table (LUT) stores operation-level latency or power measurements for GNN operators on specific devices.  
It is built in two steps:

1. **Operation Data Measurement**
   - Select all GNN operations used in the target search/design space (e.g., KNN, Combine, pooling, aggregate).
   - For each operation and each `(in_dim, out_dim)` pair:
     - Perform **multiple independent measurements** under identical hardware/software settings.
     - Measure both **latency** (ms) and/or **power consumption** (W), depending on the LUT type.
   - Store results in TXT files with the following format:
        ```
        operation_name,in_dim,out_dim,value
        ```
        where:
        - `operation_name`: Name of the GNN operation.
        - `in_dim`: Input feature dimension.
        - `out_dim`: Output feature dimension.
        - `value`: Measured latency (ms) or power (W).
     
        **Example (latency LUT raw file):**
        ```
        aggregate,3,6,0.22633870442708334
        aggregate,6,12,0.27776615960257395
        aggregate,12,24,0.43201446533203125
        aggregate,24,48,0.7267793019612631
        ```
2. **LUT Packaging**
   - Convert the cleaned TXT files into PKL format using:
     ```bash
     python LUT/generate_LUT.py
     ```
   - The resulting PKL files will be stored in `LUT/` and used directly during dataset construction.

---



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


> **Note:**  
> This repository is currently undergoing **code cleanup** and **engineering tests**.  
> Citation information will be **updated in the near future**.  
> A complete and stable release — including organized code, verified datasets, and detailed usage instructions — will be published soon.  
> We welcome your **use** and **citation** of this work.


