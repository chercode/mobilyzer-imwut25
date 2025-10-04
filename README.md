# MobiLyzer: Fine-grained Mobile Liquid Analyzer

This repository provides the official implementation of the  **IMWUT 2025/Ubicomp 2026** paper:   

**MobiLyzer: Fine-grained Mobile Liquid Analyzer**  
*[Shahrzad Mirzaei]()*<sup>1</sup>, *[Mariam Bebawy]()*<sup>1</sup>, *[Amr Mohamed Sharafeldin]()*<sup>1</sup>, and *[Mohamed Hefeeda]()*<sup>1,2</sup>  

<sup>1</sup> School of Computing Science, Simon Fraser University, Canada

<sup>2</sup>  Qatar Computing Research Institute, Qatar

 [Paper – pending]() | [Supplementary – pending]() | [Demo Video – pending]() | [Application – pending]()

![mobilyzer](figures/Picture1.png)  

---

## Overview and Components

 MobiLyzer supports tasks such as:  

- **Fraud detection** (e.g., adulterated olive oil)  
- **Quality assessment** (e.g., Extra virgin olive oil vs Refined olive oil)  
- **Origin labeling** (e.g., Italian olive oil vs American olive oil)
- **Composition labeling** (Fat, Protein, and sugar composition analysis)
- **Medical diagnostics** (e.g., urine analysis) 

![System Overview](figures/overview.png)

The repository is organized into four main modules:

1. **Intrinsic Decomposition**  
   Separation of reflectance and illumination components to recover material properties without the effect of iilumination.

2. **Hyperspectral Reconstruction**  
   Truthful reconstruction of spectral signatures from RGB+NIR smartphone images. 

3. **Liquid Analysis**  
    classification for fraud detection, composition analysis, and diagnosis.  

4. **Mobile Application**  
   A lightweight Android app for real-time liquid analysis on smartphones. [Link – pending]()

---

## Repository layout

```text
mobilyzer/
├─ reconstruction/                 # TSR training & evaluation
│  ├─ architecture/                # Spectral Reconstruction backbone and TSR heads
│  ├─ helper_matrices/             # S, B, PS, PB (for cmf & NIR centers)
│  ├─ train.py                     # Train TSR (truthful spectral recon)
│  ├─ test.py                      # Quantitative eval (SAM/SID/PSNR/ΔE/MAE)
│  ├─ utils.py, losses.py, utils_truthful.py  # TSR heads
│  └─ hsi_dataset_mobilyzer.py     # HSI dataset loader
├─ classification/                 # 1D-CNN classifiers on signatures
│  ├─ dataset.py                   # loads signature tensors & labels
│  ├─ train.py                     # k-fold training
│  └─ test.py                      # hold-out evaluation & reports
├─ datasets/
│  ├─ HSI/                         # Specim IQ VNIR (204 bands, and RGB+NIR)
│  └─ phone/                       # (RGB+NIR from smartphone camera)
├─ models/
│  ├─ HSI/                         # pre-trained models for reconstruction
│  └─ phone/                       # pre-trained models for classification
├─ figures/                        # paper figures & diagrams
├─ LICENSE
├─ environment.yml
└─ README.md
``` 
---

## Quickstart

### 1) Environment

- Python ≥ 3.9 (Conda recommended)  
- CUDA-enabled PyTorch (tested with 1.8.1 and 2.1+)  
- See `environment.yml` for pinned versions.  

```bash
conda env create -f environment.yml
conda activate mobilyzer
```
### 2) Install PyTorch

GPU:

```bash

python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1 triton==3.0.0

```

CPU:

```bash

python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1 triton==3.0.0

```

## Dataset
### HSI Dataset

- We provide five hyperspectral imaging (HSI) datasets corresponding to different liquids: **evoo**, **milk**, **honey**, **medicine**, and **urine**. Each dataset is organized under its respective folder (e.g., `evoo`).  

- The directory structure is as follows:  
  ```bash
  datasets/
  ├─ data/
  │  ├─ evoo/
  │  ├─ milk/
  │  ├─ honey/
  │  ├─ medicine/
  │  └─ urine/
  │
  │  Each liquid folder contains:
  │  ├─ HSI/           # Hyperspectral data (spectra + aligned RGB/NIR)
  │  └─ phone/         # Smartphone data organized by each case
  │     ├─ fraud/
  │     ├─ origin/
  │     └─ quality/
  ```
- To train the TSR from scratch, evaluate it, and perform liquid analysis, you need to download one or more of the following datasets:

    - [evoo]() ( GB) - [milk]() ( GB) - [honey]() ( GB) - [medicine]() ( GB) - [urine]() ( GB)

- After downloading, unzip the dataset(s) and place them under the datasets directory. Please note that additional storage (comparable to the dataset size) is required to reproduce the reconstruction results.
   
- You can also download [all datasets together]() ( GB). 

### Smartphone Dataset
If you are only interested in liquid analysis using pretrained models (without training TSR from scratch), you may download one or more of the following smartphone-based datasets for each liquid:

  - [evoo]() ( GB)  - [milk]() ( GB)  - [honey]() ( GB)  - [medicine]() ( GB)  - [urine]() ( GB)


We also evaluate MobiLyzer on three additional smartphones. Their datasets are provided below for evaluation:

  - [oneplus]() ( GB)  - [ulefone]() ( GB)  - [doogee]() ( GB)
    
## Pretrained Models

To simplify reproduction of the results, we provide pretrained models for both Truthful Spectral Reconstruction (TSR) and liquid analysis:

TSR Models: Trained on the HSI datasets to reconstruct spectral information from smartphone images.

Classification Models: Trained on reconstructed spectra (or smartphone images directly) to perform liquid identification, fraud detection, and quality analysis.

Download the pretrained models from: pretrained_models.zip
 (GB).
After extraction, place the folder under the project root (e.g., models/).

This allows you to directly run the reconstruction and classification pipelines without retraining, significantly reducing compute and storage requirements.


## Getting Started

### 1) Intrinsic Decomposition

In the dataset structure, intrinsic decomposition outputs are stored under:

```bash
datasets/phone/<liquid>/intrinsic/

```
Each folder contains normalized RGB albedo images, which should be used as the input to the TSR reconstruction model instead of raw phone captures.

We adapted the intrinsic decomposition model from [Careaga et al.](https://github.com/compphoto/Intrinsic) and re-engineered it for efficiency on smartphones, including ONNX export and quantization for reduced memory footprint and faster inference. Please find more information in our paper.

### 2) Truthful Spectral Reconstruction(TSR)
Training TSR from Scratch
```bash
python3 reconstruction/train.py --data_root /path/to/dataset/liquid/HSI/ --liquid evoo --epochs 20 --batch_size 16 --lr 1e-4 --patch_size 64
```
Using Pre-trained TSR Models
```bash

python3 reconstruction/test.py --model_path models/HSI/TSR_evoo_best.pth --input_dir datasets/phone/evoo/origin/IT --output_dir datasets/phone/evoo/origin/reconstructed/ --liquid evoo


**Inputs:** use ``` datasets/phone/<liquid>/<task>/intrinsic/ ``` (albedo RGB) + the matching nir/ folder.

**NIR choice:** set ``` --nir 850 ``` or ```--nir 940``` to match the device (e.g., night-vision cameras ≈850 nm; FaceID cameras ≈940 nm).

**Bands:** ```--bands 68``` is the default we provide; higher values increase compute with little/no accuracy gain.

**Outputs:** ```.mat``` per sample saved under .../reconstructed/. which will be the input to our classification model.

### 3) Liquid Analysis

Train a 1D-CNN on reconstructed spectral signatures (68-D vectors) for tasks like fraud, origin, quality, and medical:

data_root must point to reconstructed spectra produced by TSR

```bash
python3 classification/train.py \
  --data_root /path/to/datasets/phone/evoo/<task>/reconstructed/ \
  --liquid evoo \
  --n_splits 4 \
  --max_epochs 100 \
  --batch_size 256

```
Evaluate a pretrained classifier:
```bash
python3 classification/test.py \
  --models_dir models/phone/evoo/<task>/ \
  --liquid evoo \
  --n_splits 4
```

Notes

**Inputs:** each sample in ```.../reconstructed/``` should be a spectrum (.mat) with 68 bands from TSR.

**Liquid:** 

**Splits:** --n_splits 4 performs stratified K-fold (recommended);

Outputs: 

Reports: the test command prints accuracy/precision/recall (and confusion matrix if supported) and saves them next to the checkpoints.


### 4) Mobile Application

XXX Link

## Citation

If you use MobiLyzer in your research, please cite our paper:
```bibtex
@article{Mirzaei2025MobiLyzer,
  author = {Mirzaei, Shahrzad and Bebawy, Mariam and Sharafeldin, Amr Mohamed and Hefeeda, Mohamed},
  title = {MobiLyzer: Fine-grained Mobile Liquid Analyzer},
  year = {2025},
  volume = {4},
  number = {9},
  journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
  doi = {10.1145/3770678}
}
```


