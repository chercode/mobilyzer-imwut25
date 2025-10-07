# MobiLyzer: Fine-grained Mobile Liquid Analyzer

This repository provides the official implementation of the  **IMWUT 2025/Ubicomp 2026** paper:   

**MobiLyzer: Fine-grained Mobile Liquid Analyzer**  
*[Shahrzad Mirzaei](https://github.com/chercode)*<sup>1</sup>, *[Mariam Bebawy](https://github.com/mariam-bebawy)*<sup>1</sup>, *[Amr Mohamed Sharafeldin](https://github.com/AmrMSharafeldin)*<sup>1</sup>, and *[Mohamed Hefeeda](https://github.com/mhefeeda)*<sup>1,2</sup>  

<span style="font-size:9px"><sup>1</sup> School of Computing Science, Simon Fraser University, Canada</span>

<small><sup>2</sup>  Qatar Computing Research Institute, Qatar</small>

 [Paper](https://doi.org/10.1145/3770678) | [Supplementary](https://doi.org/10.1145/3770678) | [Phone Application](https://github.com/chercode/MobiLyzer-Android/tree/release)

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
   A lightweight Android app for real-time liquid analysis on smartphones. [MobiLyzer-Android Application](https://github.com/chercode/MobiLyzer-Android/tree/release)

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
git clone https://github.com/chercode/mobilyzer-imwut25.git
cd mobilyzer-imwut25
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
---
## Dataset
### HSI Dataset

- We provide five hyperspectral imaging (HSI) datasets corresponding to different liquids: **evoo**, **milk**, **honey**, **medicine**, and **urine**. Each dataset is organized under its respective folder (e.g., `evoo`).  

- The data directory structure is as follows:  
  ```bash

  ├─ data/
  │  ├─ evoo/
  │  ├─ milk/
  │  ├─ honey/
  │  ├─ medicine/
  │  └─ urine/
  │
  │  Each liquid folder contains:
  │  ├─ HSI/           # Hyperspectral data (spectra + aligned RGB/NIR)
  │  └─ phone/ 
  │     └─ task/    # Smartphone data organized by each task
  │        ├─ raw/
  │        ├─ intrinsic/
  │        └─ reconstructed/
  
  ```
- To train the TSR from scratch, evaluate it, and perform liquid analysis, you need to download one or more of the following datasets:

    - [evoo](https://drive.google.com/file/d/1MPZ9KFyL8dOD6WHQziyKszDlxNdgPI9M/view?usp=drive_link) - [milk](https://drive.google.com/file/d/1WdLbpQ-5lGT2PUenaWBeUghlGuBN6358/view?usp=drive_link) - [honey](https://drive.google.com/file/d/114q591eZA7Xc0EIpnqADOzZzDhzxQF-q/view?usp=drive_link) - [medicine](https://drive.google.com/file/d/1kwVfoS6TJwna59noSxkJQXhxHjAXO3ly/view?usp=drive_link) - [urine](https://drive.google.com/file/d/13WJgscmSxbSLzv6kDy7zcJBo5HFmmsao/view?usp=drive_link)

- After downloading, unzip the dataset(s) and place them under the ```data``` directory. Please note that additional storage (comparable to the dataset size) is required to reproduce the reconstruction results.

### Smartphone Dataset
If you are only interested in liquid analysis using pretrained models (without training TSR from scratch), you may download one or more of the following smartphone-based datasets for each liquid:

  - [evoo](https://drive.google.com/file/d/1vVkTpHR5Vmfibuwv5dhxGcaGEGzUlSTe/view?usp=drive_link)  - [milk](https://drive.google.com/file/d/14bj_4kMPE6mGZYLaoW62R1HiUTI7Lqx7/view?usp=drive_link)  - [honey](https://drive.google.com/file/d/1ja5R-EqC99WC64I4iG8htLDviWD_KCxq/view?usp=drive_link)  - [medicine](https://drive.google.com/file/d/1z3rBU-J-wkhV6XUbLUGle1MioCMyQAHv/view?usp=drive_link)  - [urine](https://drive.google.com/file/d/1W9sR09mh2ud64CYaH8ApXnRdhMHAhHV6/view?usp=drive_link)


We also evaluate MobiLyzer on three additional smartphones. Their datasets are provided below for evaluation:

  - [oneplus](https://drive.google.com/file/d/1piJaskqZZylVrtkYXIJ_1PhTo5JU8us5/view?usp=drive_link)  - [ulefone](https://drive.google.com/file/d/1Z3JH7IS2CtznKguhhGv8XMGesy4guD1L/view?usp=drive_link)  - [doogee](https://drive.google.com/file/d/1kQa5154KubGk8z210C_sovdS3ukk-KoO/view?usp=drive_link)
    
## Pretrained Models

To simplify reproduction of the results, we provide pretrained models for both Truthful Spectral Reconstruction (TSR) and liquid analysis:

HSI Models: Trained on the HSI datasets to reconstruct spectral information from smartphone images.

Classification Models: Trained on reconstructed spectra (or smartphone images directly) to perform liquid identification, fraud detection, and quality analysis.

Download the pretrained models from: [pretrained_models]()

After extraction, place the folder under the project root, ```models/```.

This allows you to directly run the reconstruction and classification pipelines without retraining.

---
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
python3 reconstruction/train.py --data_root /path/to/dataset/liquid/HSI/ --split_root /path/to/dataset/liquid/HSI/ --outf ../models/HSI/ --nir 940 --bands 68
```
Using Pre-trained TSR Models
```bash

python3 reconstruction/evaluate_mobile.py --model_path models/HSI/pretrained_model.pth --input_dir /path/to/dataset/liquid/phone/intrinsic --output_dir /path/to/dataset/liquid/phone/reconstructed/ --nir 940 --bands 68
```

**Inputs:** use ``` datasets/phone/<liquid>/<task>/intrinsic/ ``` (albedo RGB) + the matching nir/ folder.

**NIR choice:** set ``` --nir 850 ``` or ```--nir 940``` to match the device (e.g., night-vision cameras ≈850 nm; FaceID cameras ≈940 nm).

**Bands:** ```--bands 68``` is the default we provide; higher values increase compute with little/no accuracy gain.

**Outputs:** ```.mat``` per sample saved under .../reconstructed/. which will be the input to our classification model.

**Phone Models:** By default, the pipeline uses Pixel settings. For other devices, add `--phone [model]`:
```bash
# Options: oneplus, doogee, ulefone, pixel (default)
python3 reconstruction/evaluate_mobile.py --phone oneplus --model_path models/HSI/pretrained_model.pth --input_dir /path/to/dataset/liquid/phone/intrinsic --output_dir /path/to/dataset/liquid/phone/reconstructed/

### 3) Liquid Analysis

Train our classification model on reconstructed spectral signatures (68-D vectors) for different liquids and tasks:

data_root must point to reconstructed spectra produced by TSR

```bash
python3 classification/train.py --data_root /path/to/datasets/phone/<liquid>/<task>/phone/reconstructed/ --liquid evoo

```
Evaluate a pretrained classifier:
```bash
python3 classification/test.py --models_dir /path/to/models/phone/evoo/<task>/ --liquid evoo 
```

**Inputs**
- Each sample in `.../reconstructed/` is a `.mat` spectrum with **68 bands** produced by TSR.
- `--models_dir` should point to a folder containing one or more `.pth` checkpoints compatible with the chosen `<liquid>` and `<task>`.

**Liquid**
- Set with `--liquid {evoo|milk|honey|medicine|urine}`.
- Must match the data under `datasets/phone/<liquid>/...` and the model trained for that liquid.

**Splits**
- `--n_splits 4` runs **K-Fold** (recommended) for robust evaluation.
---

### 4) Mobile Application

We created an Android Application for MobiLyzer, which can be found [here](https://github.com/chercode/MobiLyzer-Android/tree/release) on GitHub.

---

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


