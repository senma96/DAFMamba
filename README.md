# DFECrack

[![DOI](https://zenodo.org/badge/1223222632.svg)](https://doi.org/10.5281/zenodo.20109959)

**Direction-Aware and Frequency-Enhanced Lightweight Network for Efficient Crack Segmentation**

This is the official repository for the paper submitted to *The Visual Computer*. If you use this code or the pretrained models, please cite our paper.

## Overview

DFECrack is a lightweight crack segmentation network built on the visual state-space model (Mamba). It tackles three key limitations of existing approaches:

- **Direction-Aware Scanning (DAS)** — scans feature maps along bidirectional axial directions and adaptively merges responses via a learned pixel-level orientation field, preserving crack continuity.
- **Adaptive Channel-Spatial Fusion (ACSF)** — selectively gates local convolutional and global Mamba features at the pixel level with O(N) complexity.
- **Adaptive Frequency Enhancement (AFE)** — separates low-frequency structure from high-frequency boundary through spectral decomposition for cross-scale frequency coordination.

DFECrack achieves state-of-the-art mIoU on four public benchmarks with only **3.44M parameters** and **18.70 GFLOPs**.

## Results

| Dataset   | ODS   | OIS   | P     | R     | F1    | mIoU  |
|-----------|-------|-------|-------|-------|-------|-------|
| TUT       | 82.31 | 82.77 | 84.20 | 84.45 | 84.33 | 85.03 |
| Crack500  | 74.26 | 75.17 | 79.11 | 77.84 | 78.47 | 79.16 |
| DeepCrack | 91.83 | 92.38 | 92.20 | 94.28 | 93.23 | 92.29 |
| CrackMap  | 79.56 | 79.60 | 78.14 | 81.27 | 79.67 | 82.47 |

## Getting Started

### Prerequisites

- Python 3.10
- PyTorch 2.1.0 + CUDA 11.8
- [mamba-ssm](https://github.com/state-spaces/mamba) 1.2.0
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) 1.4.0

### Installation

```bash
conda create -n DFECrack python=3.10 -y
conda activate DFECrack

# PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Mamba dependencies
pip install causal-conv1d==1.4.0
pip install mamba-ssm==1.2.0

# mmcv ecosystem
pip install openmim
mim install mmcv-full==1.7.2

# Other dependencies
pip install einops timm opencv-python scipy pillow tqdm torchinfo
```

### Pretrained Weights

Download pretrained checkpoints from [Google Drive](https://drive.google.com/drive/folders/1-ghCowtD4XZzhl7nyTBWP_2VFt-H7CYf) and place them under `checkpoint/`:

```
DFECrack/
└── checkpoint/
    ├── checkpoint_best_TUT.pth
    ├── checkpoint_best_Crack500.pth
    ├── checkpoint_best_DeepCrack.pth
    └── checkpoint_best_CrackMap.pth
```

SHA-256 checksums:

| File | SHA-256 |
|------|---------|
| `checkpoint_best_TUT.pth` | `5f5d415525c5bfb98acf6a5f39856c755bdea60e28881e97a39ae386e739734f` |
| `checkpoint_best_Crack500.pth` | `8e178654a9e7c349b6a0b6056f23a0c80c915a4fa9b172aea28d215d349bb4c9` |
| `checkpoint_best_DeepCrack.pth` | `35c7bd09bbda07ed31afa4f34aad10f63d748188f2b61cb08507806b7e583cf0` |
| `checkpoint_best_CrackMap.pth` | `2e2f3193a807b0218f5bc1c21f9201f8be880307e1f48c59bf43e834a0e78262` |

## Dataset Preparation

Download datasets from [Google Drive](https://drive.google.com/drive/folders/1-ghCowtD4XZzhl7nyTBWP_2VFt-H7CYf) or prepare them manually. Each dataset should follow this structure:

```
<dataset_root>/
├── <DatasetName>/
│   ├── train_img/     # training images
│   ├── train_lab/     # training labels (binary masks)
│   ├── val_img/       # validation images
│   ├── val_lab/       # validation labels
│   ├── test_img/      # test images
│   └── test_lab/      # test labels (binary masks)
```

For example:

```
/your/data/path/
├── TUT/
│   ├── train_img/
│   ├── train_lab/
│   ├── val_img/
│   ├── val_lab/
│   ├── test_img/
│   └── test_lab/
├── Crack500/
├── DeepCrack/
└── CrackMap/
```

Pass the full path to a specific dataset via `--dataset_path`, e.g. `--dataset_path /your/data/path/TUT`.

## Training

```bash
# TUT (with boundary supervision)
python main.py --dataset_path /your/data/path/TUT --use_boundary

# Other datasets
python main.py --dataset_path /your/data/path/Crack500
python main.py --dataset_path /your/data/path/DeepCrack
python main.py --dataset_path /your/data/path/CrackMap
```

## Inference

```bash
python test.py --dataset_path /your/data/path/TUT \
  --test_checkpoint ./checkpoint/checkpoint_best_TUT.pth
```

## Evaluation

After inference, evaluate predictions:

```bash
python eval/evaluate.py --results_dir ./results/<DatasetName>/<DatasetName>_results
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{ma2026dfecrack,
  title={Direction-Aware and Frequency-Enhanced Lightweight Network for Efficient Crack Segmentation},
  author={Ma, Shuaisen and Xu, Mingyue and Wu, Wen},
  year={2026}
}
```

## Acknowledgements

We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba), [Mamba](https://github.com/state-spaces/mamba), [SCSegamba](https://github.com/Karl1109/SCSegamba), and [CrackMamba](https://github.com/shengyu27/CrackMamba) for their excellent open-source implementations.

## License

This project is released under the [Apache License 2.0](LICENSE).
