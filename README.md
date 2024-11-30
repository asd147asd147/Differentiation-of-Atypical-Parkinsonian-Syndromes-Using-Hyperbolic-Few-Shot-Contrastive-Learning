# Differentiation-of-Atypical-Parkinsonian-Syndromes-Using-Hyperbolic-Few-Shot-Contrastive-Learning

![Overall Framework Flowchart](https://github.com/asd147asd147/Differentiation-of-Atypical-Parkinsonian-Syndromes-Using-Hyperbolic-Few-Shot-Contrastive-Learning/assets/55697983/1761ccb9-600f-4229-890b-b5bf38271015)

## Introduction

This repository contains the implementation for the paper *"Differentiating atypical parkinsonian syndromes with hyperbolic few-shot contrastive learning"* (DOI: [10.1016/j.neuroimage.2024.120940](https://doi.org/10.1016/j.neuroimage.2024.120940)). The study introduces a novel framework leveraging hyperbolic geometry and few-shot learning techniques to classify atypical parkinsonian syndromes (APS) effectively, addressing challenges such as limited data availability and hierarchical relationships in clinical features.

### Key Highlights
- Classification of APS types (MSA-P, MSA-C, PSP, PD) based on hierarchical relationships in iron accumulation patterns.
- Utilization of hyperbolic embedding for better representation of clinical hierarchical data.
- Few-shot learning with contrastive loss for improved performance on limited datasets.

## Prerequisites

This implementation assumes a deep learning environment with specific dependencies. Using Docker is highly recommended to ensure compatibility.

### Experiment Environment
- **Operating System**: Ubuntu (Linux OS supporting NVIDIA Docker)
- **Deep Learning Framework**: PyTorch v1.10.0
- **CUDA Version**: 11.3
- **cuDNN Version**: 8

To replicate the environment:
1. Build the Docker image using the provided `Dockerfile`.
2. Install Python dependencies listed in `requirements.txt`.

## Dataset

The dataset used in this research is private and not included in the repository. To use this code, organize your dataset as follows:

Dataset   
├── MSAP   
├── MSAC   
├── PSP   
├── PD   
└── NC   

## Training and Testing

### Run Training
Once the environment and dataset are ready, execute the following command to start training and testing:

```bash
python main.py
```

Model Outputs   
- Weights: Create a weights directory in the project root to store model weights.

![UMAP](https://github.com/asd147asd147/Differentiation-of-Atypical-Parkinsonian-Syndromes-Using-Hyperbolic-Few-Shot-Contrastive-Learning/assets/55697983/eacdb273-8e93-437d-90b2-def110611423)

```base
@article{choi2024differentiation,
  title={Differentiating atypical parkinsonian syndromes with hyperbolic few-shot contrastive learning},
  author={Choi, Won June and HwangBo, Jin and Duong, Quan Anh and Lee, Jae-Hyeok and Gahm, Jin Kyu},
  journal={NeuroImage},
  volume={304},
  pages={120940},
  year={2024},
  publisher={Elsevier}
}
```
