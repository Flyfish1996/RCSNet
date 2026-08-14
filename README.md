# RCSNet: Rhythm Commonality-Specificity Network for MI-EEG Decoding

Official PyTorch implementation of the paper:  
**"Rhythm Commonality–Specificity Network for Motor Imagery EEG Decoding"**

> **Note**: This repository is structured for anonymous peer-review. All author and institutional details have been removed.

---

## 📖 Introduction

Motor Imagery Brain-Computer Interfaces (MI-BCIs) decode motor intentions directly from electroencephalography (EEG) signals. Although sensorimotor rhythms ($\mu/\alpha$ and $\beta$ bands) provide key physiological bases for discriminating motor imagery, existing deep learning methods either process broadband signals uniformly or treat multiple rhythms as monolithic units, neglecting what information is **shared across rhythms** and what remains **specific to individual rhythms**.

To address this, we propose **RCSNet** (**Rhythm Commonality–Specificity Network**), which explicitly models both cross-rhythm commonality and rhythm-dependent specificity:
1. **Rhythm View Construction (RVC)**: Utilizes learnable Sinc-based depthwise filters to dynamically decompose raw EEG signals into multiple frequency-differentiated rhythmic views.
2. **Dual-Path Rhythm Encoding**: Encodes each view into separate rhythm-common and rhythm-specific representation spaces using shallow spatio-temporal encoders.
3. **Commonality–Specificity Loss ($\mathcal{L}_{\mathrm{CS}}$)**: Enforces cross-rhythm consistency on common representations and common-specific diversity within each view.
4. **Deep Temporal Encoding (DTE)**: Captures long-term temporal dependencies using temporal learners (e.g., GRU, Transformer, CNN) on aggregated common features and independent specific features.
5. **Uncertainty-Aware Fusion (UAF)**: Adaptively fuses predictions from common and specific branches during inference based on sample-wise predictive entropy.

---

## 🏗️ Repository Structure

```text
├── rcsnet.py                    # RCSNet model architecture, STConvEncoder, DTE, and RCSNetLoss
├── train_model.py               # Single-fold training loop, metrics calculation, and UAF evaluation
├── run_experiments.py           # Experiment runner for BCI Competition IV-2a dataset
├── run_experiments_openbmi.py   # Experiment runner for OpenBMI (Lee2019_MI) dataset
├── DataPrepare.py               # Data loading and dataset preprocessing interface
├── prepocess_in_model.py        # Bandpass filtering and window cropping helper functions
├── requirements.txt             # Environment dependencies
└── README.md                    # Project documentation
```

---

## 🛠️ Environment Setup

We recommend using Anaconda with Python 3.8+ and PyTorch with CUDA acceleration.

```bash
# Clone the anonymous repository
git clone https://github.com/anonymous/RCSNet.git
cd RCSNet

# Create and activate conda environment
conda create -n rcsnet python=3.8 -y
conda activate rcsnet

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn mne
```

---

## 🚀 Usage & Quick Start

### 1. BCI Competition IV-2a Dataset
To run 4-class MI decoding experiments on the BCI Competition IV-2a dataset (22 channels, 9 subjects):

```bash
python run_experiments.py
```

### 2. OpenBMI (Lee2019_MI) Dataset
To run 2-class MI decoding experiments on the OpenBMI dataset (62 channels, 21-subject valid subset):

```bash
python run_experiments_openbmi.py
```

All evaluation metrics (Accuracy, Precision, F1-Score, and Cohen's Kappa) will be automatically computed and saved as formatted `.txt` files under the `./results/` directory.

---

## ⚙️ Model Hyperparameters

Default hyperparameters for `RCSNet` can be adjusted in the experiment scripts:

```python
model_para = {
    'K': 3,                  # Number of learnable rhythmic views
    'rdm_filters': 32,       # Number of shallow spatio-temporal channels
    'time_out': 1,           # Pooled temporal feature dimension
    'seq_model_type': 'gru', # Temporal sequence learner ('gru', 'lstm', 'transformer', 'cnn')
    'beta': 0.5              # Weight for Commonality-Specificity Loss
}
```

---

## 🙏 Acknowledgements

We sincerely thank the authors of the following open-source projects and codebases for providing invaluable baselines, reference implementations, and theoretical insights for this study:

* **EEGNet** — Lawhern et al., *Journal of Neural Engineering*, 2018. [[Paper](https://doi.org/10.1088/1741-2552/aace8c)]
* **ShallowConvNet (ShallowNet)** — Schirrmeister et al., *Human Brain Mapping*, 2017. [[Paper](https://doi.org/10.1002/hbm.23730)] | [[Code](https://github.com/braindecode/braindecode)]
* **FBCNet**  — Mane et al., *arXiv*, 2021. [[Paper](https://arxiv.org/abs/2104.01233)] | [[Code](https://github.com/ravikiran-mane/FBCNet)]
* **LightConvNet** — Ma et al., *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 2023. [[Paper](https://doi.org/10.1109/TNSRE.2023.3299355)] | [[Code](https://github.com/Ma-Xinzhi/LightConvNet)]
* **LMDA-Net**  — Miao et al., *NeuroImage*, 2023. [[Paper](https://doi.org/10.1016/j.neuroimage.2023.120209)] | [[Code](https://github.com/MiaoZhengQing/LMDA-Code)]
* **EEGConformer** — Song et al., *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 2023. [[Paper](https://doi.org/10.1109/TNSRE.2022.3230250)] | [[Code](https://github.com/eeyhsong/EEG-Conformer)]
* **EEG-TransNet (TransNet)** — Ma et al., *Computers in Biology and Medicine*, 2024. [[Paper](https://doi.org/10.1016/j.compbiomed.2024.108504)] | [[Code](https://github.com/Ma-Xinzhi/EEG-TransNet)]
* **MSVTNet** — Liu et al., *IEEE Journal of Biomedical and Health Informatics*, 2024. [[Paper](https://doi.org/10.1109/JBHI.2024.3450753)] | [[Code](https://github.com/SheepTAO/MSVTNet)]
* **IFNet** — Wang et al., *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 2023. [[Paper](https://doi.org/10.1109/TNSRE.2023.3257319)] | [[Code](https://github.com/Jiaheng-Wang/IFNet)]
* **SST-DPN** — Han et al., *Knowledge-Based Systems*, 2025. [[Paper](https://doi.org/10.1016/j.knosys.2025.113315)] | [[Code](https://github.com/Open-EXG/SST-DPN)]

特别感谢 **EEGConformer** [[12]] [[Paper](https://doi.org/10.1109/TNSRE.2022.3230250) | [Code](https://github.com/eeyhsong/EEG-Conformer)] 为本研究提供的训练范式与评估管线（Training Paradigm and Evaluation Pipeline）。本项目的数据加载、模型训练与评估框架在此基础上进行了进一步深化与定制。

Special thanks to EEGConformer [[Paper](https://doi.org/10.1109/TNSRE.2022.3230250) | [Code](https://github.com/eeyhsong/EEG-Conformer)] for providing the training paradigm and evaluation pipeline that served as an important foundation for this study. Building upon this framework, we further customized and extended the data loading, model training, and evaluation procedures to suit the specific requirements of our work.

