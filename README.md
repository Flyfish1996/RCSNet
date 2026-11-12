# RCSNet: Rhythm Commonality-Specificity Network for MI-EEG Decoding

Official PyTorch implementation of the paper:  
**"Rhythm Commonality–Specificity Learning for Multi-Rhythm Motor Imagery EEG Decoding"**

---

## 📖 Introduction

Motor imagery brain–computer interfaces (MI-BCIs) decode motor intentions from electroencephalography (EEG) signals. MI-related neural activity is characterized by rhythmic modulations distributed across multiple frequency ranges, particularly the $\mu/\alpha$ and $\beta$ rhythms. Existing deep learning methods have substantially improved MI-EEG decoding, but the shared task-related information across rhythms and the information specific to individual rhythms remain insufficiently characterized.

To address this issue, we propose **RCSNet (Rhythm Commonality–Specificity Network)**, a multi-rhythm MI-EEG decoding framework that jointly models cross-rhythm commonality and rhythm-specific information.

RCSNet consists of five main components:

1. **Rhythm View Construction (RVC)**  
   Learnable channel-wise Sinc filters construct multiple frequency-differentiated rhythmic views directly from EEG signals.

2. **Dual-Path Rhythm Encoding**  
   Each rhythmic view is encoded through independent common and specific pathways to obtain rhythm-common and rhythm-specific representations.

3. **Commonality–Specificity Loss ($\mathcal{L}_{\mathrm{CS}}$)**  
   Cross-rhythm consistency is encouraged among common representations, while redundancy between common and specific representations is reduced within each rhythm.

4. **Deep Temporal Encoding (DTE)**  
   Temporal dependencies are further modeled for the aggregated common representation and individual rhythm-specific representations. The implementation supports GRU, LSTM, Transformer, and temporal CNN encoders.

5. **Uncertainty-Aware Fusion (UAF)**  
   During inference, predictions from the common and rhythm-specific branches are adaptively combined according to their sample-wise predictive uncertainty.

---

## 🏗️ Repository Structure

```text
├── RCSNet.py                             # RCSNet model architecture, STConvEncoder, DTE, and RCSNetLoss
├── train_model_withMean_oneStage.py      # Single-fold training loop, metrics calculation
├── Classify_BNCI2014_001_WithinSub.py    # Experiment runner for BCI Competition IV-2a dataset
├── Classify_Lee2019_MI_WithinSub.py      # Experiment runner for OpenBMI (Lee2019_MI) dataset
├── DataPrepare.py                        # Data loading and dataset preprocessing interface
├── prepocess_in_model.py                 # Bandpass filtering and window cropping helper functions
└── README.md                             # Project documentation
```

---

## 🛠️ Environment Setup

The implementation is based on PyTorch. We recommend Python 3.8 or later and a CUDA-enabled environment for GPU training.

Install the main dependencies with:

```bash
pip install torch
pip install numpy scipy scikit-learn mne
```

The exact PyTorch version can be selected according to the installed CUDA version.

---

## 🚀 Usage & Quick Start

### 1. BCI Competition IV-2a Dataset
To run 4-class MI decoding experiments on the BCI Competition IV-2a dataset (22 channels, 9 subjects):

```bash
python Classify_BNCI2014_001_WithinSub.py
```

### 2. OpenBMI (Lee2019_MI) Dataset
To run 2-class MI decoding experiments on the OpenBMI dataset (62 channels, 21-subject valid subset):

```bash
python Classify_Lee2019_MI_WithinSub.py
```

All evaluation metrics (Accuracy, Precision, F1-Score, and Cohen's Kappa) will be automatically computed and saved as formatted `.txt` files under the `./results/` directory.

---

## ⚙️ Model Hyperparameters

Default hyperparameters for `RCSNet` can be adjusted in the experiment scripts:

```python
model_para = {
    'K': 3,                  # Number of learnable rhythmic views
    'rdm_filters': 20,       # Number of shallow spatio-temporal channels
    'time_out': 16,           # Pooled temporal feature dimension
    'seq_model_type': 'gru', # Temporal sequence learner ('gru', 'lstm', 'transformer', 'cnn')
    'beta': 0.05              # Weight for Commonality-Specificity Loss
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

Special thanks to EEGConformer [[Paper](https://doi.org/10.1109/TNSRE.2022.3230250) | [Code](https://github.com/eeyhsong/EEG-Conformer)] for providing the training paradigm and evaluation pipeline that served as an important foundation for this study. Building upon this framework, we further customized and extended the data loading, model training, and evaluation procedures to suit the specific requirements of our work.

