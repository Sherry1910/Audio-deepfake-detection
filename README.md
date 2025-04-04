# 🎧 Audio Deepfake Detection using LCNN + FTANet

This project implements a deepfake audio detector using the ASVspoof2019 LA dataset. The model architecture is based on a Light Convolutional Neural Network (LCNN), with an optional enhancement using FTANet-style temporal attention. It supports both notebook-based experimentation and script-based training.

---

## 🚀 Features

- LCNN-based architecture optimized for spoofed audio detection.
- Optional FTANet temporal attention mechanism for improved feature learning.
- Jupyter Notebook for interactive training and evaluation.
- Script-based pipeline (`train.py`) for automation.
- Clean modular code structure with PyTorch.
- Well-documented implementation & model selection analysis.

---

## 🛠️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Sherry1910/Audio-deepfake-detection.git
cd Audio-deepfake-detection

pip install -r requirements.txt

jupyter notebook notebook.ipynb

python train.py

audio-deepfake-detection/
├── models/
│   └── lcnn.py
├── notebook.ipynb
├── train.py
├── requirements.txt
├── README.md
├── documentation.md
└── .gitignore

