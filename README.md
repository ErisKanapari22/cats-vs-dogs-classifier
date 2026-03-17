# 🐶🐱 Cats vs Dogs Image Classifier

A Deep Learning project that classifies images as **Cat** or **Dog** using a Convolutional Neural Network (CNN) built with PyTorch.

## 🚀 Project Overview

This project demonstrates how to build an end-to-end image classification pipeline:

- Data loading and preprocessing
- Data augmentation
- CNN model building
- Training and evaluation
- CLI-based prediction on custom images

The goal is to create a **clean, professional, and portfolio-ready ML project**.

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- Matplotlib
- PIL (Python Imaging Library)

## 📁 Project Structure
cats-vs-dogs-classifier/
```
│
├── data/
│ ├── raw/
│ └── processed/
│
├── outputs/
│ ├── accuracy_plot.png
│ └── loss_plot.png
│
├── src/
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ └── predict.py
│
├── main.py
├── predict.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier
```
2. Create virtual environment:
```
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

---

# 🧩 PART 5 — Training


## 🧠 Training the Model

Run the training script:

```bash
python main.py

- During training, the model will:

- Learn from the dataset

- Print training & validation accuracy

- Save performance plots in the outputs/ folder
```


---

# 🧩 PART 6 — Prediction (CLI 🔥)


## 🔍 Predict on Custom Image

Run:

```bash
python predict.py path_to_image.jpg
```
```python predict.py data/samples/test.jpg```
```Prediction: Dog (0.87 confidence)```


---

# 🧩 PART 7 — Results


## 📊 Results

- Achieved ~72% accuracy on validation data
- Performance improves with more training and tuning

### Training Performance

Accuracy and loss plots are saved in:

## 🚀 Future Improvements

- Use Transfer Learning (ResNet, MobileNet)
- Increase dataset size
- Hyperparameter tuning
- Deploy as a web app (Flask / FastAPI)