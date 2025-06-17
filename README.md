# 🚀 Land Cover Classification with PyTorch

This project uses deep learning to classify satellite images from the **EuroSAT** dataset into different land cover types such as forest, crops, water, etc.

---

## 📁 Project Structure

landcover\_classifier/
├── configs/             # YAML configuration
├── models/              # CNN architecture (ResNet)
├── utils/               # Data loading, metrics, helpers
├── train.py             # Epoch-wise training
├── test.py              # Final evaluation and confusion matrix
├── main.py              # Main script
├── requirements.txt     # Dependencies
└── README.md            # This guide

---

## 📦 Dataset: EuroSAT

* Source: [EuroSAT on Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
* Expected structure:

```
data/
└── EuroSAT/
    ├── AnnualCrop/
    ├── Forest/
    ├── River/
    ├── ... (10 classes)
```

When running `main.py`, the dataset will be automatically split into:

```
data/
├── train/
└── test/
```

---

## ⚙️ Installation

1. Clone this repository:

```bash
git clone https://github.com/tu_usuario/landcover_classifier.git
cd landcover_classifier
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate     # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Configuration

Edit the file `configs/config.yaml` to change hyperparameters like the model, batch size, epochs, etc.

```yaml
model_name: resnet18
batch_size: 64
epochs: 20
learning_rate: 0.0003
num_classes: 10
input_size: 224
checkpoint_path: checkpoints/landcover_model.pth
use_gpu: true
```

---

## 🚀 Training

Once the dataset is downloaded:

```bash
python main.py
```

This will:

* Split the dataset into training and validation
* Train the model
* Save checkpoints
* Log metrics to TensorBoard

---

## 📊 Visualization

```bash
tensorboard --logdir runs/
```

Open your browser at [http://localhost:6006](http://localhost:6006)

---

## ✅ Final Results

* Accuracy and F1 score per epoch
* Confusion matrix
* Trained model in `checkpoints/`
* Ready to use in APIs, dashboards, or research

---

## 🧠 Dataset Classes

* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake
