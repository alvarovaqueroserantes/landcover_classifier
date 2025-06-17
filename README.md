# 🛰️ Land Cover Classification with PyTorch

Este proyecto utiliza deep learning para clasificar imágenes satelitales del dataset **EuroSAT** en diferentes tipos de cobertura terrestre como bosques, cultivos, agua, etc.

---

## 📁 Estructura del Proyecto

landcover_classifier/
├── configs/ # Configuración YAML
├── models/ # Arquitectura CNN (ResNet)
├── utils/ # Carga de datos, métricas, helpers
├── train.py # Entrenamiento por época
├── test.py # Evaluación final y matriz de confusión
├── main.py # Script principal
├── requirements.txt # Dependencias
└── README.md # Esta guía

yaml
Copiar
Editar

---

## 📦 Dataset: EuroSAT

- Fuente: [EuroSAT on Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
- Estructura esperada:

data/
└── EuroSAT/
├── AnnualCrop/
├── Forest/
├── River/
├── ... (10 clases)

css
Copiar
Editar

Al ejecutar `main.py`, el dataset se dividirá automáticamente en:

data/
├── train/
└── test/

yaml
Copiar
Editar

---

## ⚙️ Instalación

1. Clona este repositorio:

```bash
git clone https://github.com/tu_usuario/landcover_classifier.git
cd landcover_classifier
Crea y activa un entorno virtual:

bash
Copiar
Editar
python -m venv venv
.\venv\Scripts\activate     # En Windows
Instala las dependencias:

bash
Copiar
Editar
pip install -r requirements.txt
🛠️ Configuración
Edita el archivo configs/config.yaml para cambiar hiperparámetros como el modelo, batch size, epochs, etc.

yaml
Copiar
Editar
model_name: resnet18
batch_size: 64
epochs: 20
learning_rate: 0.0003
num_classes: 10
input_size: 224
checkpoint_path: checkpoints/landcover_model.pth
use_gpu: true
🚀 Entrenamiento
Una vez tengas el dataset descargado:

bash
Copiar
Editar
python main.py
Esto:

Divide el dataset en entrenamiento y validación

Entrena el modelo

Guarda checkpoints

Registra métricas en TensorBoard

📊 Visualización
bash
Copiar
Editar
tensorboard --logdir runs/
Abre tu navegador en http://localhost:6006

✅ Resultado Final
Accuracy y F1 por época

Matriz de confusión

Modelo entrenado en checkpoints/

Listo para usar en APIs, dashboards o investigación

🧠 Clases del Dataset
AnnualCrop

Forest

HerbaceousVegetation

Highway

Industrial

Pasture

PermanentCrop

Residential

River

SeaLake