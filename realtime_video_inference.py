import cv2
import torch
import numpy as np
import os
from torchvision import transforms
from models.resnet_landcover import get_model
import yaml
from PIL import Image

# === Cargar configuración ===
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Cargar clases ===
class_names = sorted(os.listdir("data/EuroSAT"))

# === Etiquetas personalizadas (abreviadas)
custom_labels = {
    "AnnualCrop": "Crop",
    "Forest": "For",
    "HerbaceousVegetation": "Veg",
    "Highway": "Hwy",
    "Industrial": "Ind",
    "Pasture": "Pas",
    "PermanentCrop": "PCrp",
    "Residential": "Res",
    "River": "Riv",
    "SeaLake": "Sea"
}

# === Dispositivo ===
device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")

# === Cargar modelo entrenado ===
model = get_model(config["model_name"], config["num_classes"])
model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device)["model_state"])
model.to(device)
model.eval()

# === Transformaciones ===
transform = transforms.Compose([
    transforms.Resize((config["input_size"], config["input_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# === Captura de video ===
cap = cv2.VideoCapture(r"C:\Users\alvar\Downloads\225862_small.mp4")
grid_size = config["input_size"]  # tamaño del bloque (ej. 224)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    output_frame = frame.copy()

    # === Sliding window ===
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            crop = frame[y:y+grid_size, x:x+grid_size]

            if crop.shape[0] != grid_size or crop.shape[1] != grid_size:
                continue  # ignorar bloques incompletos

            # Preprocesamiento
            image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            # Inferencia
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                label = class_names[pred.item()]
                short_label = custom_labels.get(label, label[:3])  # abreviado

            # Dibujar resultado
            cv2.rectangle(output_frame, (x, y), (x + grid_size, y + grid_size), (0, 255, 0), 1)
            cv2.putText(output_frame, short_label, (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Mostrar en ventana
    cv2.imshow("Grid Classification", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
