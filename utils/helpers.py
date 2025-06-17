import os
import random
import numpy as np
import torch

def seed_everything(seed=42):
    """
    Establece todas las semillas para asegurar resultados reproducibles.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔁 Semilla fijada en {seed}")

def load_checkpoint_if_available(model, optimizer, checkpoint_path):
    """
    Carga un checkpoint si existe y restaura el modelo y optimizador.

    Returns:
        start_epoch (int): época desde la que continuar el entrenamiento
    """
    if os.path.exists(checkpoint_path):
        print(f"📦 Cargando checkpoint desde {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint.get("epoch", 0)
    else:
        print("ℹ️ No se encontró checkpoint. Entrenando desde cero.")
        return 0
