import torch.nn as nn
import torchvision.models as models

def get_model(model_name: str, num_classes: int):
    """
    Devuelve un modelo CNN preentrenado adaptado al número de clases deseado.

    Args:
        model_name (str): Nombre del modelo base (e.g., 'resnet18').
        num_classes (int): Número de clases de salida.

    Returns:
        nn.Module: Modelo de clasificación listo para entrenar.
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    return model
