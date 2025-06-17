import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter

def evaluate_model(model, dataloader, device, writer: SummaryWriter, epoch: int):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcular métricas
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"[Validation] Epoch {epoch+1} | Accuracy: {acc:.4f} | F1 Score (macro): {f1:.4f}")

    # Loguear en TensorBoard
    writer.add_scalar("Accuracy/val", acc * 100, epoch)
    writer.add_scalar("F1_score/val", f1, epoch)

    return acc, f1
