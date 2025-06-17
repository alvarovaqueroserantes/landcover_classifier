import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.resnet_landcover import get_model
from utils.dataset import get_dataloaders
from utils.helpers import seed_everything, load_checkpoint_if_available
from utils.metrics import evaluate_model
from train import train_one_epoch
from test import test_model

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config("configs/config.yaml")
    seed_everything(42)

    device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Data loaders
    train_loader, val_loader = get_dataloaders(
        batch_size=config["batch_size"],
        input_size=config["input_size"]
    )

    # Model
    model = get_model(config["model_name"], config["num_classes"])
    model.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Checkpoint
    start_epoch = load_checkpoint_if_available(model, optimizer, config["checkpoint_path"])

    # TensorBoard
    writer = SummaryWriter()

    # Training loop
    for epoch in range(start_epoch, config["epochs"]):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer)
        evaluate_model(model, val_loader, device, writer, epoch)

        # Save checkpoint
        os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, config["checkpoint_path"])

    writer.close()

    # Final evaluation
    print("✅ Final Evaluation on Validation Set:")
    test_model(model, val_loader, device)

if __name__ == "__main__":
    main()
