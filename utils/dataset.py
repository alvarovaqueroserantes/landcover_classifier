import os
import shutil
import random
from sklearn.model_selection import train_test_split

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

def prepare_train_val_split(base_dir="data/EuroSAT", train_dir="data/train", val_dir="data/test", val_split=0.2, seed=42):
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print("✅ Conjuntos train/test ya preparados.")
        return

    print("🛠️ Dividiendo datos en train/test...")

    random.seed(seed)
    classes = os.listdir(base_dir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for cls in classes:
        class_path = os.path.join(base_dir, cls)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        train_imgs, val_imgs = train_test_split(images, test_size=val_split, random_state=seed)

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, cls, img))
        for img in val_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, cls, img))

    print("✅ División completada.")

def get_albumentations_transform(input_size, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

class AlbumentationsDataset(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root)
        self.albumentations_transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = np.array(image)
        image = self.albumentations_transform(image=image)["image"]
        return image, label

def get_dataloaders(batch_size=64, input_size=224):
    prepare_train_val_split()

    train_transform = get_albumentations_transform(input_size, is_train=True)
    val_transform = get_albumentations_transform(input_size, is_train=False)

    train_dataset = AlbumentationsDataset("data/train", transform=train_transform)
    val_dataset = AlbumentationsDataset("data/test", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader
