import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PrescriptionDataset(Dataset):
    """
    Dataset for handwritten medicine name recognition
    """

    def __init__(
        self,
        base_dir,
        split="Training",
        transform=None,
        augment=False
    ):
        """
        Args:
            base_dir (str): path to dataset folder
            split (str): Training / Validation / Testing
            transform: base preprocessing transforms
            augment (bool): whether to apply augmentation
        """

        self.base_dir = base_dir
        self.split = split
        self.split_path = os.path.join(base_dir, split)

        # Load CSV
        self.csv_file = self._find_csv()
        self.df = pd.read_csv(self.csv_file)

        # Find image folder
        self.image_dir = self._find_image_folder()

        # Base transform (always applied)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((64, 256)),   # OCR standard size
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        # Optional augmentation (only for training)
        self.augment = augment and split == "Training"

        self.augmentation = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

        print(f"\n[INFO] Loaded {split} dataset")
        print(f"CSV: {self.csv_file}")
        print(f"Images: {self.image_dir}")
        print(f"Samples: {len(self.df)}")
        print(f"Augmentation: {self.augment}")

    def _find_csv(self):
        """Find CSV file automatically"""
        for file in os.listdir(self.split_path):
            if file.endswith(".csv"):
                return os.path.join(self.split_path, file)
        raise FileNotFoundError(f"No CSV found in {self.split_path}")

    def _find_image_folder(self):
        """Find image folder automatically"""
        for folder in os.listdir(self.split_path):
            full_path = os.path.join(self.split_path, folder)
            if os.path.isdir(full_path):
                return full_path
        raise FileNotFoundError(f"No image folder found in {self.split_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row["IMAGE"]
        label = str(row["MEDICINE_NAME"]).lower().strip()

        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("L")  # instead of RGB

        # Apply augmentation (only training)
        if self.augment:
            image = self.augmentation(image)

        # Apply base transform
        image = self.transform(image)

        return {
            "image": image,          # tensor
            "label": label,          # string
            "image_path": img_path
        }
    
    