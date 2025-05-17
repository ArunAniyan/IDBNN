import json
import os
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# Add this import at the top
from .augmentations import get_augmentation_pipeline


class DetectionDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (800, 800),
        augment: bool = True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment and split == "train"
        self.annotations = self._load_annotations()

        # Get augmentation pipeline
        self.transform = get_augmentation_pipeline(train=self.augment)

    def _load_annotations(self) -> List[Dict]:
        ann_path = os.path.join(self.root_dir, f"annotations_{self.split}.json")
        with open(ann_path) as f:
            annotations = json.load(f)
        return annotations["annotations"]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        ann = self.annotations[idx]
        img_path = os.path.join(self.root_dir, "images", ann["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Convert boxes to tensor
        boxes = torch.as_tensor(ann["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(ann["labels"], dtype=torch.int64)

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        # Apply transformations
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target


class DetectionDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (800, 800),
        augment: bool = True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment and split == "train"
        self.annotations = self._load_annotations()
        self.transform = self._get_transforms()

    def _load_annotations(self) -> List[Dict]:
        ann_path = os.path.join(self.root_dir, f"annotations_{self.split}.json")
        with open(ann_path) as f:
            annotations = json.load(f)
        return annotations["annotations"]

    def _get_transforms(self):
        if self.augment:
            return T.Compose(
                [
                    T.Resize(self.image_size),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            return T.Compose(
                [
                    T.Resize(self.image_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        ann = self.annotations[idx]
        img_path = os.path.join(self.root_dir, "images", ann["file_name"])
        image = Image.open(img_path).convert("RGB")

        boxes = torch.as_tensor(ann["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(ann["labels"], dtype=torch.int64)

        image = self.transform(image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# In train.py - Update the get_dataloaders function:


def get_dataloaders(
    root_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (800, 800),
) -> Dict[str, DataLoader]:
    # Training dataset with augmentations
    train_dataset = DetectionDataset(
        root_dir=root_dir,
        split="train",
        image_size=image_size,
        augment=True,  # Enable augmentations
    )

    # Validation dataset without augmentations
    val_dataset = DetectionDataset(
        root_dir=root_dir,
        split="val",
        image_size=image_size,
        augment=False,  # Disable augmentations
    )

    # Training data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # Drop last incomplete batch
    )

    # Validation data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Typically use batch size 1 for validation
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return {"train": train_loader, "val": val_loader}
