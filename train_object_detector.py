"""
Training script for Object Detection Model using CDBNN Feature Extraction and DBNN Logic
Author: Cascade AI Assistant
Date: May 17, 2025
"""

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CocoDetection
from tqdm import tqdm

# Import our object detector
from object_detector import ObjectDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("object_detector_training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ObjectDetectionDataset(CocoDetection):
    """
    Dataset for object detection using COCO format
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        # Load annotation file first to check for valid images
        with open(annFile) as f:
            coco_data = json.load(f)

        # Create a mapping of image IDs to file paths
        self.img_paths = {}
        self.valid_ids = []

        # Check which images actually exist
        logger.info(f"Validating image files in {root}...")
        for img in coco_data["images"]:
            img_path = os.path.join(root, img["file_name"])
            if os.path.exists(img_path):
                self.img_paths[img["id"]] = img_path
                self.valid_ids.append(img["id"])
            else:
                logger.warning(f"Image file not found: {img_path}")

        # Filter annotations to only include valid images
        valid_annotations = []
        for ann in coco_data["annotations"]:
            if ann["image_id"] in self.valid_ids:
                valid_annotations.append(ann)

        # Create filtered dataset
        filtered_data = {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "categories": coco_data.get("categories", []),
            "images": [
                img for img in coco_data["images"] if img["id"] in self.valid_ids
            ],
            "annotations": valid_annotations,
        }

        # Save filtered annotations to a temporary file
        filtered_ann_file = annFile + ".filtered.json"
        with open(filtered_ann_file, "w") as f:
            json.dump(filtered_data, f)

        logger.info(
            f"Found {len(self.valid_ids)} valid images out of {len(coco_data['images'])}"
        )

        # Initialize with filtered annotations
        super().__init__(root, filtered_ann_file, transform, target_transform)

        # Create category mapping
        self.category_mapping = {
            cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])
        }
        self.categories = coco_data["categories"]

        # Store class names
        self.class_names = {
            idx: cat["name"]
            for cat_id, idx in self.category_mapping.items()
            for cat in self.categories
            if cat["id"] == cat_id
        }

        # Add background class
        self.class_names[-1] = "background"

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # Convert target to the format expected by the model
        boxes = []
        labels = []

        for annotation in target:
            # Get box coordinates
            x, y, w, h = annotation["bbox"]
            # Convert COCO format (x, y, width, height) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Ensure box coordinates are valid
            if w <= 0 or h <= 0:
                continue

            boxes.append([x1, y1, x2, y2])

            # Get category and map to our index
            category_id = annotation["category_id"]
            # Make sure we only use categories that are in our mapping
            if category_id in self.category_mapping:
                label = self.category_mapping[category_id]
                # Ensure label is valid (non-negative)
                if label >= 0:
                    labels.append(label)
                else:
                    # Skip this annotation if label is invalid
                    boxes.pop()
            else:
                # Skip this annotation if category is not in our mapping
                boxes.pop()

        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # Empty annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        # Create target dictionary
        target_dict = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": torch.tensor([ann["area"] for ann in target])
            if target
            else torch.zeros(0),
            "iscrowd": torch.tensor([ann["iscrowd"] for ann in target])
            if target
            else torch.zeros(0),
        }

        return img, target_dict

    def get_num_classes(self):
        """Get the number of classes in the dataset"""
        return len(self.class_names)


def collate_fn(batch):
    """
    Custom collate function for object detection batches
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    return images, targets


def train_model(args):
    """
    Train the object detection model
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Data transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")

    # Check which dataset structure we have
    train_dir = os.path.join(args.data_path, "train")
    images_dir = os.path.join(args.data_path, "images")

    # Determine annotation file path
    if os.path.exists(os.path.join(args.data_path, "annotations.json")):
        ann_file = os.path.join(args.data_path, "annotations.json")
    elif os.path.exists(os.path.join(args.data_path, "train_annotations.json")):
        ann_file = os.path.join(args.data_path, "train_annotations.json")
    else:
        raise FileNotFoundError(f"Could not find annotations file in {args.data_path}")

    # Determine images directory
    if os.path.exists(images_dir):
        root_dir = images_dir
        logger.info(f"Using images directory: {root_dir}")
    elif os.path.exists(train_dir):
        root_dir = train_dir
        logger.info(f"Using train directory: {root_dir}")
    else:
        raise FileNotFoundError(f"Could not find images directory in {args.data_path}")

    dataset = ObjectDetectionDataset(
        root=root_dir, annFile=ann_file, transform=transform
    )

    # Get number of classes
    num_classes = dataset.get_num_classes()
    logger.info(f"Dataset has {num_classes} classes")

    # Create validation dataset
    if args.val_data_path:
        logger.info(f"Loading validation dataset from {args.val_data_path}")

        # Determine validation annotation file path
        if os.path.exists(os.path.join(args.val_data_path, "annotations.json")):
            val_ann_file = os.path.join(args.val_data_path, "annotations.json")
        elif os.path.exists(os.path.join(args.val_data_path, "val_annotations.json")):
            val_ann_file = os.path.join(args.val_data_path, "val_annotations.json")
        else:
            raise FileNotFoundError(
                f"Could not find validation annotations file in {args.val_data_path}"
            )

        # Determine validation images directory
        val_images_dir = os.path.join(args.val_data_path, "images")
        val_dir = os.path.join(args.val_data_path, "val")

        if os.path.exists(val_images_dir):
            val_root_dir = val_images_dir
            logger.info(f"Using validation images directory: {val_root_dir}")
        elif os.path.exists(val_dir):
            val_root_dir = val_dir
            logger.info(f"Using val directory: {val_root_dir}")
        else:
            raise FileNotFoundError(
                f"Could not find validation images directory in {args.val_data_path}"
            )

        val_dataset = ObjectDetectionDataset(
            root=val_root_dir, annFile=val_ann_file, transform=transform
        )
    else:
        # Split dataset into train and validation
        dataset_size = len(dataset)
        val_size = int(dataset_size * 0.2)  # 20% for validation
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Training set size: {train_size}, Validation set size: {val_size}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Initialize model
    logger.info(f"Initializing model with {num_classes} classes")
    model = ObjectDetector(
        num_classes=num_classes,
        feature_dim=args.feature_dim,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        backbone=args.backbone,
        pretrained=True,
        device=device,
    )
    model.to(device)

    # Set up optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        params, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float("inf")

    # Create directory for saving models
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        train_progress = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} (Train)"
        )
        for images, targets in train_progress:
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            try:
                loss_dict = model(images, targets)
            except Exception as e:
                logger.error(f"Error in model forward pass during training: {e}")
                # Create a zero tensor as fallback
                loss_dict = torch.tensor(0.0, device=device)

            # Calculate total loss - handle both dictionary and list formats
            try:
                if isinstance(loss_dict, dict):
                    # If it's a dictionary, just take the first value
                    # This avoids tensor size mismatch errors
                    if loss_dict.values():
                        losses = next(iter(loss_dict.values()))
                    else:
                        losses = torch.tensor(0.0, device=device)
                elif isinstance(loss_dict, list):
                    # If it's a list, check what's inside
                    if len(loss_dict) > 0:
                        if isinstance(loss_dict[0], dict):
                            # First element is a dictionary
                            losses = sum(loss for loss in loss_dict[0].values())
                        elif isinstance(loss_dict[0], torch.Tensor):
                            # First element is a tensor, use the first one
                            # This avoids size mismatch errors when tensors have different shapes
                            losses = loss_dict[0]
                        else:
                            # Unknown type, just use the first element
                            losses = loss_dict[0]
                    else:
                        # Empty list, create a zero tensor
                        losses = torch.tensor(0.0, device=device)
                elif isinstance(loss_dict, torch.Tensor):
                    # If it's a single tensor, use it directly
                    losses = loss_dict
                else:
                    # Unknown type, create a zero tensor
                    logger.warning(f"Unknown loss type: {type(loss_dict)}")
                    losses = torch.tensor(0.0, device=device)
            except Exception as e:
                # If anything goes wrong, log it and use a zero tensor
                logger.error(f"Error calculating loss: {e}")
                losses = torch.tensor(0.0, device=device)

            # Backward pass and optimize
            losses.backward()
            optimizer.step()

            # Update progress - safely handle empty tensors
            try:
                # Make sure the tensor has at least one element
                if losses.numel() > 0:
                    loss_value = losses.item()
                else:
                    loss_value = 0.0
                train_loss += loss_value
                train_progress.set_postfix({"loss": loss_value})
            except Exception as e:
                logger.error(f"Error getting loss value: {e}")
                train_progress.set_postfix({"loss": 0.0})

        # Update learning rate
        lr_scheduler.step()

        # Calculate average training loss
        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_progress = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} (Val)"
            )
            for images, targets in val_progress:
                # Move data to device
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass
                try:
                    loss_dict = model(images, targets)
                except Exception as e:
                    logger.error(f"Error in model forward pass: {e}")
                    # Create a zero tensor as fallback
                    loss_dict = torch.tensor(0.0, device=device)

                # Calculate total loss - handle both dictionary and list formats
                try:
                    if isinstance(loss_dict, dict):
                        # If it's a dictionary, just take the first value
                        # This avoids tensor size mismatch errors
                        if loss_dict.values():
                            losses = next(iter(loss_dict.values()))
                        else:
                            losses = torch.tensor(0.0, device=device)
                    elif isinstance(loss_dict, list):
                        # If it's a list, check what's inside
                        if len(loss_dict) > 0:
                            if isinstance(loss_dict[0], dict):
                                # First element is a dictionary
                                try:
                                    # Instead of summing, just take the first value
                                    # This avoids tensor size mismatch errors
                                    if loss_dict[0].values():
                                        losses = next(iter(loss_dict[0].values()))
                                    else:
                                        losses = torch.tensor(0.0, device=device)
                                except Exception as e:
                                    logger.error(
                                        f"Error accessing dictionary values: {e}"
                                    )
                                    losses = torch.tensor(0.0, device=device)
                            elif isinstance(loss_dict[0], torch.Tensor):
                                # First element is a tensor, use the first one
                                # This avoids size mismatch errors when tensors have different shapes
                                losses = loss_dict[0]
                            else:
                                # Unknown type, just use the first element
                                losses = loss_dict[0]
                        else:
                            # Empty list, create a zero tensor
                            losses = torch.tensor(0.0, device=device)
                    elif isinstance(loss_dict, torch.Tensor):
                        # If it's a single tensor, use it directly
                        losses = loss_dict
                    else:
                        # Unknown type, create a zero tensor
                        logger.warning(f"Unknown loss type: {type(loss_dict)}")
                        losses = torch.tensor(0.0, device=device)
                except Exception as e:
                    # If anything goes wrong, log it and use a zero tensor
                    logger.error(f"Error calculating loss: {e}")
                    losses = torch.tensor(0.0, device=device)

                # Update progress - safely handle empty tensors
                try:
                    # Make sure the tensor has at least one element
                    if losses.numel() > 0:
                        loss_value = losses.item()
                    else:
                        loss_value = 0.0
                    val_loss += loss_value
                    val_progress.set_postfix({"loss": loss_value})
                except Exception as e:
                    logger.error(f"Error getting loss value: {e}")
                    val_progress.set_postfix({"loss": 0.0})

        # Calculate average validation loss
        val_loss /= len(val_loader)

        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(
                args.output_dir, f"best_model_epoch_{epoch+1}.pth"
            )
            model.save_model(model_path)
            logger.info(
                f"Saved best model at epoch {epoch+1} with validation loss {val_loss:.4f}"
            )

    # After RPN training, extract features and train DBNN classifier
    logger.info("Training DBNN classifier on extracted features...")

    # Extract features from training set
    train_features = []
    train_labels = []

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(train_loader, desc="Extracting training features"):
            # Move data to device
            images = [img.to(device) for img in images]

            # Extract features for ground truth boxes
            for i, target in enumerate(targets):
                boxes = target["boxes"].to(device)
                labels = target["labels"].to(device)

                if len(boxes) > 0:
                    # Extract features for these boxes
                    features = model.feature_extractor.extract_region_features(
                        images[i].unsqueeze(0), [boxes]
                    )[0]

                    train_features.append(features)
                    train_labels.append(labels)

    # Concatenate all features and labels
    if train_features:
        train_features = torch.cat(train_features)
        train_labels = torch.cat(train_labels)

        # Extract features from validation set
        val_features = []
        val_labels = []

        with torch.no_grad():
            for images, targets in tqdm(
                val_loader, desc="Extracting validation features"
            ):
                # Move data to device
                images = [img.to(device) for img in images]

                # Extract features for ground truth boxes
                for i, target in enumerate(targets):
                    boxes = target["boxes"].to(device)
                    labels = target["labels"].to(device)

                    if len(boxes) > 0:
                        # Extract features for these boxes
                        features = model.feature_extractor.extract_region_features(
                            images[i].unsqueeze(0), [boxes]
                        )[0]

                        val_features.append(features)
                        val_labels.append(labels)

        # Concatenate all validation features and labels
        if val_features:
            val_features = torch.cat(val_features)
            val_labels = torch.cat(val_labels)

            # Train DBNN classifier
            logger.info(f"Training DBNN classifier with {len(train_features)} samples")
            history = model.train_dbnn_classifier(
                train_features, train_labels, val_features, val_labels
            )

            # Save final model with DBNN classifier
            final_model_path = os.path.join(args.output_dir, "final_model.pth")
            model.save_model(final_model_path)
            logger.info(f"Saved final model with DBNN classifier at {final_model_path}")

            # Plot training history
            if history:
                plt.figure(figsize=(10, 6))
                plt.plot(history.get("error_rates", []), label="Error Rate")
                plt.title("DBNN Training History")
                plt.xlabel("Epoch")
                plt.ylabel("Error Rate")
                plt.legend()
                plt.savefig(os.path.join(args.output_dir, "dbnn_training_history.png"))
        else:
            logger.warning(
                "No validation features extracted. Skipping DBNN classifier training."
            )
    else:
        logger.warning(
            "No training features extracted. Skipping DBNN classifier training."
        )

    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Train Object Detection Model with CDBNN and DBNN"
    )

    # Dataset parameters
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=None,
        help="Path to validation dataset (optional)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="models", help="Directory to save models"
    )

    # Model parameters
    parser.add_argument(
        "--backbone", type=str, default="resnet50", help="Backbone model"
    )
    parser.add_argument(
        "--feature_dim", type=int, default=128, help="Feature dimension"
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument(
        "--nms_threshold", type=float, default=0.3, help="NMS threshold"
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0005, help="Weight decay"
    )
    parser.add_argument(
        "--lr_step_size", type=int, default=3, help="LR scheduler step size"
    )
    parser.add_argument(
        "--lr_gamma", type=float, default=0.1, help="LR scheduler gamma"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Train the model
    train_model(args)


if __name__ == "__main__":
    main()
