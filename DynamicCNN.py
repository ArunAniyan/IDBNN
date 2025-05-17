import argparse
import json
import os
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from astropy.io import fits
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------Fully Functional version : April 28 7:42 am---------------------------------------------
# ---- author : Ninan Sajeeth Philip, Artificial Intelligence Research and Intelligent Systems
# updated and made suitable for different types and sizes of data. April 29 7:44 pm
# udated 8:56 pm
# -------------------------------------------------------------------------------------------------------------------------------
# --------------------------
# Dataset Handling Components
# --------------------------
class CustomDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        target_size=None,
        mode="train",
        class_metadata=None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.mode = mode
        self.class_metadata = class_metadata
        # Add this for prediction mode
        if mode == "predict" and class_metadata:
            self.known_classes = set(class_metadata["classes"])
            self.class_to_idx = class_metadata["class_to_idx"]
        if mode in ["train", "val"]:
            self.classes, self.class_to_idx = self._find_classes()
            self.samples = self._make_dataset()
        else:
            self.samples = self._load_prediction_samples()

    def _find_classes(self):
        # Step 1: Find the first directory containing images
        image_dir = None
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if self._is_valid_file(os.path.join(root, file)):
                    image_dir = root
                    break
            if image_dir is not None:
                break

        if image_dir is None:
            raise FileNotFoundError(f"No images found in {self.root_dir}")

        # Step 2: Get parent directory
        parent_dir = os.path.dirname(image_dir)

        # Step 3: Collect subdirectories of parent_dir that contain images
        classes = []
        for entry in os.scandir(parent_dir):
            if entry.is_dir():
                # Check if this subdirectory contains any images
                has_images = False
                for root_sub, _, files_sub in os.walk(entry.path):
                    for file_sub in files_sub:
                        if self._is_valid_file(os.path.join(root_sub, file_sub)):
                            has_images = True
                            break
                    if has_images:
                        classes.append(entry.name)
                        break

        if not classes:
            raise FileNotFoundError(f"No valid class directories found in {parent_dir}")

        # Update root_dir to parent_dir to reflect the correct dataset structure
        self.root_dir = parent_dir

        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self):
        instances = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.root_dir, target_class)
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if self._is_valid_file(path):
                        item = (path, class_index)
                        instances.append(item)
        return instances

    def _load_prediction_samples(self):
        samples = []
        for root, _, files in os.walk(self.root_dir):
            for fname in files:
                path = os.path.join(root, fname)
                if self._is_valid_file(path):
                    # Determine label from path structure
                    label = self._find_label_from_path(path)
                    samples.append((path, label))
        return samples

    def _find_label_from_path(self, path):
        if not self.class_metadata:
            return "dummy_label"

        # Check all parent directories for known classes
        current_dir = os.path.dirname(path)
        while True:
            dir_name = os.path.basename(current_dir)
            if dir_name in self.known_classes:
                return dir_name
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached root
                break
            current_dir = parent_dir

        return "dummy_label"

    def _is_valid_file(self, path):
        valid_extensions = (".png", ".jpg", ".jpeg", ".dcm", ".fits", ".fit")
        return path.lower().endswith(valid_extensions)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.mode in ["train", "val"]:
            path, target = self.samples[index]
            img = self._load_image(path)
            if self.transform:
                img = self.transform(img)
            return img, target, path
        else:
            try:
                path, target = self.samples[index]
            except:
                path = self.samples[index]
            img = self._load_image(path)
            if self.transform:
                img = self.transform(img)
            return img, path

    def _load_image(self, path):
        try:
            if path.lower().endswith(".dcm"):
                return self._load_dicom(path)
            elif path.lower().endswith((".fits", ".fit")):
                return self._load_fits(path)
            else:
                return self._load_standard_image(path)
        except Exception as e:
            raise RuntimeError(f"Error loading {path}: {str(e)}")

    def _load_dicom(self, path):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(float)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        return Image.fromarray(img.astype("uint8")).convert("L")

    def _load_fits(self, path):
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(float)
            data = (data - data.min()) / (data.max() - data.min()) * 255
            return Image.fromarray(data.astype("uint8")).convert("L")

    def _load_standard_image(self, path):
        img = Image.open(path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        elif img.mode not in ["RGB", "L"]:
            img = img.convert("L")
        return img


# --------------------------
# Model Components
# --------------------------
def calculate_max_depth(H, W):
    """Calculate maximum layers before spatial collapse."""
    max_depth = 0
    current_H, current_W = H, W
    while current_H >= 2 and current_W >= 2:  # Ensure at least 2x2 after pooling
        current_H = current_H // 2  # MaxPool2d(2) reduces size
        current_W = current_W // 2
        max_depth += 1
    return max_depth


class DynamicJNet(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        depth=3,
        initial_filters=32,
        input_size=(224, 224),
        skip_connections=True,
        reduced_dim=128,
    ):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.skip_connections = skip_connections
        self.depth = depth
        self.input_size = input_size

        # Precompute encoder spatial sizes
        self.encoder_spatial_sizes = []
        current_H, current_W = input_size
        self.encoder_spatial_sizes.append((current_H, current_W))

        # Encoder
        current_channels = in_channels
        for i in range(depth):
            out_channels = initial_filters * (2**i)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            current_channels = out_channels
            current_H = current_H // 2
            current_W = current_W // 2
            self.encoder_spatial_sizes.append((current_H, current_W))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_channels, reduced_dim, 3, padding=1),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(inplace=True),
        )

        # Decoder with dynamic skip connections
        current_channels = reduced_dim
        for i in reversed(range(depth - 1)):  # One less layer than encoder
            out_channels = initial_filters * (2**i)
            if skip_connections:
                decoder_in = current_channels + initial_filters * (2**i)
            else:
                decoder_in = current_channels
            self.decoder_convs.append(
                nn.Sequential(
                    nn.Conv2d(decoder_in, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            current_channels = out_channels

        # Final classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)

    def forward(self, x):
        skip_features = []

        # Encoder path
        for layer in self.encoder:
            x = layer(x)
            skip_features.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i in range(len(self.decoder_convs)):
            target_size = self.encoder_spatial_sizes[-(i + 2)]
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            if self.skip_connections and i < len(skip_features) - 1:
                x = torch.cat([x, skip_features[-(i + 2)]], dim=1)
            x = self.decoder_convs[i](x)

        # Classification
        x = self.adaptive_pool(x).squeeze()
        return self.classifier(x), x, 0


class DynamicCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        depth=3,
        initial_filters=32,
        input_size=(28, 28),
        bottleneck_dim=128,  # New: Target feature dimension
        sparsity_weight=0.01,  # New: Weight for L1 regularization
        initial_kernel_type="standard",  # Add parameter here
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels

        # Add kernel adaptation
        if initial_kernel_type == "diagonal_enhanced":
            self._initialize_diagonal_kernels()

        # Automatically calculate and limit depth
        H, W = input_size
        self.max_depth = calculate_max_depth(H, W)
        self.depth = min(depth, self.max_depth)  # Use smaller of config vs max

        # Dynamic convolutional layers
        for i in range(self.depth):
            out_channels = initial_filters * (2**i)
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            current_channels = out_channels

        # Adaptive components
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Feature bottleneck (add this after adaptive pooling)
        self.bottleneck = nn.Sequential(
            nn.Linear(current_channels, min(bottleneck_dim, 128)),  # Enforce ceiling
            nn.BatchNorm1d(min(bottleneck_dim, 128)),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(
            bottleneck_dim if bottleneck_dim else current_channels, num_classes
        )

        # Sparsity regularization
        self.sparsity_weight = sparsity_weight
        self.feature_dim = bottleneck_dim if bottleneck_dim else current_channels

        # Warn if depth was reduced
        if self.depth < depth:
            print(
                f"⚠️ Adjusted depth from {depth} to {self.depth} for input size {input_size}"
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # Apply bottleneck
        x = self.bottleneck(x)

        # Sparsity: Add L1 penalty to bottleneck output
        sparsity_loss = torch.norm(x, p=1, dim=1).mean() * self.sparsity_weight

        logits = self.classifier(x)
        return logits, x, sparsity_loss  # Return sparsity loss

    def _initialize_diagonal_kernels(self):
        """Custom kernel init for diagonal feature detection"""
        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer[0], nn.Conv2d):
                    # Create diagonal edge detectors
                    diagonal_filter = torch.tensor(
                        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
                    ).repeat(1, 1, 1, 1)
                    layer[0].weight.data = diagonal_filter


# --------------------------
# Training Utilities
# --------------------------
def kl_divergence_loss(features, labels, eps=1e-6):
    unique_labels = torch.unique(labels)
    if len(unique_labels) < 2:
        return torch.tensor(0.0, device=features.device)

    # Normalize features to stabilize variance
    features = F.normalize(features, p=2, dim=1)  # L2 normalization

    class_stats = []
    for lbl in unique_labels:
        mask = labels == lbl
        cls_feat = features[mask]
        if cls_feat.size(0) == 0:
            continue
        mean = cls_feat.mean(dim=0)
        var = cls_feat.var(dim=0, unbiased=False) + eps  # Increased epsilon
        class_stats.append((mean, var))

    kl_loss = 0.0
    n_pairs = 0
    for i in range(len(class_stats)):
        for j in range(i + 1, len(class_stats)):
            mean_i, var_i = class_stats[i]
            mean_j, var_j = class_stats[j]

            # KL divergence with stability
            kl = 0.5 * (torch.sum(torch.log(var_j + eps) - torch.log(var_i + eps)))
            kl += 0.5 * torch.sum((var_i + (mean_i - mean_j) ** 2) / (var_j + eps))
            kl -= 0.5 * mean_i.size(0)

            kl_loss += kl
            n_pairs += 1

    return kl_loss / n_pairs if n_pairs > 0 else torch.tensor(0.0)


from tqdm import tqdm


# --------------------------
# Model Path Management
# --------------------------
def get_model_dir(config):
    return os.path.join(
        "data", config["dataset"]["name"], f"Model_{config['model']['type']}"
    )


def get_model_path(config):
    model_type = config["model"]["type"]
    return os.path.join(get_model_dir(config), f"best_{model_type}.pth")


# --------------------------
# Metadata Management Utilities
# --------------------------
def get_metadata_path(config):
    return os.path.join("data", config["dataset"]["name"], "class_metadata.json")


def prune_features(model, threshold="auto", verbose=True):
    """Prune features with dynamic threshold adjustment"""
    with torch.no_grad():
        for name, param in model.bottleneck.named_parameters():
            if "weight" in name:
                # Always calculate absolute weights first
                abs_weights = torch.abs(param.data)  # <-- Moved outside conditionals

                # Calculate dynamic threshold if needed
                if threshold == "auto":
                    threshold_val = torch.quantile(abs_weights, 0.25)
                else:
                    threshold_val = threshold

                # Calculate pruning statistics
                original_nonzero = torch.sum(abs_weights > 1e-6).item()
                mask = (abs_weights > threshold_val).float()

                # Apply pruning
                param.data *= mask

                # Calculate post-pruning stats
                new_nonzero = torch.sum(torch.abs(param.data) > 1e-6).item()
                sparsity = (
                    1 - new_nonzero / original_nonzero if original_nonzero > 0 else 0
                )

                if verbose:
                    print(f"\n🔧 Pruning {name}:")
                    print(f"   Threshold: {threshold_val:.4f}")
                    print(f"   Weights: {original_nonzero} → {new_nonzero}")
                    print(f"   Sparsity: {sparsity:.1%}")
                    print(f"   Shape: {param.shape}")

                return threshold_val  # Return actual threshold used


def validate_metadata(metadata, dataset):
    """Check if metadata matches current dataset structure"""
    # Get current mappings
    current_classes = sorted(dataset.class_to_idx.keys())  # Ensure sorted comparison
    current_mapping = dataset.class_to_idx

    # Check if metadata classes match sorted current classes
    if metadata["classes"] != current_classes:
        return False

    # Check if class_to_idx matches exactly
    if metadata["class_to_idx"] != current_mapping:
        return False

    return True


def save_metadata(config, dataset):
    metadata_path = get_metadata_path(config)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # Generate sorted class list from class_to_idx keys
    sorted_classes = sorted(dataset.class_to_idx.keys())
    metadata = {
        "classes": sorted_classes,
        "class_to_idx": dataset.class_to_idx,  # Directly use the dataset's mapping
        "dataset_version": str(os.path.getmtime(dataset.root_dir)),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def load_metadata(config):
    metadata_path = get_metadata_path(config)
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path) as f:
        return json.load(f)


# --------------------------
# Training Utilities (Add this new function)
# --------------------------
def calculate_confusion_weights(confusion_matrix):
    """Calculate class weights based on confusion patterns"""
    # Add device-aware operations
    device = confusion_matrix.device
    row_sums = confusion_matrix.sum(dim=1, keepdim=True)
    row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
    normalized = confusion_matrix / row_sums
    weights = 1.0 - torch.diag(normalized)
    return (weights / weights.sum() * weights.numel()).to(device)


def train(model, train_loader, val_loader, config, device, full_dataset):
    model_dir = get_model_dir(config)
    os.makedirs(model_dir, exist_ok=True)
    model_path = get_model_path(config)
    # Try to load existing model
    if os.path.exists(model_path):
        print(f"🔁 Found existing model at {model_path}, loading weights")
        model.load_state_dict(torch.load(model_path))
        prune_features(model, threshold=0.1)  # Prune existing model immediately

    # Metadata handling
    existing_metadata = load_metadata(config)

    if existing_metadata:
        # Validate against current dataset
        is_valid = validate_metadata(existing_metadata, full_dataset)

        # Additional validation using directory modification time
        current_version = str(os.path.getmtime(full_dataset.root_dir))
        if existing_metadata.get("dataset_version") != current_version:
            is_valid = False

        if not is_valid:
            print("⚠️ Existing metadata outdated. Generating new version.")
            metadata = save_metadata(config, full_dataset)
        else:
            print("✅ Using existing validated metadata")
            metadata = existing_metadata
    else:
        print("🆕 Creating new class metadata")
        metadata = save_metadata(config, full_dataset)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training_params"]["learning_rate"]
    )
    # criterion = nn.CrossEntropyLoss()
    metric = config["training_params"]["early_stopping_metric"]
    if metric == "loss":
        best_metric = float("inf")
    else:
        best_metric = 0
    patience_counter = 0
    early_stop = False
    metrics = {
        "epoch": 0,
        "tr_loss": float("inf"),
        "tr_acc": 0.0,
        "val_loss": float("inf"),
        "val_acc": 0.0,
        "best_metric": 0.0,
        "patience": 0,
    }

    progress_bar = tqdm(
        desc="Training Progress",
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
        postfix={
            k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()
        },
    )

    epoch = 0
    while epoch < config["training_params"]["max_epochs"] and not early_stop:
        epoch += 1
        metrics["epoch"] = epoch
        metrics["patience"] = config["training_params"]["patience"] - patience_counter

        # Training phase
        num_classes = len(full_dataset.classes)  # Get actual number of classes
        confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        train_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            leave=False,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
        )

        for inputs, labels, _ in train_iter:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            # Forward pass
            outputs, features, sparsity_loss = model(inputs)

            # Custom CE loss calculation (per-class mean)
            per_sample_loss = F.cross_entropy(outputs, labels, reduction="none")
            unique_labels = torch.unique(labels)
            if len(unique_labels) == 0:
                ce_loss = torch.tensor(0.0, device=device)
            else:
                class_losses = []
                for lbl in unique_labels:
                    mask = labels == lbl
                    class_loss = per_sample_loss[mask].mean()
                    class_losses.append(class_loss)
                ce_loss = (
                    torch.sum(torch.stack(class_losses))
                    if class_losses
                    else torch.tensor(0.0)
                )

            kl_loss = kl_divergence_loss(features, labels)
            loss = (
                ce_loss
                + config["training_params"]["kl_weight"] * kl_loss
                + sparsity_loss
            )

            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_iter.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{(predicted == labels).sum().item()/labels.size(0)*100:.2f}%",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        # Auto-adjust KL weights for confused classes
        class_kl_weights = calculate_confusion_weights(confusion_matrix).to(device)
        loss = ce_loss + (class_kl_weights[labels] * kl_loss).mean() + sparsity_loss

        # Calculate epoch metrics
        metrics["tr_loss"] = total_loss / len(train_loader)
        metrics["tr_acc"] = 100 * correct / total

        # Add periodic pruning during training
        if config["training_params"].get("prune_during_training", False):
            if epoch % 5 == 0:  # Prune every 5 epochs
                current_threshold = prune_features(
                    model,
                    threshold=config["model"].get("prune_threshold", "auto"),
                    verbose=True,
                )
                config["model"]["actual_prune_threshold"] = float(current_threshold)

        # Validation phase
        val_acc, val_loss = validate(model, val_loader, device)  # Removed criterion
        metrics["val_loss"] = val_loss
        metrics["val_acc"] = val_acc

        # Early stopping logic
        current_metric = (
            val_acc
            if config["training_params"]["early_stopping_metric"] == "accuracy"
            else val_loss
        )
        if metric == "loss":
            if current_metric < best_metric:
                best_metric = current_metric
                metrics["best_metric"] = (
                    val_acc
                    if config["training_params"]["early_stopping_metric"] == "accuracy"
                    else val_loss
                )
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    f"data/{config['dataset']['name']}/Model/best_model.pth",
                )
            else:
                patience_counter += 1
                if patience_counter >= config["training_params"]["patience"]:
                    early_stop = True
                    progress_bar.set_postfix_str(
                        "Early stopping triggered", refresh=True
                    )

        else:
            if current_metric > best_metric:
                best_metric = current_metric
                metrics["best_metric"] = (
                    val_acc
                    if config["training_params"]["early_stopping_metric"] == "accuracy"
                    else val_loss
                )
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    f"data/{config['dataset']['name']}/Model/best_model.pth",
                )
            else:
                patience_counter += 1
                if patience_counter >= config["training_params"]["patience"]:
                    early_stop = True
                    progress_bar.set_postfix_str(
                        "Early stopping triggered", refresh=True
                    )

        # Update progress bar
        progress_bar.set_postfix(
            {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()}
        )
        progress_bar.update(1)

        # Print epoch summary
        tqdm.write(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {metrics['tr_loss']:.4f} | Acc: {metrics['tr_acc']:.2f}% | "
            f"Val Loss: {metrics['val_loss']:.4f} | Acc: {metrics['val_acc']:.2f}% | "
            f"Patience: {patience_counter}/{config['training_params']['patience']}"
        )

    progress_bar.close()
    print("\nGenerating final artifacts...")

    class_metadata = {
        "classes": full_dataset.classes,
        "class_to_idx": full_dataset.class_to_idx,
    }
    metadata_path = os.path.join(
        "data", config["dataset"]["name"], "class_metadata.json"
    )
    with open(metadata_path, "w") as f:
        json.dump(class_metadata, f, indent=2)

    # Load best model for feature extraction
    model.load_state_dict(
        torch.load(f"data/{config['dataset']['name']}/Model/best_model.pth")
    )
    print("\nApplying final feature pruning...")
    prune_features(model, threshold=0.1)

    # Final pruning and saving
    print("\n🏁 Final model pruning:")
    final_threshold = prune_features(
        model, threshold=config["model"].get("prune_threshold", "auto"), verbose=True
    )

    # Save pruned model version
    pruned_model_path = f"data/{config['dataset']['name']}/Model/pruned_model.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "prune_threshold": final_threshold,
            "config": config,
        },
        pruned_model_path,
    )
    print(f"💾 Saved pruned model to {pruned_model_path}")

    # Extract features from training set
    train_features, train_labels, train_paths = extract_features(
        model, train_loader, device
    )
    # Create CSV file path
    csv_path = os.path.join(
        "data", config["dataset"]["name"], f"{config['dataset']['name']}.csv"
    )

    # Save to CSV
    save_features_to_csv(
        train_features, train_labels, train_paths, csv_path, class_metadata=metadata
    )

    # Update config with CSV info
    config["file_path"] = csv_path
    config["column_names"] = ["filepath", "target"] + [
        f"feature_{i}" for i in range(train_features.shape[1])
    ]

    # Update config to use pruned model
    config["dataset"]["model_path"] = pruned_model_path

    # Save final config
    config_path = os.path.join("data", config["dataset"]["name"], "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Artifacts generated:\n- {csv_path}\n- {config_path}")

    # Create .conf file
    conf_path = os.path.join(
        "data", config["dataset"]["name"], f"{config['dataset']['name']}.conf"
    )
    num_features = train_features.shape[1]
    column_names = [f"feature_{i}" for i in range(num_features)] + ["target"]

    conf_data = {
        "file_path": csv_path,
        "column_names": column_names,
        "separator": ",",
        "has_header": True,
        "target_column": "target",
        "modelType": "Histogram",
        "feature_group_size": 2,
        "max_combinations": 10000,
        "bin_sizes": [128],
        "active_learning": {
            "tolerance": 1.0,
            "cardinality_threshold_percentile": 95,
            "strong_margin_threshold": 0.3,
            "marginal_margin_threshold": 0.1,
            "min_divergence": 0.1,
        },
        "training_params": {
            "trials": 100,
            "epochs": 1000,
            "learning_rate": 0.001,
            "batch_size": 128,
            "test_fraction": 0.2,
            "random_seed": 42,
            "minimum_training_accuracy": 0.95,
            "cardinality_threshold": 0.9,
            "cardinality_tolerance": 4,
            "n_bins_per_dim": 21,
            "enable_adaptive": True,
            "invert_DBNN": True,
            "reconstruction_weight": 0.5,
            "feedback_strength": 0.3,
            "inverse_learning_rate": 0.001,
            "Save_training_epochs": True,
            "training_save_path": "training_data",
            "enable_vectorized": False,
            "vectorization_warning_acknowledged": False,
            "compute_device": "auto",
            "use_interactive_kbd": False,
            "class_preference": True,
        },
        "execution_flags": {
            "train": True,
            "train_only": False,
            "predict": True,
            "fresh_start": False,
            "use_previous_model": True,
            "gen_samples": False,
        },
    }

    with open(conf_path, "w") as f:
        json.dump(conf_data, f, indent=2)

    print(f"- {conf_path}")

    return best_metric


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        val_iter = tqdm(
            val_loader,
            desc="Validating",
            leave=False,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            postfix={"loss": "0.000", "acc": "0.00%"},
        )

        for inputs, labels, _ in val_iter:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _, _ = model(inputs)

            # Custom CE loss calculation (same as in training)
            per_sample_loss = F.cross_entropy(outputs, labels, reduction="none")
            unique_labels = torch.unique(labels)
            if len(unique_labels) == 0:
                loss = torch.tensor(0.0, device=device)
            else:
                class_losses = []
                for lbl in unique_labels:
                    mask = labels == lbl
                    class_loss = per_sample_loss[mask].mean()
                    class_losses.append(class_loss)
                loss = torch.mean(torch.stack(class_losses))

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            batch_total = labels.size(0)
            correct += (predicted == labels).sum().item()
            total += batch_total

            val_iter.set_postfix(
                {
                    "loss": f"{loss.item():.3f}",
                    "acc": f"{(predicted == labels).sum().item()/batch_total*100:.2f}%",
                }
            )

    return 100 * correct / total, total_loss / len(val_loader)


def predict(model, loader, device):
    model.eval()
    features = []
    paths = []
    predictions = []

    with torch.no_grad():
        pred_iter = tqdm(
            loader,
            desc="Predicting",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            postfix={"processed": "0"},
        )

        for batch in pred_iter:
            if len(batch) == 2:  # Predict mode returns (image, path)
                inputs, batch_paths = batch
                inputs = inputs.to(device)
                outputs, feats, _ = model(inputs)
            else:
                inputs, _, batch_paths = batch
                inputs = inputs.to(device)
                outputs, feats, _ = model(inputs)

            features.append(feats.cpu())
            paths.extend(batch_paths)
            predictions.append(torch.argmax(outputs, 1).cpu())

            pred_iter.set_postfix({"processed": f"{len(paths)}/{len(loader.dataset)}"})

    return (torch.cat(features).numpy(), paths, torch.cat(predictions).numpy())


# --------------------------
# Modified Training Feature Saving
# --------------------------
def save_features_to_csv(features, labels, paths, csv_path, class_metadata=None):
    """Save features with original class names using class metadata"""
    # Convert numerical labels to class names if metadata exists
    if class_metadata is not None:
        idx_to_class = {v: k for k, v in class_metadata["class_to_idx"].items()}
        str_labels = [idx_to_class[label] for label in labels]
    else:
        str_labels = labels  # Fallback to numerical labels

    # Create DataFrame
    df = pd.DataFrame({"filepath": paths, "target": str_labels})

    # Add feature columns
    feature_df = pd.DataFrame(
        features.tolist() if isinstance(features, np.ndarray) else features,
        columns=[f"feature_{i}" for i in range(features.shape[1])],
    )
    df = pd.concat([df, feature_df], axis=1)

    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved features to {csv_path} ({len(df)} rows)")


def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    paths = []

    with torch.no_grad():
        iter = tqdm(
            loader,
            desc="Extracting features",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
        )

        for batch in iter:
            if len(batch) == 3:  # Training/validation mode
                inputs, lbls, pths = batch
                inputs = inputs.to(device)  # Move inputs to the device
                _, feats, _ = model(inputs)
                labels.extend(lbls.tolist())
            else:  # Prediction mode
                inputs, pths = batch
                inputs = inputs.to(device)
                _, feats, _ = model(inputs)
                lbls = [item[1] for item in loader.dataset.samples]
                labels.extend(lbls)

            features.append(feats.cpu().numpy())
            paths.extend(pths)

    return np.concatenate(features), labels, paths


# --------------------------
# Configuration Management
# --------------------------
def find_dataset_root(data_dir):
    current_dir = os.path.abspath(data_dir)
    while True:
        try:
            os.listdir(current_dir)
        except PermissionError:
            print(f"⚠️ Permission denied accessing {current_dir}, moving up...")
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                break
            current_dir = parent_dir
            continue
        # Check if current_dir contains valid class directories
        class_candidates = []
        for entry in os.listdir(current_dir):
            entry_path = os.path.join(current_dir, entry)
            if os.path.isdir(entry_path):
                # Check if the subdirectory contains images
                if any(
                    is_image_file(os.path.join(entry_path, f))
                    for f in os.listdir(entry_path)
                ):
                    class_candidates.append(entry_path)

        if class_candidates:
            return current_dir  # Found dataset root

        # Move up one directory level
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached filesystem root
            break
        current_dir = parent_dir

    raise ValueError(
        f"No valid dataset structure found in {data_dir} or its parents. Required structure: a directory containing subdirectories with images."
    )


def extract_low_level_features(img_path):
    """Simplified feature extractor using basic image properties"""
    try:
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        img = img.resize((32, 32))  # Standardize size
        return np.array(img).flatten()  # Use raw pixels as features
    except:
        return np.zeros(32 * 32)  # Fallback for corrupted images


def analyze_class_separability(root_dir, classes):
    """Measure initial feature distance between classes"""
    separability = defaultdict(float)
    feats_per_class = {}

    # Extract raw features from first 100 samples per class
    for cls in classes:
        samples = [
            os.path.join(root_dir, cls, f)
            for f in os.listdir(os.path.join(root_dir, cls))[:100]
        ]
        feats = [extract_low_level_features(img) for img in samples]
        feats_per_class[cls] = np.mean(feats, axis=0)  # Mean feature vector per class

    # Compare all class pairs using PyTorch's cosine similarity
    for (cls1, feats1), (cls2, feats2) in combinations(feats_per_class.items(), 2):
        # Convert numpy arrays to tensors and flatten
        t1 = torch.tensor(feats1.flatten(), dtype=torch.float32)
        t2 = torch.tensor(feats2.flatten(), dtype=torch.float32)

        # Calculate cosine similarity
        similarity = F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0), dim=1).item()
        separability[f"{cls1}-{cls2}"] = similarity

    return separability


def analyze_stroke_complexity(root_dir, classes):
    """Detect if classes require diagonal/curve features"""
    # Define diagonal kernel mask (45° and 135° directions)
    diagonal_kernel_mask = np.array(
        [[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=np.float32
    )

    edge_threshold = 1000  # Adjust based on empirical testing
    results = {"needs_diagonal_kernels": False}

    for cls in classes:
        class_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(class_dir):
            continue

        sample_file = os.listdir(class_dir)[0]
        sample_path = os.path.join(class_dir, sample_file)

        try:
            img = Image.open(sample_path).convert("L")
            img_array = np.array(img)

            # Edge detection with Canny
            edges = cv2.Canny(img_array, 50, 150)

            # Calculate diagonal edges using convolution
            diagonal_edges = cv2.filter2D(edges, -1, diagonal_kernel_mask)
            diagonal_score = np.sum(np.abs(diagonal_edges))

            if diagonal_score > edge_threshold:
                results["needs_diagonal_kernels"] = True
                break  # No need to check other samples if found

        except Exception as e:
            print(f"Error processing {sample_path}: {str(e)}")
            continue

    return results


def calculate_adaptive_bottleneck(num_classes, feature_dim, separability):
    """Set bottleneck size based on class complexity"""
    min_dim = 64
    max_dim = 128

    # If any class pair has high similarity, increase bottleneck
    if min(separability.values()) > 0.7:  # Highly similar classes
        return min(max_dim, int(feature_dim * 0.8))
    else:
        return max(min_dim, int(feature_dim * 0.2))


def create_transforms(config, stroke_analysis):
    transforms_list = []

    if stroke_analysis["needs_diagonal_kernels"]:
        transforms_list += [
            transforms.RandomRotation(15),
            transforms.RandomPerspective(distortion_scale=0.2),
        ]

    transforms_list += [
        transforms.Resize(config["dataset"]["input_size"]),
        transforms.ToTensor(),
        transforms.Normalize(config["dataset"]["mean"], config["dataset"]["std"]),
    ]

    return transforms.Compose(transforms_list)


def create_default_config(name, data_dir, resize=None):
    # Automatically determine name from data directory if not provided
    if name == "dataset":
        name = os.path.basename(os.path.normpath(data_dir))

    config = {
        "dataset": {
            "name": name,
            "type": "custom",
            "in_channels": 3,
            "num_classes": 0,
            "input_size": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resize_images": False,
            "train_dir": data_dir,
        },
        "model": {
            "type": "jnet",  # 'cnn' or 'jnet'
            "cnn_params": {
                "bottleneck_dim": 128,  # Set default ceiling here
                "sparsity_weight": 0.01,
            },
            "jnet_params": {"skip_connections": True, "reduced_dim": 128},
            "depth": 3,
            "initial_filters": 32,
        },
        "training_params": {
            "trials": 100,
            "epochs": 100,
            "learning_rate": 0.001,
            "kl_weight": 0.1,
            "batch_size": 32,
            "test_fraction": 0.2,
            "random_seed": 42,
            "max_epochs": 1000,  # Absolute maximum epochs to run
            "patience": 20,  # Number of epochs to wait without improvement
            "early_stopping_metric": "accuracy",  # accuracy, loss
            "early_stopping_mode": "max",  # min,max
        },
    }

    try:
        # Find the dataset root directory that contains subdirectories with images
        dataset_root = find_dataset_root(data_dir)
        if config["dataset"]["name"] == "dataset":
            config["dataset"]["name"] = os.path.basename(dataset_root)
        config["dataset"]["train_dir"] = dataset_root

        # Get class directories
        class_dirs = []
        for entry in os.listdir(dataset_root):
            entry_path = os.path.join(dataset_root, entry)
            if os.path.isdir(entry_path):
                # Check if this directory has images
                if any(
                    is_image_file(os.path.join(entry_path, f))
                    for f in os.listdir(entry_path)
                ):
                    class_dirs.append(entry_path)

        if not class_dirs:
            raise ValueError(f"No valid class directories found in {dataset_root}")

        config["dataset"]["num_classes"] = len(class_dirs)

        # Image statistics analysis
        size_channels = analyze_images(
            dataset_root, [os.path.basename(d) for d in class_dirs]
        )
        sizes = [sc[0] for sc in size_channels]
        channels = [sc[1] for sc in size_channels]

        # Handle image size configuration
        if resize:
            config["dataset"]["input_size"] = list(resize)
            config["dataset"]["resize_images"] = True
        else:
            size_counts = defaultdict(int)
            for size in sizes:
                size_counts[size] += 1
            if not size_counts:
                raise ValueError("Could not determine image sizes from dataset")
            config["dataset"]["input_size"] = list(
                max(size_counts.items(), key=lambda x: x[1])[0]
            )
        # Calculate max depth based on input size
        H, W = config["dataset"]["input_size"]
        max_depth = calculate_max_depth(H, W)
        # Update config with auto-calculated depth
        config["model"]["depth"] = max_depth  # Override user-provided depth

        # Determine color channels
        channel_counts = defaultdict(int)
        for c in channels:
            channel_counts[c] += 1
        config["dataset"]["in_channels"] = max(
            channel_counts.items(), key=lambda x: x[1]
        )[0]

        # Calculate dataset statistics
        transform = transforms.Compose(
            [transforms.Resize(config["dataset"]["input_size"]), transforms.ToTensor()]
        )
        stats_dataset = CustomDataset(dataset_root, transform=transform)
        mean, std = calculate_dataset_stats(stats_dataset)
        config["dataset"]["mean"] = mean
        config["dataset"]["std"] = std

        # Create output directory structure
        output_dir = os.path.join("data", config["dataset"]["name"])
        os.makedirs(output_dir, exist_ok=True)

        # Get class names from directory structure
        class_dirs = []
        classes = []  # <-- Initialize classes list
        for entry in os.listdir(dataset_root):
            entry_path = os.path.join(dataset_root, entry)
            if os.path.isdir(entry_path):
                if any(
                    is_image_file(os.path.join(entry_path, f))
                    for f in os.listdir(entry_path)
                ):
                    class_dirs.append(entry_path)
                    classes.append(entry)  # <-- Store class names

        if not class_dirs:
            raise ValueError(f"No valid class directories found in {dataset_root}")

        # New: Class Separability Analysis
        separability_scores = analyze_class_separability(dataset_root, classes)
        config["training_params"]["kl_weight"] = 1.0 - min(separability_scores.values())

        # New: Shape Complexity Detection
        stroke_analysis = analyze_stroke_complexity(dataset_root, classes)
        if stroke_analysis["needs_diagonal_kernels"]:
            config["model"]["initial_kernel_type"] = "diagonal_enhanced"

        # Auto-set bottleneck dimension (e.g., 10% of original features)
        original_dim = config["model"]["initial_filters"] * (
            2 ** (config["model"]["depth"] - 1)
        )

        # New: Adaptive Bottleneck
        config["model"]["bottleneck_dim"] = calculate_adaptive_bottleneck(
            num_classes=len(classes),
            feature_dim=original_dim,
            separability=separability_scores,
        )
    except Exception as e:
        raise RuntimeError(
            f"Configuration creation failed: {str(e)}\n"
            f"Required structure:\n"
            f"Input directory should contain a subdirectory (dataset root) with subdirectories (classes) that have images."
        ) from e

    config["dataset"]["model_path"] = os.path.join(
        "data", config["dataset"]["name"], "Model", "best_model.pth"
    )

    return config


def is_image_file(path):
    valid_extensions = (".png", ".jpg", ".jpeg", ".dcm", ".fits", ".fit")
    return os.path.isfile(path) and path.lower().endswith(valid_extensions)


def analyze_images(root_dir, classes, sample_size=100):
    size_channels = []
    error_log = defaultdict(int)

    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for root, _, files in os.walk(class_dir):
            for fname in files:
                if len(size_channels) >= sample_size:
                    break
                try:
                    path = os.path.join(root, fname)
                    if path.lower().endswith(".dcm"):
                        ds = pydicom.dcmread(path)
                        img = ds.pixel_array
                        size = (img.shape[1], img.shape[0])
                        channels = 1
                    elif path.lower().endswith((".fits", ".fit")):
                        with fits.open(path) as hdul:
                            data = hdul[0].data
                            size = (
                                (data.shape[1], data.shape[0])
                                if data.ndim == 2
                                else (data.shape[2], data.shape[1])
                            )
                            channels = 1 if data.ndim == 2 else data.shape[0]
                    else:
                        with Image.open(path) as img:
                            size = img.size
                            channels = len(img.getbands())
                    size_channels.append((size, channels))
                except Exception as e:
                    error_log[str(e)] += 1

    if not size_channels:
        error_msg = "No valid images found. Errors encountered:\n" + "\n".join(
            [f"- {k} ({v}x)" for k, v in error_log.items()]
        )
        raise ValueError(error_msg)

    return size_channels


def calculate_dataset_stats(dataset):
    loader = DataLoader(dataset, batch_size=32, num_workers=4)
    mean = 0.0
    std = 0.0
    n_samples = 0

    for images, _, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += batch_samples

    mean /= n_samples
    std /= n_samples
    return mean.tolist(), std.tolist()


# --------------------------
# Main Execution
# --------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Dynamic CNN/J-Net Training/Prediction"
    )
    parser.add_argument("mode", choices=["train", "predict"], help="Operation mode")
    parser.add_argument(
        "--model_type",
        choices=["cnn", "jnet"],
        default="jnet",
        help="Model architecture selection",
    )
    parser.add_argument("input", help="Input path (directory/archive/torchvision name)")
    parser.add_argument("--config", help="Custom config file path")
    parser.add_argument(
        "--name", default="dataset", type=str, help="Dataset name for config"
    )
    parser.add_argument("--resize", nargs=2, type=int, help="Resize dimensions W H")
    parser.add_argument(
        "--output", default="predictions.csv", help="Output path for predictions"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Handle input data
    data_dir = handle_input_data(args.input)

    # Load or create configuration
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = create_default_config(args.name, data_dir, args.resize)

    # Create model directory
    model_dir = os.path.join("data", config["dataset"]["name"], "Model")
    os.makedirs(model_dir, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type:
        config["model"]["type"] = args.model_type
    # Model initialization

    if config["model"]["type"] == "jnet":
        model = DynamicJNet(
            in_channels=config["dataset"]["in_channels"],
            num_classes=config["dataset"]["num_classes"],
            depth=config["model"]["depth"],
            initial_filters=config["model"]["initial_filters"],
            input_size=config["dataset"]["input_size"],
            **config["model"]["jnet_params"],
        ).to(device)
    else:
        model = DynamicCNN(
            in_channels=config["dataset"]["in_channels"],
            num_classes=config["dataset"]["num_classes"],
            depth=config["model"]["depth"],
            initial_filters=config["model"]["initial_filters"],
            input_size=config["dataset"][
                "input_size"
            ],  # Critical for depth calculation
            initial_kernel_type=config["model"].get("initial_kernel_type", "standard"),
        ).to(device)

    # Load existing model if available
    if os.path.exists(config["dataset"]["model_path"]):
        print(f"🔁 Loading existing model from {config['dataset']['model_path']}")
        model.load_state_dict(torch.load(config["dataset"]["model_path"]))

    if args.mode == "train":
        # Training implementation
        try:
            # Dataset and transforms
            transform = transforms.Compose(
                [
                    transforms.Resize(config["dataset"]["input_size"]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        config["dataset"]["mean"], config["dataset"]["std"]
                    ),
                ]
            )

            # Create dataset and split train/val
            full_dataset = CustomDataset(
                config["dataset"]["train_dir"], transform=transform, mode="train"
            )

            # Split dataset
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["training_params"]["batch_size"],
                shuffle=True,
                num_workers=4,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["training_params"]["batch_size"],
                shuffle=False,
                num_workers=4,
            )

            # Initialize model
            if config["model"]["type"] == "jnet":
                model = DynamicJNet(
                    in_channels=config["dataset"]["in_channels"],
                    num_classes=config["dataset"]["num_classes"],
                    depth=config["model"]["depth"],
                    initial_filters=config["model"]["initial_filters"],
                    input_size=config["dataset"]["input_size"],
                    **config["model"]["jnet_params"],
                ).to(device)
            else:
                model = DynamicCNN(
                    in_channels=config["dataset"]["in_channels"],
                    num_classes=config["dataset"]["num_classes"],
                    depth=config["model"]["depth"],
                    initial_filters=config["model"]["initial_filters"],
                    input_size=config["dataset"][
                        "input_size"
                    ],  # Critical for depth calculation
                ).to(device)

            # Print model summary
            print(f"Model architecture:\n{model}")
            print(
                f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
            )

            # Train the model
            train(model, train_loader, val_loader, config, device, full_dataset)

            # Save final artifacts
            save_dir = f"data/{config['dataset']['name']}"
            os.makedirs(save_dir, exist_ok=True)

            # Save model
            model_path = os.path.join(save_dir, "model.pth")
            torch.save(model.state_dict(), model_path)

            # Save config
            config_path = os.path.join(save_dir, f"{config['dataset']['name']}.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            # Save CSV with features
            metadata_path = os.path.join(
                "data", config["dataset"]["name"], "class_metadata.json"
            )
            with open(metadata_path) as f:
                class_metadata = json.load(f)  # This is the correctly loaded variable
            features, labels, paths = extract_features(model, train_loader, device)
            csv_path = os.path.join(save_dir, f"{config['dataset']['name']}.csv")
            save_features_to_csv(
                features, labels, paths, csv_path, class_metadata=class_metadata
            )

            print(f"Training completed. Artifacts saved to {save_dir}")

        except Exception as e:
            print(f"Training failed: {str(e)}")
            raise

    elif args.mode == "predict":
        try:
            metadata_path = os.path.join(
                "data", config["dataset"]["name"], "class_metadata.json"
            )
            if not os.path.exists(metadata_path):
                raise RuntimeError(
                    f"❌ No training metadata found at {metadata_path}\n"
                    "Please train a model first before prediction"
                )

            with open(metadata_path) as f:
                class_metadata = json.load(f)

            # Verify dataset version if available
            if "dataset_version" in class_metadata:
                current_version = str(os.path.getmtime(pred_dataset.root_dir))
                if metadata["dataset_version"] != current_version:
                    print(
                        "⚠️ Warning: Prediction data directory modified since training"
                    )

            # Load model
            if config["model"]["type"] == "jnet":
                model = DynamicJNet(
                    in_channels=config["dataset"]["in_channels"],
                    num_classes=config["dataset"]["num_classes"],
                    depth=config["model"]["depth"],
                    initial_filters=config["model"]["initial_filters"],
                    input_size=config["dataset"]["input_size"],
                    **config["model"]["jnet_params"],
                ).to(device)
            else:
                model = DynamicCNN(
                    in_channels=config["dataset"]["in_channels"],
                    num_classes=config["dataset"]["num_classes"],
                    depth=config["model"]["depth"],
                    initial_filters=config["model"]["initial_filters"],
                    input_size=config["dataset"][
                        "input_size"
                    ],  # Critical for depth calculation
                ).to(device)

            # model_path = f"data/{config['dataset']['name']}/model.pth"
            model_path = f"data/{config['dataset']['name']}/Model/pruned_model.pth"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            print("Model loaded successfully")

            # Create prediction dataset
            transform = transforms.Compose(
                [
                    transforms.Resize(config["dataset"]["input_size"]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        config["dataset"]["mean"], config["dataset"]["std"]
                    ),
                ]
            )

            pred_dataset = CustomDataset(
                data_dir,
                transform=transform,
                mode="predict",
                class_metadata=class_metadata,
            )
            pred_loader = DataLoader(
                pred_dataset,
                batch_size=config["training_params"]["batch_size"],
                shuffle=False,
                num_workers=4,
            )
            print(f"Found {len(pred_dataset)} prediction samples")

            # Run prediction
            features, paths, predictions = predict(model, pred_loader, device)

            # Save results
            output_dir = os.path.dirname(args.output) if "/" in args.output else "."
            os.makedirs(output_dir, exist_ok=True)
            training_csv_path = os.path.join(
                "data", config["dataset"]["name"], f"{config['dataset']['name']}.csv"
            )
            save_predictions(features, paths, predictions, args.output, class_metadata)
            print(f"Predictions saved to {args.output}")

        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            raise


def handle_input_data(input_path):
    """Process different input types and return directory path"""
    # Handle torchvision datasets
    try:
        from torchvision.datasets import __all__ as tv_datasets

        if input_path in tv_datasets:
            return download_torchvision_data(input_path)
    except ImportError:
        pass

    # Handle compressed files
    if os.path.isfile(input_path):
        temp_dir = tempfile.mkdtemp()
        try:
            if zipfile.is_zipfile(input_path):
                with zipfile.ZipFile(input_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)
            elif tarfile.is_tarfile(input_path):
                with tarfile.open(input_path, "r") as tar_ref:
                    tar_ref.extractall(temp_dir)
            else:
                raise ValueError(f"Unsupported file format: {input_path}")

            # Find actual data directory
            data_root = find_data_root(temp_dir)
            if not data_root:
                raise ValueError("No valid data found in extracted archive")
            return data_root
        except:
            shutil.rmtree(temp_dir)
            raise

    # Handle directories
    if os.path.isdir(input_path):
        data_root = find_data_root(input_path)
        if not data_root:
            raise ValueError(f"No valid data found in {input_path}")
        return data_root

    raise ValueError(f"Could not process input path: {input_path}")


def download_torchvision_data(name):
    """Download standard torchvision dataset"""
    data_dir = f"data/{name}"
    os.makedirs(data_dir, exist_ok=True)

    try:
        if name == "MNIST":
            dataset = torchvision.datasets.MNIST(
                root=data_dir, download=True, train=True
            )
        elif name == "CIFAR10":
            dataset = torchvision.datasets.CIFAR10(
                root=data_dir, download=True, train=True
            )
        # Add more datasets as needed
        else:
            raise ValueError(f"Unsupported torchvision dataset: {name}")
    except Exception:
        shutil.rmtree(data_dir)
        raise

    return os.path.join(data_dir, "raw")


def find_data_root(directory):
    """Find the directory containing actual data files"""
    # Look for common structures
    for root, dirs, files in os.walk(directory):
        # Check for image files
        if any(
            f.lower().endswith((".png", ".jpg", ".jpeg", ".dcm", ".fits"))
            for f in files
        ):
            return root
        # Check for train/test split
        if "train" in dirs and "test" in dirs:
            return directory
    return None


def predict(model, loader, device):
    model.eval()
    features = []
    paths = []
    predictions = []

    with torch.no_grad():
        pred_iter = tqdm(
            loader,
            desc="Predicting",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            postfix={"processed": "0"},
        )

        for batch in pred_iter:
            if len(batch) == 2:  # Predict mode returns (image, path)
                inputs, batch_paths = batch
                inputs = inputs.to(device)
                outputs, feats, _ = model(inputs)  # <-- Add third unpack value
                paths.extend(batch_paths)
                predictions.append(torch.argmax(outputs, 1).cpu())

                features.append(feats.cpu())

            pred_iter.set_postfix({"processed": f"{len(paths)}/{len(loader.dataset)}"})

    return (torch.cat(features).numpy(), paths, torch.cat(predictions).numpy())


# --------------------------
# Modified Prediction Saving
# --------------------------
def save_predictions(features, paths, labels, output_path, class_metadata):
    """Save predictions using class metadata"""
    # Load class metadata
    metadata_path = os.path.join(
        "data", config["dataset"]["name"], "class_metadata.json"
    )
    with open(metadata_path) as f:
        json.load(f)

    # Convert numeric labels to class names
    idx_to_class = {v: k for k, v in class_metadata["class_to_idx"].items()}
    class_names = [idx_to_class[label] for label in labels]

    # Load training CSV structure
    train_csv = os.path.join(
        "data", config["dataset"]["name"], f"{config['dataset']['name']}.csv"
    )
    train_df = pd.read_csv(train_csv, nrows=1)

    # Create prediction DataFrame with class names
    df = pd.DataFrame({"filepath": paths, "target": class_names})
    # Add features with matching column order
    feature_cols = [c for c in train_df.columns if c.startswith("feature_")]
    feature_df = pd.DataFrame(features, columns=feature_cols)
    df = pd.concat([df, feature_df], axis=1)

    # Reorder columns to match training data
    df = df[train_df.columns]

    # Save results
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
