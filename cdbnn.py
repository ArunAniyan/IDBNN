# Working, fully functional with predcition 30/March/2025
# Revisions on Mar30 2025 Stable version 8:56 AM
# Added distance correlations to filter the output features. April 12, 3:45 am
# Fixed a bug in Prediction mode model loading April 14 2025 9:32 am
# Finalised completely working module as on 15th April 2025
# Feature Dimension can now be input during training instead of hardcoding to 128 April 30 11:13 PM
# Last updated with configurable data_name on May 1 1:07 am
# Infinite training loop added May 1 1:30 am
# ----------Bug fixes and improved version - April 5 4:24 pm----------------------------------------------
# ---- author : Ninan Sajeeth Philip, Artificial Intelligence Research and Intelligent Systems
# -------------------------------------------------------------------------------------------------------------------------------

import argparse
import bz2
import csv
import gzip
import json
import logging
import lzma
import os
import random
import shutil
import subprocess
import sys
import tarfile
import traceback
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import kornia.filters as KF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import correlation
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import resize
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output"""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @staticmethod
    def color_value(current_value, previous_value=None, higher_is_better=True):
        """Color a value based on whether it improved or declined"""
        if previous_value is None:
            return f"{current_value:.4f}"

        if higher_is_better:
            if current_value > previous_value:
                return f"{Colors.GREEN}{current_value:.4f}{Colors.ENDC}"
            elif current_value < previous_value:
                return f"{Colors.RED}{current_value:.4f}{Colors.ENDC}"
        else:  # lower is better
            if current_value < previous_value:
                return f"{Colors.GREEN}{current_value:.4f}{Colors.ENDC}"
            elif current_value > previous_value:
                return f"{Colors.RED}{current_value:.4f}{Colors.ENDC}"

        return f"{current_value:.4f}"  # No color if equal

    @staticmethod
    def highlight_dataset(name):
        """Highlight dataset name in red"""
        return f"{Colors.RED}{name}{Colors.ENDC}"

    @staticmethod
    def highlight_time(time_value):
        """Color time values based on threshold"""
        if time_value < 10:
            return f"{Colors.GREEN}{time_value:.2f}{Colors.ENDC}"
        elif time_value < 30:
            return f"{Colors.YELLOW}{time_value:.2f}{Colors.ENDC}"
        else:
            return f"{Colors.RED}{time_value:.2f}{Colors.ENDC}"


class DistanceCorrelationFeatureSelector:
    """Helper class to select features based on distance correlation criteria"""

    def __init__(self, upper_threshold=0.85, lower_threshold=0.01):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def calculate_distance_correlations(self, features, labels):
        """Calculate distance correlations between features and labels"""
        n_features = features.shape[1]
        label_corrs = np.zeros(n_features)

        # Calculate correlation with labels
        for i in range(n_features):
            label_corrs[i] = 1 - correlation(features[:, i], labels)

        return label_corrs

    def select_features(self, features, labels):
        """Select features based on distance correlation criteria"""
        label_corrs = self.calculate_distance_correlations(features, labels)

        # Get indices of features that meet upper threshold
        selected_indices = [
            i for i, corr in enumerate(label_corrs) if corr >= self.upper_threshold
        ]

        # Sort by correlation strength (descending)
        selected_indices.sort(key=lambda i: -label_corrs[i])

        # Remove features that are too correlated with each other
        final_indices = []
        feature_matrix = features[:, selected_indices]

        for i, idx in enumerate(selected_indices):
            keep = True
            for j in final_indices:
                # Calculate correlation between features
                corr = 1 - correlation(
                    feature_matrix[:, i], feature_matrix[:, selected_indices.index(j)]
                )
                if corr > self.lower_threshold:
                    keep = False
                    break
            if keep:
                final_indices.append(idx)

        return final_indices, label_corrs


class PredictionManager:
    """Manages prediction on new images using a trained model."""

    def __init__(self, config: Dict, device: str = None):
        """
        Initialize the PredictionManager.

        Args:
            config (Dict): Configuration dictionary.
            device (str, optional): Device to use (e.g., 'cuda' or 'cpu'). Defaults to None.
        """

        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.heatmap_attn = config["model"].get("heatmap_attn", False)
        self.checkpoint_manager = UnifiedCheckpoint(config)
        self.model = self._load_model()

    def _extract_archive(self, archive_path: str, extract_dir: str) -> str:
        """Extract a compressed archive to a directory."""
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        os.makedirs(extract_dir, exist_ok=True)

        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(extract_dir)
        elif archive_path.endswith(".tar"):
            with tarfile.open(archive_path, "r:") as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
        return extract_dir

    def _get_image_files(self, input_path: str) -> List[str]:
        """Get a list of image files from the input path."""
        if not isinstance(input_path, (str, bytes, os.PathLike)):
            raise ValueError(
                f"input_path must be a string or PathLike object, got {type(input_path)}"
            )

        image_files = []
        dataset_name = self.config["dataset"]["name"]
        if os.path.isfile(input_path):
            # Single image file
            if input_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                return [input_path]
            # Compressed archive
            elif input_path.lower().endswith((".zip", ".tar.gz", ".tgz", ".tar")):
                extract_dir = os.path.join(
                    os.path.dirname(input_path), f"{dataset_name}/extracted"
                )
                os.makedirs(extract_dir, exist_ok=True)
                self._extract_archive(input_path, extract_dir)
                return self._get_image_files(
                    extract_dir
                )  # Recursively process extracted files
        elif os.path.isdir(input_path):
            # Directory of images - recursively search for image files
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                    ):
                        image_files.append(os.path.join(root, file))
            return image_files
        else:
            raise ValueError(f"Invalid input path: {input_path}")

    def _load_model(self) -> nn.Module:
        """Load the trained model with all components"""
        model = ModelFactory.create_model(self.config)
        model.to(self.device)

        checkpoint_path = self.checkpoint_manager.checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.warning("CUDA not available. Falling back to CPU.")
                checkpoint = torch.load(
                    checkpoint_path, map_location=torch.device("cpu")
                )
            else:
                raise e

        # Find the best state with both KL divergence and class encoding if enabled
        state_key = None
        for key in checkpoint["model_states"]:
            if "phase2" in key and "kld" in key and "cls" in key:
                state_key = key
                break

        if state_key is None:
            raise ValueError(
                "No suitable checkpoint state found with both KL divergence and class encoding"
            )

        state_dict = checkpoint["model_states"][state_key]["best"]["state_dict"]

        # Load the state dict
        model.load_state_dict(state_dict, strict=False)

        # Handle clustering temperature conversion if needed
        if model.use_kl_divergence:
            if hasattr(model, "clustering_temperature"):
                if isinstance(model.clustering_temperature, float):
                    # Convert float to tensor
                    model.clustering_temperature = torch.tensor(
                        [model.clustering_temperature],
                        dtype=torch.float32,
                        device=self.device,
                    )
                elif not isinstance(model.clustering_temperature, torch.Tensor):
                    model.clustering_temperature = torch.tensor(
                        [1.0], dtype=torch.float32, device=self.device  # default value
                    )

        # Verify both components were loaded correctly
        if model.use_kl_divergence:
            if not hasattr(model, "cluster_centers") or model.cluster_centers is None:
                raise RuntimeError("Cluster centers failed to load")
            if (
                not hasattr(model, "clustering_temperature")
                or model.clustering_temperature is None
            ):
                raise RuntimeError("Clustering temperature failed to load")

        if model.use_class_encoding and hasattr(model, "classifier"):
            if not any("classifier" in key for key in state_dict):
                raise RuntimeError("Classifier parameters failed to load")

        model.set_training_phase(2)
        model.eval()

        logger.info("Model loaded successfully with all components")
        return model

    # --------------------Prediction -----------------------
    def predict_images(
        self, data_path: str, output_csv: str = None, batch_size: int = 128
    ):
        """Predict features with actual class names and generate heatmaps"""
        # Configuration and initialization
        heatmap_enabled = self.config["model"].get("heatmap_attn", False)
        class_mapping = self._get_class_mapping(data_path)
        reverse_class_mapping = {v: k for k, v in class_mapping.items()}

        # Get image files and validate
        (
            image_files,
            class_labels,
            original_filenames,
        ) = self._get_image_files_with_labels(data_path)
        if not image_files:
            raise ValueError(f"No valid images found in {data_path}")

        # Set output paths
        if output_csv is None:
            dataset_name = self.config["dataset"]["name"]
            output_csv = os.path.join("data", dataset_name, f"{dataset_name}.csv")
        heatmap_base = os.path.join(os.path.dirname(output_csv), "heatmaps")
        os.makedirs(heatmap_base, exist_ok=True) if heatmap_enabled else None

        # Prepare CSV headers
        csv_headers = [
            "original_filename",
            "filepath",
            "label_type",
            "target",
            "cluster_assignment",
            "cluster_confidence",
        ] + [f"feature_{i}" for i in range(self.config["model"]["feature_dims"])]
        if heatmap_enabled:
            csv_headers.append("heatmap_path")

        with open(output_csv, "w", newline="") as csvfile:
            csv.writer(csvfile).writerow(csv_headers)

        # Batch processing
        for i in tqdm(
            range(0, len(image_files), batch_size), desc="Predicting features"
        ):
            batch_files = image_files[i : i + batch_size]
            batch_labels = class_labels[i : i + batch_size]
            batch_filenames = original_filenames[i : i + batch_size]
            batch_images = []

            # Load and transform images
            for filename in batch_files:
                try:
                    with Image.open(filename) as img:
                        image_tensor = (
                            self._get_transforms()(img.convert("RGB"))
                            .unsqueeze(0)
                            .to(self.device)
                        )
                        batch_images.append(image_tensor)
                except Exception as e:
                    logger.error(f"Error loading {filename}: {str(e)}")
                    continue
            if not batch_images:
                continue

            # Model inference
            with torch.no_grad():
                batch_tensor = torch.cat(batch_images, dim=0)
                feature_maps = []

                # Feature map hook
                def hook(module, input, output):
                    feature_maps.append(output.detach())

                hook_handle = self.model.encoder_layers[-1].register_forward_hook(hook)

                # Forward pass
                output = self.model(batch_tensor)
                hook_handle.remove()

                # Process outputs
                embedding = output.get(
                    "embedding", output[0] if isinstance(output, tuple) else output
                )
                features = embedding.cpu().numpy()

                # Cluster processing
                if "cluster_assignments" in output:
                    cluster_nums = output["cluster_assignments"].cpu().numpy()
                    cluster_assign = [f"Cluster_{int(c)}" for c in cluster_nums]
                    cluster_conf = (
                        output["cluster_probabilities"].max(1)[0].cpu().numpy()
                    )
                else:
                    cluster_assign = ["NA"] * len(batch_files)
                    cluster_conf = ["NA"] * len(batch_files)

                # Heatmap generation
                heatmap_paths = [""] * len(batch_files)
                if heatmap_enabled and feature_maps:
                    try:
                        fm_tensor = feature_maps[0]
                        heatmaps = torch.mean(fm_tensor, dim=1, keepdim=True)
                        heatmaps = (
                            F.interpolate(
                                heatmaps,
                                size=self.config["dataset"]["input_size"],
                                mode="bilinear",
                                align_corners=False,
                            )
                            .squeeze(1)
                            .cpu()
                            .numpy()
                        )

                        for j, filename in enumerate(batch_files):
                            rel_path = os.path.relpath(filename, data_path)
                            heatmap_path = os.path.join(heatmap_base, rel_path)
                            heatmap_path = (
                                os.path.splitext(heatmap_path)[0] + "_heatmap.png"
                            )

                            os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
                            hm = (heatmaps[j] - heatmaps[j].min()) / (
                                heatmaps[j].max() - heatmaps[j].min() + 1e-8
                            )
                            plt.imsave(heatmap_path, hm, cmap="viridis")
                            heatmap_paths[j] = os.path.relpath(
                                heatmap_path, os.path.dirname(output_csv)
                            )
                    except Exception as e:
                        logger.error(f"Heatmap error: {str(e)}")

            # CSV writing
            with open(output_csv, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for j, (filename, orig_name, true_class) in enumerate(
                    zip(batch_files, batch_filenames, batch_labels)
                ):
                    # Determine label type and target
                    is_unknown = (
                        true_class in ["unknown", ""]
                        or true_class not in reverse_class_mapping
                    )
                    label_type = "predicted" if is_unknown else "true"

                    if label_type == "true":
                        target = true_class  # Actual class name from directory
                    else:
                        target = (
                            cluster_assign[j]
                            if cluster_assign[j] != "NA"
                            else "unknown"
                        )

                    # Build row
                    row = [
                        orig_name,
                        filename,
                        label_type,
                        target,
                        cluster_assign[j],
                        cluster_conf[j],
                    ] + features[j].tolist()

                    if heatmap_enabled:
                        row.append(heatmap_paths[j])

                    writer.writerow(row)

        logger.info(f"Predictions saved to {output_csv}")

    def _get_class_mapping(self, data_path: str) -> Dict[int, str]:
        """Build class name to index mapping from directory structure"""
        class_mapping = {}
        class_dirs = []

        if os.path.isdir(data_path):
            # Get immediate subdirectories
            with os.scandir(data_path) as entries:
                for entry in entries:
                    if entry.is_dir():
                        class_dirs.append(entry.path)

            # Sort alphabetically for consistent indexing
            class_dirs = sorted(class_dirs, key=lambda x: os.path.basename(x))

            # Create mapping
            for idx, class_dir in enumerate(class_dirs):
                class_name = os.path.basename(class_dir)
                class_mapping[idx] = class_name

        return class_mapping

    # -----------------------------------------------------------

    def _get_image_files_with_labels(
        self, input_path: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Get a list of image files, their corresponding class labels, and original filenames from the input path.
        Returns:
            Tuple[List[str], List[str], List[str]]: (image_paths, class_labels, original_filenames)
        """
        image_files = []
        class_labels = []
        original_filenames = []
        dataset_name = self.config["dataset"]["name"]
        if os.path.isfile(input_path) and input_path.lower().endswith(
            (".zip", ".tar.gz", ".tgz", ".tar")
        ):
            # Handle compressed archive
            extract_dir = os.path.join(
                os.path.dirname(input_path), f"{dataset_name}/extracted"
            )
            os.makedirs(extract_dir, exist_ok=True)
            self._extract_archive(input_path, extract_dir)
            return self._get_image_files_with_labels(
                extract_dir
            )  # Recursively process extracted files

        elif os.path.isdir(input_path):
            # Directory of images - recursively search for image files
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                    ):
                        image_files.append(os.path.join(root, file))
                        original_filenames.append(file)
                        # Extract class label from subfolder name if available
                        class_label = os.path.basename(root)
                        class_labels.append(
                            class_label
                            if class_label != os.path.basename(input_path)
                            else "unknown"
                        )
        elif os.path.isfile(input_path) and input_path.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        ):
            # Single image file
            image_files.append(input_path)
            original_filenames.append(os.path.basename(input_path))
            class_labels.append("unknown")
        else:
            raise ValueError(f"Invalid input path: {input_path}")

        return image_files, class_labels, original_filenames

    def _create_dataset(
        self, image_files: List[str], transform: transforms.Compose
    ) -> Dataset:
        """Create dataset with proper channel handling for torchvision datasets."""
        if self.config.get("data_type") == "torchvision":
            # Special handling for torchvision datasets
            dataset_class = getattr(
                torchvision.datasets, self.config["dataset"]["name"].upper()
            )
            return dataset_class(
                root="data", train=False, download=True, transform=transform
            )
        else:
            # Original folder-based dataset
            class DummyDataset(Dataset):
                def __init__(self, image_files, transform):
                    self.image_files = image_files
                    self.transform = transform

                def __len__(self):
                    return len(self.image_files)

                def __getitem__(self, idx):
                    image = Image.open(self.image_files[idx])
                    if self.transform:
                        image = self.transform(image)
                    return image, 0  # Dummy label

            return DummyDataset(image_files, transform)

    def _get_transforms(self) -> transforms.Compose:
        """Get the image transforms with strict channel control."""
        transform_list = [
            transforms.Resize(tuple(self.config["dataset"]["input_size"])),
            transforms.ToTensor(),
        ]

        # Force proper channel handling
        in_channels = self.config["dataset"]["in_channels"]
        if in_channels == 1:
            transform_list.append(
                transforms.Lambda(lambda x: x[:1])
            )  # Take only first channel
        elif in_channels == 3:
            transform_list.append(
                transforms.Lambda(
                    lambda x: x if x.shape[0] == 3 else x[:1].repeat(3, 1, 1)
                )
            )

        transform_list.append(
            transforms.Normalize(
                mean=self.config["dataset"]["mean"], std=self.config["dataset"]["std"]
            )
        )

        return transforms.Compose(transform_list)

    def _save_predictions(self, predictions: Dict, output_csv: str) -> None:
        """Save predictions to a CSV file."""
        # Convert features to a DataFrame
        feature_cols = [
            f"feature_{i}" for i in range(len(predictions["features_phase1"][0]))
        ]
        df = pd.DataFrame(predictions["features_phase1"], columns=feature_cols)
        df.insert(0, "filename", predictions["filename"])

        # Save to CSV
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Predictions saved to {output_csv}")
        return model


class BaseEnhancementConfig:
    """Base class for enhancement configuration management"""

    def __init__(self, config: Dict):
        self.config = config
        self.initialize_base_config()

    def initialize_base_config(self) -> None:
        """Initialize base configuration structures"""
        if "model" not in self.config:
            self.config["model"] = {}
        if "heatmap_attn" not in self.config["model"]:
            self.config["model"]["heatmap_attn"] = False
        # Initialize autoencoder config
        if "autoencoder_config" not in self.config["model"]:
            self.config["model"]["autoencoder_config"] = {
                "phase1_learning_rate": 0.001,
                "phase2_learning_rate": 0.005,
                "reconstruction_weight": 1.0,
                "feature_weight": 0.1,
                "enable_phase2": False,
                "enhancements": {
                    "use_kl_divergence": True,
                    "use_class_encoding": False,
                    "kl_divergence_weight": 0.1,
                    "classification_weight": 0.1,
                    "clustering_temperature": 1.0,
                    "min_cluster_confidence": 0.7,
                },
            }

        # Initialize enhancement modules
        if "enhancement_modules" not in self.config["model"]:
            self.config["model"]["enhancement_modules"] = {}

        # Initialize loss functions
        if "loss_functions" not in self.config["model"]:
            self.config["model"]["loss_functions"] = {}

    def _adjust_learning_rates(self, num_enhancements: int) -> None:
        """Adjust learning rates based on number of enabled enhancements"""
        complexity_factor = max(1, num_enhancements * 0.5)
        self.config["model"]["autoencoder_config"]["phase1_learning_rate"] = (
            0.001 / complexity_factor
        )
        self.config["model"]["autoencoder_config"]["phase2_learning_rate"] = (
            0.0005 / complexity_factor
        )

    def _normalize_weights(self, enabled_enhancements: List[str]) -> None:
        """Normalize weights for enabled enhancements"""
        num_enabled = len(enabled_enhancements)
        if num_enabled > 0:
            base_reconstruction_weight = 1.0
            enhancement_weight = 1.0 / (num_enabled + 1)

            self.config["model"]["autoencoder_config"][
                "reconstruction_weight"
            ] = base_reconstruction_weight

            for loss_name, loss_config in self.config["model"][
                "loss_functions"
            ].items():
                if loss_config["enabled"]:
                    loss_config["weight"] = enhancement_weight

    def get_config(self) -> Dict:
        """Get the current configuration"""
        return self.config


class GeneralEnhancementConfig(BaseEnhancementConfig):
    """Configuration manager for general (flexible) enhancement mode"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.enhancement_configs = self._get_enhancement_configs()

    def _get_enhancement_configs(self) -> Dict:
        """Get available enhancement configurations"""
        return {
            "astronomical": {
                "name": "Astronomical",
                "desc": "star detection, galaxy structure preservation",
                "components": {
                    "structure_preservation": True,
                    "detail_preservation": True,
                    "star_detection": True,
                    "galaxy_features": True,
                    "kl_divergence": True,
                },
                "weights": {
                    "detail_weight": 1.0,
                    "structure_weight": 0.8,
                    "edge_weight": 0.7,
                },
                "loss_components": {
                    "edge_preservation": True,
                    "peak_preservation": True,
                    "detail_preservation": True,
                },
            },
            "medical": {
                "name": "Medical",
                "desc": "tissue boundary, lesion detection",
                "components": {
                    "tissue_boundary": True,
                    "lesion_detection": True,
                    "contrast_enhancement": True,
                    "subtle_feature_preservation": True,
                },
                "weights": {
                    "boundary_weight": 1.0,
                    "lesion_weight": 0.8,
                    "contrast_weight": 0.6,
                },
                "loss_components": {
                    "boundary_preservation": True,
                    "tissue_contrast": True,
                    "local_structure": True,
                },
            },
            "agricultural": {
                "name": "Agricultural",
                "desc": "texture analysis, damage detection",
                "components": {
                    "texture_analysis": True,
                    "damage_detection": True,
                    "color_anomaly": True,
                    "pattern_enhancement": True,
                    "morphological_features": True,
                },
                "weights": {
                    "texture_weight": 1.0,
                    "damage_weight": 0.8,
                    "pattern_weight": 0.7,
                },
                "loss_components": {
                    "texture_preservation": True,
                    "damage_pattern": True,
                    "color_consistency": True,
                },
            },
        }

    def configure_general_parameters(self) -> None:
        """Configure general enhancement parameters"""
        enhancements = self.config["model"]["autoencoder_config"]["enhancements"]

        print("\nConfiguring General Enhancement Parameters:")

        # KL Divergence configuration
        if input("Enable KL divergence clustering? (y/n) [y]: ").lower() != "n":
            enhancements["use_kl_divergence"] = True
            weight = input("Enter KL divergence weight (0-1) [0.1]: ").strip()
            enhancements["kl_divergence_weight"] = float(weight) if weight else 0.1
        else:
            enhancements["use_kl_divergence"] = False
            enhancements["kl_divergence_weight"] = 0.0

        # Class encoding configuration
        if input("Enable class encoding? (y/n) [y]: ").lower() != "n":
            enhancements["use_class_encoding"] = True
            weight = input("Enter classification weight (0-1) [0.1]: ").strip()
            enhancements["classification_weight"] = float(weight) if weight else 0.1
        else:
            enhancements["use_class_encoding"] = False
            enhancements["classification_weight"] = 0.0

        # Configure additional parameters if KL divergence is enabled
        if enhancements["use_kl_divergence"]:
            self._configure_clustering_parameters(enhancements)

        # Phase 2 configuration
        if (
            input(
                "Enable phase 2 training (clustering and fine-tuning)? (y/n) [y]: "
            ).lower()
            != "n"
        ):
            self.config["model"]["autoencoder_config"]["enable_phase2"] = True
        else:
            self.config["model"]["autoencoder_config"]["enable_phase2"] = False

    def _configure_clustering_parameters(self, enhancements: Dict) -> None:
        """Configure clustering-specific parameters"""
        temp = input("Enter clustering temperature (0.1-2.0) [1.0]: ").strip()
        enhancements["clustering_temperature"] = float(temp) if temp else 1.0

        conf = input("Enter minimum cluster confidence (0-1) [0.7]: ").strip()
        enhancements["min_cluster_confidence"] = float(conf) if conf else 0.7

    def configure_enhancements(self) -> None:
        """Configure enhancement features with flexible combinations"""
        enabled_enhancements = []

        print("\nConfiguring Enhancement Features for General Mode:")
        print("You can enable any combination of features\n")

        # Let user choose enhancements
        for key, enhancement in self.enhancement_configs.items():
            prompt = f"Enable {enhancement['name']} features ({enhancement['desc']})? (y/n) [n]: "
            if input(prompt).lower() == "y":
                enabled_enhancements.append(key)
                self._add_enhancement(key, enhancement)
                print(f"{enhancement['name']} features added.")

        # Normalize weights and adjust learning rates
        self._normalize_weights(enabled_enhancements)
        self._adjust_learning_rates(len(enabled_enhancements))

        # Print configuration summary
        self._print_configuration_summary(enabled_enhancements)

    def _add_enhancement(self, key: str, enhancement: Dict) -> None:
        """Add specific enhancement configuration"""
        self.config["model"]["enhancement_modules"][key] = {
            "enabled": True,
            "components": enhancement["components"],
            "weights": enhancement["weights"],
        }

        self.config["model"]["loss_functions"][f"{key}_structure"] = {
            "enabled": True,
            "weight": 1.0,  # Will be normalized later
            "components": enhancement["loss_components"],
        }

    def _print_configuration_summary(self, enabled_enhancements: List[str]) -> None:
        """Print current configuration summary"""
        print("\nCurrent Enhancement Configuration:")
        if enabled_enhancements:
            for key in enabled_enhancements:
                enhancement = self.enhancement_configs[key]
                print(f"\n{enhancement['name']} Features:")
                print(
                    "- Components:",
                    ", ".join(
                        self.config["model"]["enhancement_modules"][key][
                            "components"
                        ].keys()
                    ),
                )
                print(
                    "- Weights:",
                    ", ".join(
                        f"{k}: {v}"
                        for k, v in self.config["model"]["enhancement_modules"][key][
                            "weights"
                        ].items()
                    ),
                )
        else:
            print("\nNo enhancements enabled. Using basic autoencoder configuration.")

        print(f"\nLearning Rates:")
        print(
            f"- Phase 1: {self.config['model']['autoencoder_config']['phase1_learning_rate']}"
        )
        print(
            f"- Phase 2: {self.config['model']['autoencoder_config']['phase2_learning_rate']}"
        )

    def _generate_confusion_matrix(
        self,
        true_labels: torch.Tensor,
        pred_labels: torch.Tensor,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Generate and display a colored confusion matrix.

        Args:
            true_labels: Ground truth labels
            pred_labels: Predicted labels
            class_names: List of class names for display
        """
        if not hasattr(self, "class_names") and class_names is None:
            logger.warning("No class names available for confusion matrix")
            return

        class_names = class_names if class_names is not None else self.class_names

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels.cpu().numpy(), pred_labels.cpu().numpy())

        # Normalize by row (true labels)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Create figure
        plt.figure(figsize=(12, 10))

        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )

        plt.title("Normalized Confusion Matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # Save to file
        cm_path = os.path.join(self.log_dir, "confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix saved to {cm_path}")


class BaseAutoencoder(nn.Module):
    """Base autoencoder class with all foundational methods"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: Dict):
        """Initialize base autoencoder with shape management and all core components.

        Args:
            input_shape: Tuple of (channels, height, width)
            feature_dims: Dimension of latent space features
            config: Configuration dictionary
        """
        super().__init__()

        # Basic configuration
        self.input_shape = (
            config["dataset"]["in_channels"],  # Use configured channels
            config["dataset"]["input_size"][0],
            config["dataset"]["input_size"][1],
        )
        input_shape = self.input_shape
        self.in_channels = config["dataset"]["in_channels"]
        self.feature_dims = feature_dims
        self.config = config
        self.train_dataset = None

        # Device configuration
        self.device = torch.device(
            "cuda"
            if config["execution_flags"]["use_gpu"] and torch.cuda.is_available()
            else "cpu"
        )

        # Shape tracking initialization
        self.shape_registry = {"input": input_shape}
        self.spatial_dims = []
        current_size = input_shape[1]

        # Calculate layer dimensions
        self.layer_sizes = self._calculate_layer_sizes()

        # Track progressive spatial dimensions
        for _ in self.layer_sizes:
            self.spatial_dims.append(current_size)
            current_size = current_size // 2

        # Final dimensions
        self.final_spatial_dim = current_size
        self.flattened_size = self.layer_sizes[-1] * (self.final_spatial_dim**2)

        # Register key dimensions in shape registry
        self.shape_registry.update(
            {
                "final_spatial": (self.final_spatial_dim, self.final_spatial_dim),
                "flattened": (self.flattened_size,),
                "latent": (self.feature_dims,),
            }
        )

        # Initialize checkpoint paths
        self.checkpoint_dir = config["training"]["checkpoint_dir"]
        self.dataset_name = config["dataset"]["name"]
        self.checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{self.dataset_name}_unified.pth"
        )

        # Create network layers
        self.encoder_layers = self._create_encoder_layers()
        self.embedder = self._create_embedder()
        self.unembedder = self._create_unembedder()
        self.decoder_layers = self._create_decoder_layers()

        # Initialize enhancement components
        self.use_kl_divergence = (
            config["model"]
            .get("autoencoder_config", {})
            .get("enhancements", True)
            .get("use_kl_divergence", True)
        )

        self.use_class_encoding = (
            config["model"]
            .get("autoencoder_config", {})
            .get("enhancements", {})
            .get("use_class_encoding", True)
        )

        # Initialize classifier if class encoding is enabled
        if self.use_class_encoding:
            num_classes = config["dataset"].get("num_classes", 10)
            self.classifier = nn.Sequential(
                nn.Linear(feature_dims, feature_dims // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(feature_dims // 2, num_classes),
            )
            self.shape_registry["classifier_output"] = (num_classes,)

        # Training phase tracking
        self.training_phase = 1  # Start with phase 1

        # Initialize latent organization
        self._initialize_latent_organization()

        # Move model to appropriate device
        self.to(self.device)

        # Register shapes for encoder/decoder paths
        for idx, size in enumerate(self.layer_sizes):
            spatial_dim = self.spatial_dims[idx]
            self.shape_registry[f"encoder_{idx}"] = (size, spatial_dim, spatial_dim)
            self.shape_registry[f"decoder_{idx}"] = (size, spatial_dim, spatial_dim)

        # Initialize training metrics
        self.best_loss = float("inf")
        self.best_accuracy = 0.0
        self.current_epoch = 0
        self.history = defaultdict(list)
        # Initialize clustering parameters
        self._initialize_clustering(config)

    # --------------------------Distance Correlations ----------
    def get_high_confidence_samples(self, dataloader, threshold=0.9):
        """Identify high-confidence predictions for semi-supervised learning"""
        self.eval()
        confident_samples = []

        with torch.no_grad():
            for data, _ in dataloader:  # Note: ignoring true labels
                output = self(data)
                if "cluster_probabilities" in output:
                    probs, preds = output["cluster_probabilities"].max(1)
                    mask = probs > threshold
                    confident_samples.append((data[mask], preds[mask]))

        return torch.cat(confident_samples) if confident_samples else None

    def _select_features_using_distance_correlation(self, features, labels, config):
        """Select features based on distance correlation criteria"""
        selector = DistanceCorrelationFeatureSelector(
            upper_threshold=config["distance_correlation_upper"],
            lower_threshold=config["distance_correlation_lower"],
        )

        selected_indices, corr_values = selector.select_features(features, labels)

        # Create new feature matrix with only selected features
        selected_features = features[:, selected_indices]
        feature_names = [f"feature_{i}" for i in selected_indices]

        return selected_features, feature_names, corr_values

    def save_features(
        self,
        train_features: Dict[str, torch.Tensor],
        test_features: Dict[str, torch.Tensor],
        output_path: str,
    ) -> None:
        """
        Save features for training and test sets with distance correlation feature selection.
        Handles both adaptive and non-adaptive modes with proper configuration.

        Args:
            train_features: Features from training set (dict with 'embeddings', 'labels', etc.)
            test_features: Features from test set (same format as train_features)
            output_path: Full path for output CSV file (e.g., 'data/dataset/features.csv')
        """
        try:
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)

            # Load distance correlation config (with proper error handling)
            dc_config = self._get_distance_correlation_config(output_dir)
            use_dc = dc_config.get("use_distance_correlation", True)

            # Get adaptive mode setting
            enable_adaptive = self.config["model"].get("enable_adaptive", True)

            # Process features based on mode
            if enable_adaptive:
                # Adaptive mode - single combined output
                features_df = self._prepare_features_dataframe(
                    train_features, dc_config if use_dc else None
                )

                # Save features
                features_df.to_csv(output_path, index=False)
                logger.info(f"Saved features to {output_path} (adaptive mode)")

                # Save metadata
                feature_columns = [
                    c for c in features_df.columns if c.startswith("feature_")
                ]
                self._save_feature_metadata(
                    output_dir, feature_columns, dc_config if use_dc else None
                )
            else:
                # Non-adaptive mode - separate train/test files
                train_df = self._prepare_features_dataframe(
                    train_features, dc_config if use_dc else None
                )
                test_df = self._prepare_features_dataframe(
                    test_features, dc_config if use_dc else None
                )

                # Save features
                train_path = output_path.replace(".csv", "_train.csv")
                test_path = output_path.replace(".csv", "_test.csv")

                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)
                logger.info(f"Saved train features to {train_path}")
                logger.info(f"Saved test features to {test_path}")

                # Save metadata (using train features as reference)
                feature_columns = [
                    c for c in train_df.columns if c.startswith("feature_")
                ]
                self._save_feature_metadata(
                    output_dir, feature_columns, dc_config if use_dc else None
                )

        except Exception as e:
            logger.error(f"Error in save_features: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to save features: {str(e)}")

    def _prepare_features_dataframe(
        self, features: Dict[str, torch.Tensor], dc_config: Optional[Dict]
    ) -> pd.DataFrame:
        """
        Prepare DataFrame from features with optional distance correlation selection.

        Args:
            features: Dictionary containing 'embeddings', 'labels', etc.
            dc_config: Distance correlation config or None if disabled

        Returns:
            pd.DataFrame: Processed features with metadata
        """
        data = {}

        # Convert features to numpy
        embeddings = features["embeddings"].cpu().numpy()
        labels = features.get("labels", torch.zeros(len(embeddings))).cpu().numpy()

        # Apply feature selection if enabled
        if dc_config and dc_config.get("use_distance_correlation", True):
            selector = DistanceCorrelationFeatureSelector(
                upper_threshold=dc_config["distance_correlation_upper"],
                lower_threshold=dc_config["distance_correlation_lower"],
            )
            selected_indices, corr_values = selector.select_features(embeddings, labels)

            # Store selected features with metadata
            for new_idx, orig_idx in enumerate(selected_indices):
                data[f"feature_{new_idx}"] = embeddings[:, orig_idx]
                data[f"original_feature_idx_{new_idx}"] = orig_idx
                data[f"feature_{new_idx}_correlation"] = corr_values[orig_idx]
        else:
            # Store all features without selection
            for i in range(embeddings.shape[1]):
                data[f"feature_{i}"] = embeddings[:, i]

        # Add labels and metadata
        if "class_names" in features:
            data["target"] = features["class_names"]
        else:
            data["target"] = labels

        if "filenames" in features:
            data["filename"] = features["filenames"]

        return pd.DataFrame(data)

    def _get_distance_correlation_config(self, output_dir: str) -> Dict:
        """
        Safely load or create distance correlation configuration.

        Args:
            output_dir: Directory where config should be stored

        Returns:
            Dictionary with configuration parameters
        """
        config_path = os.path.join(output_dir, "feature_selection_config.json")
        default_config = {
            "use_distance_correlation": True,
            "distance_correlation_upper": 0.85,
            "distance_correlation_lower": 0.01,
        }

        try:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                # Validate loaded config
                return {**default_config, **config}
            else:
                with open(config_path, "w") as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            logger.warning(f"Could not load/create config: {str(e)} - using defaults")
            return default_config

    def generate_configuration(self, features_path: str, output_dir: str = None) -> str:
        """
        Generate final .conf configuration file from extracted features.
        Should be called after save_features().

        Args:
            features_path: Path to saved features CSV
            output_dir: Optional output directory (default: same as features)

        Returns:
            Path to generated .conf file
        """
        try:
            if output_dir is None:
                output_dir = os.path.dirname(features_path)
            os.makedirs(output_dir, exist_ok=True)

            # Load features and metadata
            features_df = pd.read_csv(features_path)
            metadata_path = os.path.join(output_dir, "feature_selection_metadata.json")

            # Get config settings
            dc_config = self._get_distance_correlation_config(output_dir)
            use_dc = dc_config.get("use_distance_correlation", True)

            # Prepare base configuration
            config = {
                "dataset": self.config["dataset"]["name"],
                "feature_count": len(
                    [c for c in features_df.columns if c.startswith("feature_")]
                ),
                "generated_at": datetime.now().isoformat(),
                "feature_selection": {
                    "method": "distance_correlation" if use_dc else "none",
                    "parameters": dc_config if use_dc else {},
                },
            }

            # Add feature-specific info if selection was used
            if use_dc and os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)
                config["feature_selection"].update(
                    {
                        "original_feature_count": metadata["original_feature_count"],
                        "selected_features": [
                            {
                                "name": col,
                                "original_index": features_df[
                                    f"original_feature_idx_{i}"
                                ][0],
                                "correlation": features_df[f"feature_{i}_correlation"][
                                    0
                                ],
                            }
                            for i, col in enumerate(
                                [
                                    c
                                    for c in features_df.columns
                                    if c.startswith("feature_")
                                ]
                            )
                        ],
                    }
                )

            # Save .conf file
            conf_path = os.path.join(
                output_dir, f"{self.config['dataset']['name']}.conf"
            )
            with open(conf_path, "w") as f:
                json.dump(config, f, indent=4)

            logger.info(f"Configuration file generated at {conf_path}")
            return conf_path

        except Exception as e:
            logger.error(f"Error generating configuration: {str(e)}")
            raise

    def _save_feature_metadata(
        self, output_dir: str, feature_columns: List[str], dc_config: Dict = None
    ):
        """Save comprehensive metadata about the saved features and feature selection process.

        Args:
            output_dir (str): Directory where features are being saved
            feature_columns (List[str]): List of feature column names that were saved
            dc_config (Dict): Distance correlation configuration (optional)
        """
        # Prepare base metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "feature_info": {
                "total_features": len(feature_columns),
                "feature_columns": feature_columns,
                "feature_selection": {
                    "method": "distance_correlation"
                    if dc_config and dc_config.get("use_distance_correlation", True)
                    else "none",
                    "parameters": dc_config
                    if dc_config and dc_config.get("use_distance_correlation", True)
                    else None,
                },
            },
            "model_config": {
                "type": self.__class__.__name__,
                "feature_dims": self.feature_dims,
                "training_phase": self.training_phase,
                "enhancements": {
                    "use_kl_divergence": self.use_kl_divergence,
                    "use_class_encoding": self.use_class_encoding,
                },
            },
            "dataset_info": {
                "name": self.config["dataset"]["name"],
                "input_size": self.config["dataset"]["input_size"],
                "channels": self.config["dataset"]["in_channels"],
            },
        }

        # Save the metadata
        metadata_path = os.path.join(output_dir, "feature_extraction_metadata.json")

        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Saved comprehensive feature metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving feature metadata: {str(e)}")
            raise

    def _features_to_dataframe(
        self, feature_dict: Dict[str, torch.Tensor], dc_config: Dict
    ) -> pd.DataFrame:
        """
        Convert features dictionary to a pandas DataFrame with optional feature selection.

        Args:
            feature_dict (Dict): Dictionary containing features and metadata
            dc_config (Dict): Distance correlation configuration

        Returns:
            pd.DataFrame: DataFrame containing selected features and metadata
        """
        data_dict = {}

        # Get base length from embeddings
        base_length = (
            len(feature_dict["embeddings"]) if "embeddings" in feature_dict else 0
        )
        if base_length == 0:
            raise ValueError("No embeddings found in features")

        # Process embeddings
        features = feature_dict["embeddings"].cpu().numpy()

        # Apply feature selection if enabled
        if dc_config.get("use_distance_correlation", True) and "labels" in feature_dict:
            labels = feature_dict["labels"].cpu().numpy()

            # Select features based on distance correlation
            selector = DistanceCorrelationFeatureSelector(
                upper_threshold=dc_config["distance_correlation_upper"],
                lower_threshold=dc_config["distance_correlation_lower"],
            )
            selected_indices, corr_values = selector.select_features(features, labels)

            # Store only selected features
            for new_idx, orig_idx in enumerate(selected_indices):
                data_dict[f"feature_{new_idx}"] = features[:, orig_idx]
                data_dict[f"original_feature_idx_{new_idx}"] = orig_idx
                data_dict[f"feature_{new_idx}_correlation"] = corr_values[orig_idx]
        else:
            # Include all features if selection is disabled
            for i in range(features.shape[1]):
                data_dict[f"feature_{i}"] = features[:, i]

        # Process labels and class names
        if "class_names" in feature_dict:
            if len(feature_dict["class_names"]) == base_length:
                data_dict["target"] = feature_dict["class_names"]
        elif "labels" in feature_dict:
            if len(feature_dict["labels"]) == base_length:
                data_dict["target"] = feature_dict["labels"].cpu().numpy()

        # Include additional metadata if available
        optional_fields = ["indices", "filenames"]
        for field in optional_fields:
            if field in feature_dict and len(feature_dict[field]) == base_length:
                data_dict[field] = feature_dict[field]

        return pd.DataFrame(data_dict)

    def _save_feature_selection_metadata(
        self, features: Dict[str, torch.Tensor], dc_config: Dict, output_dir: str
    ) -> None:
        """
        Save metadata about feature selection process.

        Args:
            features (Dict): Original feature dictionary
            dc_config (Dict): Distance correlation configuration
            output_dir (str): Directory to save metadata files
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "config": dc_config,
            "original_feature_count": features["embeddings"].shape[1],
            "description": "Feature selection metadata",
        }

        metadata_path = os.path.join(output_dir, "feature_selection_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved feature selection metadata to {metadata_path}")

    # --------------------------
    def _initialize_clustering(self, config: Dict):
        """Initialize clustering parameters with existence check"""
        self.use_kl_divergence = config["model"]["autoencoder_config"]["enhancements"][
            "use_kl_divergence"
        ]

        if self.use_kl_divergence:
            # Only initialize if not already exists
            if not hasattr(self, "cluster_centers"):
                num_clusters = config["dataset"].get("num_classes", 10)
                self.register_buffer(
                    "cluster_centers", torch.randn(num_clusters, self.feature_dims)
                )

        # ALWAYS initialize temperature as a tensor buffer
        temp_value = config["model"]["autoencoder_config"]["enhancements"][
            "clustering_temperature"
        ]
        if not hasattr(self, "clustering_temperature"):
            self.register_buffer(
                "clustering_temperature",
                torch.tensor([float(temp_value)], dtype=torch.float32),
            )
        else:
            # Ensure existing temperature is a tensor
            if not isinstance(self.clustering_temperature, torch.Tensor):
                self.clustering_temperature = torch.tensor(
                    [float(temp_value)], dtype=torch.float32, device=self.device
                )

    def state_dict(self, *args, **kwargs):
        """Extend state dict to include all necessary components"""
        state = super().state_dict(*args, **kwargs)

        # Add clustering parameters if they exist
        if hasattr(self, "cluster_centers"):
            state["cluster_centers"] = self.cluster_centers
        # Ensure temperature is saved as tensor
        if hasattr(self, "clustering_temperature"):
            if not isinstance(self.clustering_temperature, torch.Tensor):
                self.clustering_temperature = torch.tensor(
                    [float(self.clustering_temperature)],
                    dtype=torch.float32,
                    device=self.device,
                )
            state["clustering_temperature"] = self.clustering_temperature

        # Add classifier if it exists
        if hasattr(self, "classifier"):
            state["classifier_state"] = self.classifier.state_dict()

        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict including all components"""
        # Load main model state
        missing_keys, unexpected_keys = super().load_state_dict(
            state_dict, strict=False
        )

        # Load clustering parameters
        if "cluster_centers" in state_dict:
            if not hasattr(self, "cluster_centers"):
                self.register_buffer("cluster_centers", state_dict["cluster_centers"])
            else:
                self.cluster_centers.data.copy_(state_dict["cluster_centers"])

        # Handle clustering temperature
        if "clustering_temperature" in state_dict:
            temp = state_dict["clustering_temperature"]
            if not isinstance(temp, torch.Tensor):
                temp = torch.tensor(
                    [float(temp)], dtype=torch.float32, device=self.device
                )

            if not hasattr(self, "clustering_temperature"):
                self.register_buffer("clustering_temperature", temp)
            else:
                self.clustering_temperature.data.copy_(temp)

        # Load classifier if it exists
        if "classifier_state" in state_dict and hasattr(self, "classifier"):
            self.classifier.load_state_dict(state_dict["classifier_state"])

        if strict:
            if missing_keys:
                raise RuntimeError(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys: {unexpected_keys}")

    # --------------------------
    def set_dataset(self, dataset: Dataset):
        """Store dataset reference"""
        self.train_dataset = dataset

    def _initialize_latent_organization(self):
        """Initialize latent space organization components with existence checks"""
        self.use_kl_divergence = (
            self.config["model"]
            .get("autoencoder_config", {})
            .get("enhancements", {})
            .get("use_kl_divergence", True)
        )
        self.use_class_encoding = (
            self.config["model"]
            .get("autoencoder_config", {})
            .get("enhancements", {})
            .get("use_class_encoding", True)
        )

        if self.use_class_encoding and not hasattr(self, "classifier"):
            num_classes = self.config["dataset"].get("num_classes", 10)
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dims, self.feature_dims // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.feature_dims // 2, num_classes),
            )

        if self.use_kl_divergence:
            if not hasattr(self, "cluster_centers"):
                num_clusters = self.config["dataset"].get("num_classes", 10)
                self.cluster_centers = nn.Parameter(
                    torch.randn(num_clusters, self.feature_dims)
                )
            if not hasattr(self, "clustering_temperature"):
                self.clustering_temperature = (
                    self.config["model"]
                    .get("autoencoder_config", {})
                    .get("enhancements", {})
                    .get("clustering_temperature", 1.0)
                )

    def set_training_phase(self, phase: int):
        """Set the training phase (1 or 2) with proper cluster initialization"""
        self.training_phase = phase
        if phase == 2 and self.use_kl_divergence:
            if not hasattr(self, "cluster_centers"):
                # Initialize only if not already initialized
                num_clusters = self.config["dataset"].get("num_classes", 10)
                self.cluster_centers = nn.Parameter(
                    torch.randn(num_clusters, self.feature_dims, device=self.device)
                )
                self.clustering_temperature = (
                    self.config["model"]
                    .get("autoencoder_config", {})
                    .get("enhancements", {})
                    .get("clustering_temperature", 1.0)
                )

    def _initialize_cluster_centers(self):
        """Initialize cluster centers using k-means"""
        self.eval()
        with torch.no_grad():
            if self.train_dataset is None:
                raise ValueError("Dataset not set. Call set_dataset() before training.")

            # Use stored dataset reference
            dataloader = DataLoader(self.train_dataset, batch_size=1000, shuffle=True)
            batch_data, _ = next(iter(dataloader))
            embeddings = self.encode(batch_data.to(self.device))
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]

            # Use k-means to initialize cluster centers
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=self.cluster_centers.size(0), n_init=20)
            kmeans.fit(embeddings.cpu().numpy())
            self.cluster_centers.data = torch.tensor(
                kmeans.cluster_centers_, device=self.device
            )

    def _calculate_layer_sizes(self) -> List[int]:
        """Calculate progressive channel sizes for encoder/decoder"""
        base_channels = 32
        sizes = []
        current_size = base_channels

        min_dim = min(self.input_shape[1:])
        max_layers = int(np.log2(min_dim)) - 2

        for _ in range(max_layers):
            sizes.append(current_size)
            if current_size < 1024:
                current_size *= 2

        logging.info(f"Layer sizes: {sizes}")
        return sizes

    def _create_encoder_layers(self) -> nn.ModuleList:
        """Create encoder layers"""
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for size in self.layer_sizes:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, size, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(size),
                    nn.LeakyReLU(0.2),
                )
            )
            in_channels = size

        return layers

    def _create_embedder(self) -> nn.Sequential:
        """Create embedder layers"""
        return nn.Sequential(
            nn.Linear(self.flattened_size, self.feature_dims),
            nn.BatchNorm1d(self.feature_dims),
            nn.LeakyReLU(0.2),
        )

    def _create_unembedder(self) -> nn.Sequential:
        """Create unembedder layers"""
        return nn.Sequential(
            nn.Linear(self.feature_dims, self.flattened_size),
            nn.BatchNorm1d(self.flattened_size),
            nn.LeakyReLU(0.2),
        )

    def _create_decoder_layers(self) -> nn.ModuleList:
        """Create decoder layers"""
        layers = nn.ModuleList()
        in_channels = self.layer_sizes[-1]

        for i in range(len(self.layer_sizes) - 1, -1, -1):
            out_channels = self.in_channels if i == 0 else self.layer_sizes[i - 1]
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(out_channels) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2) if i > 0 else nn.Tanh(),
                )
            )
            in_channels = out_channels

        return layers

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Basic encoding process"""
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return self.embedder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Basic decoding process"""
        x = self.unembedder(x)
        x = x.view(
            x.size(0),
            self.layer_sizes[-1],
            self.final_spatial_dim,
            self.final_spatial_dim,
        )

        for layer in self.decoder_layers:
            x = layer(x)

        return x

    def forward(
        self, x: torch.Tensor
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with flexible output format"""
        embedding = self.encode(x)
        if isinstance(embedding, tuple):
            embedding = embedding[0]
        reconstruction = self.decode(embedding)

        if self.training_phase == 2:
            # Return dictionary format in phase 2
            output = {"embedding": embedding, "reconstruction": reconstruction}

            if self.use_class_encoding and hasattr(self, "classifier"):
                class_logits = self.classifier(embedding)
                output["class_logits"] = class_logits
                output["class_predictions"] = class_logits.argmax(dim=1)

            if self.use_kl_divergence:
                latent_info = self.organize_latent_space(embedding)
                output.update(latent_info)

            return output
        else:
            # Return tuple format in phase 1
            return embedding, reconstruction

    def get_encoding_shape(self) -> Tuple[int, ...]:
        """Get the shape of the encoding at each layer"""
        return tuple([size for size in self.layer_sizes])

    def get_spatial_dims(self) -> List[List[int]]:
        """Get the spatial dimensions at each layer"""
        return self.spatial_dims.copy()

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to image array with proper normalization"""
        tensor = tensor.detach().cpu()
        if len(tensor.shape) == 3:
            tensor = tensor.permute(1, 2, 0)

        mean = torch.tensor(self.config["dataset"]["mean"]).view(1, 1, -1)
        std = torch.tensor(self.config["dataset"]["std"]).view(1, 1, -1)
        tensor = tensor * std + mean

        return (tensor.clamp(0, 1) * 255).numpy().astype(np.uint8)

    def save_image(self, tensor: torch.Tensor, path: str):
        """Save tensor as image with proper normalization"""
        img_array = self._tensor_to_image(tensor)
        img = Image.fromarray(img_array)

        # Ensure target size
        target_size = tuple(self.config["dataset"]["input_size"])
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)

        img.save(path, quality=99, optimize=True)
        logging.debug(f"Saved image to {path} with size {img.size}")

    def plot_reconstruction_samples(
        self, inputs: torch.Tensor, save_path: Optional[str] = None
    ) -> None:
        """Visualize original and reconstructed images"""
        self.eval()
        with torch.no_grad():
            embedding = self.encode(inputs)
            reconstructions = self.decode(embedding)

        num_samples = min(inputs.size(0), 8)
        fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))

        for i in range(num_samples):
            # Original
            axes[0, i].imshow(self._tensor_to_image(inputs[i]))
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_title("Original")

            # Reconstruction
            axes[1, i].imshow(self._tensor_to_image(reconstructions[i]))
            axes[1, i].axis("off")
            if i == 0:
                axes[1, i].set_title("Reconstructed")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Reconstruction samples saved to {save_path}")
        plt.close()

    def extract_features(
        self, loader: DataLoader, dataset_type: str = "train"
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from a DataLoader with improved label handling.

        Args:
            loader (DataLoader): DataLoader for the dataset.
            dataset_type (str): Type of dataset ("train" or "test"). Defaults to "train".

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing extracted features and metadata.
        """
        self.eval()
        all_embeddings = []
        all_labels = []
        all_indices = []  # Store file indices
        all_filenames = []  # Store filenames
        all_class_names = []  # Store actual class names

        try:
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(
                    tqdm(loader, desc=f"Extracting {dataset_type} features")
                ):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Get metadata if available, otherwise use placeholders
                    if hasattr(loader.dataset, "get_additional_info"):
                        # Custom dataset with metadata
                        indices = [
                            loader.dataset.get_additional_info(idx)[0]
                            for idx in range(len(inputs))
                        ]
                        filenames = [
                            loader.dataset.get_additional_info(idx)[1]
                            for idx in range(len(inputs))
                        ]

                        # Improved class name handling
                        if hasattr(loader.dataset, "reverse_encoder"):
                            class_names = [
                                loader.dataset.reverse_encoder[label.item()]
                                for label in labels
                            ]
                        elif hasattr(loader.dataset, "classes"):
                            class_names = [
                                loader.dataset.classes[label.item()] for label in labels
                            ]
                        else:
                            class_names = [f"class_{label.item()}" for label in labels]
                    else:
                        # Dataset without metadata (e.g., torchvision)
                        indices = [
                            f"unavailable_{batch_idx}_{i}" for i in range(len(inputs))
                        ]
                        filenames = [
                            f"unavailable_{batch_idx}_{i}" for i in range(len(inputs))
                        ]

                        # Better fallback for class names
                        if hasattr(loader.dataset, "classes"):
                            class_names = [
                                loader.dataset.classes[label.item()] for label in labels
                            ]
                        else:
                            class_names = [str(label.item()) for label in labels]

                    # Extract embeddings
                    embeddings = self.encode(inputs)
                    if isinstance(embeddings, tuple):
                        embeddings = embeddings[0]

                    # Append to lists
                    all_embeddings.append(embeddings)
                    all_labels.append(labels)
                    all_indices.extend(indices)
                    all_filenames.extend(filenames)
                    all_class_names.extend(class_names)

                # Concatenate all results
                embeddings = torch.cat(all_embeddings)
                labels = torch.cat(all_labels)

                feature_dict = {
                    "embeddings": embeddings,
                    "labels": labels,
                    "indices": all_indices,
                    "filenames": all_filenames,
                    "class_names": all_class_names,  # Now contains proper class names in all cases
                }

                return feature_dict

        except Exception as e:
            logger.error(f"Error during feature extraction: {str(e)}")
            raise

    def get_enhancement_features(
        self, embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Hook method for enhanced models to add specialized features.
        Override this in derived classes to add model-specific features.
        """
        return {}

    def _get_enhancement_columns(
        self, feature_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        """Extract enhancement-specific features for saving"""
        enhancement_dict = {}

        # Handle class probabilities
        if "class_probabilities" in feature_dict:
            probs = feature_dict["class_probabilities"].cpu().numpy()
            for i in range(probs.shape[1]):
                enhancement_dict[f"class_{i}_probability"] = probs[:, i]

        # Handle cluster assignments
        if "cluster_assignments" in feature_dict:
            enhancement_dict["cluster_assignment"] = (
                feature_dict["cluster_assignments"].cpu().numpy()
            )

        # Handle cluster probabilities
        if "cluster_probabilities" in feature_dict:
            cluster_probs = feature_dict["cluster_probabilities"].cpu().numpy()
            for i in range(cluster_probs.shape[1]):
                enhancement_dict[f"cluster_{i}_probability"] = cluster_probs[:, i]

        # Add confidence scores if available
        if "class_logits" in feature_dict:
            logits = feature_dict["class_logits"].cpu().numpy()
            enhancement_dict["prediction_confidence"] = softmax(logits, axis=1).max(
                axis=1
            )

        return enhancement_dict

    def organize_latent_space(
        self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Organize latent space using KL divergence with consistent behavior for prediction"""
        output = {"embeddings": embeddings}  # Keep on same device as input

        if self.use_kl_divergence and hasattr(self, "cluster_centers"):
            # Ensure cluster centers are on same device
            cluster_centers = self.cluster_centers.to(embeddings.device)
            self.clustering_temperature

            # Calculate distances to cluster centers
            distances = torch.cdist(embeddings, cluster_centers)

            # Convert distances to probabilities (soft assignments)
            q_dist = 1.0 / (1.0 + (distances / self.clustering_temperature) ** 2)
            q_dist = q_dist / q_dist.sum(dim=1, keepdim=True)

            if labels is not None:
                # Create target distribution if labels are provided
                p_dist = torch.zeros_like(q_dist)
                for i in range(self.cluster_centers.size(0)):
                    mask = labels == i
                    if mask.any():
                        p_dist[mask, i] = 1.0
            else:
                # During prediction, use current distribution as target
                p_dist = q_dist.detach()  # Stop gradient for target

            output.update(
                {
                    "cluster_probabilities": q_dist,
                    "target_distribution": p_dist,
                    "cluster_assignments": q_dist.argmax(dim=1),
                    "cluster_confidence": q_dist.max(dim=1)[0],
                }
            )

        if self.use_class_encoding and hasattr(self, "classifier"):
            class_logits = self.classifier(embeddings)
            output.update(
                {
                    "class_logits": class_logits,
                    "class_predictions": class_logits.argmax(dim=1),
                    "class_probabilities": F.softmax(class_logits, dim=1),
                }
            )

        return output


class AstronomicalStructurePreservingAutoencoder(BaseAutoencoder):
    """Autoencoder specialized for astronomical imaging features"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: Dict):
        super().__init__(input_shape, feature_dims, config)

        # Get enhancement configurations
        self.enhancement_config = config["model"]["enhancement_modules"]["astronomical"]

        # Initial channel transformation layer
        self.initial_transform = nn.Sequential(
            nn.Conv2d(self.in_channels, self.layer_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.layer_sizes[0]),
            nn.LeakyReLU(0.2),
        )

        # Detail preservation module with multiple scales
        self.detail_preserving = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.layer_sizes[0],
                        self.layer_sizes[0],
                        kernel_size=k,
                        padding=k // 2,
                    ),
                    nn.BatchNorm2d(self.layer_sizes[0]),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.layer_sizes[0], self.layer_sizes[0], kernel_size=1),
                )
                for k in [3, 5, 7]
            ]
        )

        # Star detection module
        self.star_detector = nn.Sequential(
            nn.Conv2d(
                self.layer_sizes[0], self.layer_sizes[0], kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(self.layer_sizes[0]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.layer_sizes[0], self.layer_sizes[0], kernel_size=1),
            nn.Sigmoid(),
        )

        # Galaxy feature enhancement
        self.galaxy_enhancer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(size, size, kernel_size=3, padding=d, dilation=d),
                    nn.BatchNorm2d(size),
                    nn.LeakyReLU(0.2),
                )
                for size, d in zip(self.layer_sizes, [1, 2, 4])
            ]
        )

        # Initialize loss function
        self.structure_loss = (
            AstronomicalStructureLoss() if self.enhancement_config["enabled"] else None
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced encoding with astronomical feature preservation"""
        features = {}

        # Initial channel transformation
        x = self.initial_transform(x)

        if self.enhancement_config["components"]["detail_preservation"]:
            # Multi-scale detail extraction
            detail_features = [module(x) for module in self.detail_preserving]
            features["details"] = sum(detail_features) / len(detail_features)
            x = x + 0.1 * features["details"]

        if self.enhancement_config["components"]["star_detection"]:
            # Star detection
            features["stars"] = self.star_detector(x)
            x = x * (1 + 0.1 * features["stars"])

        # Regular encoding path with galaxy enhancement
        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if self.enhancement_config["components"]["galaxy_features"]:
                if idx < len(self.galaxy_enhancer):
                    galaxy_features = self.galaxy_enhancer[idx](x)
                    x = x + 0.1 * galaxy_features

        x = x.view(x.size(0), -1)
        embedding = self.embedder(x)

        return embedding, features

    def decode(
        self, embedding: torch.Tensor, features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Enhanced decoding with structure preservation"""
        x = self.unembedder(embedding)
        x = x.view(
            x.size(0),
            self.layer_sizes[-1],
            self.final_spatial_dim,
            self.final_spatial_dim,
        )

        for layer in self.decoder_layers:
            x = layer(x)

        # Final channel transformation back to input channels
        x = nn.Conv2d(self.layer_sizes[0], self.in_channels, kernel_size=1)(x)

        # Add preserved features if available
        if self.enhancement_config["components"]["detail_preservation"]:
            if "details" in features:
                x = x + 0.1 * features["details"]

        if self.enhancement_config["components"]["star_detection"]:
            if "stars" in features:
                x = x * (1 + 0.1 * features["stars"])

        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced feature preservation"""
        embedding, features = self.encode(x)
        reconstruction = self.decode(embedding, features)

        output = {"embedding": embedding, "reconstruction": reconstruction}

        if self.training and self.enhancement_config["enabled"]:
            loss = self.structure_loss(reconstruction, x)
            output["loss"] = loss

        return output


class MedicalStructurePreservingAutoencoder(BaseAutoencoder):
    """Autoencoder specialized for medical imaging features"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: Dict):
        super().__init__(input_shape, feature_dims, config)

        # Get enhancement configurations
        self.enhancement_config = config["model"]["enhancement_modules"]["medical"]

        # Tissue boundary detection
        self.boundary_detector = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=4),
            nn.PReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Lesion detection module
        self.lesion_detector = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=3, padding=d, dilation=d),
                    nn.InstanceNorm2d(32),
                    nn.PReLU(),
                )
                for d in [1, 2, 4]
            ]
        )

        # Contrast enhancement module
        self.contrast_enhancer = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2),
            nn.InstanceNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, self.in_channels, kernel_size=1),
        )

        # Initialize loss function
        self.structure_loss = (
            MedicalStructureLoss() if self.enhancement_config["enabled"] else None
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced encoding with medical feature preservation"""
        features = {}

        if self.enhancement_config["components"]["tissue_boundary"]:
            features["boundaries"] = self.boundary_detector(x)
            x = x * (1 + 0.1 * features["boundaries"])

        # Regular encoding path with lesion detection
        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if self.enhancement_config["components"]["lesion_detection"]:
                if idx < len(self.lesion_detector):
                    lesion_features = self.lesion_detector[idx](x)
                    x = x + 0.1 * lesion_features

        if self.enhancement_config["components"]["contrast_enhancement"]:
            features["contrast"] = self.contrast_enhancer(x)
            x = x + 0.1 * features["contrast"]

        x = x.view(x.size(0), -1)
        embedding = self.embedder(x)

        return embedding, features

    def decode(
        self, embedding: torch.Tensor, features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Enhanced decoding with structure preservation"""
        x = self.unembedder(embedding)
        x = x.view(
            x.size(0),
            self.layer_sizes[-1],
            self.final_spatial_dim,
            self.final_spatial_dim,
        )

        for layer in self.decoder_layers:
            x = layer(x)

        # Add preserved features
        if self.enhancement_config["components"]["tissue_boundary"]:
            x = x * (1 + 0.1 * features.get("boundaries", 0))

        if self.enhancement_config["components"]["contrast_enhancement"]:
            x = x + 0.1 * features.get("contrast", 0)

        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced feature preservation"""
        embedding, features = self.encode(x)
        reconstruction = self.decode(embedding, features)

        output = {"embedding": embedding, "reconstruction": reconstruction}

        if self.training and self.enhancement_config["enabled"]:
            loss = self.structure_loss(reconstruction, x)
            output["loss"] = loss

        return output


class AgriculturalPatternAutoencoder(BaseAutoencoder):
    """Autoencoder specialized for agricultural imaging features"""

    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: Dict):
        super().__init__(input_shape, feature_dims, config)

        # Get enhancement configurations
        self.enhancement_config = config["model"]["enhancement_modules"]["agricultural"]

        # Ensure channel numbers are compatible with groups
        texture_groups = min(
            4, self.in_channels
        )  # Adjust groups based on input channels
        intermediate_channels = 32 - (32 % texture_groups)  # Ensure divisible by groups

        self.texture_analyzer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.in_channels,
                        intermediate_channels,
                        kernel_size=k,
                        padding=k // 2,
                        groups=texture_groups,
                    ),
                    nn.InstanceNorm2d(intermediate_channels),
                    nn.PReLU(),
                    nn.Conv2d(
                        intermediate_channels,
                        intermediate_channels,
                        kernel_size=k,
                        padding=k // 2,
                        groups=texture_groups,
                    ),
                )
                for k in [3, 5, 7]
            ]
        )

        # Damage pattern detector
        damage_intermediate_channels = 32 - (32 % self.in_channels)  # Ensure divisible
        self.damage_detector = nn.Sequential(
            nn.Conv2d(
                self.in_channels, damage_intermediate_channels, kernel_size=3, padding=1
            ),
            nn.PReLU(),
            nn.Conv2d(damage_intermediate_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Color anomaly detection
        self.color_analyzer = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(32, self.in_channels, kernel_size=1),
        )

        # Initialize loss function
        self.pattern_loss = (
            AgriculturalPatternLoss() if self.enhancement_config["enabled"] else None
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced encoding with pattern preservation"""
        features = {}

        if self.enhancement_config["components"]["texture_analysis"]:
            texture_features = [module(x) for module in self.texture_analyzer]
            features["texture"] = sum(texture_features) / len(texture_features)
            x = x + 0.1 * features["texture"]

        if self.enhancement_config["components"]["damage_detection"]:
            features["damage"] = self.damage_detector(x)

        if self.enhancement_config["components"]["color_anomaly"]:
            features["color"] = self.color_analyzer(x)
            x = x + 0.1 * features["color"]

        # Regular encoding path
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        embedding = self.embedder(x)

        return embedding, features

    def decode(
        self, embedding: torch.Tensor, features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Enhanced decoding with pattern preservation"""
        x = self.unembedder(embedding)
        x = x.view(
            x.size(0),
            self.layer_sizes[-1],
            self.final_spatial_dim,
            self.final_spatial_dim,
        )

        for layer in self.decoder_layers:
            x = layer(x)

        # Add preserved features
        if self.enhancement_config["components"]["texture_analysis"]:
            x = x + 0.1 * features.get("texture", 0)

        if self.enhancement_config["components"]["damage_detection"]:
            damage_mask = features.get("damage", torch.zeros_like(x))
            x = x * (1 + 0.2 * damage_mask)

        if self.enhancement_config["components"]["color_anomaly"]:
            x = x + 0.1 * features.get("color", 0)

        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced pattern preservation"""
        embedding, features = self.encode(x)
        reconstruction = self.decode(embedding, features)

        output = {"embedding": embedding, "reconstruction": reconstruction}

        if self.training and self.enhancement_config["enabled"]:
            loss = self.pattern_loss(reconstruction, x)
            output["loss"] = loss

        return output


class AstronomicalStructureLoss(nn.Module):
    """Loss function specialized for astronomical imaging features"""

    def __init__(self):
        super().__init__()

        # Edge detection filters
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        # Point source detection filter (for stars)
        self.point_filter = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        # Multi-scale structure filters
        self.structure_filters = [
            self._create_gaussian_kernel(sigma) for sigma in [0.5, 1.0, 2.0]
        ]

        # Scale-space filters for galaxy features
        self.scale_filters = [
            self._create_log_kernel(sigma) for sigma in [1.0, 2.0, 4.0]
        ]

    def _create_gaussian_kernel(
        self, sigma: float, kernel_size: int = 7
    ) -> torch.Tensor:
        """Create Gaussian kernel for smoothing"""
        x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
        x = x.view(-1, 1)
        y = x.t()
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        return (gaussian / gaussian.sum()).view(1, 1, kernel_size, kernel_size)

    def _create_log_kernel(self, sigma: float, kernel_size: int = 7) -> torch.Tensor:
        """Create Laplacian of Gaussian kernel for blob detection"""
        x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
        x = x.view(-1, 1)
        y = x.t()
        r2 = x**2 + y**2
        log = (1 - r2 / (2 * sigma**2)) * torch.exp(-r2 / (2 * sigma**2))
        return (log / log.abs().sum()).view(1, 1, kernel_size, kernel_size)

    def forward(
        self, reconstruction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate loss with astronomical feature preservation"""
        device = reconstruction.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        self.point_filter = self.point_filter.to(device)
        self.structure_filters = [f.to(device) for f in self.structure_filters]
        self.scale_filters = [f.to(device) for f in self.scale_filters]

        # Basic reconstruction loss with intensity weighting
        intensity_weights = (target > target.mean()).float() * 2 + 1
        recon_loss = F.mse_loss(
            reconstruction * intensity_weights, target * intensity_weights
        )

        # Edge and gradient preservation
        rec_grad_x = F.conv2d(reconstruction, self.sobel_x, padding=1)
        rec_grad_y = F.conv2d(reconstruction, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)

        gradient_loss = F.mse_loss(rec_grad_x, target_grad_x) + F.mse_loss(
            rec_grad_y, target_grad_y
        )

        # Point source (star) preservation
        rec_points = F.conv2d(reconstruction, self.point_filter, padding=1)
        target_points = F.conv2d(target, self.point_filter, padding=1)
        point_loss = F.mse_loss(rec_points, target_points)

        # Multi-scale structure preservation
        structure_loss = 0
        for filter in self.structure_filters:
            rec_struct = F.conv2d(reconstruction, filter, padding=filter.size(-1) // 2)
            target_struct = F.conv2d(target, filter, padding=filter.size(-1) // 2)
            structure_loss += F.mse_loss(rec_struct, target_struct)

        # Scale-space feature preservation
        scale_loss = 0
        for filter in self.scale_filters:
            rec_scale = F.conv2d(reconstruction, filter, padding=filter.size(-1) // 2)
            target_scale = F.conv2d(target, filter, padding=filter.size(-1) // 2)
            scale_loss += F.mse_loss(rec_scale, target_scale)

        # Peak intensity preservation (for bright stars)
        peak_loss = F.l1_loss(
            torch.max_pool2d(reconstruction, kernel_size=3, stride=1, padding=1),
            torch.max_pool2d(target, kernel_size=3, stride=1, padding=1),
        )

        # Combine losses with weights
        total_loss = (
            recon_loss
            + 2.0 * gradient_loss
            + 1.5 * point_loss
            + 1.0 * structure_loss
            + 1.0 * scale_loss
            + 2.0 * peak_loss
        )

        return total_loss


class MedicalStructureLoss(nn.Module):
    """Loss function specialized for medical imaging features"""

    def __init__(self):
        super().__init__()

        # Edge detection filters
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        # Multi-scale analysis filters
        self.feature_filters = [
            self._create_gaussian_kernel(sigma) for sigma in [0.5, 1.0, 2.0]
        ]

        # Structure filters for tissue boundaries
        self.boundary_filters = [
            self._create_dog_kernel(sigma) for sigma in [1.0, 2.0, 4.0]
        ]

    def _create_gaussian_kernel(
        self, sigma: float, kernel_size: int = 7
    ) -> torch.Tensor:
        """Create Gaussian kernel for smoothing"""
        x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
        x = x.view(-1, 1)
        y = x.t()
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        return (gaussian / gaussian.sum()).view(1, 1, kernel_size, kernel_size)

    def _create_dog_kernel(
        self, sigma: float, k: float = 1.6, kernel_size: int = 7
    ) -> torch.Tensor:
        """Create Difference of Gaussians kernel for edge detection"""
        x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
        x = x.view(-1, 1)
        y = x.t()
        r2 = x**2 + y**2
        g1 = torch.exp(-r2 / (2 * sigma**2))
        g2 = torch.exp(-r2 / (2 * (k * sigma) ** 2))
        dog = g1 / sigma**2 - g2 / (k * sigma) ** 2
        return (dog / dog.abs().sum()).view(1, 1, kernel_size, kernel_size)

    def forward(
        self, reconstruction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate loss with medical feature preservation"""
        device = reconstruction.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        self.feature_filters = [f.to(device) for f in self.feature_filters]
        self.boundary_filters = [f.to(device) for f in self.boundary_filters]

        # Reconstruction loss with tissue weighting
        tissue_weights = (target > target.mean()).float() * 2 + 1
        recon_loss = F.mse_loss(
            reconstruction * tissue_weights, target * tissue_weights
        )

        # Gradient preservation for tissue boundaries
        rec_grad_x = F.conv2d(reconstruction, self.sobel_x, padding=1)
        rec_grad_y = F.conv2d(reconstruction, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)

        gradient_loss = F.mse_loss(rec_grad_x, target_grad_x) + F.mse_loss(
            rec_grad_y, target_grad_y
        )

        # Multi-scale feature preservation
        feature_loss = 0
        for filter in self.feature_filters:
            rec_features = F.conv2d(
                reconstruction, filter, padding=filter.size(-1) // 2
            )
            target_features = F.conv2d(target, filter, padding=filter.size(-1) // 2)
            feature_loss += F.mse_loss(rec_features, target_features)

        # Boundary preservation
        boundary_loss = 0
        for filter in self.boundary_filters:
            rec_bound = F.conv2d(reconstruction, filter, padding=filter.size(-1) // 2)
            target_bound = F.conv2d(target, filter, padding=filter.size(-1) // 2)
            boundary_loss += F.mse_loss(rec_bound, target_bound)

        # Local contrast preservation for lesion detection
        rec_std = torch.std(F.unfold(reconstruction, kernel_size=5), dim=1)
        target_std = torch.std(F.unfold(target, kernel_size=5), dim=1)
        contrast_loss = F.mse_loss(rec_std, target_std)

        # Combine losses with weights
        total_loss = (
            recon_loss
            + 1.5 * gradient_loss
            + 1.0 * feature_loss
            + 2.0 * boundary_loss
            + 1.0 * contrast_loss
        )

        return total_loss


class AgriculturalPatternLoss(nn.Module):
    """Loss function optimized for agricultural pest and disease detection"""

    def __init__(self):
        super().__init__()
        self.texture_filters = None  # Will be initialized on first use
        self.pattern_filters = None  # Will be initialized on first use

        # Color analysis filters
        self.color_filters = [
            torch.eye(3, dtype=torch.float32).view(3, 3, 1, 1),
            torch.tensor([[0.299, 0.587, 0.114]], dtype=torch.float32).view(1, 3, 1, 1),
        ]

    def _create_gabor_kernel(
        self, frequency: float, angle: float, sigma: float = 3.0, size: int = 7
    ) -> torch.Tensor:
        """Create Gabor filter for texture analysis"""
        # Convert angle to radians and create as tensor
        angle_rad = torch.tensor(angle * np.pi / 180)

        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-size // 2, size // 2, size),
            torch.linspace(-size // 2, size // 2, size),
            indexing="ij",
        )

        # Compute rotated coordinates
        cos_angle = torch.cos(angle_rad)
        sin_angle = torch.sin(angle_rad)
        x_rot = x_grid * cos_angle + y_grid * sin_angle
        y_rot = -x_grid * sin_angle + y_grid * cos_angle

        # Create Gabor filter components
        gaussian = torch.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))
        sinusoid = torch.cos(2 * np.pi * frequency * x_rot)

        # Combine and normalize
        kernel = (gaussian * sinusoid).view(1, 1, size, size)
        return kernel / kernel.abs().sum()

    def _create_pattern_kernel(self, size: int) -> torch.Tensor:
        """Create kernel for local pattern analysis"""
        kernel = torch.ones(size, size, dtype=torch.float32)
        center = size // 2
        kernel[center, center] = -(size**2) + 1
        return (kernel / kernel.abs().sum()).view(1, 1, size, size)

    def _initialize_filters(self, device: torch.device):
        """Initialize filters if not already done"""
        if self.texture_filters is None:
            self.texture_filters = [
                self._create_gabor_kernel(frequency=f, angle=a).to(device)
                for f in [0.1, 0.2, 0.3]
                for a in [0, 45, 90, 135]
            ]

        if self.pattern_filters is None:
            self.pattern_filters = [
                self._create_pattern_kernel(size=s).to(device) for s in [3, 5, 7]
            ]

        self.color_filters = [f.to(device) for f in self.color_filters]

    def forward(
        self, reconstruction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate loss with agricultural pattern preservation"""
        device = reconstruction.device
        self._initialize_filters(device)

        # Basic reconstruction loss
        recon_loss = F.mse_loss(reconstruction, target)

        # Texture preservation loss
        texture_loss = 0
        for filter in self.texture_filters:
            rec_texture = F.conv2d(reconstruction, filter, padding=filter.size(-1) // 2)
            target_texture = F.conv2d(target, filter, padding=filter.size(-1) // 2)
            texture_loss += F.mse_loss(rec_texture, target_texture)
        texture_loss = texture_loss / len(self.texture_filters)

        # Pattern preservation loss
        pattern_loss = 0
        for filter in self.pattern_filters:
            rec_pattern = F.conv2d(reconstruction, filter, padding=filter.size(-1) // 2)
            target_pattern = F.conv2d(target, filter, padding=filter.size(-1) // 2)
            pattern_loss += F.mse_loss(rec_pattern, target_pattern)
        pattern_loss = pattern_loss / len(self.pattern_filters)

        # Color preservation loss
        color_loss = 0
        for filter in self.color_filters:
            rec_color = F.conv2d(reconstruction, filter)
            target_color = F.conv2d(target, filter)
            color_loss += F.mse_loss(rec_color, target_color)
        color_loss = color_loss / len(self.color_filters)

        # Local contrast preservation
        contrast_loss = F.mse_loss(
            torch.std(reconstruction, dim=[2, 3]), torch.std(target, dim=[2, 3])
        )

        # Combine losses with weights
        total_loss = (
            recon_loss
            + 2.0 * texture_loss
            + 1.5 * pattern_loss
            + 1.0 * color_loss
            + 0.5 * contrast_loss
        )

        return total_loss

    def _analyze_texture_statistics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze texture statistics for pattern detection"""
        stats = {}

        # Calculate local statistics using texture filters
        texture_responses = []
        for filter in self.texture_filters:
            response = F.conv2d(x, filter.to(x.device), padding=filter.size(-1) // 2)
            texture_responses.append(response)

        # Compute texture energy
        stats["energy"] = torch.mean(
            torch.stack([r.pow(2).mean() for r in texture_responses])
        )

        # Compute texture contrast
        stats["contrast"] = torch.mean(
            torch.stack([r.std() for r in texture_responses])
        )

        return stats

    def _analyze_pattern_distribution(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze pattern distribution for anomaly detection"""
        stats = {}

        # Calculate pattern responses at different scales
        pattern_responses = []
        for filter in self.pattern_filters:
            response = F.conv2d(x, filter.to(x.device), padding=filter.size(-1) // 2)
            pattern_responses.append(response)

        # Compute pattern density
        stats["density"] = torch.mean(
            torch.stack([r.abs().mean() for r in pattern_responses])
        )

        # Compute pattern variability
        stats["variability"] = torch.mean(
            torch.stack([r.var() for r in pattern_responses])
        )

        return stats


def safe_get_scalar(value):
    """Safely convert any numeric value to Python float"""
    if isinstance(value, torch.Tensor):
        return value.item()
    elif isinstance(value, (float, int)):
        return float(value)
    else:
        raise ValueError(f"Cannot convert {type(value)} to scalar")


class EnhancedLossManager:
    """Manager for handling specialized loss functions"""

    def __init__(self, config: Dict):
        self.config = config
        self.loss_functions = {}
        self.initialize_loss_functions()

    def initialize_loss_functions(self):
        """Initialize appropriate loss functions based on configuration"""
        enhancement_modules = self.config["model"]["enhancement_modules"]

        # Initialize astronomical loss if enabled
        if enhancement_modules["astronomical"]["enabled"]:
            self.loss_functions["astronomical"] = AstronomicalStructureLoss()

        # Initialize medical loss if enabled
        if enhancement_modules["medical"]["enabled"]:
            self.loss_functions["medical"] = MedicalStructureLoss()

        # Initialize agricultural loss if enabled
        if enhancement_modules["agricultural"]["enabled"]:
            self.loss_functions["agricultural"] = AgriculturalPatternLoss()

    def get_loss_function(self, image_type: str) -> Optional[nn.Module]:
        """Get appropriate loss function for image type"""
        return self.loss_functions.get(image_type)

    def calculate_loss(
        self, reconstruction: torch.Tensor, target: torch.Tensor, image_type: str
    ) -> Dict[str, torch.Tensor]:
        """Calculate loss with appropriate enhancements"""
        loss_fn = self.get_loss_function(image_type)
        if loss_fn is None:
            return {"loss": F.mse_loss(reconstruction, target)}

        result = loss_fn(reconstruction, target)

        # Ensure we always return a dictionary with tensor loss
        if isinstance(result, dict):
            if "loss" in result:
                if isinstance(result["loss"], torch.Tensor):
                    return result
                else:
                    return {
                        "loss": torch.tensor(
                            float(result["loss"]), device=reconstruction.device
                        )
                    }
            else:
                return {"loss": F.mse_loss(reconstruction, target)}
        elif isinstance(result, torch.Tensor):
            return {"loss": result}
        else:
            return {"loss": torch.tensor(float(result), device=reconstruction.device)}


class UnifiedCheckpoint:
    """Manages a unified checkpoint file containing multiple model states"""

    def __init__(self, config: Dict):
        self.config = config
        self.checkpoint_dir = config["training"]["checkpoint_dir"]
        self.dataset_name = config["dataset"]["name"]
        self.checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{self.dataset_name}_unified.pth"
        )
        self.current_state = None
        self.model_type = config["model"].get(
            "encoder_type", "autoenc"
        )  # Track model type

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Load or initialize checkpoint
        self.load_checkpoint()

    def load_checkpoint(self):
        """Load existing checkpoint or initialize new one"""
        if os.path.exists(self.checkpoint_path):
            self.current_state = torch.load(self.checkpoint_path)
            logger.info(f"Loaded unified checkpoint from {self.checkpoint_path}")
        else:
            self.current_state = {
                "model_states": {},
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "config": self.config,
                },
            }
            logger.info("Initialized new unified checkpoint")

    def get_state_key(self, phase: int, model: nn.Module) -> str:
        """Generate unique key including model type"""
        components = [f"phase{phase}", f"model_{self.model_type}"]

        if phase == 2:
            if model.use_kl_divergence:
                components.append("kld")
            if model.use_class_encoding:
                components.append("cls")

            image_type = self.config["dataset"].get("image_type", "general")
            if image_type != "general":
                components.append(image_type)

        return "_".join(components)

    def save_model_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        phase: int,
        epoch: int,
        loss: float,
        is_best: bool = False,
    ):
        """Save model state including all components"""
        state_key = self.get_state_key(phase, model)

        # Prepare complete state dictionary
        state_dict = {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "phase": phase,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "kl_divergence": model.use_kl_divergence,
                "class_encoding": model.use_class_encoding,
                "image_type": self.config["dataset"].get("image_type", "general"),
                "clustering_params": {
                    "num_clusters": model.cluster_centers.size(0)
                    if hasattr(model, "cluster_centers")
                    else 0,
                    "temperature": model.clustering_temperature.item()
                    if hasattr(model, "clustering_temperature")
                    and isinstance(model.clustering_temperature, torch.Tensor)
                    else 1.0,
                },
            },
        }

        # Update model_states in the checkpoint
        if state_key not in self.current_state["model_states"]:
            self.current_state["model_states"][state_key] = {
                "current": None,
                "best": None,
                "history": [],
            }

        self.current_state["model_states"][state_key]["current"] = state_dict
        if is_best:
            self.current_state["model_states"][state_key]["best"] = state_dict

        # Save checkpoint
        torch.save(self.current_state, self.checkpoint_path)

    def load_model_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        phase: int,
        load_best: bool = False,
    ) -> Optional[Dict]:
        """Load model state from unified checkpoint"""
        state_key = self.get_state_key(phase, model)

        if state_key not in self.current_state["model_states"]:
            logger.info(f"No existing state found for {state_key}")
            return None

        # Get appropriate state
        state_dict = self.current_state["model_states"][state_key][
            "best" if load_best else "current"
        ]
        if state_dict is None:
            return None

        # Load state
        model.load_state_dict(state_dict["state_dict"])
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])

        logger.info(
            f"Loaded {'best' if load_best else 'current'} state for {state_key}"
        )
        return state_dict

    def get_best_loss(self, phase: int, model: nn.Module) -> float:
        """Get best loss for current configuration"""
        state_key = self.get_state_key(phase, model)
        if state_key in self.current_state["model_states"]:
            best_state = self.current_state["model_states"][state_key]["best"]
            if best_state is not None:
                loss = best_state["loss"]
                return loss.item() if hasattr(loss, "item") else float(loss)
        return float("inf")

    def print_checkpoint_summary(self):
        """Print summary of checkpoint contents"""
        print("\nUnified Checkpoint Summary:")
        print("-" * 50)
        print(f"Dataset: {self.dataset_name}")
        print(f"Last Updated: {self.current_state['metadata']['last_updated']}")
        print("\nModel States:")

        for state_key, state in self.current_state["model_states"].items():
            print(f"\n{state_key}:")
            if state["current"] is not None:
                print(
                    f"  Current - Epoch: {state['current']['epoch']}, "
                    f"Loss: {state['current']['loss']:.4f}"
                )
            if state["best"] is not None:
                print(
                    f"  Best    - Epoch: {state['best']['epoch']}, "
                    f"Loss: {state['best']['loss']:.4f}"
                )
            print(f"  History - {len(state['history'])} entries")


class ModelFactory:
    """Factory for creating appropriate model based on configuration"""

    @staticmethod
    def create_model(config: Dict) -> nn.Module:
        """Create model with proper channel handling."""
        input_shape = (
            config["dataset"]["in_channels"],  # Use configured channels
            config["dataset"]["input_size"][0],
            config["dataset"]["input_size"][1],
        )
        feature_dims = config["model"]["feature_dims"]

        image_type = config["dataset"].get("image_type", "general")

        # Create appropriate model with proper channel handling
        if image_type == "astronomical":
            model = AstronomicalStructurePreservingAutoencoder(
                input_shape, feature_dims, config
            )
        elif image_type == "medical":
            model = MedicalStructurePreservingAutoencoder(
                input_shape, feature_dims, config
            )
        elif image_type == "agricultural":
            model = AgriculturalPatternAutoencoder(input_shape, feature_dims, config)
        else:
            model = BaseAutoencoder(input_shape, feature_dims, config)

        # Verify channel compatibility
        if hasattr(model, "in_channels"):
            if model.in_channels != config["dataset"]["in_channels"]:
                logger.warning(
                    f"Model expects {model.in_channels} channels but config specifies {config['dataset']['in_channels']}"
                )

        return model


# Update the training loop to handle the new feature dictionary format
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    config: Dict,
    loss_manager: EnhancedLossManager,
) -> Dict[str, List]:
    """Two-phase training implementation with checkpoint handling"""
    # Store dataset reference in model
    model.set_dataset(train_loader.dataset)

    history = defaultdict(list)

    # Initialize starting epoch and phase
    start_epoch = getattr(model, "current_epoch", 0)
    current_phase = getattr(model, "training_phase", 1)

    # Phase 1: Pure reconstruction (if not already completed)
    if current_phase == 1:
        logger.info("Starting/Resuming Phase 1: Pure reconstruction training")
        model.set_training_phase(1)
        optimizer = optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])

        phase1_history = _train_phase(
            model,
            train_loader,
            optimizer,
            loss_manager,
            config["training"]["epochs"],
            1,
            config,
            start_epoch=start_epoch,
        )
        history.update(phase1_history)

        # Reset start_epoch for phase 2
        start_epoch = 0
    else:
        logger.info("Phase 1 already completed, skipping")

    # Phase 2: Latent space organization
    if config["model"]["autoencoder_config"]["enhancements"].get("enable_phase2", True):
        if current_phase < 2:
            logger.info("Starting Phase 2: Latent space organization")
            model.set_training_phase(2)
        else:
            logger.info("Resuming Phase 2: Latent space organization")

        # Lower learning rate for fine-tuning
        optimizer = optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])

        phase2_history = _train_phase(
            model,
            train_loader,
            optimizer,
            loss_manager,
            config["training"]["epochs"],
            2,
            config,
            start_epoch=start_epoch if current_phase == 2 else 0,
        )

        # Merge histories
        for key, value in phase2_history.items():
            history[f"phase2_{key}"] = value

    return history


def _get_checkpoint_identifier(model: nn.Module, phase: int, config: Dict) -> str:
    """
    Generate unique identifier for checkpoint based on phase and active enhancements.
    """
    # Start with phase identifier
    identifier = f"phase{phase}"

    # Add active enhancements
    if phase == 2:
        active_enhancements = []
        if model.use_kl_divergence:
            active_enhancements.append("kld")
        if model.use_class_encoding:
            active_enhancements.append("cls")

        # Add specialized enhancements
        if config["dataset"].get("image_type") != "general":
            image_type = config["dataset"]["image_type"]
            if (
                config["model"]["enhancement_modules"]
                .get(image_type, {})
                .get("enabled", True)
            ):
                active_enhancements.append(image_type)

        if active_enhancements:
            identifier += "_" + "_".join(sorted(active_enhancements))

    return identifier


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    phase: int,
    loss: float,
    config: Dict,
    is_best: bool = False,
):
    """
    Save training checkpoint with phase and enhancement-specific handling.
    """
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get unique identifier for this configuration
    identifier = _get_checkpoint_identifier(model, phase, config)
    dataset_name = config["dataset"]["name"]

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "phase": phase,
        "training_phase": model.training_phase,
        "loss": loss,
        "identifier": identifier,
        "config": config,
        "active_enhancements": {
            "kl_divergence": model.use_kl_divergence,
            "class_encoding": model.use_class_encoding,
            "image_type": config["dataset"].get("image_type", "general"),
        },
    }

    # Always save latest checkpoint
    latest_path = os.path.join(
        checkpoint_dir, f"{dataset_name}_{identifier}_latest.pth"
    )
    torch.save(checkpoint, latest_path)

    # Save phase-specific best model if applicable
    if is_best:
        best_path = os.path.join(
            checkpoint_dir, f"{dataset_name}_{identifier}_best.pth"
        )
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model for {identifier} with loss: {loss:.4f}")

    logger.info(f"Saved checkpoint for {identifier} at epoch {epoch + 1}")


def load_best_checkpoint(model: nn.Module, phase: int, config: Dict) -> Optional[Dict]:
    """
    Load the best checkpoint for the given phase and enhancement combination.
    """
    checkpoint_dir = config["training"]["checkpoint_dir"]
    identifier = _get_checkpoint_identifier(model, phase, config)
    dataset_name = config["dataset"]["name"]
    best_path = os.path.join(checkpoint_dir, f"{dataset_name}_{identifier}_best.pth")

    if os.path.exists(best_path):
        logger.info(f"Loading best checkpoint for {identifier}")
        return torch.load(best_path, map_location=model.device)
    return None


def update_phase_specific_metrics(
    model: nn.Module, phase: int, config: Dict
) -> Dict[str, Any]:
    """
    Track and return phase-specific metrics and best values.
    """
    metrics = {}
    _get_checkpoint_identifier(model, phase, config)

    # Try to load existing best metrics
    checkpoint = load_best_checkpoint(model, phase, config)
    if checkpoint:
        metrics["best_loss"] = checkpoint.get("loss", float("inf"))
        metrics["best_epoch"] = checkpoint.get("epoch", 0)
    else:
        metrics["best_loss"] = float("inf")
        metrics["best_epoch"] = 0

    return metrics


def _train_phase(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_manager: EnhancedLossManager,
    epochs: int,
    phase: int,
    config: Dict,
    start_epoch: int = 0,
) -> Dict[str, List]:
    """Training logic with class-balanced loss calculation"""

    history = defaultdict(list)
    device = next(model.parameters()).device
    best_loss = float("inf")
    min_thr = float(config["model"]["autoencoder_config"]["convergence_threshold"])
    checkpoint_manager = UnifiedCheckpoint(config)
    use_classwise = config["training"].get("use_classwise_acc", True)  # Config flag
    patience_counter = 0

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            num_batches = len(train_loader)

            pbar = tqdm(
                train_loader, desc=f"Phase {phase} - Epoch {epoch+1}", leave=False
            )
            for batch_idx, (data, labels) in enumerate(pbar):
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                if phase == 1:
                    # Phase 1: Reconstruction loss only
                    reconstruction = model.decode(model.encode(data))
                    loss = F.mse_loss(reconstruction, data)
                else:
                    # Phase 2: Enhanced losses
                    output = model(data)
                    reconstruction = output["reconstruction"]
                    embedding = output["embedding"]

                    # Base reconstruction loss
                    recon_loss = F.mse_loss(reconstruction, data)

                    # Initialize total loss
                    total_loss = recon_loss

                    # KL Divergence loss
                    if model.use_kl_divergence:
                        latent_info = model.organize_latent_space(embedding, labels)
                        kl_loss = F.kl_div(
                            latent_info["cluster_probabilities"].log(),
                            latent_info["target_distribution"],
                            reduction="batchmean",
                        )
                        total_loss += (
                            config["model"]["autoencoder_config"]["enhancements"][
                                "kl_divergence_weight"
                            ]
                            * kl_loss
                        )

                    # Class-balanced classification loss
                    if model.use_class_encoding and hasattr(model, "classifier"):
                        class_logits = model.classifier(embedding)

                        if use_classwise:  # Class-balanced loss calculation
                            class_losses = []
                            for cls in torch.unique(labels):
                                mask = labels == cls
                                if mask.sum() > 0:  # Handle empty classes
                                    cls_loss = F.cross_entropy(
                                        class_logits[mask],
                                        labels[mask],
                                        reduction="mean",
                                    )
                                    class_losses.append(cls_loss)

                            if class_losses:  # Prevent empty list
                                class_loss = torch.mean(torch.stack(class_losses))
                            else:
                                class_loss = torch.tensor(0.0, device=device)
                        else:  # Standard loss
                            class_loss = F.cross_entropy(class_logits, labels)

                        total_loss += (
                            config["model"]["autoencoder_config"]["enhancements"][
                                "classification_weight"
                            ]
                            * class_loss
                        )

                        # Calculate accuracy (class-balanced or standard)
                        with torch.no_grad():
                            preds = torch.argmax(class_logits, dim=1)
                            if use_classwise:
                                class_acc = []
                                for cls in torch.unique(labels):
                                    mask = labels == cls
                                    if mask.sum() > 0:
                                        cls_acc = (
                                            (preds[mask] == labels[mask]).float().mean()
                                        )
                                        class_acc.append(cls_acc)
                                acc = (
                                    torch.mean(torch.stack(class_acc))
                                    if class_acc
                                    else 0.0
                                )
                            else:
                                acc = (preds == labels).float().mean()
                            running_acc += acc.item()

                    loss = total_loss

                # Backpropagation
                loss.backward()
                optimizer.step()

                # Update metrics
                running_loss += loss.item()
                avg_loss = running_loss / (batch_idx + 1)
                status = {
                    "loss": f"{Colors.color_value(avg_loss, best_loss, False)}",
                    "best": f"{best_loss:.4f}",
                    "patience": f"{patience_counter}",
                }

                if phase == 2 and model.use_class_encoding:
                    avg_acc = running_acc / (batch_idx + 1)
                    status["acc"] = f"{avg_acc:.2%}"

                pbar.set_postfix(status)

                # Cleanup
                del data, loss
                torch.cuda.empty_cache()

            # Epoch statistics
            avg_loss = running_loss / num_batches
            history[f"phase{phase}_loss"].append(avg_loss)

            if phase == 2 and model.use_class_encoding:
                avg_acc = running_acc / num_batches
                history[f"phase{phase}_accuracy"].append(avg_acc)

            # Checkpointing
            if (best_loss - avg_loss) > min_thr:
                best_loss = avg_loss
                patience_counter = 0
                checkpoint_manager.save_model_state(
                    model, optimizer, phase, epoch, avg_loss, True
                )
            else:
                patience_counter += 1

            if patience_counter >= config["training"].get("early_stopping", {}).get(
                "patience", 5
            ):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

    return history


class ReconstructionManager:
    """Manages model prediction with unified checkpoint loading"""

    def __init__(self, config: Dict):
        self.config = config
        self.checkpoint_manager = UnifiedCheckpoint(config)
        self.device = torch.device(
            "cuda"
            if config["execution_flags"]["use_gpu"] and torch.cuda.is_available()
            else "cpu"
        )

    def load_model_for_prediction(self) -> Tuple[nn.Module, Dict]:
        """Load appropriate model configuration based on user input"""
        # Get available configurations from checkpoint
        available_configs = self._get_available_configurations()

        if not available_configs:
            raise ValueError("No trained models found in checkpoint")

        # Show available configurations
        print("\nAvailable Model Configurations:")
        for idx, (key, config) in enumerate(available_configs.items(), 1):
            print(f"{idx}. {key}")
            if config.get("best") and config["best"].get("loss"):
                print(f"   Best Loss: {config['best']['loss']:.4f}")
            print(f"   Features: {self._get_config_description(config)}")

        # Get user selection
        while True:
            try:
                choice = int(input("\nSelect configuration (number): ")) - 1
                if 0 <= choice < len(available_configs):
                    selected_key = list(available_configs.keys())[choice]
                    selected_config = available_configs[selected_key]
                    break
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

        # Create and load model
        model = self._create_model_from_config(selected_config)
        state_dict = (
            selected_config["best"]
            if selected_config.get("best")
            else selected_config["current"]
        )
        model.load_state_dict(state_dict["state_dict"])
        model.eval()

        return model, selected_config

    def _get_available_configurations(self) -> Dict:
        """Get available model configurations from checkpoint"""
        return self.checkpoint_manager.current_state["model_states"]

    def _get_config_description(self, config: Dict) -> str:
        """Generate human-readable description of model configuration"""
        features = []

        if config["current"]["config"].get("kl_divergence"):
            features.append("KL Divergence")
        if config["current"]["config"].get("class_encoding"):
            features.append("Class Encoding")

        image_type = config["current"]["config"].get("image_type")
        if image_type and image_type != "general":
            features.append(f"{image_type.capitalize()} Enhancement")

        return ", ".join(features) if features else "Basic Autoencoder"

    def _create_model_from_config(self, config: Dict) -> nn.Module:
        """Create model instance based on configuration"""
        input_shape = (
            self.config["dataset"]["in_channels"],
            self.config["dataset"]["input_size"][0],
            self.config["dataset"]["input_size"][1],
        )
        feature_dims = self.config["model"]["feature_dims"]

        logger.info(f"Main configuration saved: {self.config_path}")
        image_type = config["dataset"].get("image_type", "general")

        # Set model configuration based on saved state
        self.config["model"]["autoencoder_config"]["enhancements"].update(
            {
                "use_kl_divergence": config["current"]["config"]["kl_divergence"],
                "use_class_encoding": config["current"]["config"]["class_encoding"],
            }
        )

        # Create appropriate model
        image_type = config["current"]["config"]["image_type"]
        if image_type == "astronomical":
            model = AstronomicalStructurePreservingAutoencoder(
                input_shape, feature_dims, self.config
            )
        elif image_type == "medical":
            model = MedicalStructurePreservingAutoencoder(
                input_shape, feature_dims, self.config
            )
        elif image_type == "agricultural":
            model = AgriculturalPatternAutoencoder(
                input_shape, feature_dims, self.config
            )
        else:
            model = BaseAutoencoder(input_shape, feature_dims, self.config)

        return model.to(self.device)

    def predict_from_csv(
        self, csv_path: Optional[str] = None, output_dir: Optional[str] = None
    ):
        """Generate predictions from features in CSV"""
        # Load model
        model, config = self.load_model_for_prediction()
        model.eval()  # Ensure model is in evaluation mode

        # Determine input CSV path
        if csv_path is None:
            dataset_name = self.config["dataset"]["name"]
            base_dir = os.path.join("data", dataset_name)

            if self.config.get("execution_flags", {}).get("invert_DBNN", False):
                csv_path = os.path.join(base_dir, "reconstructed_input.csv")
            else:
                csv_path = os.path.join(base_dir, f"{dataset_name}.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load and process CSV
        df = pd.read_csv(csv_path)
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        features = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(
            self.device
        )

        # Verify feature dimensions
        expected_dims = self.config["model"]["feature_dims"]
        if features.size(1) != expected_dims:
            raise ValueError(
                f"Feature dimension mismatch: got {features.size(1)}, expected {expected_dims}"
            )

        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(
                "data", self.config["dataset"]["name"], "predictions"
            )
        os.makedirs(output_dir, exist_ok=True)

        # Get image type and enhancement modules
        image_type = self.config["dataset"].get("image_type", "general")
        enhancement_modules = self.config["model"].get("enhancement_modules", {})

        outputs = []
        batch_size = self.config["training"].get("batch_size", 128)

        with torch.no_grad():
            for i in tqdm(
                range(0, len(features), batch_size), desc="Generating predictions"
            ):
                batch = features[i : i + batch_size]

                try:
                    if config["current"]["phase"] == 1:
                        # Phase 1: Direct decoding
                        reconstruction = model.decode(batch)
                        output = {"reconstruction": reconstruction}
                    else:
                        # Phase 2: Full forward pass with enhancements
                        # First decode the features
                        reconstruction = model.decode(batch)

                        # Then run through full model if needed for enhancements
                        if (
                            image_type != "general"
                            and image_type in enhancement_modules
                        ):
                            enhanced_output = model(
                                reconstruction
                            )  # Get enhanced features
                            output = {
                                "reconstruction": enhanced_output["reconstruction"]
                                if "reconstruction" in enhanced_output
                                else reconstruction,
                                "embedding": enhanced_output.get("embedding", batch),
                            }

                            # Add enhancement-specific outputs
                            if isinstance(
                                model, AstronomicalStructurePreservingAutoencoder
                            ):
                                output.update(
                                    self._apply_astronomical_enhancements(
                                        enhanced_output
                                    )
                                )
                            elif isinstance(
                                model, MedicalStructurePreservingAutoencoder
                            ):
                                output.update(
                                    self._apply_medical_enhancements(enhanced_output)
                                )
                            elif isinstance(model, AgriculturalPatternAutoencoder):
                                output.update(
                                    self._apply_agricultural_enhancements(
                                        enhanced_output
                                    )
                                )
                        else:
                            output = {"reconstruction": reconstruction}

                    outputs.append(self._process_output(output))

                except Exception as e:
                    logger.error(f"Error processing batch {i}: {str(e)}")
                    raise

        # Combine and save results
        combined_output = self._combine_outputs(outputs)
        self._save_predictions(combined_output, output_dir, config)

    def _save_enhancement_outputs(
        self, predictions: Dict[str, np.ndarray], output_dir: str
    ):
        """Save enhancement-specific outputs"""
        # Save astronomical features
        if "star_features" in predictions:
            star_dir = os.path.join(output_dir, "star_detection")
            os.makedirs(star_dir, exist_ok=True)
            for idx, feat in enumerate(predictions["star_features"]):
                img_path = os.path.join(star_dir, f"stars_{idx}.png")
                self._save_feature_map(feat, img_path)

        # Save medical features
        if "boundary_features" in predictions:
            boundary_dir = os.path.join(output_dir, "boundary_detection")
            os.makedirs(boundary_dir, exist_ok=True)
            for idx, feat in enumerate(predictions["boundary_features"]):
                img_path = os.path.join(boundary_dir, f"boundary_{idx}.png")
                self._save_feature_map(feat, img_path)

        # Save agricultural features
        if "texture_features" in predictions:
            texture_dir = os.path.join(output_dir, "texture_analysis")
            os.makedirs(texture_dir, exist_ok=True)
            for idx, feat in enumerate(predictions["texture_features"]):
                img_path = os.path.join(texture_dir, f"texture_{idx}.png")
                self._save_feature_map(feat, img_path)

    def _save_feature_map(self, feature_map: np.ndarray, path: str):
        """Save feature map as image"""
        # Normalize feature map to 0-255 range
        feature_map = (
            (feature_map - feature_map.min())
            / (feature_map.max() - feature_map.min() + 1e-8)
            * 255
        ).astype(np.uint8)
        Image.fromarray(feature_map).save(path)

    def _process_output(self, output: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Process model output into numpy arrays"""
        processed = {}
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                processed[key] = value.detach().cpu().numpy()
            else:
                processed[key] = value
        return processed

    def _combine_outputs(
        self, outputs: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Combine batched outputs"""
        combined = {}
        for key in outputs[0].keys():
            combined[key] = np.concatenate([out[key] for out in outputs])
        return combined

    def _save_predictions(
        self, predictions: Dict[str, np.ndarray], output_dir: str, config: Dict
    ):
        """Save predictions with appropriate format based on configuration"""
        os.makedirs(output_dir, exist_ok=True)

        # Save reconstructions as images
        if "reconstruction" in predictions:
            recon_dir = os.path.join(output_dir, "reconstructions")
            os.makedirs(recon_dir, exist_ok=True)

            for idx, recon in enumerate(predictions["reconstruction"]):
                img = self._tensor_to_image(torch.tensor(recon))
                img_path = os.path.join(recon_dir, f"reconstruction_{idx}.png")
                Image.fromarray(img).save(img_path)

        # Save enhancement-specific outputs
        self._save_enhancement_outputs(predictions, output_dir)

        # Save predictions to CSV
        pred_path = os.path.join(output_dir, "predictions.csv")
        pred_dict = {}

        # Add all numeric predictions to CSV
        for key, value in predictions.items():
            if isinstance(value, np.ndarray) and value.ndim <= 2:
                if value.ndim == 1:
                    pred_dict[key] = value
                else:
                    for i in range(value.shape[1]):
                        pred_dict[f"{key}_{i}"] = value[:, i]

        if pred_dict:
            pd.DataFrame(pred_dict).to_csv(pred_path, index=False)

        logger.info(f"Predictions saved to {output_dir}")

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to image array with proper normalization"""
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        tensor = tensor.cpu()

        # Denormalize using dataset mean and std
        mean = torch.tensor(self.config["dataset"]["mean"]).view(1, 1, -1)
        std = torch.tensor(self.config["dataset"]["std"]).view(1, 1, -1)
        tensor = tensor * std + mean

        return (tensor.clamp(0, 1) * 255).numpy().astype(np.uint8)

    def _apply_astronomical_enhancements(
        self, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply astronomical-specific enhancements"""
        enhanced = {}
        if "reconstruction" in output:
            enhanced["reconstruction"] = output["reconstruction"]
            if "star_features" in output:
                enhanced["star_features"] = output["star_features"]
            if "galaxy_features" in output:
                enhanced["galaxy_features"] = output["galaxy_features"]
        return enhanced

    def _apply_medical_enhancements(
        self, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply medical-specific enhancements"""
        enhanced = {}
        if "reconstruction" in output:
            enhanced["reconstruction"] = output["reconstruction"]
            if "boundary_features" in output:
                enhanced["boundary_features"] = output["boundary_features"]
            if "lesion_features" in output:
                enhanced["lesion_features"] = output["lesion_features"]
        return enhanced

    def _apply_agricultural_enhancements(
        self, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply agricultural-specific enhancements"""
        enhanced = {}
        if "reconstruction" in output:
            enhanced["reconstruction"] = output["reconstruction"]
            if "texture_features" in output:
                enhanced["texture_features"] = output["texture_features"]
            if "damage_features" in output:
                enhanced["damage_features"] = output["damage_features"]
        return enhanced


# ----------------------------------------------
class ClusteringLoss(nn.Module):
    """Loss function for clustering in latent space using KL divergence"""

    def __init__(self, num_clusters: int, feature_dims: int, temperature: float = 1.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        # Learnable cluster centers
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, feature_dims))

    def forward(
        self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate distances to cluster centers
        distances = torch.cdist(embeddings, self.cluster_centers)

        # Convert distances to probabilities (soft assignments)
        q_dist = 1.0 / (1.0 + (distances / self.temperature) ** 2)
        q_dist = q_dist / q_dist.sum(dim=1, keepdim=True)

        if labels is not None:
            # If labels are provided, create target distribution
            p_dist = torch.zeros_like(q_dist)
            for i in range(self.num_clusters):
                mask = labels == i
                if mask.any():
                    p_dist[mask, i] = 1.0
        else:
            # Self-supervised target distribution (following DEC paper)
            p_dist = (q_dist**2) / q_dist.sum(dim=0, keepdim=True)
            p_dist = p_dist / p_dist.sum(dim=1, keepdim=True)

        # Calculate KL divergence loss
        kl_loss = F.kl_div(q_dist.log(), p_dist, reduction="batchmean")

        # Return both loss and cluster assignments
        return kl_loss, q_dist.argmax(dim=1)


class EnhancedAutoEncoderLoss(nn.Module):
    """Combined loss function for enhanced autoencoder with clustering and classification"""

    def __init__(
        self,
        num_classes: int,
        feature_dims: int,
        reconstruction_weight: float = 1.0,
        clustering_weight: float = 0.1,
        classification_weight: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.clustering_weight = clustering_weight
        self.classification_weight = classification_weight

        self.clustering_loss = ClusteringLoss(
            num_clusters=num_classes, feature_dims=feature_dims, temperature=temperature
        )
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        input_data: torch.Tensor,
        reconstruction: torch.Tensor,
        embedding: torch.Tensor,
        classification_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, input_data)

        # Clustering loss
        cluster_loss, cluster_assignments = self.clustering_loss(embedding, labels)

        # Classification loss
        if labels is not None:
            class_loss = self.classification_loss(classification_logits, labels)
        else:
            # Use cluster assignments as pseudo-labels when true labels unavailable
            class_loss = self.classification_loss(
                classification_logits, cluster_assignments
            )

        # Combine losses
        total_loss = (
            self.reconstruction_weight * recon_loss
            + self.clustering_weight * cluster_loss
            + self.classification_weight * class_loss
        )

        return total_loss, cluster_assignments, classification_logits.argmax(dim=1)


class DetailPreservingLoss(nn.Module):
    """Loss function that preserves fine details and enhances class differences.

    Components:
    1. Laplacian filtering - Preserves high-frequency details and edges
    2. Gram matrix analysis - Maintains texture patterns
    3. Frequency domain loss - Emphasizes high-frequency components
    """

    def __init__(self, detail_weight=1.0, texture_weight=0.8, frequency_weight=0.6):
        super().__init__()
        self.detail_weight = detail_weight
        self.texture_weight = texture_weight
        self.frequency_weight = frequency_weight

        # High-pass filters for detail detection
        self.laplacian = KF.Laplacian(3)
        self.sobel = KF.SpatialGradient()

    def forward(self, prediction, target):
        # Base reconstruction loss
        recon_loss = F.mse_loss(prediction, target)

        # High-frequency detail preservation
        pred_lap = self.laplacian(prediction)
        target_lap = self.laplacian(target)
        detail_loss = F.l1_loss(pred_lap, target_lap)

        # Texture preservation using Gram matrices
        pred_gram = self._gram_matrix(prediction)
        target_gram = self._gram_matrix(target)
        texture_loss = F.mse_loss(pred_gram, target_gram)

        # Frequency domain loss
        freq_loss = self._frequency_loss(prediction, target)

        # Combine losses with weights
        total_loss = (
            recon_loss
            + self.detail_weight * detail_loss
            + self.texture_weight * texture_loss
            + self.frequency_weight * freq_loss
        )

        return total_loss

    def _gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)

    def _frequency_loss(self, prediction, target):
        # Convert to frequency domain
        pred_freq = torch.fft.fft2(prediction)
        target_freq = torch.fft.fft2(target)

        # Compute magnitude spectrum
        pred_mag = torch.abs(pred_freq)
        target_mag = torch.abs(target_freq)

        # Focus on high-frequency components
        high_freq_mask = self._create_high_freq_mask(pred_mag.shape)
        high_freq_mask = high_freq_mask.to(prediction.device)

        pred_high = pred_mag * high_freq_mask
        target_high = target_mag * high_freq_mask

        return F.mse_loss(pred_high, target_high)

    def _create_high_freq_mask(self, shape):
        _, _, h, w = shape
        mask = torch.ones((h, w))
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4

        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        dist_from_center = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
        mask[dist_from_center < radius] = 0.2

        return mask.unsqueeze(0).unsqueeze(0)


class StructuralLoss(nn.Module):
    """Loss function to enhance image structures like contours and regions"""

    def __init__(self, edge_weight=1.0, smoothness_weight=0.5):
        super().__init__()
        self.edge_weight = edge_weight
        self.smoothness_weight = smoothness_weight
        self.sobel = KF.SpatialGradient()

    def forward(self, prediction, target):
        # Basic reconstruction loss
        recon_loss = F.mse_loss(prediction, target)

        # Edge detection loss using Sobel filters
        pred_edges = self.sobel(prediction)
        target_edges = self.sobel(target)
        edge_loss = F.mse_loss(pred_edges, target_edges)

        # Smoothness loss to preserve continuous regions
        smoothness_loss = torch.mean(
            torch.abs(prediction[:, :, :, :-1] - prediction[:, :, :, 1:])
        ) + torch.mean(torch.abs(prediction[:, :, :-1, :] - prediction[:, :, 1:, :]))

        return (
            recon_loss
            + self.edge_weight * edge_loss
            + self.smoothness_weight * smoothness_loss
        )


class ColorEnhancementLoss(nn.Module):
    """Loss function to enhance color variations across channels"""

    def __init__(self, channel_weight=0.5, contrast_weight=0.3):
        super().__init__()
        self.channel_weight = channel_weight
        self.contrast_weight = contrast_weight

    def forward(self, prediction, target):
        # Basic reconstruction loss
        recon_loss = F.mse_loss(prediction, target)

        # Channel correlation loss
        pred_corr = self._channel_correlation(prediction)
        target_corr = self._channel_correlation(target)
        channel_loss = F.mse_loss(pred_corr, target_corr)

        # Color contrast loss
        contrast_loss = self._color_contrast_loss(prediction, target)

        return (
            recon_loss
            + self.channel_weight * channel_loss
            + self.contrast_weight * contrast_loss
        )

    def _channel_correlation(self, x):
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1)
        mean = torch.mean(x_flat, dim=2).unsqueeze(2)
        x_centered = x_flat - mean
        corr = torch.bmm(x_centered, x_centered.transpose(1, 2))
        return corr / (h * w)

    def _color_contrast_loss(self, prediction, target):
        pred_std = torch.std(prediction, dim=[2, 3])
        target_std = torch.std(target, dim=[2, 3])
        return F.mse_loss(pred_std, target_std)


class MorphologyLoss(nn.Module):
    """Loss function to enhance morphological features"""

    def __init__(self, shape_weight=0.7, symmetry_weight=0.3):
        super().__init__()
        self.shape_weight = shape_weight
        self.symmetry_weight = symmetry_weight

    def forward(self, prediction, target):
        # Basic reconstruction loss
        recon_loss = F.mse_loss(prediction, target)

        # Shape preservation loss using moment statistics
        shape_loss = self._moment_loss(prediction, target)

        # Symmetry preservation loss
        symmetry_loss = self._symmetry_loss(prediction, target)

        return (
            recon_loss
            + self.shape_weight * shape_loss
            + self.symmetry_weight * symmetry_loss
        )

    def _moment_loss(self, prediction, target):
        # Calculate spatial moments to capture shape characteristics
        pred_moments = self._calculate_moments(prediction)
        target_moments = self._calculate_moments(target)
        return F.mse_loss(pred_moments, target_moments)

    def _calculate_moments(self, x):
        b, c, h, w = x.size()
        y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w))
        y_grid = y_grid.float().to(x.device) / h
        x_grid = x_grid.float().to(x.device) / w

        moments = []
        for i in range(b):
            for j in range(c):
                img = x[i, j]
                m00 = torch.sum(img)
                if m00 != 0:
                    m10 = torch.sum(img * y_grid)
                    m01 = torch.sum(img * x_grid)
                    m20 = torch.sum(img * y_grid * y_grid)
                    m02 = torch.sum(img * x_grid * x_grid)
                    moments.append(
                        torch.stack([m00, m10 / m00, m01 / m00, m20 / m00, m02 / m00])
                    )
                else:
                    moments.append(torch.zeros(5).to(x.device))

        return torch.stack(moments).view(b, c, -1)

    def _symmetry_loss(self, prediction, target):
        # Compare horizontal and vertical symmetry
        h_pred = self._horizontal_symmetry(prediction)
        h_target = self._horizontal_symmetry(target)
        v_pred = self._vertical_symmetry(prediction)
        v_target = self._vertical_symmetry(target)

        return F.mse_loss(h_pred, h_target) + F.mse_loss(v_pred, v_target)

    def _horizontal_symmetry(self, x):
        return F.mse_loss(x, torch.flip(x, [-1]))

    def _vertical_symmetry(self, x):
        return F.mse_loss(x, torch.flip(x, [-2]))


# Set sharing strategy at the start
torch.multiprocessing.set_sharing_strategy("file_system")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention module for feature maps"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        # Query, Key, and Value transformations
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Output transformation
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()

        # Compute queries, keys, and values
        queries = (
            self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        )  # (B, H*W, C')
        keys = self.key(x).view(batch_size, -1, height * width)  # (B, C', H*W)
        values = self.value(x).view(batch_size, -1, height * width)  # (B, C, H*W)

        # Compute attention scores
        attention_scores = torch.bmm(queries, keys)  # (B, H*W, H*W)
        attention_scores = F.softmax(attention_scores, dim=-1)  # Normalize scores

        # Apply attention to values
        out = torch.bmm(values, attention_scores.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(
            batch_size, channels, height, width
        )  # Reshape to original dimensions

        # Combine with input
        out = self.gamma * out + x  # Residual connection
        return out


class DCTLayer(nn.Module):  # Do a cosine Transform
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.dct(x)

    def dct(self, x):
        x = x.permute(0, 2, 3, 1)  # Change to [batch, height, width, channels]
        x = torch.fft.fft(x, dim=1)
        x = torch.fft.fft(x, dim=2)
        x = x.real
        x = x.permute(0, 3, 1, 2)  # Change back to [batch, channels, height, width]
        return x


class DynamicAutoencoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        feature_dims: int,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.input_shape = (
            config["dataset"]["in_channels"],  # Use configured channels
            config["dataset"]["input_size"][0],
            config["dataset"]["input_size"][1],
        )
        input_shape = self.input_shape
        self.in_channels = config["dataset"]["in_channels"]
        self.feature_dims = feature_dims
        self.num_classes = num_classes

        # Calculate progressive spatial dimensions
        self.spatial_dims = []
        current_size = input_shape[1]  # Start with height (assuming square)
        self.layer_sizes = self._calculate_layer_sizes()

        for _ in self.layer_sizes:
            self.spatial_dims.append(current_size)
            current_size = current_size // 2

        self.final_spatial_dim = current_size
        # Calculate flattened size after all conv layers
        self.flattened_size = self.layer_sizes[-1] * (self.final_spatial_dim**2)

        # Encoder layers with self-attention
        self.encoder_layers = nn.ModuleList()
        in_channels = self.in_channels  # Start with input channels
        for size in self.layer_sizes:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, size, 3, stride=2, padding=1),
                    nn.BatchNorm2d(size),
                    nn.LeakyReLU(0.2),
                    SelfAttention(size),
                )
            )
            in_channels = size

        # Class-aware embedding
        if self.num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, feature_dims)

        # Embedder layers
        self.embedder = nn.Sequential(
            nn.Linear(self.flattened_size, feature_dims),
            nn.BatchNorm1d(feature_dims),
            nn.LeakyReLU(0.2),
        )

        # Unembedder (decoder start)
        self.unembedder = nn.Sequential(
            nn.Linear(feature_dims, self.flattened_size),
            nn.BatchNorm1d(self.flattened_size),
            nn.LeakyReLU(0.2),
        )

        # Decoder layers with self-attention and DCT
        self.decoder_layers = nn.ModuleList()
        in_channels = self.layer_sizes[-1]

        # Build decoder layers in reverse
        for i in range(len(self.layer_sizes) - 1, -1, -1):
            out_channels = self.in_channels if i == 0 else self.layer_sizes[i - 1]
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(out_channels) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2) if i > 0 else nn.Tanh(),
                    SelfAttention(out_channels),
                    DCTLayer()
                    if i == 0
                    else nn.Identity(),  # Apply DCT at the final layer
                )
            )
            in_channels = out_channels

    def _calculate_layer_sizes(self) -> List[int]:
        """Calculate progressive channel sizes for encoder/decoder"""
        base_channels = 32
        sizes = []
        current_size = base_channels

        min_dim = min(self.input_shape[1:])
        max_layers = int(np.log2(min_dim)) - 2

        for _ in range(max_layers):
            sizes.append(current_size)
            if current_size < 128:
                current_size *= 2

        logger.info(f"Layer sizes: {sizes}")
        return sizes

    def encode(
        self, x: torch.Tensor, class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode input images to feature space"""
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Input has {x.size(1)} channels, expected {self.in_channels}"
            )

        for layer in self.encoder_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.embedder(x)

        if self.num_classes is not None and class_labels is not None:
            class_emb = self.class_embedding(class_labels)
            x = x + class_emb

        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode features back to image space"""
        x = self.unembedder(x)
        x = x.view(
            x.size(0),
            self.layer_sizes[-1],
            self.final_spatial_dim,
            self.final_spatial_dim,
        )

        for layer in self.decoder_layers:
            x = layer(x)

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Output has {x.size(1)} channels, expected {self.in_channels}"
            )

        return x

    def forward(
        self, x: torch.Tensor, class_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder"""
        embedding = self.encode(x, class_labels)
        reconstruction = self.decode(embedding)
        return embedding, reconstruction

    def get_encoding_shape(self) -> Tuple[int, ...]:
        """Get the shape of the encoding at each layer"""
        return tuple([size for size in self.layer_sizes])

    def get_spatial_dims(self) -> List[List[int]]:
        """Get the spatial dimensions at each layer"""
        return self.spatial_dims.copy()


class AutoencoderLoss(nn.Module):
    """Composite loss function for autoencoder training"""

    def __init__(self, reconstruction_weight: float = 1.0, feature_weight: float = 0.1):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.feature_weight = feature_weight

    def forward(
        self,
        input_data: torch.Tensor,
        reconstruction: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, input_data)

        # Feature distribution loss (encourage normal distribution)
        feature_loss = torch.mean(torch.abs(embedding.mean(dim=0))) + torch.mean(
            torch.abs(embedding.std(dim=0) - 1)
        )

        return (
            self.reconstruction_weight * recon_loss + self.feature_weight * feature_loss
        )


class StructurePreservingAutoencoder(DynamicAutoencoder):
    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int, config: Dict):
        super().__init__(input_shape, feature_dims)

        # Add residual connections for detail preservation
        self.skip_connections = nn.ModuleList()

        # Enhanced encoder with more layers for fine detail capture
        self.detail_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.PReLU(),
                    nn.Conv2d(
                        32, 32, kernel_size=3, padding=1, groups=4
                    ),  # Group convolution for local feature preservation
                    nn.BatchNorm2d(32),
                    nn.PReLU(),
                )
                for _ in range(3)  # Multiple detail preservation layers
            ]
        )

        # Structure-aware decoder components
        self.structure_decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.PReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=4),
                    nn.BatchNorm2d(32),
                    nn.PReLU(),
                )
                for _ in range(3)
            ]
        )

        # Edge detection and preservation module
        self.edge_detector = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, self.in_channels, kernel_size=3, padding=1),
        )

        # Local contrast enhancement
        self.contrast_enhancement = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2),
            nn.PReLU(),
            nn.Conv2d(32, self.in_channels, kernel_size=5, padding=2),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced encoding with detail preservation"""
        skip_features = []

        # Regular encoding path
        for layer in self.encoder_layers:
            x = layer(x)
            skip_features.append(x)

            # Apply detail preservation at each scale
            if len(skip_features) <= len(self.detail_encoder):
                x = (
                    self.detail_encoder[len(skip_features) - 1](x) + x
                )  # Residual connection

        x = x.view(x.size(0), -1)
        x = self.embedder(x)

        return x, skip_features

    def decode(
        self, x: torch.Tensor, skip_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Enhanced decoding with structure preservation"""
        x = self.unembedder(x)
        x = x.view(
            x.size(0),
            self.layer_sizes[-1],
            self.final_spatial_dim,
            self.final_spatial_dim,
        )

        for idx, layer in enumerate(self.decoder_layers):
            x = layer(x)

            # Apply structure preservation
            if idx < len(self.structure_decoder):
                x = self.structure_decoder[idx](x) + x

            # Add skip connections from encoder
            if idx < len(skip_features):
                x = (
                    x + skip_features[-(idx + 1)]
                )  # Add features from corresponding encoder layer

        # Enhance edges and local contrast
        edges = self.edge_detector(x)
        contrast = self.contrast_enhancement(x)

        # Combine all features
        x = x + 0.1 * edges + 0.1 * contrast

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with enhanced detail preservation"""
        # Extract edges for detail preservation
        self.edge_detector(x)

        # Main autoencoder path with skip connections
        embedding, skip_features = self.encode(x)
        reconstruction = self.decode(embedding, skip_features)

        # Enhance final reconstruction with edge and contrast features
        reconstruction = (
            reconstruction
            + 0.1 * self.edge_detector(reconstruction)
            + 0.1 * self.contrast_enhancement(reconstruction)
        )

        return embedding, reconstruction


class CustomImageDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform=None,
        csv_file: Optional[str] = None,
        target_size: int = 256,
        overlap: float = 0.5,
        config: Optional[Dict] = None,
        data_name: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.config = config if config is not None else {}
        self.data_name = data_name.lower() if data_name else None

        # Handle target size configuration
        if self.config.get("resize_images", False):
            size = 256
        else:
            input_cfg = self.config.get("dataset", {})
            size = input_cfg.get("input_size", 256)

        if isinstance(size, int):
            self.target_size = size
        elif isinstance(size, (list, tuple)):
            self.target_size = size[0]  # Use first dimension
        else:
            self.target_size = 256  # Final fallback

        self.overlap = overlap
        self.image_files = []
        self.labels = []
        self.file_indices = []
        self.filenames = []
        self.label_encoder = {}
        self.reverse_encoder = {}
        self.preprocessed_images = []

        # Determine valid class directories (only those containing images)
        valid_classes = []
        for entry in os.listdir(data_dir):
            entry_path = os.path.join(data_dir, entry)
            if os.path.isdir(entry_path):
                # Check if directory contains image files
                has_images = any(
                    fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
                    for fname in os.listdir(entry_path)
                )
                if has_images:
                    valid_classes.append(entry)

        # Use either custom data_name or folder name
        if not self.data_name:
            if self.config.get("dataset"):
                self.data_name = self.config["dataset"].get("name", "dataset")
            else:
                self.data_name = (
                    os.path.basename(os.path.normpath(data_dir)).lower() or "dataset"
                )

        # Create label mappings only for valid classes
        valid_classes = sorted(valid_classes)
        for idx, class_name in enumerate(valid_classes):
            self.label_encoder[class_name] = idx
            self.reverse_encoder[idx] = class_name

        # Collect images from valid classes
        for class_idx, class_name in enumerate(valid_classes):
            class_dir = os.path.join(data_dir, class_name)
            image_list = [
                fname
                for fname in os.listdir(class_dir)
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ]

            for img_name in image_list:
                self.image_files.append(os.path.join(class_dir, img_name))
                self.labels.append(class_idx)
                self.filenames.append(img_name)
                self.file_indices.append(len(self.image_files) - 1)

        # Fallback to CSV if no directory structure found
        if csv_file and os.path.exists(csv_file) and not self.image_files:
            self.data = pd.read_csv(csv_file)
            # Add CSV processing logic here if needed

        # Final configuration fallbacks
        if not self.data_name:
            self.data_name = "dataset"

        self.resize_images = self.config.get("resize_images", False)
        if self.resize_images:
            self.target_size = 256

        # Update config with determined name
        if self.config:
            if "dataset" not in self.config:
                self.config["dataset"] = {}
            self.config["dataset"]["name"] = self.data_name

        # Preprocess all images
        self._preprocess_all_images()

    def _preprocess_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Preprocess an image tensor to ensure it is suitable for the CNN.
        - If the image is smaller than target_size and resize_images is True, resize it to target_size.
        - If the image is larger than target_size, split it into sliding windows of target_size.

        Args:
            image_tensor (torch.Tensor): Input image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Processed image tensor(s). If windowing is used, returns a batch of tensors.
        """
        _, h, w = image_tensor.shape

        # Resize if image is smaller than target_size and resize_images is True
        if self.resize_images and (h < self.target_size or w < self.target_size):
            image_tensor = resize(
                image_tensor, (self.target_size, self.target_size), antialias=True
            )
            return image_tensor.unsqueeze(0)  # Add batch dimension

        # Split into sliding windows if image is larger than target_size
        if h > self.target_size or w > self.target_size:
            stride = int(
                self.target_size * (1 - self.overlap)
            )  # Stride based on overlap
            windows = []

            # Extract windows
            for y in range(0, h - self.target_size + 1, stride):
                for x in range(0, w - self.target_size + 1, stride):
                    window = image_tensor[
                        :, y : y + self.target_size, x : x + self.target_size
                    ]
                    windows.append(window)

            # Stack windows into a batch
            return torch.stack(windows)

        # If image is already target_size, return as is
        return image_tensor.unsqueeze(0)

    def _preprocess_all_images(self):
        """
        Preprocess all images to ensure consistent shapes (256x256).
        """
        # Create a directory to store preprocessed images (if saving to disk)
        self.preprocessed_dir = os.path.join(self.data_dir, "preprocessed")
        os.makedirs(self.preprocessed_dir, exist_ok=True)

        # Preprocess images with a progress bar
        for idx, img_path in enumerate(
            tqdm(self.image_files, desc=f"Preprocessing images")
        ):
            image = Image.open(img_path).convert("RGB")
            image_tensor = transforms.ToTensor()(image)

            # Resize or window the image
            preprocessed_tensors = self._preprocess_image(image_tensor)

            # Save preprocessed images to disk (optional)
            for i, tensor in enumerate(preprocessed_tensors):
                save_path = os.path.join(
                    self.preprocessed_dir, f"{self.filenames[idx]}_window{i}.pt"
                )
                torch.save(tensor, save_path)

            # Store preprocessed tensors (or paths to the preprocessed images)
            self.preprocessed_images.append(preprocessed_tensors)

    def __len__(self):
        return len(self.image_files)

    def get_additional_info(self, idx):
        """Retrieve additional information (file_index and filename) for a given index."""
        return self.file_indices[idx], self.filenames[idx]

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        file_index = self.file_indices[idx]  # Retrieve file index
        filename = self.filenames[idx]  # Retrieve filename

        if self.transform:
            image = self.transform(image)

        # Return only image and label during training
        return image, label


class DatasetProcessor:
    SUPPORTED_FORMATS = {
        "zip": zipfile.ZipFile,
        "tar": tarfile.TarFile,
        "tar.gz": tarfile.TarFile,
        "tgz": tarfile.TarFile,
        "gz": gzip.GzipFile,
        "bz2": bz2.BZ2File,
        "xz": lzma.LZMAFile,
    }

    SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")

    def __init__(
        self,
        datafile: str = "MNIST",
        datatype: str = "torchvision",
        output_dir: str = "data",
        config: Optional[Dict] = None,
        data_name: Optional[str] = None,
    ):  # Add data_name parameter
        self.datafile = datafile
        self.datatype = datatype.lower()
        self.output_dir = output_dir
        self.config = config if config is not None else {}

        # Determine dataset name
        if data_name:
            self.dataset_name = data_name.lower()
        else:
            if self.datatype == "torchvision":
                self.dataset_name = self.datafile.lower()
            else:
                # Handle both files and directories
                path_obj = Path(self.datafile)
                if path_obj.is_dir():
                    self.dataset_name = path_obj.name.lower()
                else:
                    self.dataset_name = path_obj.stem.lower() or "dataset"

        self.dataset_dir = os.path.join("data", self.dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.config_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.json")
        self.conf_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.conf")
        self.dbnn_conf_path = os.path.join(self.dataset_dir, "adaptive_dbnn.conf")

    def _extract_archive(self, archive_path: str) -> str:
        """Extract compressed archive to temporary directory"""
        extract_dir = os.path.join(self.dataset_dir, "temp_extract")
        os.makedirs(extract_dir, exist_ok=True)

        file_ext = Path(archive_path).suffix.lower()
        if file_ext.startswith("."):
            file_ext = file_ext[1:]

        if file_ext == "zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        elif file_ext in ["tar", "tgz"] or archive_path.endswith("tar.gz"):
            with tarfile.open(archive_path, "r:*") as tar_ref:
                tar_ref.extractall(extract_dir)
        elif file_ext == "gz":
            output_path = os.path.join(extract_dir, Path(archive_path).stem)
            with gzip.open(archive_path, "rb") as gz_file:
                with open(output_path, "wb") as out_file:
                    shutil.copyfileobj(gz_file, out_file)
        elif file_ext == "bz2":
            output_path = os.path.join(extract_dir, Path(archive_path).stem)
            with bz2.open(archive_path, "rb") as bz2_file:
                with open(output_path, "wb") as out_file:
                    shutil.copyfileobj(bz2_file, out_file)
        elif file_ext == "xz":
            output_path = os.path.join(extract_dir, Path(archive_path).stem)
            with lzma.open(archive_path, "rb") as xz_file:
                with open(output_path, "wb") as out_file:
                    shutil.copyfileobj(xz_file, out_file)
        else:
            raise ValueError(f"Unsupported archive format: {file_ext}")

        return extract_dir

    def _process_data_path(self, data_path: str) -> str:
        """Process input data path, handling compressed files if necessary"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")

        file_ext = Path(data_path).suffix.lower()
        if file_ext.startswith("."):
            file_ext = file_ext[1:]

        # Check if it's a compressed file
        if file_ext in self.SUPPORTED_FORMATS or data_path.endswith("tar.gz"):
            logger.info(f"Extracting compressed file: {data_path}")
            extract_dir = self._extract_archive(data_path)

            # Find the main data directory
            contents = os.listdir(extract_dir)
            if len(contents) == 1 and os.path.isdir(
                os.path.join(extract_dir, contents[0])
            ):
                return os.path.join(extract_dir, contents[0])
            return extract_dir

        return data_path

    def process(self) -> Tuple[str, Optional[str]]:
        """Process dataset and return paths to train and test directories"""
        if self.datatype == "torchvision":
            return self._process_torchvision()
        else:
            # Process the data path first
            processed_path = self._process_data_path(self.datafile)
            return self._process_custom(processed_path)

    def _handle_existing_directory(self, path: str):
        """Handle existing directory by either removing it or merging its contents."""
        if os.path.exists(path):
            response = input(
                f"The directory '{path}' already exists. Do you want to (R)emove it or (M)erge its contents? [R/M]: "
            ).lower()
            if response == "r":
                shutil.rmtree(path)
                os.makedirs(path)
            elif response == "m":
                # Merge contents (no action needed, as shutil.copytree will handle it with dirs_exist_ok=True)
                pass
            else:
                raise ValueError(
                    "Invalid choice. Please choose 'R' to remove or 'M' to merge."
                )

    def _process_custom(self, data_path: str) -> Tuple[str, Optional[str]]:
        """Process custom dataset structure"""
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")

        # Access enable_adaptive from training_params
        try:
            enable_adaptive = self.config["model"].get("enable_adaptive", True)
        except:
            enable_adaptive = True
            print(f"Enable Adaptive mode is set {enable_adaptive} in process custom")
        # Check if dataset already has train/test structure
        if os.path.isdir(os.path.join(data_path, "train")) and os.path.isdir(
            os.path.join(data_path, "test")
        ):
            # Check if adaptive_fit_predict is active
            if enable_adaptive:
                # Handle existing train directory
                self._handle_existing_directory(train_dir)
                # Merge train and test folders into a single train folder

                # Copy train data
                shutil.copytree(
                    os.path.join(data_path, "train"), train_dir, dirs_exist_ok=True
                )

                return (
                    train_dir,
                    test,
                )  # return train folder populated with both train and test data and the test folder for consistency.

            else:
                # Normal processing with separate train and test folders
                if os.path.exists(train_dir):
                    shutil.rmtree(train_dir)
                if os.path.exists(test_dir):
                    shutil.rmtree(test_dir)

                shutil.copytree(os.path.join(data_path, "train"), train_dir)
                shutil.copytree(os.path.join(data_path, "test"), test_dir)
                return train_dir, test_dir
        # Handle single directory with class subdirectories
        if not os.path.isdir(data_path):
            raise ValueError(f"Invalid dataset path: {data_path}")

        class_dirs = [
            d
            for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ]

        if not class_dirs:
            raise ValueError(f"No class directories found in {data_path}")

        # Ask user about train/test split
        response = input("Create train/test split? (y/n): ").lower()
        if response == "y":
            test_size = float(input("Enter test size (0-1, default: 0.2): ") or "0.2")
            return self._create_train_test_split(data_path, test_size)
        else:
            # Use all data for training
            os.makedirs(train_dir, exist_ok=True)
            for class_dir in class_dirs:
                src = os.path.join(data_path, class_dir)
                dst = os.path.join(train_dir, class_dir)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            return train_dir, None

    def cleanup(self):
        """Clean up temporary files"""
        temp_dir = os.path.join(self.dataset_dir, "temp_extract")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    # ------------------------
    def get_transforms(self, config: Dict, is_train: bool = True) -> transforms.Compose:
        """Get transforms based on configuration"""
        transform_list = []

        # Handle resolution and channel conversion first
        target_size = tuple(config["dataset"]["input_size"])
        target_channels = config["dataset"]["in_channels"]

        # Resolution adjustment
        transform_list.append(transforms.Resize(target_size))

        # Channel conversion
        if target_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))

        # Training augmentations
        if is_train and config.get("augmentation", {}).get("enabled", True):
            aug_config = config["augmentation"]
            if aug_config.get("random_crop", {}).get("enabled", False):
                transform_list.append(transforms.RandomCrop(target_size, padding=4))
            if aug_config.get("horizontal_flip", {}).get("enabled", False):
                transform_list.append(transforms.RandomHorizontalFlip())
            if aug_config.get("color_jitter", {}).get("enabled", False):
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    )
                )

        # Final transforms
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config["dataset"]["mean"], std=config["dataset"]["std"]
                ),
            ]
        )

        return transforms.Compose(transform_list)

    def _generate_main_config(self, train_dir: str) -> Dict:
        """Generate main configuration with all necessary parameters"""
        input_size, in_channels = self._detect_image_properties(train_dir)
        class_dirs = [
            d
            for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ]
        num_classes = len(class_dirs)

        mean = [0.5] if in_channels == 1 else [0.485, 0.456, 0.406]
        std = [0.5] if in_channels == 1 else [0.229, 0.224, 0.225]
        feature_dims = min(128, np.prod(input_size) // 4)

        return {
            "dataset": {
                "name": self.dataset_name,
                "type": self.datatype,
                "in_channels": in_channels,
                "num_classes": num_classes,
                "input_size": list(input_size),
                "mean": mean,
                "std": std,
                "resize_images": False,
                "train_dir": train_dir,
                "test_dir": os.path.join(os.path.dirname(train_dir), "test"),
            },
            "model": {
                "encoder_type": "autoenc",
                "enable_adaptive": True,  # Default value
                "feature_dims": feature_dims,
                "learning_rate": 0.001,
                "optimizer": {
                    "type": "Adam",
                    "weight_decay": 0.0001,
                    "momentum": 0.9,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-08,
                },
                "scheduler": {
                    "type": "ReduceLROnPlateau",
                    "factor": 0.1,
                    "patience": 10,
                    "min_lr": 1e-06,
                    "verbose": True,
                },
                "autoencoder_config": {
                    "reconstruction_weight": 1.0,
                    "feature_weight": 0.1,
                    "convergence_threshold": 0.0001,
                    "min_epochs": 10,
                    "patience": 5,
                    "enhancements": {
                        "enabled": True,
                        "use_kl_divergence": True,
                        "use_class_encoding": False,
                        "kl_divergence_weight": 0.5,
                        "classification_weight": 0.5,
                        "clustering_temperature": 1.0,
                        "min_cluster_confidence": 0.7,
                    },
                },
                "loss_functions": {
                    "structural": {
                        "enabled": True,
                        "weight": 1.0,
                        "params": {"edge_weight": 1.0, "smoothness_weight": 0.5},
                    },
                    "color_enhancement": {
                        "enabled": True,
                        "weight": 0.8,
                        "params": {"channel_weight": 0.5, "contrast_weight": 0.3},
                    },
                    "morphology": {
                        "enabled": True,
                        "weight": 0.6,
                        "params": {"shape_weight": 0.7, "symmetry_weight": 0.3},
                    },
                    "detail_preserving": {
                        "enabled": True,
                        "weight": 0.8,
                        "params": {
                            "detail_weight": 1.0,
                            "texture_weight": 0.8,
                            "frequency_weight": 0.6,
                        },
                    },
                    "astronomical_structure": {
                        "enabled": True,
                        "weight": 1.0,
                        "components": {
                            "edge_preservation": True,
                            "peak_preservation": True,
                            "detail_preservation": True,
                        },
                    },
                    "medical_structure": {
                        "enabled": True,
                        "weight": 1.0,
                        "components": {
                            "boundary_preservation": True,
                            "tissue_contrast": True,
                            "local_structure": True,
                        },
                    },
                    "agricultural_pattern": {
                        "enabled": True,
                        "weight": 1.0,
                        "components": {
                            "texture_preservation": True,
                            "damage_pattern": True,
                            "color_consistency": True,
                        },
                    },
                },
                "enhancement_modules": {
                    "astronomical": {
                        "enabled": True,
                        "components": {
                            "structure_preservation": True,
                            "detail_preservation": True,
                            "star_detection": True,
                            "galaxy_features": True,
                            "kl_divergence": True,
                        },
                        "weights": {
                            "detail_weight": 1.0,
                            "structure_weight": 0.8,
                            "edge_weight": 0.7,
                        },
                    },
                    "medical": {
                        "enabled": True,
                        "components": {
                            "tissue_boundary": True,
                            "lesion_detection": True,
                            "contrast_enhancement": True,
                            "subtle_feature_preservation": True,
                        },
                        "weights": {
                            "boundary_weight": 1.0,
                            "lesion_weight": 0.8,
                            "contrast_weight": 0.6,
                        },
                    },
                    "agricultural": {
                        "enabled": True,
                        "components": {
                            "texture_analysis": True,
                            "damage_detection": True,
                            "color_anomaly": True,
                            "pattern_enhancement": True,
                            "morphological_features": True,
                        },
                        "weights": {
                            "texture_weight": 1.0,
                            "damage_weight": 0.8,
                            "pattern_weight": 0.7,
                        },
                    },
                },
            },
            "training": {
                "batch_size": 128,
                "epochs": 200,
                "num_workers": min(4, os.cpu_count() or 1),
                "checkpoint_dir": os.path.join(self.dataset_dir, "checkpoints"),
                "validation_split": 0.2,
                "invert_DBNN": True,
                "reconstruction_weight": 0.5,
                "feedback_strength": 0.3,
                "inverse_learning_rate": 0.1,
                "use_classwise_acc": True,  # classwise accuracy has priority
                "early_stopping": {"patience": 5, "min_delta": 0.001},
            },
            "augmentation": {
                "enabled": True,
                "random_crop": {"enabled": True, "padding": 4},
                "random_rotation": {"enabled": True, "degrees": 10},
                "horizontal_flip": {"enabled": True, "probability": 0.5},
                "vertical_flip": {"enabled": False},
                "color_jitter": {
                    "enabled": True,
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.2,
                    "hue": 0.1,
                },
                "normalize": {"enabled": True, "mean": mean, "std": std},
            },
            "execution_flags": {
                "mode": "train_and_predict",
                "use_gpu": torch.cuda.is_available(),
                "mixed_precision": True,
                "distributed_training": False,
                "debug_mode": False,
                "use_previous_model": True,
                "fresh_start": False,
            },
            "output": {
                "features_file": os.path.join(
                    self.dataset_dir, f"{self.dataset_name}.csv"
                ),
                "model_dir": os.path.join(self.dataset_dir, "models"),
                "visualization_dir": os.path.join(self.dataset_dir, "visualizations"),
            },
        }

    def _generate_dataset_conf(self, feature_dims: int) -> Dict:
        """Generate dataset-specific configuration"""
        return {
            "file_path": os.path.join(self.dataset_dir, f"{self.dataset_name}.csv"),
            "column_names": [f"feature_{i}" for i in range(feature_dims)] + ["target"],
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
                "strong_margin_threshold": 0.01,
                "marginal_margin_threshold": 0.01,
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

    def _generate_dbnn_config(self, main_config: Dict) -> Dict:
        """Generate DBNN-specific configuration"""
        return {
            "training_params": {
                "trials": main_config["training"]["epochs"],
                "epochs": main_config["training"]["epochs"],
                "learning_rate": main_config["model"]["learning_rate"],
                "batch_size": 128,
                "test_fraction": 0.2,
                "random_seed": 42,
                "minimum_training_accuracy": 0.95,
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "n_bins_per_dim": 128,
                "enable_adaptive": True,
                "invert_DBNN": main_config["training"].get("invert_DBNN", False),
                "reconstruction_weight": 0.5,
                "feedback_strength": 0.3,
                "inverse_learning_rate": 0.1,
                "Save_training_epochs": False,
                "training_save_path": os.path.join(self.dataset_dir, "training_data"),
                "modelType": "Histogram",
                "compute_device": "auto",
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

    def generate_default_config(self, train_dir: str) -> Dict:
        """Generate and manage all configuration files"""
        os.makedirs(self.dataset_dir, exist_ok=True)
        logger.info(
            f"Starting configuration generation for dataset: {self.dataset_name}"
        )

        # 1. Generate and handle main configuration (json)
        logger.info("Generating main configuration...")
        config = self._generate_main_config(train_dir)
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    existing_config = json.load(f)
                    logger.info(f"Found existing main config, merging...")
                    config = self._merge_configs(existing_config, config)
            except json.JSONDecodeError:
                logger.warning(
                    f"Invalid JSON in {self.config_path}, using default template"
                )

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Main configuration saved: {self.config_path}")

        # 2. Generate and handle dataset.conf using _generate_dataset_conf
        logger.info("Generating dataset configuration...")
        dataset_conf = self._generate_dataset_conf(config["model"]["feature_dims"])
        if os.path.exists(self.conf_path):
            try:
                with open(self.conf_path) as f:
                    existing_dataset_conf = json.load(f)
                    logger.info(f"Found existing dataset config, merging...")
                    dataset_conf = self._merge_configs(
                        existing_dataset_conf, dataset_conf
                    )
            except json.JSONDecodeError:
                logger.warning(
                    f"Invalid JSON in {self.conf_path}, using default template"
                )

        with open(self.conf_path, "w") as f:
            json.dump(dataset_conf, f, indent=4)
        logger.info(f"Dataset configuration saved: {self.conf_path}")

        # 3. Generate and handle adaptive_dbnn.conf using _generate_dbnn_config
        logger.info("Generating DBNN configuration...")
        dbnn_config = self._generate_dbnn_config(config)
        if os.path.exists(self.dbnn_conf_path):
            try:
                with open(self.dbnn_conf_path) as f:
                    existing_dbnn_config = json.load(f)
                    logger.info(f"Found existing DBNN config, merging...")
                    dbnn_config = self._merge_configs(existing_dbnn_config, dbnn_config)
            except json.JSONDecodeError:
                logger.warning(
                    f"Invalid JSON in {self.dbnn_conf_path}, using default template"
                )

        with open(self.dbnn_conf_path, "w") as f:
            json.dump(dbnn_config, f, indent=4)
        logger.info(f"DBNN configuration saved: {self.dbnn_conf_path}")

        # Return the main config for further use
        return config

    def _merge_configs(self, existing: Dict, default: Dict) -> Dict:
        """Recursively merge configs, preserving existing values"""
        result = existing.copy()
        for key, value in default.items():
            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = self._merge_configs(result[key], value)
        return result

    def _ensure_required_configs(self, config: Dict) -> Dict:
        """Ensure all required configurations exist"""
        if "loss_functions" not in config["model"]:
            config["model"]["loss_functions"] = {}

        if "autoencoder" not in config["model"]["loss_functions"]:
            config["model"]["loss_functions"]["autoencoder"] = {
                "enabled": True,
                "type": "AutoencoderLoss",
                "weight": 1.0,
                "params": {"reconstruction_weight": 1.0, "feature_weight": 0.1},
            }

        return config

    def _detect_image_properties(self, folder_path: str) -> Tuple[Tuple[int, int], int]:
        """Detect actual image properties but use config values if specified"""
        # Load existing config if available
        config_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                if "dataset" in config:
                    dataset_config = config["dataset"]
                    if all(
                        key in dataset_config for key in ["input_size", "in_channels"]
                    ):
                        logger.info("Using image properties from config file")
                        return (
                            tuple(dataset_config["input_size"]),
                            dataset_config["in_channels"],
                        )

        # Fall back to detection from files
        size_counts = defaultdict(int)
        channel_counts = defaultdict(int)
        samples_checked = 0

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS):
                    try:
                        with Image.open(os.path.join(root, file)) as img:
                            tensor = transforms.ToTensor()(img)
                            height, width = tensor.shape[1], tensor.shape[2]
                            channels = tensor.shape[0]

                            size_counts[(width, height)] += 1
                            channel_counts[channels] += 1
                            samples_checked += 1

                            if samples_checked >= 50:
                                break
                    except Exception as e:
                        logger.warning(f"Could not process image {file}: {str(e)}")
                        continue

            if samples_checked >= 50:
                break

        if not size_counts:
            raise ValueError(f"No valid images found in {folder_path}")

        input_size = max(size_counts.items(), key=lambda x: x[1])[0]
        in_channels = max(channel_counts.items(), key=lambda x: x[1])[0]

        return input_size, in_channels

    def _process_torchvision(self) -> Tuple[str, str]:
        """Process torchvision dataset"""
        dataset_name = self.datafile.upper()
        if not hasattr(datasets, dataset_name):
            raise ValueError(f"Torchvision dataset {dataset_name} not found")

        # Setup paths in dataset-specific directory
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Download and process datasets
        transform = transforms.ToTensor()

        train_dataset = getattr(datasets, dataset_name)(
            root=self.output_dir, train=True, download=True, transform=transform
        )

        test_dataset = getattr(datasets, dataset_name)(
            root=self.output_dir, train=False, download=True, transform=transform
        )

        # Save images with class directories
        def save_dataset_images(dataset, output_dir, split_name):
            logger.info(f"Processing {split_name} split...")

            class_to_idx = getattr(dataset, "class_to_idx", None)
            if class_to_idx:
                idx_to_class = {v: k for k, v in class_to_idx.items()}

            with tqdm(total=len(dataset), desc=f"Saving {split_name} images") as pbar:
                for idx, (img, label) in enumerate(dataset):
                    class_name = idx_to_class[label] if class_to_idx else str(label)
                    class_dir = os.path.join(output_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)

                    if isinstance(img, torch.Tensor):
                        img = transforms.ToPILImage()(img)

                    img_path = os.path.join(class_dir, f"{idx}.png")
                    img.save(img_path)
                    pbar.update(1)

        save_dataset_images(train_dataset, train_dir, "training")
        save_dataset_images(test_dataset, test_dir, "test")

        return train_dir, test_dir

    def _create_train_test_split(
        self, source_dir: str, test_size: float
    ) -> Tuple[str, str]:
        """Create train/test split from source directory"""
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for class_name in tqdm(os.listdir(source_dir), desc="Processing classes"):
            class_path = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            # Create class directories
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # Get all image files
            image_files = [
                f
                for f in os.listdir(class_path)
                if f.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS)
            ]

            # Random split
            random.shuffle(image_files)
            split_idx = int((1 - test_size) * len(image_files))
            train_files = image_files[:split_idx]
            test_files = image_files[split_idx:]

            # Copy files
            for fname in train_files:
                shutil.copy2(
                    os.path.join(class_path, fname),
                    os.path.join(train_class_dir, fname),
                )

            for fname in test_files:
                shutil.copy2(
                    os.path.join(class_path, fname), os.path.join(test_class_dir, fname)
                )

        return train_dir, test_dir


class ConfigManager:
    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
        self.editor = os.environ.get("EDITOR", "nano")

    def _open_editor(self, filepath: str) -> bool:
        """Open file in editor and return if changed"""
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                json.dump({}, f, indent=4)

        mtime = os.path.getmtime(filepath)
        try:
            subprocess.call([self.editor, filepath])
            changed = os.path.getmtime(filepath) > mtime
            if changed:
                # Validate JSON after editing
                with open(filepath) as f:
                    json.load(f)  # Just to validate
                return True
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in edited file {filepath}")
            return False
        except Exception as e:
            logger.error(f"Error opening editor: {str(e)}")
            return False
        return False

    def _validate_json(self, filepath: str) -> Tuple[bool, Dict]:
        """Validate JSON file structure"""
        try:
            with open(filepath) as f:
                data = json.load(f)
            return True, data
        except Exception as e:
            logger.error(f"Error validating {filepath}: {str(e)}")
            return False, {}

    def merge_configs(self, existing: Dict, template: Dict) -> Dict:
        """Recursively merge template into existing config, adding missing entries"""
        result = existing.copy()
        for key, value in template.items():
            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = self.merge_configs(result[key], value)
        return result

    def manage_config(self, filepath: str, template: Dict) -> Dict:
        """Manage configuration file without overwriting existing content"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            if os.path.exists(filepath):
                try:
                    with open(filepath) as f:
                        existing_config = json.load(f)
                    # Merge template into existing config
                    merged_config = self.merge_configs(existing_config, template)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in {filepath}, using template")
                    merged_config = template
            else:
                # For new file, use template
                merged_config = template

            # Save if file doesn't exist or changes were made
            if not os.path.exists(filepath) or merged_config != template:
                with open(filepath, "w") as f:
                    json.dump(merged_config, f, indent=4)
                logger.info(f"Updated configuration file: {filepath}")

            return merged_config

        except Exception as e:
            logger.error(f"Error managing config {filepath}: {str(e)}")
            return template

    def manage_csv(self, filepath: str, headers: List[str]) -> bool:
        """Manage CSV file"""
        if not os.path.exists(filepath):
            logger.info(f"Creating new CSV file: {filepath}")
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            return True

        with open(filepath) as f:
            reader = csv.reader(f)
            try:
                existing_headers = next(reader)
                if existing_headers != headers:
                    logger.warning("CSV headers don't match expected structure")
                    response = input(
                        f"Would you like to edit {filepath}? (y/n): "
                    ).lower()
                    if response == "y":
                        return self._open_editor(filepath)
            except StopIteration:
                logger.error("Empty CSV file detected")
                return False

        return True

    def _detect_image_properties(self, folder_path: str) -> Tuple[Tuple[int, int], int]:
        """Detect image size and channels from dataset"""
        img_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        size_counts = defaultdict(int)
        channel_counts = defaultdict(int)

        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in img_formats):
                    try:
                        with Image.open(os.path.join(root, file)) as img:
                            tensor = transforms.ToTensor()(img)
                            height, width = tensor.shape[1], tensor.shape[2]
                            channels = tensor.shape[0]

                            size_counts[(width, height)] += 1
                            channel_counts[channels] += 1
                    except Exception as e:
                        logger.warning(f"Could not read image {file}: {str(e)}")
                        continue

            if sum(size_counts.values()) >= 50:
                break

        if not size_counts:
            raise ValueError(f"No valid images found in {folder_path}")

        input_size = max(size_counts, key=size_counts.get)
        in_channels = max(channel_counts, key=channel_counts.get)

        return input_size, in_channels


class EnhancedConfigManager(ConfigManager):
    """Enhanced configuration manager with support for specialized imaging features"""

    def __init__(self, config_dir: str):
        super().__init__(config_dir)
        self.editor = os.environ.get("EDITOR", "nano")

    def verify_enhancement_config(self, config: Dict) -> Dict:
        """Verify and add enhancement-specific configurations"""
        if "model" not in config:
            config["model"] = {}

        # Add enhancement modules configuration
        config["model"].setdefault(
            "enhancement_modules",
            {
                "astronomical": {
                    "enabled": False,
                    "components": {
                        "structure_preservation": True,
                        "detail_preservation": True,
                        "star_detection": True,
                        "galaxy_features": True,
                        "kl_divergence": True,
                    },
                    "weights": {
                        "detail_weight": 1.0,
                        "structure_weight": 0.8,
                        "edge_weight": 0.7,
                    },
                },
                "medical": {
                    "enabled": False,
                    "components": {
                        "tissue_boundary": True,
                        "lesion_detection": True,
                        "contrast_enhancement": True,
                        "subtle_feature_preservation": True,
                    },
                    "weights": {
                        "boundary_weight": 1.0,
                        "lesion_weight": 0.8,
                        "contrast_weight": 0.6,
                    },
                },
                "agricultural": {
                    "enabled": False,
                    "components": {
                        "texture_analysis": True,
                        "damage_detection": True,
                        "color_anomaly": True,
                        "pattern_enhancement": True,
                        "morphological_features": True,
                    },
                    "weights": {
                        "texture_weight": 1.0,
                        "damage_weight": 0.8,
                        "pattern_weight": 0.7,
                    },
                },
            },
        )

        # Add loss function configurations
        config["model"].setdefault("loss_functions", {})
        loss_functions = config["model"]["loss_functions"]

        loss_functions.setdefault(
            "astronomical_structure",
            {
                "enabled": False,
                "weight": 1.0,
                "components": {
                    "edge_preservation": True,
                    "peak_preservation": True,
                    "detail_preservation": True,
                },
            },
        )

        loss_functions.setdefault(
            "medical_structure",
            {
                "enabled": False,
                "weight": 1.0,
                "components": {
                    "boundary_preservation": True,
                    "tissue_contrast": True,
                    "local_structure": True,
                },
            },
        )

        loss_functions.setdefault(
            "agricultural_pattern",
            {
                "enabled": False,
                "weight": 1.0,
                "components": {
                    "texture_preservation": True,
                    "damage_pattern": True,
                    "color_consistency": True,
                },
            },
        )

        return config

    def configure_image_type(self, config: Dict, image_type: str) -> Dict:
        """Configure enhancement modules for specific image type"""
        if "dataset" not in config:
            config["dataset"] = {}

        config["dataset"]["image_type"] = image_type

        # Disable all enhancement modules first
        for module in config["model"]["enhancement_modules"]:
            config["model"]["enhancement_modules"][module]["enabled"] = False
            config["model"]["loss_functions"][f"{module}_structure"]["enabled"] = False

        # Enable specific module if not general
        if (
            image_type != "general"
            and image_type in config["model"]["enhancement_modules"]
        ):
            config["model"]["enhancement_modules"][image_type]["enabled"] = True
            config["model"]["loss_functions"][f"{image_type}_structure"][
                "enabled"
            ] = True

        return config

    def interactive_setup(self, config: Dict) -> Dict:
        """Interactive configuration setup for enhancements"""
        print("\nEnhanced Autoencoder Configuration")
        print("=================================")

        # Ensure enhancement config exists
        config = self.verify_enhancement_config(config)

        # Configure based on image type
        image_type = config["dataset"]["image_type"]
        if image_type != "general":
            module = config["model"]["enhancement_modules"][image_type]

            print(f"\nConfiguring {image_type} components:")

            # Configure components
            for component in module["components"]:
                current = module["components"][component]
                response = input(
                    f"Enable {component}? (y/n) [{['n', 'y'][current]}]: "
                ).lower()
                if response in ["y", "n"]:
                    module["components"][component] = response == "y"

            # Configure weights
            print(f"\nConfiguring {image_type} weights (0-1):")
            for weight_name, current_value in module["weights"].items():
                while True:
                    try:
                        new_value = input(f"{weight_name} [{current_value}]: ")
                        if new_value:
                            value = float(new_value)
                            if 0 <= value <= 1:
                                module["weights"][weight_name] = value
                                break
                            else:
                                print("Weight must be between 0 and 1")
                        else:
                            break
                    except ValueError:
                        print("Please enter a valid number")

            # Configure loss function
            loss_config = config["model"]["loss_functions"][f"{image_type}_structure"]
            print(f"\nConfiguring loss function components:")
            for component in loss_config["components"]:
                current = loss_config["components"][component]
                response = input(
                    f"Enable {component}? (y/n) [{['n', 'y'][current]}]: "
                ).lower()
                if response in ["y", "n"]:
                    loss_config["components"][component] = response == "y"

            # Configure loss weight
            while True:
                try:
                    new_weight = input(f"Loss weight [{loss_config['weight']}]: ")
                    if new_weight:
                        weight = float(new_weight)
                        if weight > 0:
                            loss_config["weight"] = weight
                            break
                        else:
                            print("Weight must be positive")
                    else:
                        break
                except ValueError:
                    print("Please enter a valid number")

        return config

    def print_current_config(self, config: Dict):
        """Print current enhancement configuration"""
        print("\nCurrent Enhancement Configuration:")
        print("================================")

        image_type = config["dataset"]["image_type"]
        print(f"\nImage Type: {image_type}")

        if image_type != "general":
            module = config["model"]["enhancement_modules"][image_type]

            print("\nEnabled Components:")
            for component, enabled in module["components"].items():
                print(f"- {component}: {'✓' if enabled else '✗'}")

            print("\nComponent Weights:")
            for weight_name, value in module["weights"].items():
                print(f"- {weight_name}: {value:.2f}")

            print("\nLoss Function Configuration:")
            loss_config = config["model"]["loss_functions"][f"{image_type}_structure"]
            print(f"- Weight: {loss_config['weight']:.2f}")
            print("\nEnabled Loss Components:")
            for component, enabled in loss_config["components"].items():
                print(f"- {component}: {'✓' if enabled else '✗'}")

    def get_active_components(self, config: Dict) -> Dict:
        """Get currently active enhancement components"""
        image_type = config["dataset"]["image_type"]
        if image_type == "general":
            return {}

        module = config["model"]["enhancement_modules"][image_type]
        loss_config = config["model"]["loss_functions"][f"{image_type}_structure"]

        return {
            "type": image_type,
            "components": {k: v for k, v in module["components"].items() if v},
            "weights": module["weights"],
            "loss_components": {
                k: v for k, v in loss_config["components"].items() if v
            },
            "loss_weight": loss_config["weight"],
        }


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")
    return logger


def get_dataset(config: Dict, transform) -> Tuple[Dataset, Optional[Dataset]]:
    """Get dataset based on configuration"""
    dataset_config = config["dataset"]

    if dataset_config["type"] == "torchvision":
        train_dataset = getattr(torchvision.datasets, dataset_config["name"].upper())(
            root="./data", train=True, download=True, transform=transform
        )

        test_dataset = getattr(torchvision.datasets, dataset_config["name"].upper())(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        train_dir = dataset_config["train_dir"]
        test_dir = dataset_config.get("test_dir")

        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory not found: {train_dir}")

        train_dataset = CustomImageDataset(data_dir=train_dir, transform=transform)

        test_dataset = None
        if test_dir and os.path.exists(test_dir):
            test_dataset = CustomImageDataset(data_dir=test_dir, transform=transform)

    if config["training"].get("merge_datasets", False) and test_dataset is not None:
        return CombinedDataset(train_dataset, test_dataset), None

    return train_dataset, test_dataset


class CombinedDataset(Dataset):
    """Dataset that combines train and test sets"""

    def __init__(self, train_dataset: Dataset, test_dataset: Dataset):
        self.combined_data = ConcatDataset([train_dataset, test_dataset])

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        return self.combined_data[idx]


def update_config_with_args(config: Dict, args) -> Dict:
    """Update configuration with command line arguments"""
    if hasattr(args, "encoder_type"):
        config["model"]["encoder_type"] = args.encoder_type
    if hasattr(args, "batch_size"):
        config["training"]["batch_size"] = args.batch_size
    if hasattr(args, "epochs"):
        config["training"]["epochs"] = args.epochs
    if hasattr(args, "workers"):
        config["training"]["num_workers"] = args.workers
    if hasattr(args, "learning_rate"):
        config["model"]["learning_rate"] = args.learning_rate
    if hasattr(args, "cpu"):
        config["execution_flags"]["use_gpu"] = not args.cpu
    if hasattr(args, "debug"):
        config["execution_flags"]["debug_mode"] = args.debug

    return config


def print_usage():
    """Print usage information with examples"""
    print("\nCDBNN (Convolutional Deep Bayesian Neural Network) Image Processor")
    print("=" * 80)
    print("\nUsage:")
    print("  1. Interactive Mode:")
    print("     python cdbnn.py")
    print("\n  2. Command Line Mode:")
    print("     python cdbnn.py --data_type TYPE --data PATH [options]")

    print("\nRequired Arguments:")
    print("  --data_type     Type of dataset ('torchvision' or 'custom')")
    print("  --data          Dataset name (for torchvision) or path (for custom)")

    print("\nOptional Arguments:")
    print("  --encoder_type  Type of encoder ('cnn' or 'autoenc')")
    print("  --config        Path to configuration file (overrides other options)")
    print("  --batch_size    Batch size for training (default: 128)")
    print("  --epochs        Number of training epochs (default: 200)")
    print("  --workers       Number of data loading workers (default: 4)")
    print("  --learning_rate Learning rate (default: 0.001)")
    print("  --output-dir    Output directory (default: data)")
    print("  --cpu          Force CPU usage even if GPU is available")
    print("  --debug        Enable debug mode with verbose logging")

    print("\nExamples:")
    print("  1. Process MNIST dataset using CNN:")
    print(
        "     python cdbnn.py --data_type torchvision --data MNIST --encoder_type cnn"
    )

    print("  2. Process custom dataset using Autoencoder:")
    print(
        "     python cdbnn.py --data_type custom --data path/to/images --encoder_type autoenc"
    )


import argparse


def save_last_args(args):
    """Save arguments to JSON file"""
    args_dict = vars(args)
    with open("last_run.json", "w") as f:
        json.dump(args_dict, f, indent=4)


def load_last_args():
    """Load arguments from JSON file"""
    try:
        with open("last_run.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_interactive_args():
    """Get arguments interactively with invert DBNN support."""
    last_args = load_last_args()
    args = argparse.Namespace()
    # Add data_name prompt
    default_name = last_args.get("data_name") if last_args else None
    data_name = input(
        f"Enter dataset name (leave empty to auto-detect) [{default_name}]: "
    ).strip()
    args.data_name = data_name or default_name

    # Get mode (train/reconstruct/predict)
    while True:
        default = last_args.get("mode", "train") if last_args else "train"
        prompt = f"\nEnter mode (train/reconstruct/predict) [{default}]: "
        mode = input(prompt).strip().lower() or default
        if mode in ["train", "reconstruct", "predict"]:
            args.mode = mode
            break
        print("Invalid mode. Please enter 'train', 'reconstruct', or 'predict'")

    # Get data type
    while True:
        default = last_args.get("data_type", "") if last_args else ""
        prompt = (
            f"\nEnter dataset type (torchvision/custom) [{default}]: "
            if default
            else "\nEnter dataset type (torchvision/custom): "
        )
        data_type = input(prompt).strip().lower() or default
        if data_type in ["torchvision", "custom"]:
            args.data_type = data_type
            break
        print("Invalid type. Please enter 'torchvision' or 'custom'")

    # Get data path/name
    default = last_args.get("data", "") if last_args else ""
    prompt = f"Enter dataset name [{default}]: " if default else "Enter dataset name: "
    dataset_name = input(prompt).strip() or default

    # Handle predict mode
    if args.mode == "predict":
        # Set default model path
        default_model = f"data/{dataset_name}/checkpoints/{dataset_name}_unified.pth"
        prompt = f"Enter path to trained model [{default_model}]: "
        args.model_path = input(prompt).strip() or default_model

        # Set default input directory
        default_input = f"Data/{dataset_name}.zip" if dataset_name else ""
        prompt = f"Enter directory containing new images [{default_input}]: "
        args.input_path = input(prompt).strip() or default_input

        # Set default output CSV path
        default_csv = os.path.join("data", dataset_name, f"{dataset_name}.csv")
        prompt = f"Enter output CSV path [{default_csv}]: "
        args.output_csv = input(prompt).strip() or default_csv

    # Handle train/reconstruct modes
    else:
        # Ask about invert DBNN
        default_invert = last_args.get("invert_dbnn", True) if last_args else True
        invert_response = (
            input(f"Enable inverse DBNN mode? (y/n) [{['n', 'y'][default_invert]}]: ")
            .strip()
            .lower()
        )
        args.invert_dbnn = invert_response == "y" if invert_response else default_invert

        # If in reconstruct mode and invert DBNN is enabled, ask for input CSV
        if args.mode == "reconstruct" and args.invert_dbnn:
            default_csv = last_args.get("input_csv", "") if last_args else ""
            prompt = (
                f"Enter input CSV path (or leave empty for default) [{default_csv}]: "
            )
            args.input_csv = input(prompt).strip() or default_csv

    # Get encoder type
    while True:
        default = last_args.get("encoder_type", "autoenc") if last_args else "autoenc"
        prompt = f"Enter encoder type (cnn/autoenc) [{default}]: "
        encoder_type = input(prompt).strip().lower() or default
        if encoder_type in ["cnn", "autoenc"]:
            args.encoder_type = encoder_type
            break
        print("Invalid encoder type. Please enter 'cnn' or 'autoenc'")

    # Optional parameters
    default = last_args.get("batch_size", 128) if last_args else 128
    args.batch_size = int(input(f"Enter batch size [{default}]: ").strip() or default)

    if args.mode == "train":
        default = last_args.get("epochs", 200) if last_args else 200
        args.epochs = int(
            input(f"Enter number of epochs [{default}]: ").strip() or default
        )

    default = last_args.get("output", "data") if last_args else "data"
    args.output_dir = input(f"Enter output directory [{default}]: ").strip() or default

    # Set other defaults
    args.workers = last_args.get("workers", 4) if last_args else 4
    args.learning_rate = last_args.get("learning_rate", 0.01) if last_args else 0.01
    args.cpu = last_args.get("cpu", False) if last_args else False
    args.debug = last_args.get("debug", False) if last_args else False
    args.config = last_args.get("config", None) if last_args else None

    save_last_args(args)
    return args


def check_existing_model(dataset_dir, dataset_name):
    """Check existing model type from checkpoint"""
    checkpoint_path = os.path.join(
        dataset_dir, "checkpoints", f"{dataset_name}_best.pth"
    )
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            return checkpoint.get("config", {}).get("model", {}).get("encoder_type")
        except:
            pass
    return None


def detect_model_type_from_checkpoint(checkpoint_path):
    """Detect model architecture type from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        # Check for architecture-specific layer patterns
        is_cnn = any("conv_layers" in key for key in state_dict.keys())
        is_autoencoder = any("encoder_layers" in key for key in state_dict.keys())

        if is_cnn:
            return "cnn"
        elif is_autoencoder:
            return "autoenc"
        else:
            return None
    except Exception as e:
        logger.error(f"Error detecting model type: {str(e)}")
        return None


def configure_enhancements(config: Dict) -> Dict:
    """Interactive configuration of enhancement features"""
    enhancements = config["model"]["autoencoder_config"]["enhancements"]

    print("\nConfiguring Enhanced Autoencoder Features:")

    # KL Divergence configuration
    if input("Enable KL divergence clustering? (y/n) [n]: ").lower() != "n":
        enhancements["use_kl_divergence"] = True
        enhancements["kl_divergence_weight"] = float(
            input("Enter KL divergence weight (0-1) [0.1]: ") or 0.1
        )
    else:
        enhancements["use_kl_divergence"] = False

    # Class encoding configuration
    if input("Enable class encoding? (y/n) [y]: ").lower() == "n":
        enhancements["use_class_encoding"] = False
        enhancements["classification_weight"] = float(
            input("Enter classification weight (0-1) [0.1]: ") or 0.1
        )
    else:
        enhancements["use_class_encoding"] = True

    # Clustering configuration
    if enhancements["use_kl_divergence"]:
        enhancements["clustering_temperature"] = float(
            input("Enter clustering temperature (0.1-2.0) [1.0]: ") or 1.0
        )
        enhancements["min_cluster_confidence"] = float(
            input("Enter minimum cluster confidence (0-1) [0.7]: ") or 0.7
        )

    return config


def add_enhancement_features(config: Dict) -> Dict:
    """Add enhancement features to existing configuration"""
    # Ensure basic structure exists
    if "model" not in config:
        config["model"] = {}
    if "enhancement_modules" not in config["model"]:
        config["model"]["enhancement_modules"] = {}
    if "loss_functions" not in config["model"]:
        config["model"]["loss_functions"] = {}

    # Ask about each enhancement type
    print("\nAvailable Enhancement Features:")

    # Astronomical features
    if (
        input(
            "Add astronomical features (star detection, galaxy structure preservation)? (y/n) [n]: "
        ).lower()
        == "y"
    ):
        config["model"]["enhancement_modules"]["astronomical"] = {
            "enabled": True,
            "components": {
                "structure_preservation": True,
                "detail_preservation": True,
                "star_detection": True,
                "galaxy_features": True,
                "kl_divergence": True,
            },
            "weights": {
                "detail_weight": 1.0,
                "structure_weight": 0.8,
                "edge_weight": 0.7,
            },
        }
        config["model"]["loss_functions"]["astronomical_structure"] = {
            "enabled": True,
            "weight": 1.0,
            "components": {
                "edge_preservation": True,
                "peak_preservation": True,
                "detail_preservation": True,
            },
        }
        print("Astronomical features added.")

    # Medical features
    if (
        input(
            "Add medical features (tissue boundary, lesion detection)? (y/n) [n]: "
        ).lower()
        == "y"
    ):
        config["model"]["enhancement_modules"]["medical"] = {
            "enabled": True,
            "components": {
                "tissue_boundary": True,
                "lesion_detection": True,
                "contrast_enhancement": True,
                "subtle_feature_preservation": True,
            },
            "weights": {
                "boundary_weight": 1.0,
                "lesion_weight": 0.8,
                "contrast_weight": 0.6,
            },
        }
        config["model"]["loss_functions"]["medical_structure"] = {
            "enabled": True,
            "weight": 1.0,
            "components": {
                "boundary_preservation": True,
                "tissue_contrast": True,
                "local_structure": True,
            },
        }
        print("Medical features added.")

    # Agricultural features
    if (
        input(
            "Add agricultural features (texture analysis, damage detection)? (y/n) [n]: "
        ).lower()
        == "y"
    ):
        config["model"]["enhancement_modules"]["agricultural"] = {
            "enabled": True,
            "components": {
                "texture_analysis": True,
                "damage_detection": True,
                "color_anomaly": True,
                "pattern_enhancement": True,
                "morphological_features": True,
            },
            "weights": {
                "texture_weight": 1.0,
                "damage_weight": 0.8,
                "pattern_weight": 0.7,
            },
        }
        config["model"]["loss_functions"]["agricultural_pattern"] = {
            "enabled": True,
            "weight": 1.0,
            "components": {
                "texture_preservation": True,
                "damage_pattern": True,
                "color_consistency": True,
            },
        }
        print("Agricultural features added.")

    return config


def update_existing_config(config_path: str, new_config: Dict) -> Dict:
    """Update existing configuration while preserving current settings"""
    if os.path.exists(config_path):
        with open(config_path) as f:
            existing_config = json.load(f)

        # Merge configurations
        for key in new_config:
            if key in existing_config:
                if isinstance(existing_config[key], dict) and isinstance(
                    new_config[key], dict
                ):
                    existing_config[key].update(new_config[key])
                else:
                    existing_config[key] = new_config[key]
            else:
                existing_config[key] = new_config[key]

        return existing_config
    return new_config


def main():
    """Main function for CDBNN processing with enhancement configurations"""
    args = None
    try:
        # Setup logging
        logger = setup_logging()

        # First try to parse command line arguments
        try:
            args = parse_arguments()
        except SystemExit:
            # argparse exits when -h/--help is used
            return 0
        except:
            args = None

        # If no command line args or --interactive flag, use interactive input
        if args is None or getattr(args, "interactive", False):
            args = interactive_input()

        dataset_name = str(getattr(args, "data_name", "mnist"))

        # Process based on mode
        if args.mode == "predict":
            # Load the config
            config_path = os.path.join("data", dataset_name, f"{dataset_name}.json")
            if not os.path.exists(config_path):
                logger.error(f"Config file not found at {config_path}")
                config_path = input("Enter path to config file: ").strip()
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Config file not found at {config_path}")

            with open(config_path) as f:
                config = json.load(f)

            # Setup prediction logging
            os.makedirs("logs", exist_ok=True)
            log_file = f"logs/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            )
            logger.info(f"Logging setup complete. Log file: {log_file}")

            # Initialize the PredictionManager
            device = (
                "cuda"
                if torch.cuda.is_available() and not getattr(args, "cpu", False)
                else "cpu"
            )
            predictor = PredictionManager(config=config, device=device)

            # Set the dataset (if required)
            if hasattr(predictor.model, "set_dataset"):
                transform = predictor._get_transforms()
                dataset = predictor._create_dataset(
                    getattr(args, "input_path"), transform
                )
                predictor.model.set_dataset(dataset)
                logger.info(f"Dataset created with {len(dataset)} images")

            # Handle output path
            if not hasattr(args, "output") or not args.output:
                args.output = os.path.join("data", dataset_name, f"{dataset_name}.csv")
                logger.info(f"Using default output path: {args.output}")

            # Perform predictions
            logger.info("Starting prediction process...")
            predictor.predict_images(
                data_path=getattr(args, "input_path"),
                output_csv=getattr(args, "output"),
                batch_size=getattr(args, "batch_size", 128),
            )
            logger.info(f"Predictions saved to {args.output}")

        elif args.mode == "train":
            return handle_training_mode(args, logger)
        elif args.mode == "reconstruct":
            return handle_prediction_mode(args, logger)
        else:
            logger.error(f"Invalid mode: {args.mode}")
            return 1

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        if args and getattr(args, "debug", False):
            traceback.print_exc()
        return 1


def interactive_input():
    """Collect inputs interactively if no command line args provided"""
    print("\nInteractive Mode - Please enter the following information:")

    mode = (
        input("Enter mode (train/reconstruct/predict) [predict]: ").strip().lower()
        or "predict"
    )
    data_name = input("Enter dataset name [mnist]: ").strip() or "mnist"
    input_path = (
        input(
            f"Enter path to {'training data' if mode == 'train' else 'input images'} [Data/{data_name}.zip]: "
        ).strip()
        or f"Data/{data_name}.zip"
    )

    args = SimpleNamespace(
        mode=mode, data_name=data_name, input_path=input_path, interactive=True
    )

    # Mode-specific inputs
    if mode == "predict":
        args.model_path = (
            input(
                f"Enter path to trained model [data/{data_name}/checkpoints/{data_name}_unified.pth]: "
            ).strip()
            or f"data/{data_name}/checkpoints/{data_name}_unified.pth"
        )
        args.output = (
            input(f"Enter output CSV path [data/{data_name}/{data_name}.csv]: ").strip()
            or f"data/{data_name}/{data_name}.csv"
        )
        args.batch_size = int(input("Enter batch size [128]: ").strip() or 128)
        args.cpu = (
            input("Force CPU even if GPU available? (y/n) [n]: ").strip().lower() == "y"
        )

    elif mode == "train":
        args.epochs = int(input("Enter number of epochs [100]: ").strip() or 100)
        args.batch_size = int(input("Enter batch size [128]: ").strip() or 128)
        args.learning_rate = float(
            input("Enter learning rate [0.001]: ").strip() or 0.001
        )

    args.debug = input("Enable debug mode? (y/n) [n]: ").strip().lower() == "y"

    return args


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CDBNN Training and Prediction")

    # Main arguments
    parser.add_argument(
        "--mode",
        choices=["train", "reconstruct", "predict"],
        default="predict",
        help="Operation mode",
    )
    parser.add_argument(
        "--data_name", dest="data_name", default="dataset", help="Name of the dataset"
    )
    parser.add_argument(
        "--data_type",
        dest="data_type",
        default="custom",
        help="custom or torchvision dataset",
    )

    parser.add_argument(
        "--input_path", required=True, help="Path to input data (directory or zip file)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force interactive mode even with command line args",
    )
    parser.add_argument(
        "--encoder_type",
        choices=["autoenc", "cnn"],
        default="autoenc",
        help="Decide model type autoenc (default) or cnn",
    )

    # Prediction-specific
    parser.add_argument("--model-path", help="Path to trained model")
    parser.add_argument("--output", help="Output path for predictions")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for processing"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU even if GPU available"
    )

    # Training-specific
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for training"
    )

    # Debugging
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with verbose output"
    )

    return parser.parse_args()


def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def handle_training_mode(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle training mode operations"""
    try:
        # Setup paths
        # data_name = os.path.splitext(os.path.basename(args.data))[0]
        data_name = args.data_name
        data_dir = os.path.join("data", data_name)
        config_path = os.path.join(data_dir, f"{data_name}.json")

        # Process dataset
        processor = DatasetProcessor(
            args.input_path,
            args.data_type,
            getattr(args, "output", "data"),
            data_name=args.data_name,
        )
        train_dir, test_dir = processor.process()
        logger.info(f"Dataset processed: train_dir={train_dir}, test_dir={test_dir}")

        # Generate/verify configurations
        logger.info("Generating/verifying configurations...")
        config = processor.generate_default_config(train_dir)

        # Configure enhancements
        config = configure_image_processing(config, logger)

        # Update configuration with command line arguments
        config = update_config_with_args(config, args)

        fd = config["model"]["feature_dims"]
        feature_dims = int(
            input(f"Please specify the output feature dimensions[{ fd}]: ") or fd
        )
        config["model"]["feature_dims"] = feature_dims
        # Get model type
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        # Setup data loading
        transform = processor.get_transforms(config)
        train_dataset, test_dataset = get_dataset(config, transform)

        if train_dataset is None:
            raise ValueError("No training dataset available")

        # Create data loaders
        train_loader, test_loader = create_data_loaders(
            train_dataset, test_dataset, config
        )

        # Initialize model and loss manager
        model, loss_manager = initialize_model_components(config, logger)

        # Get training confirmation
        if not get_training_confirmation(logger):
            return 0

        # Perform training and feature extraction
        features_dict = perform_training_and_extraction(
            model, train_loader, test_loader, config, loss_manager, logger
        )

        # Save results
        save_training_results(features_dict, model, config, data_dir, data_name, logger)

        logger.info("Processing completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error in training mode: {str(e)}")
        raise


def handle_prediction_mode(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle prediction mode operations"""
    try:
        # Setup paths
        # data_name = os.path.splitext(os.path.basename(args.data))[0]
        data_name = args.data_name
        data_dir = os.path.join("data", data_name)

        # Load configuration
        config_path = os.path.join(data_dir, f"{data_name}.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        # Determine input CSV
        if args.invert_dbnn:
            input_csv = (
                args.input_csv
                if args.input_csv
                else os.path.join(data_dir, "reconstructed_input.csv")
            )
            if not os.path.exists(input_csv):
                raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        else:
            output_csv = (
                args.input_csv
                if args.input_csv
                else os.path.join(data_dir, f"{data_name}.csv")
            )

        # Setup output directory
        output_dir = os.path.join(
            data_dir, "predictions", datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(output_dir, exist_ok=True)

        # Initialize prediction manager and generate predictions
        predictor = PredictionManager(config)
        predictor.predict_from_csv(input_csv, output_dir)

        logger.info("Predictions completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error in prediction mode: {str(e)}")
        if args.debug:
            traceback.print_exc()
        return 1


def configure_image_processing(config: Dict, logger: logging.Logger) -> Dict:
    """Configure image processing type and enhancements
    # Display image type options
    print("\nSelect image type for enhanced processing:")
    image_types = ["general", "astronomical", "medical", "agricultural"]
    for i, type_name in enumerate(image_types, 1):
        print(f"{i}. {type_name}")

    # Get image type selection
    type_idx = int(input("\nSelect image type (1-4): ")) - 1
    image_type = image_types[type_idx]

    # Create appropriate configuration manager
    if image_type == "general":
    """
    image_type = "general"
    config_manager = GeneralEnhancementConfig(config)
    config_manager.configure_general_parameters()
    config_manager.configure_enhancements()
    """
    else:
        config_manager = SpecificEnhancementConfig(config, image_type)
        config_manager.configure()
    """
    # Get and update configuration
    config = config_manager.get_config()
    config["dataset"]["image_type"] = image_type

    return config


def create_data_loaders(
    train_dataset: Dataset, test_dataset: Optional[Dataset], config: Dict
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create data loaders for training and testing"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"],
        )

    return train_loader, test_loader


def initialize_model_components(
    config: Dict, logger: logging.Logger
) -> Tuple[nn.Module, EnhancedLossManager]:
    """Initialize model and loss manager"""
    logger.info(f"Initializing {config['dataset']['image_type']} enhanced model...")
    model = ModelFactory.create_model(config)
    loss_manager = EnhancedLossManager(config)
    return model, loss_manager


def get_training_confirmation(logger: logging.Logger) -> bool:
    """Get user confirmation for training"""
    if input("\nReady to start training. Proceed? (y/n): ").lower() == "n":
        logger.info("Training cancelled by user")
        return False
    return True


# Original dataset creation point
def perform_training_and_extraction(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader],
    config: Dict,
    loss_manager: EnhancedLossManager,
    logger: logging.Logger,
) -> Dict:
    """Perform model training and feature extraction"""
    # Training
    logger.info("Starting model training...")
    # HERE: train_loader is passed but train_dataset isn't stored anywhere
    train_model(model, train_loader, config, loss_manager)

    # Feature extraction
    logger.info("Extracting features...")
    # HERE: We need train_dataset but it's not accessible
    features_dict = extract_features_from_model(model, train_loader, test_loader)
    return features_dict


def extract_features_from_model(
    model: nn.Module, train_loader: DataLoader, test_loader: Optional[DataLoader]
) -> Dict:
    """Extract features from the model"""
    if isinstance(
        model,
        (
            AstronomicalStructurePreservingAutoencoder,
            MedicalStructurePreservingAutoencoder,
            AgriculturalPatternAutoencoder,
        ),
    ):
        features_dict = model.extract_features_with_class_info(train_loader)

        if test_loader:
            test_features = model.extract_features_with_class_info(test_loader)
            for key in features_dict:
                if isinstance(features_dict[key], torch.Tensor):
                    features_dict[key] = torch.cat(
                        [features_dict[key], test_features[key]]
                    )
    else:
        train_features, train_labels = model.extract_features(train_loader)
        if test_loader:
            test_features, test_labels = model.extract_features(test_loader)
            features = torch.cat([train_features, test_features])
            labels = torch.cat([train_labels, test_labels])
        else:
            features = train_features
            labels = train_labels
        features_dict = {"features": features, "labels": labels}

    return features_dict


def perform_training_and_extraction(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader],
    config: Dict,
    loss_manager: EnhancedLossManager,
    logger: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Perform model training and feature extraction"""
    # Training
    logger.info("Starting model training...")
    train_model(model, train_loader, config, loss_manager)

    # Feature extraction
    logger.info("Extracting features...")
    features_dict = model.extract_features(train_loader)

    # If test loader exists, extract and combine features
    if test_loader:
        test_features_dict = model.extract_features(test_loader)
        features_dict = merge_feature_dicts(features_dict, test_features_dict)

    return features_dict


def save_training_results(
    features_dict: Dict[str, torch.Tensor],
    model: nn.Module,
    config: Dict,
    data_dir: str,
    data_name: str,
    logger: logging.Logger,
) -> None:
    """Save training results and features"""
    # Save features
    output_path = os.path.join(data_dir, f"{data_name}.csv")
    model.save_features(features_dict, features_dict, output_path)

    # Save training history if available
    if hasattr(model, "history") and model.history:
        history_path = os.path.join(data_dir, "training_history.json")
        with open(history_path, "w") as f:
            # Convert tensor values to float for JSON serialization
            serializable_history = {
                k: [float(v) if isinstance(v, torch.Tensor) else v for v in vals]
                for k, vals in model.history.items()
            }
            json.dump(serializable_history, f, indent=4)

        # Plot training history
        plot_path = os.path.join(data_dir, "training_history.png")
        if isinstance(model, BaseAutoencoder):
            model.plot_training_history(save_path=plot_path)


def merge_feature_dicts(
    dict1: Dict[str, torch.Tensor], dict2: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Merge two feature dictionaries"""
    merged = {}
    for key in dict1.keys():
        if key in dict2:
            if isinstance(dict1[key], torch.Tensor) and isinstance(
                dict2[key], torch.Tensor
            ):
                merged[key] = torch.cat([dict1[key], dict2[key]])
            else:
                merged[key] = dict1[key]  # Keep original if not tensor
    return merged


if __name__ == "__main__":
    # print(f"{Colors.RED}The code has some bug in directly handling torchvision files. So recommendation is to use Get_Torchvision_images function instead{Colors.ENDC}")
    print("Updated on April 14/2025 Stable version")
    sys.exit(main())
