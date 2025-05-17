import argparse
import bz2
import csv
import gc
import glob
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
import time
import traceback
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import kornia.filters as KF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm


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


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extraction models"""

    def __init__(self, config: Dict, device: str = None):
        """Initialize base feature extractor"""
        self.config = self.verify_config(config)

        # Set device
        if device is None:
            self.device = torch.device(
                "cuda"
                if self.config["execution_flags"]["use_gpu"]
                and torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device

        # Initialize common parameters
        self.feature_dims = self.config["model"]["feature_dims"]
        self.learning_rate = self.config["model"].get("learning_rate", 0.001)

        # Initialize training metrics
        self.best_accuracy = 0.0
        self.best_loss = float("inf")
        self.current_epoch = 0
        self.history = defaultdict(list)
        self.training_log = []
        self.training_start_time = time.time()

        # Setup logging directory
        self.log_dir = os.path.join("Traininglog", self.config["dataset"]["name"])
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize model
        self.feature_extractor = self._create_model()

        # Load checkpoint or initialize optimizer
        if not self.config["execution_flags"].get("fresh_start", False):
            self._load_from_checkpoint()

        # Initialize optimizer if not created during checkpoint loading
        if not hasattr(self, "optimizer"):
            self.optimizer = self._initialize_optimizer()
            logger.info(f"Initialized {self.optimizer.__class__.__name__} optimizer")

        # Initialize scheduler
        self.scheduler = None
        if self.config["model"].get("scheduler"):
            self.scheduler = self._initialize_scheduler()
            if self.scheduler:
                logger.info(
                    f"Initialized {self.scheduler.__class__.__name__} scheduler"
                )

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create and return the feature extraction model"""

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer based on configuration"""
        optimizer_config = self.config["model"].get("optimizer", {})

        # Set base parameters
        optimizer_params = {
            "lr": self.learning_rate,
            "weight_decay": optimizer_config.get("weight_decay", 1e-4),
        }

        # Configure optimizer-specific parameters
        optimizer_type = optimizer_config.get("type", "Adam")
        if optimizer_type == "SGD":
            optimizer_params["momentum"] = optimizer_config.get("momentum", 0.9)
            optimizer_params["nesterov"] = optimizer_config.get("nesterov", False)
        elif optimizer_type == "Adam":
            optimizer_params["betas"] = (
                optimizer_config.get("beta1", 0.9),
                optimizer_config.get("beta2", 0.999),
            )
            optimizer_params["eps"] = optimizer_config.get("epsilon", 1e-8)

        # Get optimizer class
        try:
            optimizer_class = getattr(optim, optimizer_type)
        except AttributeError:
            logger.warning(f"Optimizer {optimizer_type} not found, using Adam")
            optimizer_class = optim.Adam
            optimizer_type = "Adam"

        # Create and return optimizer
        optimizer = optimizer_class(
            self.feature_extractor.parameters(), **optimizer_params
        )

        logger.info(
            f"Initialized {optimizer_type} optimizer with parameters: {optimizer_params}"
        )
        return optimizer

    def _initialize_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Initialize learning rate scheduler if specified in config"""
        scheduler_config = self.config["model"].get("scheduler", {})
        if not scheduler_config:
            return None

        scheduler_type = scheduler_config.get("type")
        if not scheduler_type:
            return None

        try:
            if scheduler_type == "StepLR":
                return optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get("step_size", 7),
                    gamma=scheduler_config.get("gamma", 0.1),
                )
            elif scheduler_type == "ReduceLROnPlateau":
                return optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=scheduler_config.get("factor", 0.1),
                    patience=scheduler_config.get("patience", 10),
                    verbose=True,
                )
            elif scheduler_type == "CosineAnnealingLR":
                return optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_config.get(
                        "T_max", self.config["training"]["epochs"]
                    ),
                    eta_min=scheduler_config.get("eta_min", 0),
                )
        except Exception as e:
            logger.warning(f"Failed to initialize scheduler: {str(e)}")
            return None

        return None

    def verify_config(self, config: Dict) -> Dict:
        """Verify and fill in missing configuration values"""
        if "dataset" not in config:
            raise ValueError("Configuration must contain 'dataset' section")

        # Ensure all required sections exist
        required_sections = [
            "dataset",
            "model",
            "training",
            "execution_flags",
            "likelihood_config",
            "active_learning",
        ]
        for section in required_sections:
            if section not in config:
                config[section] = {}
        # Add output/input configuration
        if "output" not in config:
            config["output"] = {}
        config["output"].setdefault("image_dir", "output/images")
        config["output"].setdefault("mode", "train")  # 'train' or 'predict'
        config["output"].setdefault(
            "csv_dir", os.path.join("data", config["dataset"]["name"])
        )
        config["output"].setdefault(
            "input_csv", None
        )  # Will be set to default if None in predict mode

        model = config.setdefault("model", {})
        model.setdefault("feature_dims", 128)
        model.setdefault("learning_rate", 0.001)
        model.setdefault("encoder_type", "cnn")
        model.setdefault("modelType", "Histogram")

        loss_functions = model.setdefault("loss_functions", {})
        # Add loss functions configuration
        loss_functions.setdefault(
            "autoencoder",
            {
                "enabled": False,
                "type": "AutoencoderLoss",
                "weight": 1.0,
                "params": {"reconstruction_weight": 1.0, "feature_weight": 0.1},
            },
        )
        loss_functions.setdefault(
            "detail_preserving",
            {
                "enabled": True,
                "type": "DetailPreservingLoss",
                "weight": 0.8,
                "params": {
                    "detail_weight": 1.0,
                    "texture_weight": 0.8,
                    "frequency_weight": 0.6,
                },
            },
        )
        loss_functions.setdefault(
            "default", {"enabled": True, "type": "CrossEntropy", "weight": 1.0}
        )
        loss_functions.setdefault(
            "structural",
            {
                "enabled": True,
                "type": "StructuralLoss",
                "weight": 0.7,
                "params": {"edge_weight": 1.0, "smoothness_weight": 0.5},
            },
        )
        loss_functions.setdefault(
            "color_enhancement",
            {
                "enabled": True,
                "type": "ColorEnhancementLoss",
                "weight": 0.5,
                "params": {"channel_weight": 0.5, "contrast_weight": 0.3},
            },
        )
        loss_functions.setdefault(
            "morphology",
            {
                "enabled": True,
                "type": "MorphologyLoss",
                "weight": 0.3,
                "params": {"shape_weight": 0.7, "symmetry_weight": 0.3},
            },
        )
        loss_functions.setdefault(
            "autoencoder",
            {
                "enabled": False,
                "type": "AutoencoderLoss",
                "weight": 1.0,
                "params": {"reconstruction_weight": 1.0, "feature_weight": 0.1},
            },
        )
        # Verify training section
        training = config.setdefault("training", {})
        training.setdefault("batch_size", 32)
        training.setdefault("epochs", 20)
        training.setdefault("num_workers", min(4, os.cpu_count() or 1))
        training.setdefault("checkpoint_dir", os.path.join("Model", "checkpoints"))
        training.setdefault("trials", 100)
        training.setdefault("test_fraction", 0.2)
        training.setdefault("random_seed", 42)
        training.setdefault("minimum_training_accuracy", 0.95)
        training.setdefault("cardinality_threshold", 0.9)
        training.setdefault("cardinality_tolerance", 4)
        training.setdefault("n_bins_per_dim", 20)
        training.setdefault("enable_adaptive", True)
        training.setdefault("invert_DBNN", False)
        training.setdefault("reconstruction_weight", 0.5)
        training.setdefault("feedback_strength", 0.3)
        training.setdefault("inverse_learning_rate", 0.1)
        training.setdefault("Save_training_epochs", False)
        training.setdefault("training_save_path", "training_data")

        # Verify execution flags
        exec_flags = config.setdefault("execution_flags", {})
        exec_flags.setdefault("mode", "train_and_predict")
        exec_flags.setdefault("use_gpu", torch.cuda.is_available())
        exec_flags.setdefault("mixed_precision", True)
        exec_flags.setdefault("distributed_training", False)
        exec_flags.setdefault("debug_mode", False)
        exec_flags.setdefault("use_previous_model", True)
        exec_flags.setdefault("fresh_start", False)
        exec_flags.setdefault("train", True)
        exec_flags.setdefault("train_only", False)
        exec_flags.setdefault("predict", True)

        # Verify likelihood config
        likelihood = config.setdefault("likelihood_config", {})
        likelihood.setdefault("feature_group_size", 2)
        likelihood.setdefault("max_combinations", 1000)
        likelihood.setdefault("bin_sizes", [20])

        # Verify active learning config
        active = config.setdefault("active_learning", {})
        active.setdefault("tolerance", 1.0)
        active.setdefault("cardinality_threshold_percentile", 95)
        active.setdefault("strong_margin_threshold", 0.3)
        active.setdefault("marginal_margin_threshold", 0.1)
        active.setdefault("min_divergence", 0.1)

        return config

    @abstractmethod
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch"""

    @abstractmethod
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""

    def train(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """Train the model"""
        early_stopping = self.config["training"].get("early_stopping", {})
        patience = early_stopping.get("patience", 5)
        min_delta = early_stopping.get("min_delta", 0.001)
        max_epochs = self.config["training"]["epochs"]

        patience_counter = 0
        best_val_metric = float("inf")

        try:
            for epoch in range(self.current_epoch, max_epochs):
                self.current_epoch = epoch

                # Training
                train_loss, train_acc = self._train_epoch(train_loader)

                # Create summary for this epoch
                epoch_dir = os.path.join(
                    "data",
                    self.config["dataset"]["name"],
                    "training_decoder_output",
                    f"epoch_{epoch}",
                )
                if os.path.exists(epoch_dir):
                    self.create_training_summary(epoch_dir)

                # Validation
                val_loss, val_acc = None, None
                if val_loader:
                    val_loss, val_acc = self._validate(val_loader)
                    current_metric = val_loss
                else:
                    current_metric = train_loss

                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_metric)
                    else:
                        self.scheduler.step()

                # Log metrics
                self.log_training_metrics(
                    epoch,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    train_loader,
                    val_loader,
                )

                # Save checkpoint
                self._save_checkpoint(is_best=False)

                # Check for improvement
                if current_metric < best_val_metric - min_delta:
                    best_val_metric = current_metric
                    patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

                # Memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return self.history

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def log_training_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        test_loss: Optional[float] = None,
        test_acc: Optional[float] = None,
        train_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ):
        """Log training metrics"""
        elapsed_time = time.time() - self.training_start_time

        metrics = {
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_time": elapsed_time,
            "elapsed_time_formatted": str(timedelta(seconds=int(elapsed_time))),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_samples": len(train_loader.dataset) if train_loader else None,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_samples": len(test_loader.dataset) if test_loader else None,
        }

        self.training_log.append(metrics)

        log_df = pd.DataFrame(self.training_log)
        log_path = os.path.join(self.log_dir, "training_metrics.csv")
        log_df.to_csv(log_path, index=False)

        logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss {train_loss:.4f}, Acc {train_acc:.2f}%"
            + (
                f", Test Loss {test_loss:.4f}, Acc {test_acc:.2f}%"
                if test_loss is not None
                else ""
            )
        )

    @abstractmethod
    def extract_features(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from data loader"""

    def save_features(
        self, features: torch.Tensor, labels: torch.Tensor, output_path: str
    ):
        """Save extracted features to CSV with user oversight"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            headers = [f"feature_{i}" for i in range(features.shape[1])] + ["target"]
            config_path = os.path.join(os.path.dirname(output_path), "config.json")

            # Create complete configuration if none exists
            if not os.path.exists(config_path):
                complete_config = {
                    "file_path": output_path,
                    "column_names": headers,
                    "separator": ",",
                    "has_header": True,
                    "target_column": "target",
                    "modelType": "Histogram",
                    "model": {
                        "feature_dims": features.shape[1],
                        "learning_rate": 0.001,
                        "encoder_type": "cnn",
                        "loss_functions": {
                            "default": {
                                "enabled": True,
                                "type": "CrossEntropy",
                                "weight": 1.0,
                            },
                            "structural": {
                                "enabled": True,
                                "type": "StructuralLoss",
                                "weight": 0.7,
                                "params": {
                                    "edge_weight": 1.0,
                                    "smoothness_weight": 0.5,
                                },
                            },
                            "color_enhancement": {
                                "enabled": True,
                                "type": "ColorEnhancementLoss",
                                "weight": 0.5,
                                "params": {
                                    "channel_weight": 0.5,
                                    "contrast_weight": 0.3,
                                },
                            },
                            "morphology": {
                                "enabled": True,
                                "type": "MorphologyLoss",
                                "weight": 0.3,
                                "params": {"shape_weight": 0.7, "symmetry_weight": 0.3},
                            },
                            "autoencoder": {
                                "enabled": False,
                                "type": "AutoencoderLoss",
                                "weight": 1.0,
                                "params": {
                                    "reconstruction_weight": 1.0,
                                    "feature_weight": 0.1,
                                },
                            },
                        },
                    },
                    "likelihood_config": {
                        "feature_group_size": 2,
                        "max_combinations": min(
                            1000, features.shape[1] * (features.shape[1] - 1) // 2
                        ),
                        "bin_sizes": [20],
                    },
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
                        "learning_rate": 0.1,
                        "test_fraction": 0.2,
                        "random_seed": 42,
                        "minimum_training_accuracy": 0.95,
                        "cardinality_threshold": 0.9,
                        "cardinality_tolerance": 4,
                        "n_bins_per_dim": 20,
                        "enable_adaptive": True,
                        "invert_DBNN": False,
                        "reconstruction_weight": 0.5,
                        "feedback_strength": 0.3,
                        "inverse_learning_rate": 0.1,
                        "Save_training_epochs": False,
                        "training_save_path": "training_data",
                    },
                    "execution_flags": {
                        "train": True,
                        "train_only": False,
                        "predict": True,
                        "fresh_start": False,
                        "use_previous_model": True,
                    },
                }
                with open(config_path, "w") as f:
                    json.dump(complete_config, f, indent=4)

            # Prompt user for editing
            print(
                "\nConfiguration file created. Would you like to edit it before training?"
            )
            response = input("Edit configuration? (y/n): ").lower()
            if response == "y":
                config_manager = ConfigManager(os.path.dirname(output_path))
                config_manager._open_editor(config_path)

            config_manager = ConfigManager(os.path.dirname(output_path))
            if config_manager.manage_csv(output_path, headers):
                chunk_size = 1000
                total_samples = features.shape[0]

                for start_idx in range(0, total_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_samples)
                    feature_dict = {
                        f"feature_{i}": features[start_idx:end_idx, i].numpy()
                        for i in range(features.shape[1])
                    }
                    feature_dict["target"] = labels[start_idx:end_idx].numpy()

                    df = pd.DataFrame(feature_dict)
                    mode = "w" if start_idx == 0 else "a"
                    header = start_idx == 0

                    df.to_csv(output_path, mode=mode, index=False, header=header)

                    del feature_dict, df
                    gc.collect()

                logger.info(f"Features saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise


class FeatureExtractorCNN(nn.Module):
    """CNN-based feature extractor model"""

    def __init__(self, in_channels: int = 3, feature_dims: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(128, feature_dims)
        self.batch_norm = nn.BatchNorm1d(feature_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if x.size(0) > 1:  # Only apply batch norm if batch size > 1
            x = self.batch_norm(x)
        return x


class DynamicAutoencoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], feature_dims: int):
        super().__init__()
        self.input_shape = input_shape  # e.g., (3, 32, 32) for CIFAR
        self.in_channels = input_shape[0]  # Store input channels explicitly
        self.feature_dims = feature_dims

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

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        in_channels = self.in_channels  # Start with input channels
        for size in self.layer_sizes:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, size, 3, stride=2, padding=1),
                    nn.BatchNorm2d(size),
                    nn.LeakyReLU(0.2),
                )
            )
            in_channels = size

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

        # Decoder layers with careful channel tracking
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
                )
            )
            in_channels = out_channels

    def _calculate_layer_sizes(self) -> List[int]:
        """Calculate progressive channel sizes for encoder/decoder"""
        # Start with input channels
        base_channels = 32  # Reduced from 64 to handle smaller images
        sizes = []
        current_size = base_channels

        # Calculate maximum number of downsampling layers based on smallest spatial dimension
        min_dim = min(self.input_shape[1:])
        max_layers = int(np.log2(min_dim)) - 2  # Ensure we don't reduce too much

        for _ in range(max_layers):
            sizes.append(current_size)
            if current_size < 256:  # Reduced from 512 to handle smaller images
                current_size *= 2

        logger.info(f"Layer sizes: {sizes}")
        return sizes

    def _calculate_flattened_size(self) -> int:
        """Calculate size of flattened feature maps before linear layer"""
        reduction_factor = 2 ** (len(self.layer_sizes) - 1)
        reduced_dims = [dim // reduction_factor for dim in self.spatial_dims]
        return self.layer_sizes[-1] * np.prod(reduced_dims)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images to feature space"""
        # Verify input channels
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Input has {x.size(1)} channels, expected {self.in_channels}"
            )

        for layer in self.encoder_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return self.embedder(x)

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

        # Verify output shape
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Output has {x.size(1)} channels, expected {self.in_channels}"
            )

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder"""
        embedding = self.encode(x)
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


class CNNFeatureExtractor(BaseFeatureExtractor):
    """CNN-based feature extractor implementation"""

    def _create_model(self) -> nn.Module:
        """Create CNN model"""
        return FeatureExtractorCNN(
            in_channels=self.config["dataset"]["in_channels"],
            feature_dims=self.feature_dims,
        ).to(self.device)

    def _load_from_checkpoint(self):
        """Load model from checkpoint"""
        checkpoint_dir = self.config["training"]["checkpoint_dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Try to find latest checkpoint
        checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                # Load model state
                self.feature_extractor.load_state_dict(checkpoint["state_dict"])

                # Initialize and load optimizer
                self.optimizer = self._initialize_optimizer()
                if "optimizer_state_dict" in checkpoint:
                    try:
                        self.optimizer.load_state_dict(
                            checkpoint["optimizer_state_dict"]
                        )
                        logger.info("Optimizer state loaded")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {str(e)}")

                # Load training state
                self.current_epoch = checkpoint.get("epoch", 0)
                self.best_accuracy = checkpoint.get("best_accuracy", 0.0)
                self.best_loss = checkpoint.get("best_loss", float("inf"))

                # Load history
                if "history" in checkpoint:
                    self.history = defaultdict(list, checkpoint["history"])

                logger.info("Checkpoint loaded successfully")

            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                self.optimizer = self._initialize_optimizer()
        else:
            logger.info("No checkpoint found, starting from scratch")
            self.optimizer = self._initialize_optimizer()

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file"""
        dataset_name = self.config["dataset"]["name"]
        checkpoint_dir = os.path.join("data", dataset_name, "checkpoints")

        if not os.path.exists(checkpoint_dir):
            return None

        # Check for best model first
        best_path = os.path.join(checkpoint_dir, f"{dataset_name}_best.pth")
        if os.path.exists(best_path):
            return best_path

        # Check for latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            return checkpoint_path

        return None

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.config["training"]["checkpoint_dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "state_dict": self.feature_extractor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_accuracy": self.best_accuracy,
            "best_loss": self.best_loss,
            "history": dict(self.history),
            "config": self.config,
        }

        # Save latest checkpoint
        dataset_name = self.config["dataset"]["name"]
        filename = f"{dataset_name}_{'best' if is_best else 'checkpoint'}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        torch.save(checkpoint, checkpoint_path)
        logger.info(
            f"Saved {'best' if is_best else 'latest'} checkpoint to {checkpoint_path}"
        )

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch"""
        self.feature_extractor.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            unit="batch",
            leave=False,
        )

        try:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.feature_extractor(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                batch_loss = running_loss / (batch_idx + 1)
                batch_acc = 100.0 * correct / total
                pbar.set_postfix(
                    {"loss": f"{batch_loss:.4f}", "acc": f"{batch_acc:.2f}%"}
                )

                # Cleanup
                del inputs, outputs, loss
                if batch_idx % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            raise

        pbar.close()
        return running_loss / len(train_loader), 100.0 * correct / total

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.feature_extractor.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.feature_extractor(inputs)
                loss = F.cross_entropy(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Cleanup
                del inputs, outputs, loss

        return running_loss / len(val_loader), 100.0 * correct / total

    def extract_features(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from data"""
        self.feature_extractor.eval()
        features = []
        labels = []

        try:
            with torch.no_grad():
                for inputs, targets in tqdm(loader, desc="Extracting features"):
                    inputs = inputs.to(self.device)
                    outputs = self.feature_extractor(inputs)
                    features.append(outputs.cpu())
                    labels.append(targets)

                    # Cleanup
                    del inputs, outputs
                    if len(features) % 50 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            return torch.cat(features), torch.cat(labels)

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def get_feature_shape(self) -> Tuple[int, ...]:
        """Get shape of extracted features"""
        return (self.feature_dims,)

    def plot_feature_distribution(
        self, loader: DataLoader, save_path: Optional[str] = None
    ):
        """Plot distribution of extracted features"""
        features, _ = self.extract_features(loader)
        features = features.numpy()

        plt.figure(figsize=(12, 6))
        plt.hist(features.flatten(), bins=50, density=True)
        plt.title("Feature Distribution")
        plt.xlabel("Feature Value")
        plt.ylabel("Density")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Feature distribution plot saved to {save_path}")
        plt.close()


class AutoEncoderFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config: Dict, device: str = None):
        super().__init__(config, device)
        self.output_image_dir = os.path.join(
            "data",
            config["dataset"]["name"],
            "output",
            "images",
            Path(config["dataset"]["name"]).stem,
        )
        os.makedirs(self.output_image_dir, exist_ok=True)

    def _create_model(self) -> nn.Module:
        """Create autoencoder model"""
        input_shape = (
            self.config["dataset"]["in_channels"],
            *self.config["dataset"]["input_size"],
        )
        return DynamicAutoencoder(
            input_shape=input_shape, feature_dims=self.feature_dims
        ).to(self.device)

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch with reconstruction visualization"""
        self.feature_extractor.train()
        running_loss = 0.0
        reconstruction_accuracy = 0.0

        # Create output directory for training reconstructions
        output_dir = os.path.join(
            "data",
            self.config["dataset"]["name"],
            "training_decoder_output",
            f"epoch_{self.current_epoch}",
        )
        os.makedirs(output_dir, exist_ok=True)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            unit="batch",
            leave=False,
        )

        for batch_idx, (inputs, _) in enumerate(pbar):
            try:
                inputs = inputs.to(self.device)

                # Log input shape and channels
                logger.debug(
                    f"Input tensor shape: {inputs.shape}, channels: {inputs.size(1)}"
                )

                self.optimizer.zero_grad()
                embedding, reconstruction = self.feature_extractor(inputs)

                # Verify reconstruction shape matches input
                if reconstruction.shape != inputs.shape:
                    raise ValueError(
                        f"Reconstruction shape {reconstruction.shape} "
                        f"doesn't match input shape {inputs.shape}"
                    )

                # Calculate loss
                loss = self._calculate_loss(inputs, reconstruction, embedding)
                loss.backward()
                self.optimizer.step()

                # Update metrics
                running_loss += loss.item()
                reconstruction_accuracy += (
                    1.0 - F.mse_loss(reconstruction, inputs).item()
                )

                # Save reconstructions periodically
                if batch_idx % 50 == 0:
                    self._save_training_batch(
                        inputs, reconstruction, batch_idx, output_dir
                    )

                # Update progress bar
                batch_loss = running_loss / (batch_idx + 1)
                batch_acc = (reconstruction_accuracy / (batch_idx + 1)) * 100
                pbar.set_postfix(
                    {"loss": f"{batch_loss:.4f}", "recon_acc": f"{batch_acc:.2f}%"}
                )

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                raise

        pbar.close()
        return (
            running_loss / len(train_loader),
            (reconstruction_accuracy / len(train_loader)) * 100,
        )

    def _calculate_loss(
        self,
        inputs: torch.Tensor,
        reconstruction: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate combined loss for autoencoder"""
        ae_config = self.config["model"]["autoencoder_config"]
        return AutoencoderLoss(
            reconstruction_weight=ae_config["reconstruction_weight"],
            feature_weight=ae_config["feature_weight"],
        )(inputs, reconstruction, embedding)

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.feature_extractor.eval()
        running_loss = 0.0
        reconstruction_accuracy = 0.0

        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(self.device)
                embedding, reconstruction = self.feature_extractor(inputs)

                loss = self._calculate_loss(inputs, reconstruction, embedding)
                running_loss += loss.item()
                reconstruction_accuracy += (
                    1.0 - F.mse_loss(reconstruction, inputs).item()
                )

                del inputs, embedding, reconstruction, loss

        return (
            running_loss / len(val_loader),
            (reconstruction_accuracy / len(val_loader)) * 100,
        )

    def extract_features(self, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features using encoder"""
        self.feature_extractor.eval()
        features = []
        labels = []

        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Extracting features"):
                inputs = inputs.to(self.device)
                embedding = self.feature_extractor.encode(inputs)
                features.append(embedding.cpu())
                labels.append(targets)

                del inputs, embedding

        return torch.cat(features), torch.cat(labels)

    def _save_training_batch(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        batch_idx: int,
        output_dir: str,
    ):
        """Save training batch images with proper error handling"""
        with torch.no_grad():
            for i in range(min(5, inputs.size(0))):
                try:
                    orig_input = inputs[i]
                    recon = reconstructions[i]

                    # Verify channel consistency
                    expected_channels = self.config["dataset"]["in_channels"]
                    if (
                        orig_input.size(0) != expected_channels
                        or recon.size(0) != expected_channels
                    ):
                        raise ValueError(
                            f"Channel mismatch: input={orig_input.size(0)}, "
                            f"recon={recon.size(0)}, expected={expected_channels}"
                        )

                    # Save images
                    orig_path = os.path.join(
                        output_dir, f"batch_{batch_idx}_sample_{i}_original.png"
                    )
                    recon_path = os.path.join(
                        output_dir, f"batch_{batch_idx}_sample_{i}_reconstruction.png"
                    )

                    self.save_training_image(orig_input, orig_path)
                    self.save_training_image(recon, recon_path)

                except Exception as e:
                    logger.error(
                        f"Error saving training sample {i} from batch {batch_idx}: {str(e)}"
                    )

    def save_training_image(self, tensor: torch.Tensor, path: str):
        """Save training image with robust channel handling"""
        try:
            tensor = tensor.detach().cpu()
            expected_channels = self.config["dataset"]["in_channels"]

            # Ensure we're working with the right shape [C, H, W]
            if len(tensor.shape) == 4:
                tensor = tensor.squeeze(0)

            if tensor.shape[0] != expected_channels:
                raise ValueError(
                    f"Expected {expected_channels} channels, got {tensor.shape[0]}"
                )

            # Move to [H, W, C] for image saving
            tensor = tensor.permute(1, 2, 0)

            # Get normalization parameters
            mean = torch.tensor(self.config["dataset"]["mean"], dtype=tensor.dtype)
            std = torch.tensor(self.config["dataset"]["std"], dtype=tensor.dtype)

            # Ensure mean/std match channel count
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)

            # Denormalize
            tensor = tensor * std + mean
            tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)

            # Handle single-channel case
            if tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)

            # Save image
            img = Image.fromarray(tensor.numpy())
            img.save(path)

        except Exception as e:
            logger.error(f"Error saving training image: {str(e)}")
            logger.error(
                f"Tensor shape at error: {tensor.shape if 'tensor' in locals() else 'unknown'}"
            )
            raise

    def predict_from_csv(self, csv_path: str):
        """Generate reconstructions from feature vectors in CSV"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        features = torch.tensor(df[feature_cols].values, dtype=torch.float32)

        # Verify feature dimensions
        expected_dims = self.config["model"]["feature_dims"]
        if features.size(1) != expected_dims:
            raise ValueError(
                f"Feature dimension mismatch: got {features.size(1)}, expected {expected_dims}"
            )

        self.feature_extractor.eval()
        output_dir = self.config["output"]["image_dir"]
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for idx, feature_vec in enumerate(
                tqdm(features, desc="Generating reconstructions")
            ):
                try:
                    # Ensure proper shape and device placement
                    feature_vec = feature_vec.to(self.device).unsqueeze(0)

                    # Generate reconstruction
                    reconstruction = self.feature_extractor.decode(feature_vec)

                    # Verify channel count
                    if reconstruction.size(1) != self.config["dataset"]["in_channels"]:
                        raise ValueError(
                            f"Reconstruction channel mismatch: got {reconstruction.size(1)}, "
                            f"expected {self.config['dataset']['in_channels']}"
                        )

                    # Save reconstructed image
                    img_path = os.path.join(output_dir, f"reconstruction_{idx}.png")
                    self.save_reconstructed_image(reconstruction[0], img_path)

                except Exception as e:
                    logger.error(f"Error processing feature vector {idx}: {str(e)}")

    def save_reconstructed_image(self, tensor: torch.Tensor, path: str):
        """Save reconstructed tensor as image with proper normalization"""
        try:
            tensor = tensor.detach().cpu()

            # Verify channel count
            if tensor.size(0) != self.config["dataset"]["in_channels"]:
                raise ValueError(
                    f"Expected {self.config['dataset']['in_channels']} channels, got {tensor.size(0)}"
                )

            # Move to [H, W, C] for image saving
            tensor = tensor.permute(1, 2, 0)

            # Get normalization parameters
            mean = torch.tensor(self.config["dataset"]["mean"], dtype=tensor.dtype)
            std = torch.tensor(self.config["dataset"]["std"], dtype=tensor.dtype)

            # Reshape for broadcasting
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)

            # Denormalize
            tensor = tensor * std + mean
            tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)

            # Handle single-channel case
            if tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)

            # Save image
            img = Image.fromarray(tensor.numpy())
            img.save(path)

        except Exception as e:
            logger.error(f"Error saving reconstructed image: {str(e)}")
            logger.error(
                f"Tensor shape at error: {tensor.shape if 'tensor' in locals() else 'unknown'}"
            )
            raise

    def plot_reconstruction_samples(
        self, loader: DataLoader, num_samples: int = 8, save_path: Optional[str] = None
    ):
        """Visualize original and reconstructed images"""
        self.feature_extractor.eval()

        # Get samples
        images, _ = next(iter(loader))
        images = images[:num_samples].to(self.device)

        with torch.no_grad():
            _, reconstructions = self.feature_extractor(images)

        # Plot results
        fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))

        for i in range(num_samples):
            # Original
            axes[0, i].imshow(self._tensor_to_image(images[i]))
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
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Reconstruction samples saved to {save_path}")
        plt.close()

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to image array with proper normalization"""
        tensor = tensor.cpu()

        # Move to [H, W, C]
        if len(tensor.shape) == 3:
            tensor = tensor.permute(1, 2, 0)

        # Denormalize
        mean = torch.tensor(self.config["dataset"]["mean"]).view(1, 1, -1)
        std = torch.tensor(self.config["dataset"]["std"]).view(1, 1, -1)
        tensor = tensor * std + mean

        # Convert to uint8
        return (tensor.clamp(0, 1) * 255).numpy().astype(np.uint8)

    def plot_latent_space(
        self,
        dataloader: DataLoader,
        num_samples: int = 1000,
        save_path: Optional[str] = None,
    ):
        """Plot 2D visualization of latent space"""
        if self.feature_dims < 2:
            logger.warning("Latent space dimension too small for visualization")
            return

        self.feature_extractor.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                if len(embeddings) * inputs.size(0) >= num_samples:
                    break

                inputs = inputs.to(self.device)
                embedding = self.feature_extractor.encode(inputs)
                embeddings.append(embedding.cpu())
                labels.extend(targets.tolist())

        embeddings = torch.cat(embeddings, dim=0)[:num_samples]
        labels = labels[:num_samples]

        # Use PCA for visualization if dimensions > 2
        if self.feature_dims > 2:
            from sklearn.decomposition import PCA

            embeddings = PCA(n_components=2).fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings[:, 0], embeddings[:, 1], c=labels, cmap="tab10"
        )
        plt.colorbar(scatter)
        plt.title("Latent Space Visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Latent space visualization saved to {save_path}")
        plt.close()

    def _load_from_checkpoint(self):
        """Load model and training state from checkpoint"""
        checkpoint_dir = self.config["training"]["checkpoint_dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Try to find latest checkpoint
        checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                # Load model state
                self.feature_extractor.load_state_dict(checkpoint["state_dict"])

                # Initialize and load optimizer
                self.optimizer = self._initialize_optimizer()
                if "optimizer_state_dict" in checkpoint:
                    try:
                        self.optimizer.load_state_dict(
                            checkpoint["optimizer_state_dict"]
                        )
                        logger.info("Optimizer state loaded")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {str(e)}")

                # Load training state
                self.current_epoch = checkpoint.get("epoch", 0)
                self.best_accuracy = checkpoint.get("best_accuracy", 0.0)
                self.best_loss = checkpoint.get("best_loss", float("inf"))

                # Load history
                if "history" in checkpoint:
                    self.history = defaultdict(list, checkpoint["history"])

                logger.info("Checkpoint loaded successfully")

            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                self.optimizer = self._initialize_optimizer()
        else:
            logger.info("No checkpoint found, starting from scratch")
            self.optimizer = self._initialize_optimizer()

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file"""
        checkpoint_dir = self.config["training"]["checkpoint_dir"]
        if not os.path.exists(checkpoint_dir):
            return None

        dataset_name = self.config["dataset"]["name"]

        # Check for best model first
        best_path = os.path.join(checkpoint_dir, f"{dataset_name}_best.pth")
        if os.path.exists(best_path):
            return best_path

        # Check for latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            return checkpoint_path

        return None

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.config["training"]["checkpoint_dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "state_dict": self.feature_extractor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_accuracy": self.best_accuracy,
            "best_loss": self.best_loss,
            "history": dict(self.history),
            "config": self.config,
        }

        # Save latest checkpoint
        dataset_name = self.config["dataset"]["name"]
        filename = f"{dataset_name}_{'best' if is_best else 'checkpoint'}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        torch.save(checkpoint, checkpoint_path)
        logger.info(
            f"Saved {'best' if is_best else 'latest'} checkpoint to {checkpoint_path}"
        )

    def create_training_summary(self, epoch_dir: str):
        """Create HTML summary of training reconstructions"""
        summary_path = os.path.join(epoch_dir, "summary.html")

        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<style>",
            ".image-pair { display: inline-block; margin: 10px; text-align: center; }",
            ".image-pair img { width: 128px; height: 128px; margin: 5px; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Training Reconstructions - Epoch {self.current_epoch + 1}</h1>",
        ]

        # Find all image pairs
        original_images = sorted(glob.glob(os.path.join(epoch_dir, "*_original.png")))

        for orig_path in original_images:
            recon_path = orig_path.replace("_original.png", "_reconstruction.png")
            if os.path.exists(recon_path):
                base_name = os.path.basename(orig_path)
                pair_id = base_name.split("_original")[0]

                html_content.extend(
                    [
                        '<div class="image-pair">',
                        f"<p>{pair_id}</p>",
                        f'<img src="{os.path.basename(orig_path)}" alt="Original">',
                        f'<img src="{os.path.basename(recon_path)}" alt="Reconstruction">',
                        "</div>",
                    ]
                )

        html_content.extend(["</body>", "</html>"])

        with open(summary_path, "w") as f:
            f.write("\n".join(html_content))

        logger.info(f"Created training summary: {summary_path}")

    def _verify_config(self):
        """Verify configuration has all required fields"""
        required_fields = {
            "dataset": ["in_channels", "input_size", "mean", "std"],
            "model": ["feature_dims", "learning_rate", "autoencoder_config"],
            "training": ["batch_size", "epochs", "checkpoint_dir"],
        }

        for section, fields in required_fields.items():
            if section not in self.config:
                raise ValueError(f"Missing config section: {section}")
            for field in fields:
                if field not in self.config[section]:
                    raise ValueError(f"Missing config field: {section}.{field}")

    def log_training_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        test_loss: Optional[float] = None,
        test_acc: Optional[float] = None,
        train_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ):
        """Log training metrics"""
        elapsed_time = time.time() - self.training_start_time

        metrics = {
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_time": elapsed_time,
            "elapsed_time_formatted": str(timedelta(seconds=int(elapsed_time))),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_samples": len(train_loader.dataset) if train_loader else None,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_samples": len(test_loader.dataset) if test_loader else None,
        }

        self.training_log.append(metrics)

        log_df = pd.DataFrame(self.training_log)
        log_path = os.path.join(self.log_dir, "training_metrics.csv")
        log_df.to_csv(log_path, index=False)

        logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss {train_loss:.4f}, Acc {train_acc:.2f}%"
            + (
                f", Test Loss {test_loss:.4f}, Acc {test_acc:.2f}%"
                if test_loss is not None
                else ""
            )
        )

    def _initialize_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Initialize learning rate scheduler if specified in config"""
        scheduler_config = self.config["model"].get("scheduler", {})
        if not scheduler_config:
            return None

        scheduler_type = scheduler_config.get("type")
        if not scheduler_type:
            return None

        try:
            if scheduler_type == "StepLR":
                return optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get("step_size", 7),
                    gamma=scheduler_config.get("gamma", 0.1),
                )
            elif scheduler_type == "ReduceLROnPlateau":
                return optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=scheduler_config.get("factor", 0.1),
                    patience=scheduler_config.get("patience", 10),
                    verbose=True,
                )
            elif scheduler_type == "CosineAnnealingLR":
                return optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_config.get(
                        "T_max", self.config["training"]["epochs"]
                    ),
                    eta_min=scheduler_config.get("eta_min", 0),
                )
        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {str(e)}")
            return None

        return None

    def save_features(
        self, features: torch.Tensor, labels: torch.Tensor, output_path: str
    ):
        """Save extracted features to CSV with user oversight"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            headers = [f"feature_{i}" for i in range(features.shape[1])] + ["target"]
            config_path = os.path.join(os.path.dirname(output_path), "config.json")

            # Create complete configuration if none exists
            if not os.path.exists(config_path):
                complete_config = {
                    "file_path": output_path,
                    "column_names": headers,
                    "separator": ",",
                    "has_header": True,
                    "target_column": "target",
                    "modelType": "Histogram",
                    # ... (rest of config template)
                }
                with open(config_path, "w") as f:
                    json.dump(complete_config, f, indent=4)

            # Process in chunks to manage memory
            chunk_size = 1000
            total_samples = features.shape[0]

            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)
                feature_dict = {
                    f"feature_{i}": features[start_idx:end_idx, i].numpy()
                    for i in range(features.shape[1])
                }
                feature_dict["target"] = labels[start_idx:end_idx].numpy()

                df = pd.DataFrame(feature_dict)
                mode = "w" if start_idx == 0 else "a"
                header = start_idx == 0

                df.to_csv(output_path, mode=mode, index=False, header=header)

                # Cleanup
                del feature_dict, df
                gc.collect()

            logger.info(f"Features saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer based on configuration"""
        optimizer_config = self.config["model"].get("optimizer", {})

        # Set base parameters
        optimizer_params = {
            "lr": self.learning_rate,
            "weight_decay": optimizer_config.get("weight_decay", 1e-4),
        }

        # Configure optimizer-specific parameters
        optimizer_type = optimizer_config.get("type", "Adam")
        if optimizer_type == "SGD":
            optimizer_params["momentum"] = optimizer_config.get("momentum", 0.9)
            optimizer_params["nesterov"] = optimizer_config.get("nesterov", False)
        elif optimizer_type == "Adam":
            optimizer_params["betas"] = (
                optimizer_config.get("beta1", 0.9),
                optimizer_config.get("beta2", 0.999),
            )
            optimizer_params["eps"] = optimizer_config.get("epsilon", 1e-8)

        # Get optimizer class
        try:
            optimizer_class = getattr(optim, optimizer_type)
        except AttributeError:
            logger.warning(f"Optimizer {optimizer_type} not found, using Adam")
            optimizer_class = optim.Adam
            optimizer_type = "Adam"

        # Create and return optimizer
        optimizer = optimizer_class(
            self.feature_extractor.parameters(), **optimizer_params
        )

        logger.info(
            f"Initialized {optimizer_type} optimizer with parameters: {optimizer_params}"
        )
        return optimizer

    def plot_feature_distribution(
        self, loader: DataLoader, save_path: Optional[str] = None
    ):
        """Plot distribution of extracted features"""
        features, _ = self.extract_features(loader)
        features = features.numpy()

        plt.figure(figsize=(12, 6))
        plt.hist(features.flatten(), bins=50, density=True)
        plt.title("Feature Distribution")
        plt.xlabel("Feature Value")
        plt.ylabel("Density")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Feature distribution plot saved to {save_path}")
        plt.close()

    def generate_reconstructions(self):
        """Generate reconstructed images based on config mode"""
        invert_dbnn = self.config.get("execution_flags", {}).get("invert_DBNN", False)
        dataset_name = self.config["dataset"]["name"]
        base_dir = os.path.join("data", dataset_name)

        # Determine input file
        if invert_dbnn:
            input_file = os.path.join(base_dir, "reconstructed_input.csv")
        else:
            input_file = os.path.join(base_dir, f"{dataset_name}.csv")

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Read embeddings
        df = pd.read_csv(input_file)
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        embeddings = torch.tensor(df[feature_cols].values, dtype=torch.float32)

        # Generate reconstructions
        self.feature_extractor.eval()
        with torch.no_grad():
            batch_size = 32
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i : i + batch_size].to(self.device)
                reconstructions = self.feature_extractor.decode(batch)

                # Save reconstructed images
                for j, reconstruction in enumerate(reconstructions):
                    idx = i + j
                    filename = f"reconstruction_{idx}.png"
                    self.save_reconstructed_image(filename, reconstruction)

    def train(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """Train the feature extractor"""
        early_stopping = self.config["training"].get("early_stopping", {})
        patience = early_stopping.get("patience", 5)
        min_delta = early_stopping.get("min_delta", 0.001)
        max_epochs = self.config["training"]["epochs"]

        patience_counter = 0
        best_val_metric = float("inf")

        if not hasattr(self, "training_start_time"):
            self.training_start_time = time.time()

        try:
            for epoch in range(self.current_epoch, max_epochs):
                self.current_epoch = epoch

                # Training
                train_loss, train_acc = self._train_epoch(train_loader)

                # Create summary for this epoch
                epoch_dir = os.path.join(
                    "data",
                    self.config["dataset"]["name"],
                    "training_decoder_output",
                    f"epoch_{epoch}",
                )
                if os.path.exists(epoch_dir):
                    self.create_training_summary(epoch_dir)

                # Validation
                val_loss, val_acc = None, None
                if val_loader:
                    val_loss, val_acc = self._validate(val_loader)
                    current_metric = val_loss
                else:
                    current_metric = train_loss

                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_metric)
                    else:
                        self.scheduler.step()

                # Log metrics
                self.log_training_metrics(
                    epoch,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    train_loader,
                    val_loader,
                )

                # Save checkpoint
                self._save_checkpoint(is_best=False)

                # Check for improvement
                if current_metric < best_val_metric - min_delta:
                    best_val_metric = current_metric
                    patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

                # Memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return self.history

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def visualize_reconstructions(
        self,
        dataloader: DataLoader,
        num_samples: int = 8,
        save_path: Optional[str] = None,
    ):
        """Plot grid of original and reconstructed images"""
        self.feature_extractor.eval()

        # Get samples
        original_images = []
        reconstructed_images = []

        with torch.no_grad():
            for images, _ in dataloader:
                if len(original_images) >= num_samples:
                    break

                batch_images = images.to(self.device)
                _, reconstructions = self.feature_extractor(batch_images)

                original_images.extend(images.cpu())
                reconstructed_images.extend(reconstructions.cpu())

        # Select required number of samples
        original_images = original_images[:num_samples]
        reconstructed_images = reconstructed_images[:num_samples]

        # Create plot
        fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))

        for i in range(num_samples):
            # Plot original
            axes[0, i].imshow(self._tensor_to_image(original_images[i]))
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_title("Original")

            # Plot reconstruction
            axes[1, i].imshow(self._tensor_to_image(reconstructed_images[i]))
            axes[1, i].axis("off")
            if i == 0:
                axes[1, i].set_title("Reconstructed")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Reconstruction visualization saved to {save_path}")
        plt.close()

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training metrics history"""
        if not self.history:
            logger.warning("No training history available to plot")
            return

        plt.figure(figsize=(15, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        if "val_loss" in self.history:
            plt.plot(self.history["val_loss"], label="Val Loss")
        plt.title("Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_acc"], label="Train Acc")
        if "val_acc" in self.history:
            plt.plot(self.history["val_acc"], label="Val Acc")
        plt.title("Accuracy History")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        plt.close()

    def get_reconstruction(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Get reconstruction for input tensor"""
        self.feature_extractor.eval()
        with torch.no_grad():
            embedding, reconstruction = self.feature_extractor(input_tensor)
            return reconstruction

    def get_reconstruction_error(self, input_tensor: torch.Tensor) -> float:
        """Calculate reconstruction error for input tensor"""
        reconstruction = self.get_reconstruction(input_tensor)
        return F.mse_loss(reconstruction, input_tensor).item()

    def get_feature_shape(self) -> Tuple[int, ...]:
        """Get shape of extracted features"""
        return (self.feature_dims,)

    def save_model(self, path: str):
        """Save model to path"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.feature_extractor.state_dict(),
                "config": self.config,
                "feature_dims": self.feature_dims,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from path"""
        if not os.path.exists(path):
            raise ValueError(f"Model file not found: {path}")

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.feature_extractor.load_state_dict(checkpoint["state_dict"])
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


class FeatureExtractorFactory:
    """Factory class for creating feature extractors"""

    @staticmethod
    def create(config: Dict, device: Optional[str] = None) -> BaseFeatureExtractor:
        """
        Create appropriate feature extractor based on configuration.

        Args:
            config: Configuration dictionary
            device: Optional device specification

        Returns:
            Instance of appropriate feature extractor
        """
        encoder_type = config["model"].get("encoder_type", "cnn").lower()

        if encoder_type == "cnn":
            return CNNFeatureExtractor(config, device)
        elif encoder_type == "autoenc":
            return AutoEncoderFeatureExtractor(config, device)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")


def get_feature_extractor(
    config: Dict, device: Optional[str] = None
) -> BaseFeatureExtractor:
    """Convenience function to create feature extractor"""
    return FeatureExtractorFactory.create(config, device)


class CustomImageDataset(Dataset):
    """Custom dataset for loading images from directory structure"""

    def __init__(self, data_dir: str, transform=None, csv_file: Optional[str] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.label_encoder = {}
        self.reverse_encoder = {}

        if csv_file and os.path.exists(csv_file):
            self.data = pd.read_csv(csv_file)
        else:
            self.image_files = []
            self.labels = []
            unique_labels = sorted(os.listdir(data_dir))

            for idx, label in enumerate(unique_labels):
                self.label_encoder[label] = idx
                self.reverse_encoder[idx] = label

            # Save label encodings
            encoding_file = os.path.join(data_dir, "label_encodings.json")
            with open(encoding_file, "w") as f:
                json.dump(
                    {
                        "label_to_id": self.label_encoder,
                        "id_to_label": self.reverse_encoder,
                    },
                    f,
                    indent=4,
                )

            for class_name in unique_labels:
                class_dir = os.path.join(data_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                        ):
                            self.image_files.append(os.path.join(class_dir, img_name))
                            self.labels.append(self.label_encoder[class_name])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

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
    ):
        self.datafile = datafile
        self.datatype = datatype.lower()
        self.output_dir = output_dir

        if self.datatype == "torchvision":
            self.dataset_name = self.datafile.lower()
        else:
            self.dataset_name = Path(self.datafile).stem.lower()

        self.dataset_dir = os.path.join(output_dir, self.dataset_name)
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

    def _process_custom(self, data_path: str) -> Tuple[str, Optional[str]]:
        """Process custom dataset structure"""
        train_dir = os.path.join(self.dataset_dir, "train")
        test_dir = os.path.join(self.dataset_dir, "test")

        # Check if dataset already has train/test structure
        if os.path.isdir(os.path.join(data_path, "train")) and os.path.isdir(
            os.path.join(data_path, "test")
        ):
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
                "train_dir": train_dir,
                "test_dir": os.path.join(os.path.dirname(train_dir), "test"),
            },
            "model": {
                "encoder_type": "autoenc",
                "feature_dims": feature_dims,
                "learning_rate": 0.001,
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
                },
                "optimizer": {
                    "type": "Adam",
                    "weight_decay": 1e-4,
                    "momentum": 0.9,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-8,
                },
                "scheduler": {
                    "type": "ReduceLROnPlateau",
                    "factor": 0.1,
                    "patience": 10,
                    "min_lr": 1e-6,
                    "verbose": True,
                },
                "autoencoder_config": {
                    "reconstruction_weight": 1.0,
                    "feature_weight": 0.1,
                    "convergence_threshold": 0.001,
                    "min_epochs": 10,
                    "patience": 5,
                },
            },
            "training": {
                "batch_size": 32,
                "epochs": 20,
                "num_workers": min(4, os.cpu_count() or 1),
                "checkpoint_dir": os.path.join(self.dataset_dir, "checkpoints"),
                "validation_split": 0.2,
                "invert_DBNN": True,
                "reconstruction_weight": 0.5,
                "feedback_strength": 0.3,
                "inverse_learning_rate": 0.1,
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
            "max_combinations": 1000,
            "bin_sizes": [21],
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
                "learning_rate": 0.1,
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
                "inverse_learning_rate": 0.1,
                "Save_training_epochs": True,
                "training_save_path": "training_data",
                "enable_vectorized": False,
                "vectorization_warning_acknowledged": False,
                "compute_device": "auto",
                "use_interactive_kbd": False,
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
                "test_fraction": 0.2,
                "random_seed": 42,
                "minimum_training_accuracy": 0.95,
                "cardinality_threshold": 0.9,
                "cardinality_tolerance": 4,
                "n_bins_per_dim": 20,
                "enable_adaptive": True,
                "invert_DBNN": main_config["training"].get("invert_DBNN", False),
                "reconstruction_weight": 0.5,
                "feedback_strength": 0.3,
                "inverse_learning_rate": 0.1,
                "Save_training_epochs": False,
                "training_save_path": os.path.join(self.dataset_dir, "training_data"),
                "modelType": "Histogram",
                "compute_device": "auto",
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
    print("  --batch_size    Batch size for training (default: 32)")
    print("  --epochs        Number of training epochs (default: 20)")
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


def parse_arguments():
    if len(sys.argv) == 1:
        return get_interactive_args()

    parser = argparse.ArgumentParser(description="CDBNN Feature Extractor")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--data", type=str, help="dataset name/path")
    parser.add_argument(
        "--data_type", type=str, choices=["torchvision", "custom"], default="custom"
    )
    parser.add_argument(
        "--encoder_type", type=str, choices=["cnn", "autoenc"], default="cnn"
    )
    parser.add_argument("--config", type=str, help="path to configuration file")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument(
        "--output-dir", type=str, default="data", help="output directory"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--workers", type=int, default=4, help="number of workers")
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="learning rate"
    )
    parser.add_argument("--cpu", action="store_true", help="force CPU usage")
    parser.add_argument(
        "--invert-dbnn", action="store_true", help="enable inverse DBNN mode"
    )
    parser.add_argument(
        "--input-csv", type=str, help="input CSV for prediction or inverse DBNN"
    )

    return parser.parse_args()


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
    """Get arguments interactively with invert DBNN support"""
    last_args = load_last_args()
    args = argparse.Namespace()
    args.mode = (
        input("\nEnter mode (train/predict) [train]: ").strip().lower() or "train"
    )

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
    prompt = (
        f"Enter dataset name/path [{default}]: "
        if default
        else "Enter dataset name/path: "
    )
    args.data = input(prompt).strip() or default

    # Ask about invert DBNN
    default_invert = last_args.get("invert_dbnn", False) if last_args else False
    invert_response = (
        input(f"Enable inverse DBNN mode? (y/n) [{['n', 'y'][default_invert]}]: ")
        .strip()
        .lower()
    )
    args.invert_dbnn = invert_response == "y" if invert_response else default_invert

    # If in predict mode and invert DBNN is enabled, ask for input CSV
    if args.mode == "predict" and args.invert_dbnn:
        default_csv = last_args.get("input_csv", "") if last_args else ""
        prompt = f"Enter input CSV path (or leave empty for default) [{default_csv}]: "
        args.input_csv = input(prompt).strip() or default_csv

    # Get encoder type
    while True:
        default = last_args.get("encoder_type", "cnn") if last_args else "autoenc"
        prompt = f"Enter encoder type (cnn/autoenc) [{default}]: "
        encoder_type = input(prompt).strip().lower() or default
        if encoder_type in ["cnn", "autoenc"]:
            args.encoder_type = encoder_type
            break
        print("Invalid encoder type. Please enter 'cnn' or 'autoenc'")

    # Optional parameters
    default = last_args.get("batch_size", 32) if last_args else 32
    args.batch_size = int(input(f"Enter batch size [{default}]: ").strip() or default)

    default = last_args.get("epochs", 20) if last_args else 20
    args.epochs = int(input(f"Enter number of epochs [{default}]: ").strip() or default)

    default = last_args.get("output_dir", "data") if last_args else "data"
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


def main():
    args = None
    try:
        logger = setup_logging()
        args = parse_arguments()

        if args.mode == "train":
            # Setup paths
            data_name = os.path.splitext(os.path.basename(args.data))[0]
            data_dir = os.path.join("data", data_name)
            config_path = os.path.join(data_dir, f"{data_name}.json")

            # Process dataset
            processor = DatasetProcessor(
                args.data, args.data_type, getattr(args, "output_dir", "data")
            )
            train_dir, test_dir = processor.process()
            logger.info(
                f"Dataset processed: train_dir={train_dir}, test_dir={test_dir}"
            )

            # Generate/verify all configurations
            logger.info("Generating/verifying configurations...")
            config = processor.generate_default_config(train_dir)

            # If user provided a custom config, merge it
            if args.config:
                with open(args.config) as f:
                    user_config = json.load(f)
                    config = processor._merge_configs(config, user_config)
                    # Save merged config back
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=4)

            # Update configuration with command line arguments
            config = update_config_with_args(config, args)

            # Configuration editing prompt
            print(
                "\nConfiguration files have been created/updated. Would you like to edit them?"
            )
            if input("Edit configuration? (y/n): ").lower() == "y":
                config_manager = ConfigManager(processor.dataset_dir)
                config_manager._open_editor(config_path)
                config_manager._open_editor(os.path.join(data_dir, f"{data_name}.conf"))
                config_manager._open_editor(
                    os.path.join(data_dir, "adaptive_dbnn.conf")
                )
                # Reload config if edited
                with open(config_path) as f:
                    config = json.load(f)

            # Setup training
            transform = processor.get_transforms(config)
            train_dataset, test_dataset = get_dataset(config, transform)

            if train_dataset is None:
                raise ValueError("No training dataset available")

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

            logger.info(
                f"Initializing {config['model']['encoder_type']} feature extractor..."
            )
            feature_extractor = get_feature_extractor(config)

            # Training confirmation
            if input("\nReady to start training. Proceed? (y/n): ").lower() != "y":
                logger.info("Training cancelled by user")
                return 0

            # Training and feature extraction
            logger.info("Starting model training...")
            history = feature_extractor.train(train_loader, test_loader)

            logger.info("Extracting features...")
            train_features, train_labels = feature_extractor.extract_features(
                train_loader
            )

            if test_loader:
                test_features, test_labels = feature_extractor.extract_features(
                    test_loader
                )
                features = torch.cat([train_features, test_features])
                labels = torch.cat([train_labels, test_labels])
            else:
                features = train_features
                labels = train_labels

            # Save features
            # output_name = "reconstructed_input.csv" if config['training'].get('invert_DBNN', False) else f"{data_name}.csv"
            output_name = f"{data_name}.csv"
            output_path = os.path.join(data_dir, output_name)
            feature_extractor.save_features(features, labels, output_path)
            logger.info(f"Features saved to {output_path}")

            if history:
                plot_path = os.path.join(data_dir, "training_history.png")
                feature_extractor.plot_training_history(plot_path)

            logger.info("Processing completed successfully!")
            return 0

        elif args.mode == "predict":
            # Prediction mode handling (keeping existing code)
            data_name = os.path.splitext(os.path.basename(args.data))[0]
            data_dir = os.path.join("data", data_name)
            config_path = os.path.join(data_dir, f"{data_name}.json")

            # Load or generate config
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                    if hasattr(args, "invert_dbnn"):
                        config["training"]["invert_DBNN"] = args.invert_dbnn
            else:
                # Determine encoder type from checkpoint or default
                checkpoint_path = os.path.join(
                    data_dir, "checkpoints", f"{data_name}_best.pth"
                )
                encoder_type = "autoenc"
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    encoder_type = (
                        checkpoint.get("config", {})
                        .get("model", {})
                        .get("encoder_type", "autoenc")
                    )

                processor = DatasetProcessor(
                    args.data, args.data_type, getattr(args, "output_dir", "data")
                )
                train_dir, _ = processor.process()
                config = processor.generate_default_config(train_dir)
                config["model"]["encoder_type"] = encoder_type

            # Determine input CSV
            if args.input_csv:
                input_csv = args.input_csv
            else:
                inverse_csv = os.path.join(data_dir, "reconstructed_input.csv")
                default_csv = os.path.join(data_dir, f"{data_name}.csv")
                input_csv = (
                    inverse_csv
                    if os.path.exists(inverse_csv)
                    and config["training"].get("invert_DBNN", False)
                    else default_csv
                )

            if not os.path.exists(input_csv):
                raise FileNotFoundError(f"Input CSV not found: {input_csv}")

            feature_extractor = get_feature_extractor(config)
            feature_extractor.predict_from_csv(input_csv)

            output_dir = config["output"]["image_dir"]
            mode_str = (
                "inverse mode"
                if config["training"].get("invert_DBNN", False)
                else "normal mode"
            )
            logger.info(f"Images generated in {output_dir} using {mode_str}")
            return 0

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        if args and args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
