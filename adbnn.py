# --------------------------Fully Functional bug fixed version ----------------------------
# ------------------------------------May 17 2025 19:05 IST----------------------------

# Working, fully functional with predcition 31/March/2025 Stable Model
# Better Memory management 06:28am
# Tested an fully functional April 4 2025 3:34 am
# Tested for more flexibility and added posteriors with class pedictions 5 April 7:22 pm
# Enhanced mosaic images 6 April 2025 8:45 am
# Feature pair automatic recomputation disabled
# Training until patience enabled. April 7 8:14 am 2025
# Finalised completely working module as on 15th April 2025
# Tested and working well with numerica target also April 27 11:28 pm
# upgraded to use GPU in pairwise computations for both models. May 4 2025 7:28 pm
# Added automatic conf creation option. May 4 2025 10:56 pm
# Updated failed candidate selection procedure as per DBNN original concepts May 5 2025 3:57 pm
# Fixed the bug in feature selection : May 10:12:09 am
# Fixed the bug in sample selection May 11 1:27 am

import argparse

# For pdf mosaic--
import math
import os
import random
import shutil

# ----------------------------------------------------------------------------------------------------------------------------
# ---- author: Ninan Sajeeth Philip, Artificial Intelligence Research and Intelligent Systems
# -----------------------------------------------------------------------------------------------------------------------------
import subprocess
import tempfile
import time

import matplotlib
import numpy as np

# ----
import pandas as pd
import torch
from PIL import Image
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch

# --
from reportlab.platypus import Image as ReportLabImage
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

matplotlib.use("Agg")
import json
import os
import pickle
import traceback  # Add to provide debug
from collections import defaultdict
from io import StringIO
from itertools import combinations
from math import comb
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import torch

# from Invertible_DBNN import InvertibleDBNN
# ------------------------------------------------------------------------Declarations---------------------
# Device configuration - set this first since other classes need it
Train_device = "cuda" if torch.cuda.is_available() else "cpu"  # Default device
Trials = 100  # Number of epochs to wait for improvement in training
cardinality_threshold = 0.9
cardinality_tolerance = 4  # Use when the features are likely to be extremly diverse and deciimal values;4 means, precison restricted to 4 decimal places
LearningRate = 0.1
TrainingRandomSeed = 42  # None # 42
Epochs = 1000
bin_sizes = 128
n_bins_per_dim = 128
TestFraction = 0.2
Train = True  # True #False #
Train_only = False  # True #
Predict = True
Gen_Samples = False
EnableAdaptive = True  # New parameter to control adaptive training
# Assume no keyboard control by default. If you have X11 running and want to be interactive, set nokbd = False
nokbd = False  # Enables interactive keyboard when training (q and Q will not have any effect)
display = None  # Initialize display variable
import json
import os
import pickle
import traceback
from collections import defaultdict
from io import StringIO
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

# ----------------------------------------------------------------------------------------------------------------
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm


def get_dataset_name_from_path(file_path):
    """Extracts dataset name from path (e.g., 'data/mnist/file.csv' -> 'mnist')"""
    # Normalize path and split into parts
    dataset_name = file_path.split("/")[1]
    return dataset_name


class DatasetProcessor:
    """A class to handle dataset-related operations such as downloading, processing, and formatting."""

    def __init__(self, data_dir: str = "data", config_dir: str = "config"):
        """
        Initialize the DatasetProcessor.

        Args:
            data_dir: Directory to store datasets.
            config_dir: Directory to store configuration files.
        """
        self.data_dir = data_dir
        self.config_dir = config_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        self.base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        self.compressed_extensions = [".zip", ".gz", ".tar", ".7z", ".rar"]
        self.colors = Colors()

    def download_uci_dataset(self, dataset_name: str, url: str) -> str:
        """
        Download a dataset from the UCI repository.

        Args:
            dataset_name: Name of the dataset.
            url: URL to the dataset.

        Returns:
            Path to the downloaded dataset.
        """
        dataset_path = os.path.join(self.data_dir, f"{dataset_name}.csv")
        if os.path.exists(dataset_path):
            print(
                "\033[K" + f"Dataset {dataset_name} already exists at {dataset_path}."
            )
            return dataset_path

        print("\033[K" + f"Downloading dataset {dataset_name} from {url}...")
        response = requests.get(url)
        response.raise_for_status()

        # Save the dataset
        with open(dataset_path, "wb") as f:
            f.write(response.content)

        print("\033[K" + f"Dataset saved to {dataset_path}.")
        return dataset_path

    def search_uci_repository(self, query: str) -> List[Dict[str, str]]:
        """
        Search the UCI repository for datasets matching a query.

        Args:
            query: Search query.

        Returns:
            List of dictionaries containing dataset information.
        """
        search_url = (
            f"https://archive.ics.uci.edu/ml/datasets.php?format=json&query={query}"
        )
        response = requests.get(search_url)
        response.raise_for_status()

        datasets = response.json()
        print("\033[K" + f"Found {len(datasets)} datasets matching query '{query}'.")
        return datasets


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


class DatasetConfig:
    """Enhanced dataset configuration handling with support for column names and URLs"""

    DEFAULT_CONFIG = {
        "file_path": None,
        "column_names": None,
        "target_column": None,
        "separator": ",",
        "has_header": True,
        "likelihood_config": {
            "feature_group_size": 2,
            "max_combinations": 90000000,
            "bin_sizes": [128],
        },
        "active_learning": {"tolerance": 1.0, "cardinality_threshold_percentile": 95},
        "training_params": {
            "save_plots": False,  # Parameter to save plots
            "Save_training_epochs": False,  # Save the epochs parameter
            "training_save_path": "data",  # Save epochs path parameter
        },
    }

    @staticmethod
    def is_url(path: str) -> bool:
        """Check if the given path is a URL"""
        return path.startswith(("http://", "https://"))

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate that a URL is accessible"""
        try:
            response = requests.head(url)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def create_default_config(dataset_name: str) -> Dict:
        """Create a default configuration file with enhanced defaults"""
        config = DatasetConfig.DEFAULT_CONFIG.copy()
        config["file_path"] = f"{dataset_name}.csv"
        print(
            f"{Colors.RED} Creating default configuration as the cofiguration file is invalid {Colors.ENDC}"
        )
        # Try to infer column names from CSV if it exists
        if os.path.exists(config["file_path"]):
            try:
                with open(config["file_path"]) as f:
                    header = f.readline().strip()
                    config["column_names"] = header.split(config["separator"])
                    if config["column_names"]:
                        config["target_column"] = config["column_names"][-1]
            except Exception as e:
                print(
                    "\033[K"
                    + f"Warning: Could not read header from {config['file_path']}: {str(e)}"
                )

        # Add model type configuration
        config["modelType"] = "Histogram"  # Default to Histogram model

        # Add training parameters
        config["training_params"] = {
            "override_global_cardinality": False,
            "trials": 100,
            "cardinality_threshold": 0.9,
            "minimum_training_accuracy": 0.95,
            "cardinality_tolerance": 8,
            "learning_rate": 0.001,
            "random_seed": 42,
            "epochs": 1000,
            "test_fraction": 0.2,
            "n_bins_per_dim": 21,
            "enable_adaptive": True,
            "compute_device": "auto",
            "invert_DBNN": True,
            "reconstruction_weight": 0.5,
            "feedback_strength": 0.3,
            "inverse_learning_rate": 0.001,
            "save_plots": False,
            "class_preference": True,
        }
        config["active_learning"] = {
            "tolerance": 1.0,
            "cardinality_threshold_percentile": 95,
            "strong_margin_threshold": 0.01,
            "marginal_margin_threshold": 0.01,
            "min_divergence": 0.1,
        }
        config["execution_flags"] = {
            "train": True,
            "train_only": False,
            "predict": True,
            "fresh_start": False,
            "use_previous_model": True,
        }

        return config

    @staticmethod
    def load_config(dataset_name: str) -> Dict:
        """Enhanced configuration loading with interactive setup and comment removal"""
        if not dataset_name or not isinstance(dataset_name, str):
            print("\033[K" + "Error: Invalid dataset name provided.")
            return None

        config_path = os.path.join("data", dataset_name, f"{dataset_name}.conf")
        csv_path = os.path.join("data", dataset_name, f"{dataset_name}.csv")

        try:
            # Handle missing config file
            if not os.path.exists(config_path):
                # Check for corresponding CSV file
                if os.path.exists(csv_path):
                    print(
                        "\033[K"
                        + f"Configuration file not found. Starting interactive setup for {dataset_name}..."
                    )

                    # Read CSV columns
                    columns = []
                    has_header = True
                    try:
                        with open(csv_path) as f:
                            first_line = f.readline().strip()
                            columns = first_line.split(",")

                            # Verify if we should use first line as header
                            try:
                                pd.read_csv(csv_path, nrows=0)
                            except pd.errors.ParserError:
                                has_header = False
                                columns = []
                    except Exception as e:
                        print("\033[K" + f"Error reading CSV file: {str(e)}")
                        return None

                    if not has_header or not columns:
                        print(
                            "\033[K"
                            + "CSV file appears to have no header. Please provide column names."
                        )
                        columns = []
                        while True:
                            col_input = input(
                                "\033[K"
                                + "Enter comma-separated column names (including target): "
                            ).strip()
                            if col_input:
                                columns = [c.strip() for c in col_input.split(",")]
                                if len(columns) > 1:
                                    break
                            print(
                                "\033[K"
                                + "Invalid input! Must provide at least two columns."
                            )

                    print("\033[K" + f"Detected columns: {', '.join(columns)}")

                    # Get target column from user
                    target = None
                    while True:
                        target = input("\033[K" + "Enter target column name: ").strip()
                        if target in columns:
                            break
                        print(
                            "\033[K"
                            + f"Invalid column! Choose from: {', '.join(columns)}"
                        )

                    # Create base configuration
                    config = DatasetConfig.create_default_config(dataset_name)
                    config.update(
                        {
                            "file_path": csv_path,
                            "column_names": columns,
                            "target_column": target,
                            "has_header": has_header,
                            "modelType": config.get("modelType", "Histogram"),
                        }
                    )

                    # Let user edit the config
                    with tempfile.NamedTemporaryFile(
                        mode="w+", suffix=".conf", delete=False
                    ) as tmp:
                        json.dump(config, tmp, indent=2)
                        tmp_path = tmp.name

                    # Open editor
                    editor = os.environ.get("EDITOR", "nano")
                    try:
                        subprocess.call([editor, tmp_path])
                    except Exception as e:
                        print("\033[K" + f"Error opening editor: {str(e)}")
                        return None

                    # Load edited config
                    with open(tmp_path) as f:
                        edited_config = json.load(f)
                    os.remove(tmp_path)

                    # Save final config
                    os.makedirs(os.path.dirname(config_path), exist_ok=True)
                    with open(config_path, "w") as f:
                        json.dump(edited_config, f, indent=2)
                    print("\033[K" + f"Configuration saved to {config_path}")

                else:
                    print(
                        "\033[K" + f"Creating default configuration for {dataset_name}"
                    )
                    return DatasetConfig.create_default_config(dataset_name)

            # Existing configuration loading logic
            with open(config_path, encoding="utf-8") as f:
                config_text = f.read()

            def remove_comments(json_str):
                lines = []
                in_multiline_comment = False
                for line in json_str.split("\n"):
                    if "_comment" in line:
                        continue
                    if "/*" in line and "*/" in line:
                        line = line[: line.find("/*")] + line[line.find("*/") + 2 :]
                    elif "/*" in line:
                        in_multiline_comment = True
                        line = line[: line.find("/*")]
                    elif "*/" in line:
                        in_multiline_comment = False
                        line = line[line.find("*/") + 2 :]
                    elif in_multiline_comment:
                        continue
                    if "//" in line and not ("http://" in line or "https://" in line):
                        line = line.split("//")[0]
                    stripped = line.strip()
                    if stripped and not stripped.startswith("_comment"):
                        lines.append(stripped)
                return "\n".join(lines)

            clean_config = remove_comments(config_text)
            config = json.loads(clean_config)
            validated_config = DatasetConfig.DEFAULT_CONFIG.copy()
            validated_config.update(config)

            # Path handling
            if validated_config.get("file_path"):
                if not os.path.exists(validated_config["file_path"]):
                    alt_path = os.path.join("data", dataset_name, f"{dataset_name}.csv")
                    if os.path.exists(alt_path):
                        validated_config["file_path"] = alt_path
                        print("\033[K" + f"Using data file: {alt_path}")

            if not validated_config.get("file_path"):
                default_path = os.path.join("data", dataset_name, f"{dataset_name}.csv")
                if os.path.exists(default_path):
                    validated_config["file_path"] = default_path
                    print("\033[K" + f"Using default data file: {default_path}")

            # URL handling
            if DatasetConfig.is_url(validated_config.get("file_path", "")):
                url = validated_config["file_path"]
                local_path = os.path.join("data", dataset_name, f"{dataset_name}.csv")
                if not os.path.exists(local_path):
                    print("\033[K" + f"Downloading dataset from {url}")
                    if not DatasetConfig.download_dataset(url, local_path):
                        print("\033[K" + f"Failed to download dataset from {url}")
                        return None
                    print("\033[K" + f"Downloaded dataset to {local_path}")
                validated_config["file_path"] = local_path

            # Validation
            if not validated_config.get("file_path") or not os.path.exists(
                validated_config["file_path"]
            ):
                print("\033[K" + "Warning: Data file not found")
                return None

            # Validate columns
            try:
                df = pd.read_csv(validated_config["file_path"], nrows=0)
                # Only set column names if not already configured
                if (
                    "column_names" not in validated_config
                    or not validated_config["column_names"]
                ):
                    validated_config["column_names"] = df.columns.tolist()
                # Otherwise, validate configured columns exist in CSV
                else:
                    missing = set(validated_config["column_names"]) - set(df.columns)
                    if missing:
                        print(f"Configured columns missing in CSV: {missing}")
            except Exception as e:
                print("\033[K" + f"Warning: Could not validate columns: {str(e)}")
                return None

            # Validate target column exists
            target_col = validated_config.get("target_column")
            if target_col not in validated_config["column_names"]:
                print(
                    f"\033[K{Colors.RED}ERROR: Configured target column '{target_col}' not found in dataset!{Colors.ENDC}"
                )
                print(
                    f"\033[KAvailable columns: {', '.join(validated_config['column_names'])}"
                )

                # Interactive target column correction
                while True:
                    new_target = input(
                        "\033[KEnter correct target column name: "
                    ).strip()
                    if new_target in validated_config["column_names"]:
                        validated_config["target_column"] = new_target
                        # Save updated config
                        with open(config_path, "w") as f:
                            json.dump(validated_config, f, indent=2)
                        print(
                            f"\033[K{Colors.GREEN}Updated target column to '{new_target}' in configuration.{Colors.ENDC}"
                        )
                        break
                    print(
                        f"\033[K{Colors.RED}Invalid column! Choose from: {', '.join(validated_config['column_names'])}{Colors.ENDC}"
                    )
            if isinstance(target_col, (int, float)):
                # Convert numeric indices to column name
                validated_config["target_column"] = str(
                    validated_config["column_names"][target_col]
                )
            elif isinstance(target_col, str):
                pass  # Already correct
            else:
                raise ValueError(f"Invalid target column type: {type(target_col)}")

            return validated_config

        except Exception as e:
            print(
                "\033[K" + f"Error loading configuration for {dataset_name}: {str(e)}"
            )
            traceback.print_exc()
            return None

    @staticmethod
    def download_dataset(url: str, local_path: str) -> bool:
        """Download dataset from URL to local path with proper error handling"""
        try:
            print("\033[K" + f"Downloading dataset from {url}")
            response = requests.get(url, timeout=30)  # Add timeout
            response.raise_for_status()  # Check for HTTP errors

            # Handle potential text/csv content
            content = response.content.decode("utf-8")

            # Save to local file
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)

            print("\033[K" + f"Dataset downloaded successfully to {local_path}")
            return True
        except requests.exceptions.RequestException as e:
            print("\033[K" + f"Error downloading dataset: {str(e)}")
            return False
        except UnicodeDecodeError:
            # Handle binary content
            try:
                with open(local_path, "wb") as f:
                    f.write(response.content)
                print("\033[K" + f"Dataset downloaded successfully to {local_path}")
                return True
            except Exception as e:
                print("\033[K" + f"Error saving binary content: {str(e)}")
                return False
        except Exception as e:
            print("\033[K" + f"Unexpected error downloading dataset: {str(e)}")
            return False

    @staticmethod
    def get_available_datasets(create_configs: bool = False) -> List[str]:
        """Get list of available dataset configurations with better filename handling"""
        # Get all config and CSV files
        conf_files = {
            f.split(".")[0]
            for f in os.listdir()
            if f.endswith(".conf") and f != "adaptive_dbnn.conf"
        }  # Exclude own config
        csv_files = {f.split(".")[0] for f in os.listdir() if f.endswith(".csv")}

        # Filter out derived filenames
        exclude_suffixes = [
            "_last_testing",
            "_Last_testing",
            "_last_training",
            "_Last_training",
            "_predictions",
            "_training_metrics",
            "_training_metrics_metrics",
        ]

        # Filter CSV files that don't have config
        csv_without_conf = csv_files - conf_files
        csv_without_conf = {
            name
            for name in csv_without_conf
            if not any(name.endswith(suffix) for suffix in exclude_suffixes)
        }

        # Start with datasets that have config files
        datasets = conf_files

        # If requested, ask about creating configs for remaining CSVs
        if create_configs and csv_without_conf:
            print("\033[K" + "Found CSV files without configuration:")
            for csv_name in sorted(csv_without_conf):
                response = input(f"Create configuration for {csv_name}.csv? (y/n): ")
                if response.lower() == "y":
                    try:
                        DatasetConfig.create_default_config(csv_name)
                        datasets.add(csv_name)
                    except Exception as e:
                        print(
                            "\033[K" + f"Error creating config for {csv_name}: {str(e)}"
                        )

        return sorted(list(datasets))

    @staticmethod
    def validate_dataset(dataset_name: str) -> bool:
        """Validate dataset with better name handling"""
        # Check if this is a derived filename
        exclude_suffixes = [
            "_last_testing",
            "_Last_testing",
            "_last_training",
            "_Last_training",
            "_predictions",
            "_training_metrics",
            "_training_metrics_metrics",
        ]

        if any(dataset_name.endswith(suffix) for suffix in exclude_suffixes):
            print("\033[K" + f"Skipping validation for derived dataset: {dataset_name}")
            return False

        config = DatasetConfig.load_config(dataset_name)
        file_path = config["file_path"]

        # Handle URL-based datasets
        if DatasetConfig.is_url(file_path):
            if not DatasetConfig.validate_url(file_path):
                print("\033[K" + f"Warning: Dataset URL {file_path} is not accessible")
                return False

            # Download to local cache if needed
            local_path = f"{dataset_name}.csv"
            if not os.path.exists(local_path):
                if not DatasetConfig.download_dataset(file_path, local_path):
                    return False
            file_path = local_path

        if not os.path.exists(file_path):
            print("\033[K" + f"Warning: Dataset file {file_path} not found")
            return False

        return True


# ---------------------------------------Feature Filter with a #------------------------------------
def _filter_features_from_config(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Filter DataFrame to only include specified columns from config

    Args:
        df: Input DataFrame
        config: Configuration dictionary containing column names

    Returns:
        DataFrame with only the specified columns
    """
    # If no column names in config, return original DataFrame

    if "column_names" not in config or not config["column_names"]:
        print("\033[K" + "No column names specified in config. Keeping all columns.")
        return df

    # Get current DataFrame columns
    current_cols = df.columns.tolist()
    # print("\033[K" +f"Current DataFrame columns: {current_cols}")

    # Get column names from config (only those not commented out)
    requested_columns = [
        name.strip()
        for name in config["column_names"]
        if not name.strip().startswith("#")
    ]

    # If no uncommented columns found in config, return original DataFrame
    if not requested_columns:
        print(
            "\033[K"
            + "No uncommented column names found in config. Returning original DataFrame."
        )
        return df

    # Check if any requested columns exist in the DataFrame
    valid_columns = [col for col in requested_columns if col in current_cols]

    # If no valid columns found, return original DataFrame
    if not valid_columns:
        print(
            "\033[K"
            + "None of the requested columns exist in the DataFrame. Returning original DataFrame."
        )
        return df

    # Return DataFrame with only the columns to keep
    # print("\033[K" +f"Keeping only these features: {valid_columns}")
    return df[valid_columns]


# -------------------------------------------------
class ComputationCache:
    """Cache for frequently used computations"""

    def __init__(self, device):
        self.device = device
        self.bin_edges_cache = {}
        self.bin_indices_cache = {}
        self.feature_group_cache = {}
        self.max_cache_size = 1000

    def get_bin_edges(self, group_idx, edges):
        if group_idx not in self.bin_edges_cache:
            self.bin_edges_cache[group_idx] = torch.stack(
                [edge.contiguous() for edge in edges]
            ).to(self.device)
        return self.bin_edges_cache[group_idx]

    def get_feature_group(self, features, group_idx, feature_pairs):
        key = (features.shape[0], group_idx)
        if key not in self.feature_group_cache:
            if len(self.feature_group_cache) >= self.max_cache_size:
                self.feature_group_cache.clear()
            self.feature_group_cache[key] = features[
                :, feature_pairs[group_idx]
            ].contiguous()
        return self.feature_group_cache[key]


class BinWeightUpdater:
    def __init__(self, n_classes, feature_pairs, n_bins_per_dim=128, batch_size=128):
        self.n_classes = n_classes
        self.feature_pairs = feature_pairs
        self.n_bins_per_dim = n_bins_per_dim
        self.device = Train_device
        # Initialize histogram_weights as empty dictionary first
        self.histogram_weights = {}
        self.batch_size = batch_size
        # Ensure all weight tensors are contiguous
        for c in self.histogram_weights.values():
            for p in c.values():
                p = p.contiguous()
        # Create weights for each class and feature pair
        for class_id in range(n_classes):
            self.histogram_weights[class_id] = {}
            for pair_idx in range(len(feature_pairs)):
                # Initialize with default weight of 0.1
                # print("\033[K" +f"[DEBUG] Creating weights for class {class_id}, pair {pair_idx}")
                self.histogram_weights[class_id][pair_idx] = torch.full(
                    (n_bins_per_dim, n_bins_per_dim),
                    0.1,
                    dtype=torch.float32,
                    device=self.device,  # Ensure weights are created on correct device
                ).contiguous()

        # Initialize weights for each class and feature pair
        self.gaussian_weights = {}
        for class_id in range(n_classes):
            self.gaussian_weights[class_id] = {}
            for pair_idx in range(len(feature_pairs)):
                # Initialize with default weight of 0.1
                self.gaussian_weights[class_id][pair_idx] = torch.tensor(
                    0.1, dtype=torch.float32, device=self.device
                ).contiguous()

        # Verify initialization
        print("\033[K" + f"[DEBUG] Weight initialization complete. Structure:")
        print("\033[K" + f"- Number of classes: {len(self.histogram_weights)}")
        for class_id in self.histogram_weights:
            print(
                "\033[K"
                + f"- Class {class_id}: {len(self.histogram_weights[class_id])} feature pairs"
            )

        # Use a single contiguous tensor for all weights
        self.weights = torch.full(
            (n_classes, len(feature_pairs), n_bins_per_dim, n_bins_per_dim),
            0.1,
            dtype=torch.float32,
            device=self.device,  # Ensure weights are created on correct device
        ).contiguous()

        # Pre-allocate update buffers
        self.update_indices = torch.zeros(
            (3, 1000), dtype=torch.long
        )  # [dim, max_updates]
        self.update_values = torch.zeros(1000, dtype=torch.float32)
        self.update_count = 0

    def _validate_update(old_weights, new_weights, updates):
        """Sanity check for update consistency"""
        for (class_id, pair_idx, bin_i, bin_j), adj in updates.items():
            assert torch.allclose(
                old_weights[class_id][pair_idx][bin_i, bin_j] + adj,
                new_weights[class_id][pair_idx][bin_i, bin_j],
            ), "Update mismatch detected!"

    def batch_update_weights(
        self, class_indices, pair_indices, bin_indices, adjustments
    ):
        # Convert to tensors first
        class_ids = torch.tensor(class_indices, dtype=torch.long, device=self.device)
        pair_ids = torch.tensor(pair_indices, dtype=torch.long, device=self.device)
        bin_is = torch.tensor(
            [bi[0] if isinstance(bi, (list, tuple)) else bi for bi in bin_indices],
            device=self.device,
        )
        bin_js = torch.tensor(
            [bj[1] if isinstance(bj, (list, tuple)) else bj for bj in bin_indices],
            device=self.device,
        )
        adjs = torch.tensor(adjustments, dtype=torch.float32, device=self.device)

        # Group updates by (class, pair)
        unique_pairs, inverse = torch.unique(
            torch.stack([class_ids, pair_ids]), dim=1, return_inverse=True
        )

        for group_idx in range(unique_pairs.shape[1]):
            class_id, pair_id = unique_pairs[:, group_idx]
            mask = inverse == group_idx

            # Get all updates for this (class, pair)
            b_i = bin_is[mask]
            b_j = bin_js[mask]
            adj = adjs[mask]

            # Vectorized update
            self.histogram_weights[class_id.item()][pair_id.item()].index_put_(
                indices=(b_i, b_j), values=adj, accumulate=True
            )

    def get_histogram_weights(self, class_id: int, pair_idx: int) -> torch.Tensor:
        """Get weights ensuring proper dimensions"""
        class_id = int(class_id)
        pair_idx = int(pair_idx)

        if class_id not in self.histogram_weights:
            raise KeyError(f"Invalid class_id: {class_id}")
        if pair_idx not in self.histogram_weights[class_id]:
            raise KeyError(f"Invalid pair_idx: {pair_idx}")

        weights = self.histogram_weights[class_id][pair_idx]
        if len(weights.shape) != 2:
            raise ValueError(
                f"Invalid weight shape: {weights.shape}, expected 2D tensor"
            )
        if (
            weights.shape[0] != self.n_bins_per_dim
            or weights.shape[1] != self.n_bins_per_dim
        ):
            raise ValueError(
                f"Invalid weight dimensions: {weights.shape}, expected ({self.n_bins_per_dim}, {self.n_bins_per_dim})"
            )

        return weights

    def _ensure_buffers(self, batch_size):
        """Ensure buffers exist and are the right size"""
        if (
            self.batch_indices_buffer is None
            or self.batch_indices_buffer.shape[0] < batch_size
        ):
            self.batch_indices_buffer = torch.zeros(
                (batch_size, 2),
                dtype=torch.long,
                device=next(
                    iter(next(iter(self.histogram_weights.values())).values())
                ).device,
            ).contiguous()

            self.batch_adjustments_buffer = torch.zeros(
                batch_size, dtype=torch.float32, device=self.batch_indices_buffer.device
            ).contiguous()

    def update_weight(
        self, class_id: int, pair_idx: int, bin_i: int, bin_j: int, adjustment: float
    ):
        """Single weight update with proper error handling"""
        try:
            # Convert all indices to integers
            class_id = int(class_id)
            pair_idx = int(pair_idx)
            bin_i = int(bin_i)
            bin_j = int(bin_j)

            # Ensure indices are within bounds
            bin_i = min(max(0, bin_i), self.n_bins_per_dim - 1)
            bin_j = min(max(0, bin_j), self.n_bins_per_dim - 1)

            # Update the weight
            self.histogram_weights[class_id][pair_idx][bin_i, bin_j] += adjustment

        except Exception as e:
            print("\033[K" + f"Error updating weight: {str(e)}")
            print("\033[K" + f"class_id: {class_id}, pair_idx: {pair_idx}")
            print("\033[K" + f"bin_i: {bin_i}, bin_j: {bin_j}")
            print("\033[K" + f"adjustment: {adjustment}")
            raise

    def update_histogram_weights(
        self,
        failed_case,
        true_class,
        pred_class,
        bin_indices,
        posteriors,
        learning_rate,
    ):
        # Precompute all adjustments first
        adjustment = learning_rate * (
            1.0 - (posteriors[true_class] / posteriors[pred_class])
        )

        # Batch indices and values
        pair_indices = []
        bin_is = []
        bin_js = []
        adjustments = []

        for pair_idx, (bin_i, bin_j) in bin_indices.items():
            pair_indices.append(pair_idx)
            bin_is.append(bin_i)
            bin_js.append(bin_j)
            adjustments.append(adjustment)

        # Convert to tensors
        pair_ids = torch.tensor(pair_indices, dtype=torch.long, device=self.device)
        b_i = torch.tensor(bin_is, dtype=torch.long, device=self.device)
        b_j = torch.tensor(bin_js, dtype=torch.long, device=self.device)
        adjs = torch.tensor(adjustments, dtype=torch.float32, device=self.device)

        # Group by pair_idx
        unique_pairs, counts = torch.unique(pair_ids, return_counts=True)

        for pair_id in unique_pairs:
            mask = pair_ids == pair_id
            self.histogram_weights[true_class][pair_id.item()].index_put_(
                indices=(b_i[mask], b_j[mask]), values=adjs[mask], accumulate=True
            )

    def update_gaussian_weights(
        self,
        failed_case,
        true_class,
        pred_class,
        component_responsibilities,
        posteriors,
        learning_rate,
    ):
        """Update weights for Gaussian components with improved efficiency"""
        DEBUG.log(f" Updating Gaussian weights for class {true_class}")

        try:
            # Convert tensor values to Python types and validate
            true_class = (
                int(true_class) if isinstance(true_class, torch.Tensor) else true_class
            )
            pred_class = (
                int(pred_class) if isinstance(pred_class, torch.Tensor) else pred_class
            )

            if true_class not in self.gaussian_weights:
                raise ValueError(f"Invalid true_class: {true_class}")

            # Get posterior values with type checking
            true_posterior = (
                float(posteriors[true_class])
                if isinstance(posteriors, torch.Tensor)
                else posteriors[true_class]
            )
            pred_posterior = (
                float(posteriors[pred_class])
                if isinstance(posteriors, torch.Tensor)
                else posteriors[pred_class]
            )

            # Calculate adjustment based on posterior ratio with stability check
            adjustment = learning_rate * (
                1.0 - max(min(true_posterior / pred_posterior, 10), 0.1)
            )
            DEBUG.log(f" Weight adjustment: {adjustment}")

            # Process each feature pair efficiently
            for pair_idx in range(len(self.feature_pairs)):
                # Get and validate responsibility matrix
                resp_matrix = component_responsibilities[pair_idx]
                if not isinstance(resp_matrix, torch.Tensor):
                    resp_matrix = torch.tensor(resp_matrix)
                resp_matrix = resp_matrix.to(
                    self.gaussian_weights[true_class][pair_idx].device
                )

                # Ensure shapes match
                current_weights = self.gaussian_weights[true_class][pair_idx]
                if resp_matrix.shape != current_weights.shape:
                    DEBUG.log(
                        f" Shape mismatch - resp_matrix: {resp_matrix.shape}, weights: {current_weights.shape}"
                    )
                    resp_matrix = resp_matrix[
                        : current_weights.shape[0], : current_weights.shape[1]
                    ]

                # Update weights with stability check
                weight_update = resp_matrix * adjustment
                weight_update = torch.clamp(
                    weight_update, -1.0, 1.0
                )  # Prevent extreme updates
                self.gaussian_weights[true_class][pair_idx] += weight_update

                # Apply non-negativity constraint
                self.gaussian_weights[true_class][pair_idx].clamp_(min=0.0)

        except Exception as e:
            DEBUG.log(f" Error updating Gaussian weights: {str(e)}")
            DEBUG.log(" Traceback:", traceback.format_exc())
            raise

    def get_gaussian_weights(self, class_id, pair_idx):
        """Get Gaussian weights with proper type conversion and validation"""
        try:
            # Convert tensor values to Python integers
            class_id = int(class_id) if isinstance(class_id, torch.Tensor) else class_id
            pair_idx = int(pair_idx) if isinstance(pair_idx, torch.Tensor) else pair_idx

            if class_id not in self.gaussian_weights:
                raise KeyError(f"Invalid class_id: {class_id}")
            if pair_idx not in self.gaussian_weights[class_id]:
                raise KeyError(f"Invalid pair_idx: {pair_idx}")

            weights = self.gaussian_weights[class_id][pair_idx]
            DEBUG.log(
                f" Retrieved Gaussian weights for class {class_id}, pair {pair_idx}, shape: {weights.shape}"
            )
            return weights

        except Exception as e:
            DEBUG.log(f" Error retrieving Gaussian weights: {str(e)}")
            DEBUG.log(" Traceback:", traceback.format_exc())
            raise

    # Modified posterior computation for Histogram model
    def compute_histogram_posterior(self, features, bin_indices):
        batch_size = features.shape[0]
        n_classes = len(self.likelihood_params["classes"])
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        for group_idx, feature_group in enumerate(
            self.likelihood_params["feature_pairs"]
        ):
            self.likelihood_params["bin_edges"][group_idx]
            bin_probs = self.likelihood_params["bin_probs"][group_idx]

            # Get bin-specific weights
            bin_weights = self.weight_updater.get_histogram_weights(
                class_idx, group_idx
            )[bin_indices[group_idx]]

            # Apply bin-specific weights to probabilities
            weighted_probs = bin_probs * bin_weights.unsqueeze(0)

            # Continue with regular posterior computation...
            group_log_likelihoods = torch.log(weighted_probs + epsilon)
            log_likelihoods.add_(group_log_likelihoods)

        return log_likelihoods

    # Modified posterior computation for Gaussian model
    def compute_gaussian_posterior(self, features, component_responsibilities):
        batch_size = features.shape[0]
        n_classes = len(self.likelihood_params["classes"])
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        for group_idx, feature_group in enumerate(
            self.likelihood_params["feature_pairs"]
        ):
            # Get component-specific weights
            component_weights = self.weight_updater.get_gaussian_weights(
                class_idx, group_idx
            )

            # Weight the Gaussian components
            weighted_resp = component_responsibilities[group_idx] * component_weights

            # Continue with regular posterior computation...
            group_log_likelihoods = torch.log(weighted_resp.sum() + epsilon)
            log_likelihoods.add_(group_log_likelihoods)

        return log_likelihoods


from typing import Dict, Optional

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class InvertibleDBNN(nn.Module):
    """Invertible DBNN for reconstructing input features from classification probabilities."""

    def __init__(
        self,
        forward_model: nn.Module,
        feature_dims: int,
        n_classes: int,
        hidden_dims: int = 128,
        device: str = "cuda",
    ):
        """
        Initialize the Invertible DBNN.

        Args:
            forward_model (nn.Module): The forward DBNN model.
            feature_dims (int): Number of input feature dimensions.
            n_classes (int): Number of classes in the classification task.
            hidden_dims (int): Number of hidden dimensions in the inverse model.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        super().__init__()
        self.forward_model = forward_model
        self.feature_dims = feature_dims
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.device = device

        # Define the inverse model architecture
        self.inverse_model = nn.Sequential(
            nn.Linear(n_classes, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, feature_dims),
        )

        # Move model to the appropriate device
        self.to(device)

    def forward(self, class_probs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the inverse model to reconstruct features.

        Args:
            class_probs (torch.Tensor): Classification probabilities (batch_size, n_classes).

        Returns:
            torch.Tensor: Reconstructed features (batch_size, feature_dims).
        """
        return self.inverse_model(class_probs)

    def reconstruct_features(
        self,
        class_probs: torch.Tensor,
        original_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reconstruct features from classification probabilities.

        Args:
            class_probs (torch.Tensor): Classification probabilities (batch_size, n_classes).
            original_features (Optional[torch.Tensor]): Original features for validation (batch_size, feature_dims).

        Returns:
            torch.Tensor: Reconstructed features (batch_size, feature_dims).
        """
        reconstructed_features = self.forward(class_probs)

        if original_features is not None:
            # Calculate reconstruction error
            reconstruction_error = torch.mean(
                (reconstructed_features - original_features) ** 2
            )
            print("\033[K" + f"Reconstruction Error: {reconstruction_error.item():.4f}")

        return reconstructed_features

    def train_inverse_model(
        self,
        class_probs: torch.Tensor,
        original_features: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.001,
    ):
        """
        Train the inverse model to reconstruct features from classification probabilities.

        Args:
            class_probs (torch.Tensor): Classification probabilities (batch_size, n_classes).
            original_features (torch.Tensor): Original features (batch_size, feature_dims).
            epochs (int): Number of training epochs.
            lr (float): Learning rate for the optimizer.
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in tqdm(range(epochs), desc="Training Inverse Model", leave=False):
            optimizer.zero_grad()

            # Forward pass
            reconstructed_features = self.forward(class_probs)

            # Compute reconstruction loss
            loss = criterion(reconstructed_features, original_features)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print("\033[K" + f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def save_model(self, path: str):
        """Save the inverse model to a file."""
        torch.save(self.state_dict(), path)
        print("\033[K" + f"Model saved to {path}")

    def load_model(self, path: str):
        """Load the inverse model from a file."""
        self.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        print("\033[K" + f"Model loaded from {path}")


# ----------------------------------------------DBNN class-------------------------------------------------------------
class GPUDBNN:
    """GPU-Optimized Deep Bayesian Neural Network with Parallel Feature Pair Processing"""

    def __init__(
        self,
        dataset_name: str,
        learning_rate: float = LearningRate,
        max_epochs: int = Epochs,
        test_size: float = TestFraction,
        random_state: int = TrainingRandomSeed,
        device: str = None,
        fresh: bool = False,
        use_previous_model: bool = True,
        n_bins_per_dim: int = 128,
        model_type: str = "Histogram",
        mode: str = None,
    ):
        """Initialize GPUDBNN with support for continued training with fresh data"""

        # Set dataset_name and model type first
        self.mode = mode
        self.dataset_name = dataset_name
        self.model_type = model_type  # Store model type as instance variable
        self.device = Train_device
        self.computation_cache = ComputationCache(self.device)
        # Initialize train/test indices
        self.train_indices = []
        self.test_indices = None
        self._last_metrics_printed = False
        # Add new attribute for bin-specific weights
        self.weight_updater = (
            None  # Will be initialized after computing likelihood params
        )

        # Load configuration before potential cleanup
        self.config = DatasetConfig.load_config(self.dataset_name)
        self.feature_bounds = None  # Store global min/max for each
        # self.n_bins_per_dim = n_bins_per_dim
        self.n_bins_per_dim = self.config.get("likelihood_config", {}).get(
            "n_bins_per_dim", 128
        )

        # Initialize other attributes

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.test_size = test_size
        if random_state != -1:
            self.random_state = random_state
            self.shuffle_state = 1
        else:
            self.random_state = -1
            self.shuffle_state = -1
        # self.compute_dtype = torch.float64  # Use double precision for computations
        self.cardinality_tolerance = cardinality_tolerance  # Only for feature grouping
        self.fresh_start = fresh
        self.use_previous_model = use_previous_model
        # Create Model directory
        os.makedirs("Model", exist_ok=True)

        # Load configuration and data
        # DatasetConfig.load_config(self.dataset_name)
        self.target_column = self.config["target_column"]

        # Initialize model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        # Handle label encoder initialization based on mode
        if mode == "predict":
            # Strict validation for prediction mode
            if not self._load_model_components():
                raise RuntimeError(
                    "Cannot run prediction - no trained model found.\n"
                    f"Expected model files in: Model/{self.dataset_name}_*\n"
                    "Please train the model first."
                )
        elif self.use_previous_model:
            # Try to load existing model, but don't fail if it doesn't exist
            try:
                if not self._load_model_components():
                    print(
                        "\033[K"
                        + f"[INFO] No existing model found - starting fresh training"
                    )
                    self._initialize_fresh_training()
            except Exception as e:
                print("\033[K" + f"[WARNING] Failed to load previous model: {str(e)}")
                print("\033[K" + "Starting fresh training")
                self._initialize_fresh_training()
        else:
            # Fresh training requested
            self._initialize_fresh_training()

        self.likelihood_params = None
        self.feature_pairs = None
        self.bin_edges = None  # Add bin_edges attribute
        self.gaussian_params = None  # Add gaussian_params attribute for Gaussian model
        self.best_W = None
        self.best_error = float("inf")
        self.current_W = None

        # Enable cuDNN autotuner if using CUDA
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True

        # Pin memory for faster CPU->GPU transfer
        if torch.cuda.is_available():
            self.pin_memory = True
            torch.cuda.empty_cache()

        # Handle fresh start after configuration is loaded
        # Handle model state based on flags
        if use_previous_model:
            # Load previous model state
            # self.label_encoder =load_label_encoder(dataset_name)
            self._load_model_components()
            # self._load_best_weights()
            # self._load_categorical_encoders()
        else:
            # Complete fresh start
            self._clean_existing_model()

        # ------------------------------------------Adaptive Learning--------------------------------------
        super().__init__()
        self.adaptive_learning = True
        self.base_save_path = "./data"
        os.makedirs(self.base_save_path, exist_ok=True)
        self.in_adaptive_fit = False  # Set when we are in adaptive learning process
        # ------------------------------------------Adaptive Learning--------------------------------------
        # Automatically select device if none specified

        print("\033[K" + f"Using device: {self.device}")

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.test_size = test_size

        # Model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.likelihood_params = None
        self.feature_pairs = None
        self.best_W = None
        self.best_error = float("inf")
        self.current_W = None

        # Categorical feature handling
        self.categorical_encoders = {}

        # Create Model directory
        os.makedirs("Model", exist_ok=True)

        # Load dataset configuration and data
        # self.config = DatasetConfig.load_config(self.dataset_name)
        self.data = self._load_dataset()

        self.target_column = self.config["target_column"]

        # Load saved weights and encoders
        self._load_best_weights()
        self._load_categorical_encoders()

    def _initialize_fresh_training(self):
        """Initialize components for fresh training"""
        # Load dataset configuration and data
        # self.config = DatasetConfig.load_config(self.dataset_name)
        self.data = self._load_dataset()

        self.target_column = self.config["target_column"]
        if self.target_column not in self.data.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset.\n"
                f"Available columns: {list(self.data.columns)}"
            )

        # Fit label encoder and other fresh components
        self.label_encoder.fit(self.data[self.target_column])
        self.scaler = StandardScaler()
        self.feature_pairs = None
        self.likelihood_params = None

        # Mark as fresh training
        self.fresh_start = True

    def _compute_bin_edges(
        self, dataset: torch.Tensor, bin_sizes: List[int]
    ) -> List[List[torch.Tensor]]:
        """
        Vectorized computation of bin edges with GPU memory management.

        Args:
            dataset: Input tensor of shape [n_samples, n_features]
            bin_sizes: List of integers specifying bin sizes

        Returns:
            List of lists containing bin edge tensors for each feature pair
        """
        DEBUG.log("Starting vectorized _compute_bin_edges")

        # Memory management parameters
        MAX_GPU_MEM = (
            0.8 * torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else 1e10
        )
        SAFETY_FACTOR = 0.7  # Use only 70% of available memory

        # Calculate memory requirements per feature pair
        bytes_per_pair = (
            2 * 4 * len(bin_sizes)
        )  # 2 edges, 4 bytes per float, per dimension
        max_pairs_per_batch = int((MAX_GPU_MEM * SAFETY_FACTOR) / bytes_per_pair)
        max_pairs_per_batch = max(1, min(max_pairs_per_batch, len(self.feature_pairs)))

        bin_edges = []

        # Process in memory-managed batches
        for batch_start in range(0, len(self.feature_pairs), max_pairs_per_batch):
            batch_end = min(batch_start + max_pairs_per_batch, len(self.feature_pairs))
            batch_pairs = self.feature_pairs[batch_start:batch_end]

            # Vectorized min/max computation for the batch
            with torch.no_grad():
                # Stack all features needed in this batch
                feature_indices = torch.unique(
                    torch.cat(
                        [
                            torch.as_tensor(pair, device=self.device)
                            for pair in batch_pairs
                        ]
                    )
                )
                batch_data = dataset[:, feature_indices]

                # Compute min/max for all features in batch
                mins = batch_data.min(dim=0)[0]
                maxs = batch_data.max(dim=0)[0]

                # Create mapping from feature index to its position in batch_data
                feat_to_idx = {int(f): i for i, f in enumerate(feature_indices)}

                # Process each pair in batch
                for pair in batch_pairs:
                    pair_edges = []
                    for dim, feat in enumerate(pair):
                        feat_idx = feat_to_idx[int(feat)]
                        dim_min = mins[feat_idx]
                        dim_max = maxs[feat_idx]
                        padding = max((dim_max - dim_min) * 0.01, 1e-6)

                        # Get bin size for this dimension
                        bin_size = (
                            bin_sizes[0] if len(bin_sizes) == 1 else bin_sizes[dim]
                        )

                        # Vectorized edge computation
                        edges = torch.linspace(
                            dim_min - padding,
                            dim_max + padding,
                            bin_size + 1,
                            device=self.device,
                        ).contiguous()
                        pair_edges.append(edges)

                    bin_edges.append(pair_edges)

            torch.cuda.empty_cache()  # Free memory between batches

        return bin_edges


# ---------------------- -------------------------------------DBNN Class -------------------------------
class DBNNConfig:
    """Configuration class for DBNN parameters"""

    def __init__(self, **kwargs):
        # Training parameters
        self.trials = kwargs.get("trials", 100)
        self.cardinality_threshold = kwargs.get("cardinality_threshold", 0.9)
        self.cardinality_tolerance = kwargs.get("cardinality_tolerance", 4)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.random_seed = kwargs.get("random_seed", 42)
        self.epochs = kwargs.get("epochs", 1000)
        self.test_fraction = kwargs.get("test_fraction", 0.2)
        self.enable_adaptive = kwargs.get("enable_adaptive", True)
        self.batch_size = kwargs.get("batch_size", 128)

        # Model parameters
        self.model_type = kwargs.get("model_type", "Histogram")  # or 'Gaussian'
        self.n_bins_per_dim = kwargs.get("n_bins_per_dim", 128)

        # Execution flags
        self.train = kwargs.get("train", True)
        self.train_only = kwargs.get("train_only", False)
        self.predict = kwargs.get("predict", True)
        self.fresh_start = kwargs.get("fresh_start", False)
        self.use_previous_model = kwargs.get("use_previous_model", True)

        # Device configuration
        self.device = kwargs.get("device", "auto")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Paths configuration
        self.model_dir = kwargs.get("model_dir", "Model")
        self.training_data_dir = kwargs.get("training_data_dir", "training_data")

        # Debug configuration
        self.debug = kwargs.get("debug", False)

    @classmethod
    def from_file(cls, config_path: str) -> "DBNNConfig":
        """Create configuration from JSON file"""
        with open(config_path) as f:
            config_data = json.load(f)
        return cls(**config_data)


class DBNN(GPUDBNN):
    """Enhanced DBNN class that builds on GPUDBNN implementation"""

    def __init__(
        self,
        config: Optional[Union[DBNNConfig, dict]] = None,
        dataset_name: Optional[str] = None,
        mode=None,
        model_type: Optional[str] = None,
    ):
        """
        Initialize DBNN with configuration

        Args:
            config: DBNNConfig object or dictionary of parameters
            dataset_name: Name of the dataset (optional)
        """
        # Initialize configuration
        if config is None:
            config = DBNNConfig()
        elif isinstance(config, dict):
            config = DBNNConfig(**config)
        if mode is None:
            self.mode = None
        else:
            self.mode = mode

        # First load the dataset configuration
        self.data_config = (
            DatasetConfig.load_config(dataset_name) if dataset_name else None
        )
        # Metadata storage (CPU only)
        self._metadata = {
            "sample_ids": [],  # Original dataset indices
            "file_paths": [],  # Paths for image/data files
            "class_names": [],  # String class labels
            "feature_names": [],  # Column/feature names
            "aux_data": {},  # Other non-computational data
        }
        # GPU Tensors (computational only)
        self._gpu_tensors = {"features": None, "targets": None, "weights": None}
        # Map DBNNConfig to GPUDBNN parameters
        super().__init__(
            dataset_name=dataset_name,
            learning_rate=config.learning_rate,
            max_epochs=config.epochs,
            test_size=config.test_fraction,
            random_state=config.random_seed,
            fresh=config.fresh_start,
            use_previous_model=config.use_previous_model,
            model_type=model_type
            if model_type is not None
            else config.model_type,  # Pass model type from config
            mode=self.mode,
        )
        self.cardinality_threshold = self.config.get("training_params", {}).get(
            "cardinality_threshold", 0.9
        )

        # Store model configuration
        self.model_config = config
        self.training_log = pd.DataFrame()
        self.save_plots = self.config.get("training_params", {}).get(
            "save_plots", False
        )

        # Add new attributes to track the best round
        self.best_round = None  # Track the best round number
        self.best_round_initial_conditions = (
            None  # Save initial conditions of the best round
        )
        self.best_combined_accuracy = 0.00
        self.best_model_weights = None
        self.data = None
        self.global_mean = None  # Store global mean
        self.global_std = None  # Store global standard deviation
        self.global_stats_computed = False  # Flag to track if stats are computed

        # Validate dataset_name
        if not dataset_name or not isinstance(dataset_name, str):
            raise ValueError(
                "Invalid dataset_name provided. Must be a non-empty string."
            )

        # Load configuration
        self.config = DatasetConfig.load_config(dataset_name)
        if self.config is None:
            raise ValueError(
                f"Failed to load configuration for dataset: {dataset_name}"
            )

        # Initialize other attributes
        self.target_column = self.config["target_column"]
        self.batch_size = self.config.get("batch_size", 128)
        self.invertible_model = None
        # Preprocess data once during initialization
        self._is_preprocessed = False  # Flag to track preprocessing
        self._preprocess_and_split_data()  # Call preprocessing only once

    def compute_global_statistics(self, X: pd.DataFrame):
        """Compute global statistics (e.g., mean, std) for normalization."""
        batch_size = self.batch_size  # Adjust based on available memory
        n_samples = len(X)
        n_features = X.shape[1]

        # Initialize accumulators
        sum_features = np.zeros(n_features)
        sum_squared_features = np.zeros(n_features)

        # Compute sum and sum of squares in batches
        for i in range(0, n_samples, batch_size):
            batch_X = X.iloc[i : i + batch_size]
            batch_X_numeric = batch_X.select_dtypes(
                include=[np.number]
            )  # Only numeric features
            sum_features += batch_X_numeric.sum(axis=0)
            sum_squared_features += (batch_X_numeric**2).sum(axis=0)

        # Compute mean and standard deviation
        self.global_mean = sum_features / n_samples
        self.global_std = np.sqrt(
            (sum_squared_features / n_samples) - (self.global_mean**2)
        )

        # Handle zero standard deviation (replace with 1 to avoid division by zero)
        self.global_std[self.global_std == 0] = 1.0

    def _preprocess_and_split_data(self):
        """Preprocess data and split into training and testing sets."""
        # Load dataset
        self.data = self._load_dataset()

        # Preprocess features and target
        predict_mode = True if self.mode == "predict" else False
        # Load and preprocess data
        X = (
            self.data.drop(columns=[self.target_column])
            if not predict_mode
            else self.data.copy()
        )
        y = (
            self.data[self.target_column]
            if not predict_mode
            else pd.Series([-99999] * len(self.data))
        )

        # Compute global statistics for normalization
        self.compute_global_statistics(X)

        # Encode labels if not already done
        if not hasattr(self.label_encoder, "classes_"):
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)

        # Preprocess features
        X_processed = self._preprocess_data(X, is_training=True)

        # Convert to tensors on CPU first, then move to device
        self.X_tensor = X_processed.clone().detach().to(self.device)
        self.y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(self.device)

        # Split data into training and testing sets
        # Split data (use all data as "test" in prediction mode)
        if predict_mode:
            self.X_train, self.X_test = None, self.X_tensor
            self.y_train, self.y_test = None, self.y_tensor
            self.train_indices, self.test_indices = [], list(range(len(self.data)))
        else:
            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
            ) = self._get_train_test_split(self.X_tensor, self.y_tensor)

        self._is_preprocessed = True  # Mark preprocessing as complete

    def create_invertible_model(
        self, reconstruction_weight: float = 0.5, feedback_strength: float = 0.3
    ):
        """Create an invertible DBNN model"""
        if self.invertible_model is None:
            self.invertible_model = InvertibleDBNN(
                forward_model=self,
                feature_dims=self.data.shape[1] - 1,  # Exclude target column
                reconstruction_weight=reconstruction_weight,
                feedback_strength=feedback_strength,
            )
        return self.invertible_model

    def process_dataset(self, config_path: str) -> Dict:
        """
        Process dataset according to configuration file specifications

        Args:
            config_path: Path to JSON configuration file

        Returns:
            Dictionary containing processing results
        """
        # Load and validate configuration
        try:
            with open(config_path) as f:
                config_text = f.read()
            # Remove comments starting with _comment
            config_lines = [
                line for line in config_text.split("\n") if not '"_comment"' in line
            ]
            clean_config = "\n".join(config_lines)

            self.data_config = json.loads(clean_config)
        except Exception as e:
            raise ValueError(f"Error reading configuration file: {str(e)}")
        dataset_name = os.path.splitext(os.path.basename(config_path))[0]
        # Ensure file_path is set
        if not self.data_config.get("file_path"):
            default_path = os.path.join("data", dataset_name, f"{dataset_name}.csv")
            if os.path.exists(default_path):
                self.data_config["file_path"] = default_path
                print("\033[K" + f"Using default data file: {default_path}")
            else:
                raise ValueError(f"No data file found for {dataset_name}")
        # Load or create the configuration file and update global variables
        config = load_or_create_config(
            config_path=f"data/{dataset_name}/{dataset_name}.conf"
        )
        # Update global variables based on the configuration file
        global Train_device, bin_sizes, n_bins_per_dim, Trials, cardinality_threshold, cardinality_tolerance, LearningRate, TrainingRandomSeed, Epochs, TestFraction, Train, Train_only, Predict, Gen_Samples, EnableAdaptive, nokbd, display
        Train_device = config.get("compute_device", Train_device)
        print("" f"The train device is set as {Train_device}")
        self.device = Train_device
        Trials = config.get("trials", Trials)
        cardinality_threshold = config.get(
            "cardinality_threshold", cardinality_threshold
        )
        cardinality_tolerance = config.get(
            "cardinality_tolerance", cardinality_tolerance
        )
        bin_sizes = config.get("bin_sizes", 128)
        n_bins_per_dim = config.get("n_bins_per_dim", 128)
        LearningRate = config.get("learning_rate", LearningRate)
        TrainingRandomSeed = config.get("random_seed", TrainingRandomSeed)
        Epochs = config.get("epochs", Epochs)
        TestFraction = config.get("test_fraction", TestFraction)
        Train = config.get("train", Train)
        Train_only = config.get("train_only", Train_only)
        Predict = config.get("predict", Predict)
        Gen_Samples = config.get("gen_samples", Gen_Samples)
        EnableAdaptive = config.get("enable_adaptive", EnableAdaptive)
        nokbd = config.get("nokbd", nokbd)
        display = config.get("display", display)

        # Convert dictionary config to DBNNConfig object
        config_params = {
            "epochs": self.data_config.get("training_params", {}).get("epochs", Epochs),
            "learning_rate": self.data_config.get("training_params", {}).get(
                "learning_rate", LearningRate
            ),
            "model_type": self.data_config.get("modelType", "Histogram"),
            "enable_adaptive": self.data_config.get("training_params", {}).get(
                "enable_adaptive", EnableAdaptive
            ),
            "batch_size": self.data_config.get("training_params", {}).get(
                "batch_size", 128
            ),
            "training_data_dir": self.data_config.get("training_params", {}).get(
                "training_save_path", "training_data"
            ),
        }
        self.model_config = DBNNConfig(**config_params)

        # Create output directory structure
        dataset_name = os.path.splitext(
            os.path.basename(self.data_config["file_path"])
        )[0]
        output_dir = os.path.join(self.model_config.training_data_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # Update dataset name
        self.dataset_name = dataset_name

        # Load data using existing GPUDBNN method
        self.data = self._load_dataset()
        self.X_Orig = self.Original_data.drop(
            columns=[self.data_config["target_column"]]
        )
        self.batch_size = self._calculate_optimal_batch_size(len(self.data))
        # Add row tracking
        self.data["original_index"] = range(len(self.data))

        # Extract features and target

        if "target_column" not in self.data_config:
            self.data_config["target_column"] = "target"  # Set default target column
            print("\033[K" + f"Using default target column: 'target'")

        X = self.data.drop(columns=[self.data_config["target_column"]])
        y = self.data[self.data_config["target_column"]]

        # Initialize training log
        log_file = os.path.join(output_dir, f"{dataset_name}_log.csv")
        self.training_log = pd.DataFrame(
            columns=[
                "timestamp",
                "round",
                "train_size",
                "test_size",
                "train_loss",
                "test_loss",
                "train_accuracy",
                "test_accuracy",
                "training_time",
            ]
        )

        # Train model using existing GPUDBNN methods
        if self.model_config.enable_adaptive:
            results = self.adaptive_fit_predict(max_rounds=self.model_config.epochs)
        else:
            results = self.fit_predict()

        # Preprocess features and convert to tensors
        X_tensor = self._preprocess_data(X, is_training=True)  # Preprocess data
        y_tensor = torch.tensor(self.label_encoder.transform(y), dtype=torch.long).to(
            self.device
        )  # Encode labels

        # Generate predictions and true labels
        predictions, posterior = self.predict(X_tensor, batch_size=self.batch_size)
        true_labels = y_tensor.cpu().numpy()

        # Generate detailed predictions
        # predictions_df = self._generate_detailed_predictions(X_tensor, predictions, true_labels)
        predictions_df = self._generate_detailed_predictions(
            self.X_Orig, predictions, true_labels, posterior
        )

        # Save results
        results_path = os.path.join(output_dir, f"{dataset_name}_predictions.csv")
        predictions_df.to_csv(results_path, index=False)

        # Save training log
        self.training_log.to_csv(log_file, index=False)

        # Count number of features actually used (excluding high cardinality and excluded features)
        n_features = len(X.columns)
        n_excluded = len(getattr(self, "high_cardinality_columns", []))

        return {
            "results_path": results_path,
            "log_path": log_file,
            "n_samples": len(self.data),
            "n_features": n_features,
            "n_excluded": n_excluded,
            "training_results": results,
        }

    def _generate_detailed_predictions(
        self,
        X_orig: Union[pd.DataFrame, torch.Tensor],
        predictions: Union[torch.Tensor, np.ndarray],
        true_labels: Union[torch.Tensor, np.ndarray, pd.Series, List, None] = None,
        posteriors: Union[torch.Tensor, np.ndarray, None] = None,
    ) -> pd.DataFrame:
        """
        Robust predictions generator that preserves original feature values.

        Args:
            X_orig: Original input features (before preprocessing)
            predictions: Model predictions (numeric or string)
            true_labels: True labels if available (numeric or string)

        Returns:
            DataFrame with predictions and original features
        """
        # Convert predictions to numpy if they're tensors
        predictions_np = (
            predictions.cpu().numpy()
            if torch.is_tensor(predictions)
            else np.array(predictions)
        )
        # Convert posteriors if provided
        posteriors_np = (
            posteriors.cpu().numpy()
            if torch.is_tensor(posteriors)
            else np.array(posteriors)
            if posteriors is not None
            else None
        )
        pred_classes = predictions.cpu().numpy()

        # Create results DataFrame from original features
        if isinstance(X_orig, pd.DataFrame):
            results_df = X_orig.copy()
        else:
            # Handle tensor/numpy array input
            X_orig_np = (
                X_orig.cpu().numpy() if torch.is_tensor(X_orig) else np.array(X_orig)
            )
            results_df = pd.DataFrame(
                X_orig_np,
                columns=getattr(
                    self,
                    "feature_columns",
                    [f"feature_{i}" for i in range(X_orig_np.shape[1])],
                ),
            )

        # true_classes=results_df['true_class']
        # Add predictions with label decoding if possible
        if hasattr(self, "label_encoder") and hasattr(self.label_encoder, "classes_"):
            try:
                # Add predictions and confidence (max probability)
                results_df["predicted_class"] = self.label_encoder.inverse_transform(
                    pred_classes
                )

                # Add posteriors if available
                if posteriors_np is not None:
                    for i, class_name in enumerate(self.label_encoder.classes_):
                        results_df[f"prob_{class_name}"] = posteriors_np[:, i]
                results_df["prediction_confidence"] = posteriors_np[
                    np.arange(len(pred_classes)), pred_classes
                ]

            except ValueError as e:
                print(f"Note: Using raw predictions - {str(e)}")
                results_df["predicted_class"] = predictions_np
                results_df["prediction_confidence"] = posteriors_np[
                    np.arange(len(pred_classes)), pred_classes
                ]
        else:
            results_df["predicted_class"] = predictions_np
            results_df["prediction_confidence"] = posteriors_np[
                np.arange(len(pred_classes)), pred_classes
            ]

        # Handle true labels if provided
        if true_labels is not None:
            true_labels_np = (
                true_labels.cpu().numpy()
                if torch.is_tensor(true_labels)
                else true_labels.to_numpy()
                if isinstance(true_labels, (pd.Series, pd.DataFrame))
                else np.array(true_labels)
            )

            # Only try to decode if we have string labels and an encoder
            if (
                isinstance(true_labels_np.flat[0], str)
                and hasattr(self, "label_encoder")
                and hasattr(self.label_encoder, "classes_")
            ):
                try:
                    results_df["true_class"] = true_labels_np  # Keep original strings
                except Exception as e:
                    print(f"Couldn't preserve true labels: {str(e)}")
            else:
                # results_df['true_class'] = true_labels_np
                results_df["true_class"] = self.label_encoder.inverse_transform(
                    true_labels_np
                )
                # Ensure predicted_class uses string labels
                results_df["predicted_class"] = self.label_encoder.inverse_transform(
                    pred_classes
                )

        return results_df

    def _update_training_log(self, round_num: int, metrics: Dict):
        """Update training log with current metrics"""
        self.training_log = self.training_log.append(
            {
                "timestamp": pd.Timestamp.now(),
                "round": round_num,
                "train_size": metrics["train_size"],
                "test_size": metrics["test_size"],
                "train_loss": metrics["train_loss"],
                "test_loss": metrics["test_loss"],
                "train_accuracy": metrics["train_accuracy"],
                "test_accuracy": metrics["test_accuracy"],
                "training_time": metrics["training_time"],
            },
            ignore_index=True,
        )

    # --------------------------------------------------------------Class Ends ------------------------------------
    def prepare_batch(self, features):
        """Efficient batch preparation"""
        if not features.is_contiguous():
            features = features.contiguous()

        if self.device.startswith("cuda") and not features.is_cuda:
            features = features.cuda(non_blocking=True)

        return features

    def _load_dataset(self) -> pd.DataFrame:
        """Optimized dataset loader with GPU memory management"""
        DEBUG.log(f"Loading dataset: {self.dataset_name}")

        try:
            # Config validation (unchanged)
            if not self.config:
                raise ValueError(f"No config for dataset: {self.dataset_name}")

            file_path = self.config.get("file_path")
            if not file_path:
                raise ValueError("No file path in config")

            # Load data (unchanged file handling)
            if file_path.startswith(("http://", "https://")):
                df = pd.read_csv(
                    StringIO(requests.get(file_path).text),
                    sep=self.config.get("separator", ","),
                    header=0 if self.config.get("has_header", True) else None,
                    low_memory=False,
                )
            else:
                df = pd.read_csv(
                    file_path,
                    sep=self.config.get("separator", ","),
                    header=0 if self.config.get("has_header", True) else None,
                    low_memory=False,
                )
            predict_mode = True if self.mode == "predict" else False
            # Handle target column validation
            if predict_mode and self.target_column in df.columns:
                if not self._validate_target_column(df[self.target_column]):
                    print(
                        f"\033[K"
                        + f"{Colors.RED}The predict mode is {predict_mode} and target column is invalid. We will ignore it{Colors.ENDC}"
                    )
                    # Get the current column names
                    column_names = df.columns.tolist()
                    # Find the index of the target column
                    try:
                        index = column_names.index(self.target_column)
                        # Update the name
                        column_names[index] = "dummy_target"
                        # Assign the updated list back to columns
                        df.columns = column_names
                        # Update the target_column reference
                        self.target_column = None
                    except ValueError:
                        print(
                            f"\033[K"
                            + f"Warning: Target column '{self.target_column}' not found in dataset columns: {column_names}"
                        )
                        # If target column isn't found, just proceed without renaming

            # Store original data (CPU only)
            self.Original_data = df.copy()  # This is the line that was missing
            # Get prediction mode from config (not global variable)

            # Handle prediction mode (target column may not exist)
            if predict_mode and self.target_column not in df.columns:
                DEBUG.log(
                    f"Prediction mode - target column '{self.target_column}' not found"
                )
                self.X_Orig = df.copy()  # Use all columns for prediction
            else:
                # Training mode - ensure target column exists
                if self.target_column not in df.columns:
                    raise ValueError(
                        f"Target column '{self.target_column}' not found in dataset"
                    )
                self.X_Orig = df.drop(columns=[self.target_column]).copy()

            # Filter features if specified
            if "column_names" in self.config:
                df = _filter_features_from_config(df, self.config)

            # Handle target column
            target_col = self.config["target_column"]
            if isinstance(target_col, int):
                target_col = df.columns[target_col]
                self.config["target_column"] = target_col

            if not predict_mode and target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found")

            # Shuffling logic (CPU only)
            shuffle_path = os.path.join(
                self.config.get("training_params", {}).get(
                    "training_save_path", "training_data"
                ),
                os.path.splitext(os.path.basename(self.dataset_name))[0],
                "data",
                "shuffled_data.csv",
            )

            if self.fresh_start and self.random_state == -1:
                for _ in range(3):
                    df = df.sample(frac=1).reset_index(drop=True)
                os.makedirs(os.path.dirname(shuffle_path), exist_ok=True)
                df.to_csv(shuffle_path, index=False)
            elif os.path.exists(shuffle_path):
                df = pd.read_csv(shuffle_path)

            # Convert to memory-efficient format
            df = df.astype(
                {
                    col: "category" if df[col].dtype == "object" else df[col].dtype
                    for col in df.columns
                }
            )

            # Store metadata separately (never moves to GPU)
            self._metadata = {
                "file_path": file_path,
                "column_names": list(df.columns),
                "target_column": target_col,
                "original_index": df.index.values,
                "shuffle_path": shuffle_path,
            }

            # Convert target column to string once
            if self.target_column in df.columns:
                df[self.target_column] = df[self.target_column].apply(
                    str
                )  # Force string

            return df

        except Exception as e:
            DEBUG.log(f"Dataset load error: {str(e)}")
            raise RuntimeError(f"Failed to load {self.dataset_name}: {str(e)}")

    def _compute_batch_posterior(self, features: torch.Tensor, epsilon: float = 1e-10):
        """Optimized batch posterior with vectorized operations"""
        # Ensure input features are on the correct device
        features = features.to(self.device)

        # Safety checks
        if self.weight_updater is None:
            DEBUG.log(" Weight updater not initialized, initializing now...")
            self._initialize_bin_weights()
            if self.weight_updater is None:
                raise RuntimeError("Failed to initialize weight updater")

        if self.likelihood_params is None:
            raise RuntimeError("Likelihood parameters not initialized")

        batch_size = features.shape[0]
        n_classes = len(self.likelihood_params["classes"])

        # Pre-allocate tensors on the correct device
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        # Process all feature pairs at once
        feature_groups = torch.stack(
            [
                features[:, pair]
                .contiguous()
                .to(self.device)  # Ensure on correct device
                for pair in self.likelihood_params["feature_pairs"]
            ]
        ).transpose(
            0, 1
        )  # [batch_size, n_pairs, 2]

        # Compute all bin indices at once
        bin_indices_dict = {}
        for group_idx in range(len(self.likelihood_params["feature_pairs"])):
            bin_edges = self.likelihood_params["bin_edges"][group_idx]
            edges = torch.stack(
                [edge.contiguous().to(self.device) for edge in bin_edges]
            )  # Ensure on correct device

            # Vectorized binning with contiguous tensors
            indices = torch.stack(
                [
                    torch.bucketize(
                        feature_groups[:, group_idx, dim].contiguous(),
                        edges[dim].contiguous(),
                    )
                    .sub_(1)
                    .clamp_(0, self.n_bins_per_dim - 1)
                    for dim in range(2)
                ]
            )  # Shape: (2, batch_size)
            bin_indices_dict[group_idx] = indices

        # Add progress bar for feature pair processing
        n_pairs = len(self.likelihood_params["feature_pairs"])
        with tqdm(total=n_pairs, desc="Processing feature pairs", leave=False) as pbar:
            # Process all classes simultaneously
            for group_idx in range(n_pairs):
                bin_probs = self.likelihood_params["bin_probs"][
                    group_idx
                ]  # [n_classes, n_bins, n_bins]
                indices = bin_indices_dict[group_idx]  # [2, batch_size]

                # Get all weights at once
                weights = torch.stack(
                    [
                        self.weight_updater.get_histogram_weights(c, group_idx)
                        for c in range(n_classes)
                    ]
                )  # [n_classes, n_bins, n_bins]

                # Ensure weights are contiguous
                if not weights.is_contiguous():
                    weights = weights.contiguous()

                # Apply weights to probabilities
                weighted_probs = bin_probs * weights  # [n_classes, n_bins, n_bins]

                # Gather probabilities for all samples and classes at once
                probs = weighted_probs[
                    :, indices[0], indices[1]
                ]  # [n_classes, batch_size]
                log_likelihoods += torch.log(probs.t() + epsilon)

                pbar.update(1)  # Update progress bar

        # Compute posteriors efficiently
        max_log_likelihood = log_likelihoods.max(dim=1, keepdim=True)[0]
        posteriors = torch.exp(log_likelihoods - max_log_likelihood)
        posteriors /= posteriors.sum(dim=1, keepdim=True) + epsilon

        return posteriors, bin_indices_dict if self.model_type == "Histogram" else None

    # ----------------------

    def set_feature_bounds(self, dataset):
        """Initialize global feature bounds from complete dataset"""
        if self.feature_bounds is None:
            self.feature_bounds = {}
            numerical_columns = dataset.select_dtypes(
                include=["int64", "float64"]
            ).columns

            for feat_name in numerical_columns:
                feat_data = dataset[feat_name]
                min_val = feat_data.min()
                max_val = feat_data.max()
                padding = (max_val - min_val) * 0.01
                self.feature_bounds[feat_name] = {
                    "min": min_val - padding,
                    "max": max_val + padding,
                }

    def _clean_existing_model(self):
        """Remove existing model files for a fresh start"""
        try:
            files_to_remove = [
                self._get_weights_filename(),
                self._get_encoders_filename(),
                self._get_model_components_filename(),
            ]
            for file in files_to_remove:
                if os.path.exists(file):
                    os.remove(file)
                    print("\033[K" + f"Removed existing model file: {file}")
        except Exception as e:
            print("\033[K" + f"Warning: Error cleaning model files: {str(e)}")

    # ------------------------------------------Adaptive Learning--------------------------------------
    def save_epoch_data(self, epoch: int, train_indices: list, test_indices: list):
        """
        Save training and testing indices for each epoch if enabled in config
        """
        # Check if epoch saving is enabled
        save_epochs = self.config.get("training_params", {}).get(
            "Save_training_epochs", False
        )
        if not save_epochs:
            return

        # Use dataset name as subfolder
        dataset_folder = os.path.splitext(os.path.basename(self.dataset_name))[0]
        base_path = self.config.get("training_params", {}).get(
            "training_save_path", "training_data"
        )
        save_path = os.path.join(base_path, dataset_folder)

        # Create epoch directory
        epoch_dir = os.path.join(save_path, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Save indices
        try:
            with open(
                os.path.join(epoch_dir, f"{self.model_type}_train_indices.pkl"), "wb"
            ) as f:
                pickle.dump(train_indices, f)
            with open(
                os.path.join(epoch_dir, f"{self.model_type}_test_indices.pkl"), "wb"
            ) as f:
                pickle.dump(test_indices, f)
            # print("\033[K" +f"Saved epoch {epoch} data to {epoch_dir}")
        except Exception as e:
            print("\033[K" + f"Error saving epoch data: {str(e)}")

    def load_epoch_data(self, epoch: int):
        """
        Load training and testing indices for a specific epoch
        """
        epoch_dir = os.path.join(self.base_save_path, f"epoch_{epoch}")

        with open(
            os.path.join(epoch_dir, f"{self.model_type}_train_indices.pkl"), "rb"
        ) as f:
            train_indices = pickle.load(f)
        with open(
            os.path.join(epoch_dir, f"{self.model_type}_test_indices.pkl"), "rb"
        ) as f:
            test_indices = pickle.load(f)

        return train_indices, test_indices

    def _compute_cardinality_threshold(self, cardinalities, config=None):
        """
        Compute the cardinality threshold using percentile-based analysis.

        Args:
            cardinalities: List of cardinality values
            config: Configuration dictionary with threshold parameters

        Returns:
            float: Computed cardinality threshold
        """
        # Get active learning parameters from config
        active_learning_config = self.config.get("active_learning", {})
        percentile = active_learning_config.get("cardinality_threshold_percentile", 95)

        # If no cardinalities provided, create a synthetic distribution
        if not cardinalities:
            print(
                "\033[K"
                + "Warning: No cardinality data available. Using synthetic distribution based on percentile."
            )
            # Create a synthetic distribution around the percentile threshold
            cardinalities = np.array([1.0, 2.0, 5.0, 10.0, 20.0])  # Synthetic values
        else:
            cardinalities = np.array(cardinalities)

        # Compute basic statistics
        min_card = np.min(cardinalities)
        max_card = np.max(cardinalities)
        mean_card = np.mean(cardinalities)
        median_card = np.median(cardinalities)

        # Compute threshold using percentile
        threshold = np.percentile(cardinalities, percentile)

        # Print detailed analysis
        print("\033[K" + f"Cardinality Analysis:")
        print("\033[K" + f"- Using {percentile}th percentile threshold")
        print("\033[K" + f"- Distribution statistics:")
        print("\033[K" + f"  - Min: {min_card:.2f}")
        print("\033[K" + f"  - Max: {max_card:.2f}")
        print("\033[K" + f"  - Mean: {mean_card:.2f}")
        print("\033[K" + f"  - Median: {median_card:.2f}")
        print("\033[K" + f"  - Threshold: {threshold:.2f}")

        # Print number of samples that would be included
        n_included = sum(c <= threshold for c in cardinalities)
        print(
            "\033[K"
            + f"- {n_included} out of {len(cardinalities)} samples below threshold "
            f"({(n_included/len(cardinalities))*100:.1f}%)"
        )

        return threshold

    # --------------------compute sample divergence -----------------------
    def _compute_sample_divergence(
        self, sample_data: torch.Tensor, feature_pairs: List[Tuple]
    ) -> torch.Tensor:
        """Optimized divergence computation with batched feature processing"""
        device = sample_data.device
        n_samples = sample_data.shape[0]

        if n_samples <= 1:
            return torch.zeros((1, 1), device=device)

        # Use mixed precision with memory-efficient accumulation
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        distances = torch.zeros(
            (n_samples, n_samples), device=device, dtype=torch.float32
        )

        # Process feature pairs in memory-optimized batches
        batch_size = self._get_feature_batch_size(n_samples, len(feature_pairs))

        with torch.no_grad():
            for i in range(0, len(feature_pairs), batch_size):
                batch_pairs = feature_pairs[i : i + batch_size]

                # Get batch data in reduced precision
                batch_data = sample_data[:, batch_pairs].to(dtype)

                # Vectorized pairwise distance calculation
                diff = batch_data.unsqueeze(1) - batch_data.unsqueeze(
                    0
                )  # [n_samples, n_samples, batch_size, 2]
                batch_dist = torch.norm(
                    diff, p=2, dim=-1
                )  # [n_samples, n_samples, batch_size]

                # Accumulate with proper type casting
                distances += batch_dist.mean(dim=-1).to(torch.float32)

                # Explicit memory cleanup
                del batch_data, diff, batch_dist
                torch.cuda.empty_cache()

        # Normalize while maintaining numerical stability
        distances /= len(feature_pairs)
        max_val = distances.max()
        return distances / (max_val + 1e-7) if max_val > 0 else distances

    def _get_feature_batch_size(self, n_samples, n_pairs):
        """Dynamically determine optimal feature batch size based on available memory"""
        if not torch.cuda.is_available():
            return 100  # Default for CPU

        # Calculate available memory with safety margin
        free_mem = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        safety_margin = 0.8  # Use only 80% of available memory
        available_mem = free_mem * safety_margin

        # Memory estimation per feature pair batch
        per_pair_mem = n_samples * n_samples * 4  # 4 bytes per float32
        max_batches = available_mem // per_pair_mem

        return min(max(1, int(max_batches)), 256)  # Limit between 1-256

    # ---------------------------------------------------------------------------------------

    def _compute_feature_cardinalities(
        self, samples_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized computation of feature cardinalities.
        """
        cardinalities = torch.zeros(len(samples_data), device=self.device)

        # Process feature pairs in batches
        batch_size = self.batch_size  # Adjust based on memory constraints
        for i in range(0, len(samples_data), batch_size):
            batch_end = min(i + batch_size, len(samples_data))
            batch_data = samples_data[i:batch_end]

            # Compute cardinalities for each feature pair
            batch_cardinalities = torch.zeros(batch_end - i, device=self.device)
            for feat_pair in self.feature_pairs:
                pair_data = batch_data[:, feat_pair]
                # Compute unique values efficiently
                _, counts = torch.unique(pair_data, dim=0, return_counts=True)
                batch_cardinalities += len(counts)

            cardinalities[i:batch_end] = batch_cardinalities

        return cardinalities

    def _calculate_optimal_batch_size(self, sample_tensor_size):
        """Calculate optimal batch size with more conservative estimates"""
        if not torch.cuda.is_available():
            return 128  # Default for CPU

        try:
            # Get memory stats
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            free_memory = total_memory - allocated - reserved

            # Be more conservative - use only 30% of free memory
            available_memory = free_memory * 0.3

            # Calculate memory needed per sample (with safety factor)
            memory_per_sample = sample_tensor_size * 8  # More conservative estimate

            # Calculate optimal batch size
            optimal_batch_size = int(available_memory / memory_per_sample)

            # Enforce reasonable bounds
            optimal_batch_size = max(
                32, min(optimal_batch_size, 2048)
            )  # Reduced max from 4096

            DEBUG.log(f"Memory Analysis:")
            DEBUG.log(f"- Total GPU Memory: {total_memory / 1e9:.2f} GB")
            DEBUG.log(f"- Free Memory: {free_memory / 1e9:.2f} GB")
            DEBUG.log(f"- Available for batch: {available_memory / 1e9:.2f} GB")
            DEBUG.log(f"- Memory per sample: {memory_per_sample / 1e6:.2f} MB")
            print(
                f"{Colors.GREEN}The new Batch size is dynamically set to {optimal_batch_size}{Colors.ENDC}"
            )

            return optimal_batch_size

        except Exception as e:
            print(f"{Colors.RED}Error calculating batch size: {str(e)}{Colors.ENDC}")
            return 128  # Fallback value

    def _select_samples_from_failed_classes(
        self, test_predictions, y_test, test_indices, results
    ):
        """Cluster-based selection with device-aware processing"""
        from tqdm import tqdm

        # Configuration parameters
        active_learning_config = self.config.get("active_learning", {})
        min_divergence = active_learning_config.get("min_divergence", 0.1)
        max_class_addition_percent = active_learning_config.get(
            "max_class_addition_percent", 99
        )

        # Convert inputs to tensors on active device
        test_predictions = torch.as_tensor(test_predictions, device=self.device)
        y_test = torch.as_tensor(y_test, device=self.device)
        test_indices = torch.as_tensor(test_indices, device=self.device)

        all_results = results["all_predictions"]
        test_results = all_results.iloc[self.test_indices]

        # Create boolean mask using numpy arrays to avoid chained indexing
        misclassified_mask = (
            test_results["predicted_class"].to_numpy()
            != test_results["true_class"].to_numpy()
        )
        misclassified_indices = test_results.index[misclassified_mask].tolist()

        # Create mapping from original indices to test set positions
        test_pos_map = {idx: pos for pos, idx in enumerate(self.test_indices)}

        final_selected_indices = []
        unique_classes = test_results["true_class"].unique()

        # Class processing progress bar
        class_pbar = tqdm(
            unique_classes, desc="Processing classes", leave=False, position=0
        )

        for class_id in class_pbar:
            class_pbar.set_postfix_str(f"Class {class_id}")

            # Convert string class label to encoded integer
            encoded_class_id = self.label_encoder.transform([class_id])[0]

            # Get class-specific misclassified indices using proper boolean indexing
            class_mask = (
                test_results.loc[misclassified_indices, "true_class"] == class_id
            ).to_numpy()
            class_indices = np.array(misclassified_indices)[class_mask].tolist()

            # Convert original indices to test set positions
            class_positions = [
                test_pos_map[idx] for idx in class_indices if idx in test_pos_map
            ]
            if not class_positions:
                continue

            # Convert to tensor with proper dtype
            class_pos_tensor = torch.tensor(
                class_positions, dtype=torch.long, device=self.device
            )

            # Batch processing
            samples, margins, indices = [], [], []
            batch_pbar = tqdm(
                total=len(class_positions),
                desc=f"Class {class_id} batches",
                leave=False,
                position=1,
                unit="sample",
            )

            for batch_start in range(0, len(class_positions), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(class_positions))
                batch_pos = class_pos_tensor[batch_start:batch_end]

                # Get actual data indices
                batch_indices = test_indices[batch_pos]
                batch_X = self.X_tensor[batch_indices]

                # Compute posteriors using encoded class ID
                if self.model_type == "Histogram":
                    posteriors, _ = self._compute_batch_posterior(batch_X)
                else:
                    posteriors, _ = self._compute_batch_posterior_std(batch_X)

                # Device-stable calculations using encoded class ID
                max_probs, _ = torch.max(posteriors, dim=1)
                true_probs = posteriors[:, encoded_class_id]  # Use encoded class ID
                batch_margins = max_probs - true_probs

                samples.append(batch_X)
                margins.append(batch_margins)
                indices.append(batch_indices)
                batch_pbar.update(len(batch_pos))

            batch_pbar.close()

            if not samples:
                continue

            try:
                # Concatenate tensors properly
                margins = torch.cat(margins)
                indices = torch.cat(indices)
                samples = torch.cat(samples)
            except RuntimeError:
                continue

            if indices.numel() == 0:
                continue

            # --- Threshold Logic ---
            class_max_posterior = torch.max(margins)
            class_min_posterior = torch.min(margins)

            strong_threshold = class_max_posterior - active_learning_config.get(
                "strong_margin_threshold", 0.01
            )
            marginal_threshold = class_min_posterior + active_learning_config.get(
                "marginal_margin_threshold", 0.01
            )

            strong_mask = margins >= strong_threshold
            marginal_mask = margins <= marginal_threshold
            combined_mask = strong_mask | marginal_mask

            eligible_indices = indices[combined_mask]

            # --- Cluster Processing ---
            mandatory_indices = indices[
                torch.topk(margins, k=min(2, len(margins))).indices
            ]

            torch.cat([mandatory_indices, eligible_indices]).unique()
            remaining_mask = ~torch.isin(indices, mandatory_indices)
            candidate_samples = samples[remaining_mask]

            if candidate_samples.numel() > 0:
                div_matrix = self._compute_sample_divergence(
                    candidate_samples, self.feature_pairs
                )
                visited = torch.zeros(
                    len(candidate_samples), dtype=torch.bool, device=self.device
                )
                cluster_indices = []

                for i in range(len(candidate_samples)):
                    if not visited[i]:
                        cluster_mask = div_matrix[i] < min_divergence
                        cluster_members = torch.where(cluster_mask)[0]
                        if cluster_members.numel() > 0:
                            cluster_indices.append(
                                indices[remaining_mask][cluster_members[0]]
                            )
                            visited[cluster_members] = True

                if cluster_indices:
                    selected = torch.cat(
                        [mandatory_indices, torch.stack(cluster_indices)]
                    ).unique()
                else:
                    selected = mandatory_indices
            else:
                selected = mandatory_indices

            # Final selection with encoded class count check
            class_count = (
                (y_test == encoded_class_id).sum().item()
            )  # Use encoded class ID
            max_samples = max(2, int(class_count * max_class_addition_percent / 100))
            final_selected_indices.extend(selected[:max_samples].cpu().tolist())

        class_pbar.close()
        return final_selected_indices

    def _save_reconstruction_plots(
        self,
        original_features: np.ndarray,
        reconstructed_features: np.ndarray,
        true_labels: np.ndarray,
        save_path: str,
    ):
        """Generate visualization plots for reconstruction analysis"""
        plt.figure(figsize=(15, 5))

        # Feature-wise reconstruction error
        plt.subplot(131)
        errors = np.mean((original_features - reconstructed_features) ** 2, axis=0)
        plt.bar(range(len(errors)), errors)
        plt.title("Feature-wise Reconstruction Error")
        plt.xlabel("Feature Index")
        plt.ylabel("MSE")

        # Class-wise reconstruction quality
        plt.subplot(132)
        unique_classes = np.unique(true_labels)
        class_errors = []
        for class_label in unique_classes:
            mask = true_labels == class_label
            error = np.mean(
                (original_features[mask] - reconstructed_features[mask]) ** 2
            )
            class_errors.append(error)

        plt.bar(unique_classes, class_errors)
        plt.title("Class-wise Reconstruction Error")
        plt.xlabel("Class")
        plt.ylabel("MSE")

        # Error distribution
        plt.subplot(133)
        all_errors = np.mean((original_features - reconstructed_features) ** 2, axis=1)
        plt.hist(all_errors, bins=30)
        plt.title("Error Distribution")
        plt.xlabel("MSE")
        plt.ylabel("Count")

        plt.tight_layout()
        plt.savefig(f"{save_path}_reconstruction_plots.png")
        plt.close()

    def reset_to_initial_state(self):
        """Reset the model's weights and parameters to their initial state."""
        DEBUG.log("Resetting model to initial state for fresh training...")

        # Reset weights to uniform priors
        n_classes = len(self.label_encoder.classes_)
        n_pairs = len(self.feature_pairs) if self.feature_pairs is not None else 0

        if n_pairs > 0:
            self.current_W = torch.full(
                (n_classes, n_pairs),
                0.1,  # Default uniform prior
                device=self.device,
                dtype=torch.float32,
            )
            self.best_W = self.current_W.clone()
            self.best_error = float("inf")

        # Reset likelihood parameters (if applicable)
        if self.model_type == "Histogram":
            self.likelihood_params = self._compute_pairwise_likelihood_parallel(
                self.X_tensor, self.y_tensor, self.X_tensor.shape[1]
            )
        elif self.model_type == "Gaussian":
            self.likelihood_params = self._compute_pairwise_likelihood_parallel_std(
                self.X_tensor, self.y_tensor, self.X_tensor.shape[1]
            )

        # Reset weight updater
        if self.weight_updater is not None:
            self.weight_updater = BinWeightUpdater(
                n_classes=n_classes,
                feature_pairs=self.feature_pairs,
                n_bins_per_dim=self.n_bins_per_dim,
                batch_size=self.batch_size,
            )

        DEBUG.log("Model reset to initial state.")

    def adaptive_fit_predict(
        self,
        max_rounds: int = 10,
        improvement_threshold: float = 0.0001,
        load_epoch: int = None,
        batch_size: int = 128,
    ):
        """Modified adaptive training strategy with proper fresh start handling"""
        DEBUG.log(" Starting adaptive_fit_predict")
        if not EnableAdaptive:
            print("\033[K" + "Adaptive learning is disabled. Using standard training.")
            return self.fit_predict(batch_size=batch_size)

        # Record the start time
        start_time = time.time()
        start_clock = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        print(
            "\033[K"
            + f"{Colors.BOLD}{Colors.BLUE}Adaptive training started at: {start_clock}{Colors.ENDC}"
        )

        self.in_adaptive_fit = True
        train_indices = []
        test_indices = None

        try:
            # Get initial data
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            print(self.target_column)
            print("\033[K" + f" Initial data shape: X={X.shape}, y={len(y)}")
            print("\033[K" + f"Number of classes in data = {np.unique(y)}")
            print(self.data.head)
            # Initialize label encoder if not already done
            if not hasattr(self.label_encoder, "classes_"):
                self.label_encoder.fit(y)

            # Use existing label encoder
            y_encoded = self.label_encoder.transform(y)

            # Process features and initialize model components if needed
            X_processed = self._preprocess_data(X, is_training=True)
            self.X_tensor = X_processed.clone().detach().to(self.device)
            self.y_tensor = torch.LongTensor(y_encoded).to(self.device)

            # Handle model state based on flags
            model_loaded = False
            if self.use_previous_model:
                print("\033[K" + "Loading previous model state")
                if self._load_model_components():
                    # self.label_encoder =load_label_encoder(self.dataset_name)
                    # self._load_best_weights()
                    # self._load_categorical_encoders()
                    model_loaded = True

                    if not self.fresh_start:
                        # Load previous training data
                        print("\033[K" + "Loading previous training data...")
                        prev_train_file = f"{self.dataset_name}_Last_training.csv"
                        if os.path.exists(prev_train_file):
                            prev_train_data = pd.read_csv(prev_train_file)

                            # Try loading indices from last split
                            prev_train, prev_test = self.load_last_known_split()

                            if prev_train and prev_test:
                                train_indices = prev_train
                                test_indices = prev_test
                                print(
                                    f"\033[KResuming with {len(train_indices)} previous training samples"
                                )
                            else:
                                print(
                                    "\033[KNo valid previous split found, initializing new training set"
                                )
                            print(
                                "\033[K"
                                + "No previous training data found - starting fresh"
                            )
                            train_indices = []
                            test_indices = list(range(len(X)))
                else:
                    print("\033[K" + "No previous model found - starting fresh")

            if not model_loaded:
                print("\033[K" + "Initializing fresh model")
                self._clean_existing_model()
                train_indices = []
                test_indices = list(range(len(X)))

                # Initialize feature pairs for fresh start
                self.feature_pairs = self._generate_feature_combinations(
                    self.X_tensor.shape[1],
                    self.config.get("likelihood_config", {}).get(
                        "feature_group_size", 2
                    ),
                    self.config.get("likelihood_config", {}).get(
                        "max_combinations", None
                    ),
                )

            # Initialize test indices if still None
            if test_indices is None:
                test_indices = list(range(len(X)))

            # Initialize likelihood parameters if needed
            if self.likelihood_params is None:
                DEBUG.log(" Initializing likelihood parameters")
                if self.model_type == "Histogram":
                    self.likelihood_params = self._compute_pairwise_likelihood_parallel(
                        self.X_tensor, self.y_tensor, self.X_tensor.shape[1]
                    )
                elif self.model_type == "Gaussian":
                    self.likelihood_params = (
                        self._compute_pairwise_likelihood_parallel_std(
                            self.X_tensor, self.y_tensor, self.X_tensor.shape[1]
                        )
                    )
                DEBUG.log(" Likelihood parameters computed")

            # Initialize weights if needed
            if self.weight_updater is None:
                DEBUG.log(" Initializing weight updater")
                self._initialize_bin_weights()
                DEBUG.log(" Weight updater initialized")

            # Initialize model weights if needed
            if self.current_W is None:
                DEBUG.log(" Initializing model weights")
                n_classes = len(self.label_encoder.classes_)
                n_pairs = (
                    len(self.feature_pairs) if self.feature_pairs is not None else 0
                )
                if n_pairs == 0:
                    raise ValueError("Feature pairs not initialized")
                self.current_W = torch.full(
                    (n_classes, n_pairs), 0.1, device=self.device, dtype=torch.float32
                )
                if self.best_W is None:
                    self.best_W = self.current_W.clone()

            # Initialize training set if empty
            if len(train_indices) == 0:
                print("\033[K" + "Initializing new training set with minimum samples")
                # Select minimum samples from each class for initial training
                unique_classes = self.label_encoder.classes_
                for class_label in unique_classes:
                    class_indices = np.where(
                        y_encoded == self.label_encoder.transform([class_label])[0]
                    )[0]
                    if len(class_indices) < 2:
                        selected_indices = class_indices
                    else:
                        selected_indices = class_indices[:2]
                    train_indices.extend(selected_indices)

                # Update test indices
                test_indices = list(set(range(len(X))) - set(train_indices))

            DEBUG.log(f" Initial training set size: {len(train_indices)}")
            DEBUG.log(f" Initial test set size: {len(test_indices)}")
            adaptive_patience_counter = 0
            # Continue with training loop...
            while adaptive_patience_counter < 5:
                for round_num in range(max_rounds):
                    print("\033[K" + f"Round {round_num + 1}/{max_rounds}")
                    print("\033[K" + f"Training set size: {len(train_indices)}")
                    print("\033[K" + f"Test set size: {len(test_indices)}")

                    # Reset model to initial state for fresh training
                    # self.reset_to_initial_state()

                    # Save indices for this epoch
                    self.save_epoch_data(round_num, train_indices, test_indices)

                    # Create feature tensors for training
                    X_train = self.X_tensor[train_indices]
                    y_train = self.y_tensor[train_indices]

                    # Train the model
                    save_path = f"data/{self.dataset_name}/Predictions/"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    self.train_indices = train_indices
                    self.test_indices = test_indices
                    results = self.fit_predict(
                        batch_size=batch_size, save_path=save_path
                    )

                    # Check training accuracy
                    print(
                        "\033[K"
                        + f"{Colors.GREEN}Predctions on Training data{Colors.ENDC}",
                        end="\r",
                        flush=True,
                    )
                    train_accuracy = results["train_accuracy"]
                    print(
                        "\033[K" + f"Training accuracy: {train_accuracy:.4f}         "
                    )

                    # Get test accuracy from results
                    test_accuracy = results["test_accuracy"]

                    # Check if we're improving overall
                    improved = False
                    if "best_train_accuracy" not in locals():
                        best_train_accuracy = self.best_combined_accuracy
                        improved = True
                    elif (
                        self.best_combined_accuracy
                        > best_train_accuracy + improvement_threshold
                    ):
                        best_train_accuracy = self.best_combined_accuracy
                        improved = True
                        print(
                            "\033[K"
                            + f"Improved training accuracy to {train_accuracy:.4f}"
                        )

                    # Reset adaptive patience if improved
                    if improved:
                        adaptive_patience_counter = 0
                        # Save the last training and test data
                        self.save_last_split(self.train_indices, self.test_indices)
                        print(
                            "\033[K"
                            + "Saved model and data due to improved training accuracy"
                        )
                    else:
                        adaptive_patience_counter += 1
                        print(
                            "\033[K"
                            + f"No significant overall improvement. Adaptive patience: {adaptive_patience_counter}/5"
                        )
                        if (
                            adaptive_patience_counter >= 5
                        ):  # Using fixed value of 5 for adaptive patience
                            print(
                                "\033[K"
                                + f"No improvement in accuracy after 5 rounds of adding samples."
                            )
                            print(
                                "\033[K"
                                + f"Best training accuracy achieved: {best_train_accuracy:.4f}"
                            )
                            print("\033[K" + "Stopping adaptive training.")
                            break

                    # Evaluate test data using combined predictions from fit_predict
                    test_predictions = results["test_predictions"]["predicted_class"]
                    y_test = self.y_tensor[test_indices].cpu().numpy()

                    # Convert test_predictions to a NumPy array if it's a Pandas Series
                    if isinstance(test_predictions, pd.Series):
                        test_predictions = test_predictions.to_numpy()

                    # Ensure test_predictions is numeric
                    if test_predictions.dtype == np.object_:
                        # If predictions are class labels, convert them to numeric indices
                        test_predictions = self.label_encoder.transform(
                            test_predictions
                        )
                    else:
                        # If predictions are numeric but stored as object, cast to int64
                        test_predictions = test_predictions.astype(np.int64)

                    # Check if we've achieved perfect accuracy
                    if train_accuracy == 1.0:
                        if len(test_indices) == 0:
                            print(
                                "\033[K"
                                + "No more test samples available. Training complete."
                            )
                            break

                        # Get new training samples from misclassified examples
                        new_train_indices = self._select_samples_from_failed_classes(
                            test_predictions, y_test, test_indices, results
                        )

                        if not new_train_indices:
                            print(
                                "\033[K"
                                + "Achieved 100% accuracy on all data. Training complete.                                           "
                            )
                            self.in_adaptive_fit = False
                            return {"train_indices": [], "test_indices": []}

                    else:
                        # Training did not achieve 100% accuracy, select new samples
                        new_train_indices = self._select_samples_from_failed_classes(
                            test_predictions, y_test, test_indices, results
                        )

                        if not new_train_indices:
                            print(
                                "\033[K"
                                + "No suitable new samples found. Training complete."
                            )
                            break

                    if new_train_indices:
                        # Reset to the best round's initial conditions
                        if self.best_round_initial_conditions is not None:
                            print(
                                "\033[K"
                                + f"Resetting to initial conditions of best round {self.best_round}"
                            )
                            self.current_W = self.best_round_initial_conditions[
                                "weights"
                            ].clone()
                            self.likelihood_params = self.best_round_initial_conditions[
                                "likelihood_params"
                            ]
                            self.feature_pairs = self.best_round_initial_conditions[
                                "feature_pairs"
                            ]
                            self.bin_edges = self.best_round_initial_conditions[
                                "bin_edges"
                            ]
                            self.gaussian_params = self.best_round_initial_conditions[
                                "gaussian_params"
                            ]

                    # Update training and test sets with new samples
                    train_indices.extend(new_train_indices)
                    test_indices = list(set(test_indices) - set(new_train_indices))
                    print(
                        "\033[K"
                        + f"Added {len(new_train_indices)} new samples to training set"
                    )

            # Record the end time
            end_time = time.time()
            end_clock = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
            elapsed_time = end_time - start_time

            # Print the timing information
            print(
                "\033[K"
                + f"{Colors.BOLD}{Colors.BLUE}Adaptive training started at: {start_clock}{Colors.ENDC}"
            )
            print(
                "\033[K"
                + f"{Colors.BOLD}{Colors.BLUE}Adaptive training ended at: {end_clock}{Colors.ENDC}"
            )
            print(
                "\033[K"
                + f"{Colors.BOLD}{Colors.BLUE}Total adaptive training time: {elapsed_time:.2f} seconds{Colors.ENDC}"
            )

            self.in_adaptive_fit = False
            return {"train_indices": train_indices, "test_indices": test_indices}

        except Exception as e:
            DEBUG.log(f" Error in adaptive_fit_predict: {str(e)}")
            DEBUG.log(" Traceback:", traceback.format_exc())
            self.in_adaptive_fit = False
            raise

    # ------------------------------------------Adaptive Learning--------------------------------------

    def _calculate_cardinality_threshold(self):
        """Calculate appropriate cardinality threshold based on dataset characteristics"""
        n_samples = len(self.data)
        DEFAULT_THRESHOLD = 0.9
        predict_mode = True if self.mode == "predict" else False
        if predict_mode or self.target_column not in self.data.columns:
            DEBUG.log(
                "Using default cardinality threshold (prediction mode or no target)"
            )
            return DEFAULT_THRESHOLD

        n_classes = len(self.data[self.target_column].unique())

        # Base threshold from config
        base_threshold = self.config.get("training_params", {}).get(
            "cardinality_threshold", 0.9
        )  # cardinality_threshold

        # Adjust threshold based on dataset size and number of classes
        adjusted_threshold = min(
            base_threshold, max(0.1, 1.0 / np.sqrt(n_classes))  # Lower bound of 0.1
        )

        DEBUG.log(f"\nCardinality Threshold Calculation:")
        DEBUG.log(f"- Base threshold: {base_threshold}")
        DEBUG.log(f"- Number of samples: {n_samples}")
        DEBUG.log(f"- Number of classes: {n_classes}")
        DEBUG.log(f"- Adjusted threshold: {adjusted_threshold}")

        return base_threshold  # adjusted_threshold

    def _round_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Round features based on cardinality_tolerance"""
        if cardinality_tolerance == -1:
            return df
        return df.round(cardinality_tolerance)

    def _remove_high_cardinality_columns(
        self, df: pd.DataFrame, threshold: float = 0.8
    ) -> pd.DataFrame:
        """Remove high cardinality columns with more conservative approach"""
        DEBUG.log(f"Starting cardinality analysis with threshold {threshold}")

        df_filtered = df.copy()
        columns_to_drop = []
        cardinality_info = {}

        # First pass: calculate cardinality for all columns
        for column in df.columns:
            if column == self.target_column:
                continue

            unique_count = len(df[column].unique())
            unique_ratio = unique_count / len(df)
            cardinality_info[column] = {
                "unique_count": unique_count,
                "ratio": unique_ratio,
            }

            DEBUG.log(
                f"Column {column}: {unique_count} unique values, ratio {unique_ratio:.4f}"
            )

        # Determine adaptive threshold
        ratios = [info["ratio"] for info in cardinality_info.values()]
        if ratios:
            median_ratio = np.median(ratios)
            adaptive_threshold = min(threshold, max(median_ratio * 2, 0.1))
            DEBUG.log(
                f"Adaptive threshold: {adaptive_threshold} (original: {threshold})"
            )
        else:
            adaptive_threshold = threshold

        # Second pass: mark columns for dropping
        for column, info in cardinality_info.items():
            if info["ratio"] > adaptive_threshold:
                columns_to_drop.append(column)
                DEBUG.log(f"Marking {column} for removal (ratio: {info['ratio']:.4f})")

        # Ensure we keep at least some features
        if len(columns_to_drop) == len(cardinality_info):
            DEBUG.log("Would remove all features - keeping lowest cardinality ones")
            sorted_columns = sorted(
                cardinality_info.items(), key=lambda x: x[1]["ratio"]
            )
            keep_count = max(2, len(cardinality_info) // 5)  # Keep at least 2 or 20%
            columns_to_drop = [col for col, _ in sorted_columns[keep_count:]]

        # Drop columns
        if columns_to_drop:
            df_filtered = df_filtered.drop(columns=columns_to_drop)
            DEBUG.log(f"Dropped columns: {columns_to_drop}")

        DEBUG.log(
            f"Features after cardinality filtering: {df_filtered.columns.tolist()}"
        )
        return df_filtered

    def _detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect categorical columns with improved debugging"""
        DEBUG.log(" Starting categorical column detection")
        categorical_columns = []

        for column in df.columns:
            if column != self.target_column:
                dtype = df[column].dtype
                unique_count = len(df[column].unique())
                DEBUG.log(
                    f" Column {column}: dtype={dtype}, unique values={unique_count}"
                )

                if dtype == "object" or dtype.name == "category":
                    DEBUG.log(f" Adding {column} as categorical (object/category type)")
                    categorical_columns.append(column)
                elif dtype in ["int64", "float64"]:
                    threshold = min(50, len(df) * 0.05)
                    if unique_count < threshold:
                        DEBUG.log(
                            f" Adding {column} as categorical (numeric with few unique values: {unique_count} < {threshold})"
                        )
                        categorical_columns.append(column)
                    else:
                        DEBUG.log(
                            f" Keeping {column} as numeric (unique values: {unique_count} >= {threshold})"
                        )

        DEBUG.log(f" Detected categorical columns: {categorical_columns}")
        return categorical_columns

    def _preprocess_data(
        self, X: Union[pd.DataFrame, torch.Tensor], is_training: bool = True
    ) -> torch.Tensor:
        """Preprocess data with robust NaN handling and type safety."""
        DEBUG.log(f"Starting preprocessing (is_training={is_training})")

        # Check if X is a DataFrame or a tensor
        if isinstance(X, pd.DataFrame):
            DEBUG.log(f"Input shape: {X.shape}")
            DEBUG.log(f"Input columns: {X.columns.tolist()}")
            DEBUG.log(f"Input dtypes:\n{X.dtypes}")

            # In prediction mode, ensure we have all required features
            required_features = (
                self.feature_columns if hasattr(self, "feature_columns") else X.columns
            )
            missing_features = set(required_features) - set(X.columns)
            if missing_features and not predict_mode:
                raise ValueError(f"Missing required features: {missing_features}")

            # Replace NA/NaN with -99999 and keep track of locations
            X = X.copy()
            self.nan_mask = X.isna()  # Store NaN locations using pandas' isna()
            X = X.fillna(-99999)  # Replace with sentinel value

            # Convert object/string columns to numeric where possible
            for col in X.select_dtypes(include=["object", "string"]):
                try:
                    X[col] = pd.to_numeric(X[col], errors="ignore")
                except Exception as e:
                    DEBUG.log(f"Could not convert column {col} to numeric: {str(e)}")

        else:  # Tensor input
            DEBUG.log("Input is a tensor, skipping column-specific operations")
            X = X.clone().detach()
            self.nan_mask = torch.isnan(X)  # For tensor input
            X = torch.where(self.nan_mask, torch.tensor(-99999, device=X.device), X)

        # Step 1: Handle feature selection and statistics computation
        if is_training:
            # Store original columns before any filtering
            self.original_columns = (
                X.columns.tolist() if isinstance(X, pd.DataFrame) else None
            )

            # Apply feature filtering (e.g., high cardinality removal)
            if isinstance(X, pd.DataFrame):
                cardinality_threshold = self._calculate_cardinality_threshold()
                DEBUG.log(f"Cardinality threshold: {cardinality_threshold}")
                X = self._remove_high_cardinality_columns(X, cardinality_threshold)
                DEBUG.log(f"Shape after cardinality filtering: {X.shape}")

                # Store the final selected features
                self.feature_columns = X.columns.tolist()
                DEBUG.log(f"Selected feature columns: {self.feature_columns}")

                # Store removed columns for reference
                self.high_cardinality_columns = list(
                    set(self.original_columns) - set(self.feature_columns)
                )
                if self.high_cardinality_columns:
                    DEBUG.log(
                        f"Removed high cardinality columns: {self.high_cardinality_columns}"
                    )

            # Compute statistics ONLY on the selected features
            self.global_mean = X.mean(axis=0).values
            self.global_std = X.std(axis=0).values
            self.global_std[self.global_std == 0] = 1  # Avoid division by zero
            self.global_stats_computed = True
        else:
            # During prediction: enforce exact feature matching
            if isinstance(X, pd.DataFrame):
                missing = set(self.feature_columns) - set(X.columns)
                if missing:
                    raise ValueError(
                        f"Prediction data missing {len(missing)} required features: {sorted(missing)}\n"
                        f"Expected features: {self.feature_columns}\n"
                        f"Provided features: {X.columns.tolist()}"
                    )
                # Reorder to match training features exactly
                X = X[self.feature_columns]

        # Step 2: Handle categorical features
        DEBUG.log("Starting categorical encoding")
        try:
            if isinstance(X, pd.DataFrame):
                X_encoded = self._encode_categorical_features(X, is_training)
            else:
                X_encoded = X
            DEBUG.log(f"Shape after categorical encoding: {X_encoded.shape}")
            if isinstance(X_encoded, pd.DataFrame):
                DEBUG.log(f"Encoded dtypes:\n{X_encoded.dtypes}")
        except Exception as e:
            DEBUG.log(f"Error in categorical encoding: {str(e)}")
            raise

        # Step 3: Convert to numpy array
        try:
            if isinstance(X_encoded, pd.DataFrame):
                X_numpy = X_encoded.to_numpy(dtype=np.float32)
            else:
                X_numpy = (
                    X_encoded.cpu().numpy()
                    if torch.is_tensor(X_encoded)
                    else np.array(X_encoded, dtype=np.float32)
                )
            DEBUG.log(f"Numpy array shape: {X_numpy.shape}")
        except Exception as e:
            DEBUG.log(f"Error converting to numpy: {str(e)}")
            raise

        # Step 4: Standardize using the correct stats
        try:
            X_scaled = (X_numpy - self.global_mean) / self.global_std
            DEBUG.log("Scaling successful")
        except Exception as e:
            DEBUG.log(f"Standard scaling failed: {str(e)}. Using manual scaling")
            means = np.nanmean(X_numpy, axis=0)
            stds = np.nanstd(X_numpy, axis=0)
            stds[stds == 0] = 1
            X_scaled = (X_numpy - means) / stds

        # Step 5: Convert to tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

        # Step 6: Compute feature pairs and bin edges (training only)
        if is_training:
            remaining_feature_indices = list(range(len(self.feature_columns)))
            DEBUG.log(
                f"Computing feature pairs from {len(remaining_feature_indices)} features"
            )

            self.feature_pairs = self._generate_feature_combinations(
                remaining_feature_indices,
                self.config.get("likelihood_config", {}).get("feature_group_size", 2),
                self.config.get("likelihood_config", {}).get("max_combinations", None),
            )
            DEBUG.log(f"Generated {len(self.feature_pairs)} feature pairs")

            self.bin_edges = self._compute_bin_edges(
                X_tensor,
                self.config.get("likelihood_config", {}).get("bin_sizes", [128]),
            )
            DEBUG.log(f"Computed bin edges for {len(self.bin_edges)} feature pairs")

        DEBUG.log(f"Final preprocessed shape: {X_scaled.shape}")
        return X_tensor

    # -----------------------------------------PDF mosaic -----------------------------------------------------
    def _generate_prediction_analysis_files(self, predictions_df, true_labels=None):
        """Generate CSV files for failed and correct predictions"""
        dataset_name = self.dataset_name
        output_dir = f"data/{dataset_name}/Predictions/"
        os.makedirs(output_dir, exist_ok=True)

        # Save all predictions
        predictions_df.to_csv(
            f"{output_dir}{dataset_name}_all_predictions.csv", index=False
        )

        # Only generate failure/success files if we have true labels
        if true_labels is not None:
            # Add true labels to dataframe if not already present
            if "true_class" not in predictions_df.columns:
                predictions_df["true_class"] = true_labels

            # Create failed predictions CSV
            failed_predictions = predictions_df[
                predictions_df["predicted_class"] != predictions_df["true_class"]
            ]
            failed_predictions.to_csv(
                f"{output_dir}{dataset_name}_failed_predictions.csv", index=False
            )

            # Create correct predictions CSV
            correct_predictions = predictions_df[
                predictions_df["predicted_class"] == predictions_df["true_class"]
            ]
            correct_predictions.to_csv(
                f"{output_dir}{dataset_name}_correct_predictions.csv", index=False
            )

            return {
                "all_predictions": f"{output_dir}{dataset_name}_all_predictions.csv",
                "failed_predictions": f"{output_dir}{dataset_name}_failed_predictions.csv",
                "correct_predictions": f"{output_dir}{dataset_name}_correct_predictions.csv",
            }

        return {"all_predictions": f"{output_dir}{dataset_name}_all_predictions.csv"}

    def generate_class_pdf_mosaics(self, predictions_df, output_dir, columns=4, rows=4):
        """
        Generate PDF mosaics with configurable grid layout (columns x rows per page).
        Captions are clickable hyperlinks to the original image paths.

        Args:
            predictions_df: DataFrame containing predictions and image paths.
            output_dir: Directory to save the PDF files.
            columns: Number of columns per page (default: 4).
            rows: Number of rows per page (default: 4).
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Add custom styles
        styles = getSampleStyleSheet()
        if "Caption" not in styles:
            from reportlab.lib.styles import ParagraphStyle

            styles.add(
                ParagraphStyle(
                    name="Caption",
                    parent=styles["Normal"],
                    fontSize=8,
                    leading=9,
                    spaceBefore=2,
                    spaceAfter=2,
                    alignment=1,  # Center aligned
                )
            )

        # Add hyperlink style
        if "Hyperlink" not in styles:
            styles.add(
                ParagraphStyle(
                    name="Hyperlink",
                    parent=styles["Caption"],
                    textColor=colors.blue,
                    underline=1,
                )
            )

        # Group by predicted class
        class_groups = predictions_df.groupby("predicted_class")
        images_per_page = columns * rows

        for class_name, group_df in class_groups:
            safe_name = "".join(c if c.isalnum() else "_" for c in str(class_name))
            pdf_path = os.path.join(output_dir, f"class_{safe_name}_mosaic.pdf")

            # Sort by prediction confidence (highest first)
            sorted_df = group_df.sort_values("prediction_confidence", ascending=False)
            n_images = len(sorted_df)
            n_pages = math.ceil(n_images / images_per_page)

            # PDF setup with margins
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=letter,
                rightMargin=0.5 * inch,
                leftMargin=0.5 * inch,
                topMargin=0.5 * inch,
                bottomMargin=0.5 * inch,
            )
            elements = []

            # Calculate image dimensions based on page size and grid
            usable_width = letter[0] - inch  # Account for margins
            usable_height = letter[1] - 2 * inch
            img_width = usable_width / columns
            img_height = (
                usable_height / rows
            ) * 0.85  # 85% of row height for image, 15% for caption

            # Single progress bar for the entire class
            with tqdm(
                total=n_images,
                desc=f"{str(class_name)[:15]:<15}",
                unit="img",
                bar_format="{l_bar}{bar:40}{r_bar}{bar:-40b}",
                leave=False,
            ) as pbar:
                processed_images = 0

                for page_num in range(n_pages):
                    start_idx = page_num * images_per_page
                    end_idx = min(start_idx + images_per_page, n_images)
                    page_images = sorted_df.iloc[start_idx:end_idx]

                    # Page header (skip for first page)
                    if page_num > 0:
                        elements.extend(
                            [
                                Spacer(1, 0.25 * inch),
                                Paragraph(
                                    f"Page {page_num+1} of {n_pages}", styles["Normal"]
                                ),
                                Spacer(1, 0.25 * inch),
                            ]
                        )

                    elements.append(
                        Paragraph(
                            f"Class: {class_name} (Sorted by Confidence)",
                            styles["Heading2"],
                        )
                    )
                    elements.append(Spacer(1, 0.1 * inch))

                    # Create image grid table
                    table_data = []
                    row_data = []

                    for _, row in page_images.iterrows():
                        img_path = row["filepath"]
                        img_name = os.path.basename(img_path)
                        confidence = row["prediction_confidence"]

                        try:
                            # Verify and load image
                            with PILImage.open(img_path) as img:
                                img.verify()

                            # Create clickable caption with hyperlink
                            caption_text = f'<link href="{img_path}">{img_name[:15]}...</link><br/>Conf: {confidence:.2%}'
                            caption = Paragraph(caption_text, styles["Hyperlink"])

                            # Create table cell with image and caption
                            cell_content = [
                                ReportLabImage(
                                    img_path,
                                    width=img_width * 0.9,
                                    height=img_height * 0.85,
                                ),
                                caption,
                            ]
                            row_data.append(cell_content)

                            # Start new row when current row is full
                            if len(row_data) == columns:
                                table_data.append(row_data)
                                row_data = []

                        except Exception as e:
                            print(f"\033[K⚠️ Error loading {img_path}: {str(e)}")
                            continue

                        # Update progress
                        processed_images += 1
                        pbar.update(1)

                    # Add any remaining images in the last row
                    if row_data:
                        # Pad with empty cells if needed
                        while len(row_data) < columns:
                            row_data.append("")
                        table_data.append(row_data)

                    # Add table to PDF elements
                    if table_data:
                        table_style = [
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ]
                        table = Table(table_data, colWidths=[img_width] * columns)
                        table.setStyle(TableStyle(table_style))
                        elements.append(table)

                    # Add page break if not the last page
                    if page_num < n_pages - 1:
                        elements.append(PageBreak())

                # Build the PDF after processing all pages
                doc.build(elements)

                # Print completion message
                print(
                    f"\033[K✅ {class_name} - Saved {n_images} images to {os.path.basename(pdf_path)}"
                )

    # --------------Option 3 ----------------
    def generate_class_pdf(
        self, image_paths: List[str], posteriors: np.ndarray, output_pdf: str
    ):
        """Generate professional multi-page PDF with 2x4 image grids per class, sorted by confidence.

        Args:
            image_paths: List of image file paths
            posteriors: Numpy array of posterior probabilities for each image
            output_pdf: Path to save the output PDF
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

        # Group images by predicted class with their posteriors
        class_groups = defaultdict(list)
        for img_path, posterior in zip(image_paths, posteriors):
            # Extract class name from filename (assuming format: class_imageid.jpg)
            class_name = os.path.basename(img_path).split("_")[0]
            class_groups[class_name].append((img_path, posterior))

        # Create PDF document
        doc = SimpleDocTemplate(
            output_pdf,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        # Define styles
        styles = getSampleStyleSheet()
        title_style = styles["Heading1"]
        caption_style = ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            alignment=1,  # Center aligned
            spaceBefore=6,
        )
        footer_style = ParagraphStyle(
            "Footer",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            alignment=2,  # Right aligned
            spaceBefore=6,
        )

        elements = []

        # Process each class
        for class_name, images in sorted(class_groups.items()):
            # Add class title page
            elements.append(Paragraph(f"Class: {class_name}", title_style))
            elements.append(Spacer(1, 0.5 * inch))
            elements.append(PageBreak())

            # Sort images by posterior (highest first)
            images.sort(key=lambda x: x[1], reverse=True)

            # Create pages with 2x4 grids
            n_images = len(images)
            n_pages = math.ceil(n_images / 8)

            for page_num in range(n_pages):
                # Create a grid for this page
                start_idx = page_num * 8
                end_idx = min(start_idx + 8, n_images)
                page_images = images[start_idx:end_idx]

                # Create a temporary image with the grid
                grid_img_path = self._create_image_grid(page_images)

                # Add to PDF with caption
                img = Image(grid_img_path, width=6 * inch, height=7.5 * inch)
                elements.append(img)

                # Add page footer
                footer_text = f"Page {page_num+1} of {n_pages} - Class: {class_name}"
                elements.append(Paragraph(footer_text, footer_style))

                if end_idx < n_images:  # Don't add break after last page
                    elements.append(PageBreak())

            # Add divider page between classes
            if class_name != sorted(class_groups.keys())[-1]:
                elements.append(Paragraph("Class Complete", title_style))
                elements.append(PageBreak())

        # Build the PDF
        doc.build(elements)

        # Clean up temporary files
        if hasattr(self, "_temp_files"):
            for f in self._temp_files:
                try:
                    os.remove(f)
                except:
                    pass

    def _create_image_grid(self, images: List[tuple]) -> str:
        """Create a single image with 2x4 grid of images and captions.

        Args:
            images: List of (image_path, posterior) tuples

        Returns:
            Path to temporary image file
        """
        from PIL import Image, ImageDraw, ImageFont

        # Grid parameters
        cols, rows = 4, 2
        img_size = 400  # Size of each individual image
        padding = 10
        caption_height = 30

        # Create blank canvas
        canvas_width = cols * (img_size + padding) + padding
        canvas_height = rows * (img_size + padding + caption_height) + padding
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        # Position images in grid
        for idx, (img_path, posterior) in enumerate(images):
            row = idx // cols
            col = idx % cols

            try:
                # Open and resize image
                img = Image.open(img_path)
                img = img.resize((img_size, img_size))

                # Calculate position
                x = col * (img_size + padding) + padding
                y = row * (img_size + padding + caption_height) + padding

                # Paste image with border
                canvas.paste(img, (x, y))
                draw.rectangle(
                    [x, y, x + img_size, y + img_size], outline="black", width=1
                )

                # Add caption
                caption = f"{os.path.basename(img_path)} (Confidence: {posterior:.2%})"
                text_width = draw.textlength(caption, font=font)
                text_x = x + (img_size - text_width) / 2
                text_y = y + img_size + 5
                draw.text((text_x, text_y), caption, fill="black", font=font)

            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                continue

        # Save temporary file
        if not hasattr(self, "_temp_files"):
            self._temp_files = []

        temp_path = os.path.join(
            tempfile.gettempdir(), f"grid_{os.getpid()}_{len(self._temp_files)}.jpg"
        )
        canvas.save(temp_path, quality=90)
        self._temp_files.append(temp_path)

        return temp_path

    # -------------------------------------------PDF mosaic Ends ------------------------------

    def _generate_feature_combinations(
        self,
        feature_indices: Union[List[int], int],
        group_size: int = 2,
        max_combinations: int = None,
    ) -> torch.Tensor:
        """
        Generate feature combinations in-memory without file I/O.
        Only generates if feature_pairs doesn't already exist.
        """
        # If we already have feature pairs, return them
        # if hasattr(self, 'feature_pairs') and self.feature_pairs is not None:
        if self.feature_pairs is not None:
            return self.feature_pairs
        # Strict check for predict mode or use_previous_model
        if (self.mode == "predict" or self.use_previous_model) and not hasattr(
            self, "feature_pairs"
        ):
            raise RuntimeError(
                f"FATAL: Required feature pairs not found in model components.\n"
                f"- Current mode: {'predict' if self.mode == 'predict' else 'train'}\n"
                f"- Use previous model: {self.use_previous_model}\n"
                f"Possible causes:\n"
                f"1. Missing model components file\n"
                f"2. Corrupted model data\n"
                f"3. Trying to predict with untrained model\n"
                f"Solution: Ensure model is properly trained first or provide correct model components."
            )
        # Convert feature_indices to list if it's an integer
        if isinstance(feature_indices, int):
            feature_indices = list(range(feature_indices))

        # Get parameters from config with defaults
        config = self.config.get("likelihood_config", {})
        group_size = group_size or config.get("feature_group_size", 2)
        max_combinations = max_combinations or config.get("max_combinations")

        print("\033[K" + f"[DEBUG] Generating feature combinations in-memory:")
        print("\033[K" + f"- n_features: {len(feature_indices)}")
        print("\033[K" + f"- group_size: {group_size}")
        print("\033[K" + f"- max_combinations: {max_combinations}")

        # Generate all possible combinations if under limit
        total_possible = comb(len(feature_indices), group_size)
        if max_combinations is None or total_possible <= max_combinations:
            print("\033[K" + "[DEBUG] Generating all possible combinations")
            all_combinations = list(combinations(feature_indices, group_size))
        else:
            print("\033[K" + f"[DEBUG] Sampling {max_combinations} random combinations")
            all_combinations = self._sample_combinations(
                feature_indices, group_size, max_combinations
            )

        # Remove duplicates and sort
        unique_combinations = list({tuple(sorted(comb)) for comb in all_combinations})
        unique_combinations = sorted(unique_combinations)
        print(
            "\033[K"
            + f"[DEBUG] Generated {len(unique_combinations)} unique feature combinations"
        )

        # Convert to tensor and store in the model
        self.feature_pairs = torch.tensor(unique_combinations, device=self.device)
        return self.feature_pairs

    def _sample_combinations(
        self, features: List[int], group_size: int, max_samples: int
    ) -> List[Tuple[int]]:
        """Memory-efficient combination sampling for large feature spaces."""
        # Use reservoir sampling for large feature spaces
        samples = set()

        # First add all possible combinations if they're few enough
        if comb(len(features), group_size) < 1e6:  # Threshold for full generation
            return list(combinations(features, group_size))

        # For very large spaces, use iterative sampling
        while len(samples) < max_samples:
            # Generate random combinations without replacement
            new_sample = tuple(sorted(random.sample(features, group_size)))
            samples.add(new_sample)

            # Progress reporting
            if len(samples) % 1000 == 0:
                print(
                    f"\033[K[DEBUG] Generated {len(samples)}/{max_samples} samples",
                    end="\r",
                )

        return list(samples)

    # -----------------------------------------------------------------------------Bin model ---------------------------
    def _compute_pairwise_likelihood_parallel(
        self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int
    ):
        """GPU-optimized version with dimension consistency fixes"""
        DEBUG.log("Starting GPU-optimized _compute_pairwise_likelihood_parallel")

        # Ensure tensors are contiguous on the computation device
        dataset = dataset.contiguous()
        labels = labels.contiguous()

        # Validate class consistency
        # unique_classes = torch.unique(labels)
        unique_classes, class_counts = torch.unique(labels, return_counts=True)
        n_classes = len(unique_classes)
        len(dataset)
        if n_classes != len(self.label_encoder.classes_):
            raise ValueError("Class count mismatch between data and label encoder")

        # Get bin configuration from model parameters
        bin_sizes = self.config.get("likelihood_config", {}).get("bin_sizes", [128])
        # n_bins = bin_sizes[0]  # Use first element of bin_sizes array
        n_bins = bin_sizes[0] if len(bin_sizes) == 1 else max(bin_sizes)
        self.n_bins_per_dim = n_bins

        # Initialize weights with consistent bin size
        self._initialize_bin_weights()

        all_bin_counts = []
        all_bin_probs = []

        # Add progress bar
        with tqdm(
            total=len(self.feature_pairs), desc="Pairwise likelihood", leave=False
        ) as pbar:
            # Process each feature pair with dimension checks
            for pair_idx, (f1, f2) in enumerate(self.feature_pairs):
                # Use precomputed bin edges that match weight dimensions
                edges = self.bin_edges[pair_idx]

                # Validate bin dimensions match weight updater settings
                assert (
                    len(edges[0]) == self.weight_updater.n_bins_per_dim + 1
                ), f"Bin edges dimension mismatch: {len(edges[0])-1} vs {self.weight_updater.n_bins_per_dim}"

                # Initialize counts tensor with proper dimensions
                pair_counts = torch.zeros(
                    (n_classes, n_bins, n_bins), dtype=torch.float32, device=self.device
                )

                for cls_idx, cls in enumerate(unique_classes):
                    cls_mask = labels == cls
                    if not torch.any(cls_mask):
                        continue

                    # Process data on GPU
                    data = dataset[cls_mask][:, [f1, f2]].contiguous()

                    # GPU-accelerated bucketization
                    indices = [
                        torch.bucketize(data[:, 0], edges[0], out_int32=True).clamp(
                            0, n_bins - 1
                        ),
                        torch.bucketize(data[:, 1], edges[1], out_int32=True).clamp(
                            0, n_bins - 1
                        ),
                    ]

                    # Flat indices computation
                    flat_indices = indices[0] * n_bins + indices[1]

                    # Bincount with validated dimensions
                    counts = torch.bincount(flat_indices, minlength=n_bins * n_bins)
                    pair_counts[cls_idx] = counts.view(n_bins, n_bins).float()

                # Laplace smoothing with dimension preservation
                smoothed = pair_counts + 1.0
                probs = smoothed / (smoothed.sum(dim=(1, 2), keepdim=True) + 1e-8)

                all_bin_counts.append(smoothed)
                all_bin_probs.append(probs)
                pbar.update(1)
        pbar.close()
        return {
            "bin_counts": all_bin_counts,
            "bin_probs": all_bin_probs,
            "bin_edges": self.bin_edges,
            "feature_pairs": self.feature_pairs,
            "classes": unique_classes,
        }

    def _process_pair_batch(
        self, dataset_cpu, labels_cpu, batch_pairs, unique_classes, bin_sizes
    ):
        """Process a batch of feature pairs on CPU"""
        batch_counts = []
        batch_probs = []

        for pair_idx, feature_group in enumerate(batch_pairs):
            feature_group = [int(x) for x in feature_group]
            group_data = dataset_cpu[:, feature_group]
            n_dims = len(feature_group)

            # Get bin sizes for this group
            group_bin_sizes = (
                bin_sizes[:n_dims] if len(bin_sizes) > 1 else [bin_sizes[0]] * n_dims
            )

            # Get bin edges for this group (already on correct device)
            group_bin_edges = self.bin_edges[pair_idx]

            # Initialize bin counts on CPU
            bin_shape = [len(unique_classes)] + group_bin_sizes
            bin_counts = torch.zeros(bin_shape, dtype=torch.float32)

            for class_idx, class_label in enumerate(unique_classes):
                class_mask = labels_cpu == class_label
                if class_mask.any():
                    class_data = group_data[class_mask]

                    # Compute bin indices
                    bin_indices = []
                    for dim in range(n_dims):
                        # Get edges for this dimension (already a tensor)
                        edges = group_bin_edges[dim]
                        indices = (
                            torch.bucketize(
                                class_data[:, dim],
                                edges.cpu(),  # Move edges to CPU to match data
                            )
                            .sub_(1)
                            .clamp_(0, group_bin_sizes[dim] - 1)
                        )
                        bin_indices.append(indices)

                    # Update counts
                    if n_dims == 1:
                        bin_counts[class_idx] = torch.bincount(
                            bin_indices[0], minlength=group_bin_sizes[0]
                        ).float()
                    else:
                        # For multi-dimensional, use scatter_add
                        flat_indices = torch.sum(
                            torch.stack(bin_indices)
                            * torch.tensor(
                                [
                                    np.prod(group_bin_sizes[i + 1 :])
                                    for i in range(n_dims)
                                ]
                            ).unsqueeze(1),
                            dim=0,
                        ).long()

                        counts = torch.zeros(
                            np.prod(group_bin_sizes), dtype=torch.float32
                        )
                        counts.scatter_add_(
                            0,
                            flat_indices,
                            torch.ones_like(flat_indices, dtype=torch.float32),
                        )
                        bin_counts[class_idx] = counts.reshape(*group_bin_sizes)

            # Apply Laplace smoothing and compute probabilities
            smoothed_counts = bin_counts + 1.0
            bin_probs = smoothed_counts / smoothed_counts.sum(
                dim=tuple(range(1, n_dims + 1)), keepdim=True
            )

            batch_counts.append(smoothed_counts)
            batch_probs.append(bin_probs)

        return batch_counts, batch_probs

    def _calculate_safe_batch_size(self, n_classes, n_bins):
        """Calculate safe batch size based on available memory"""
        # Conservative estimate - start small
        base_size = 10

        if torch.cuda.is_available():
            try:
                # Estimate memory needed per pair
                pair_mem = n_classes * (n_bins**2) * 4  # 4 bytes per float

                # Get available memory
                free_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(
                    0
                )

                # Calculate how many pairs we can fit (leave 50% buffer)
                safe_pairs = int((free_mem * 0.5) / pair_mem)
                return max(1, min(safe_pairs, 100))  # Limit between 1 and 100
            except:
                return base_size
        return base_size

    def _compute_gaussian_params(self, dataset: torch.Tensor, labels: torch.Tensor):
        """
        Compute Gaussian parameters (means and covariances) for all feature pairs once during initialization.

        Args:
            dataset: Input tensor of shape [n_samples, n_features]
            labels: Target tensor of shape [n_samples]

        Returns:
            Dictionary containing means and covariances for each feature pair and class
        """
        unique_classes = torch.unique(labels)
        n_classes = len(unique_classes)
        n_pairs = len(self.feature_pairs)

        # Initialize storage for means and covariances
        means = torch.zeros((n_classes, n_pairs, 2), device=self.device)
        covs = torch.zeros((n_classes, n_pairs, 2, 2), device=self.device)

        # Process each class
        for class_idx, class_id in enumerate(unique_classes):
            class_mask = labels == class_id
            class_data = dataset[class_mask]

            # Process each feature pair
            for pair_idx, pair in enumerate(self.feature_pairs):
                pair_data = class_data[:, pair]

                # Compute mean efficiently
                means[class_idx, pair_idx] = torch.mean(pair_data, dim=0)

                # Compute covariance with stability check
                if len(pair_data) > 1:
                    centered_data = pair_data - means[class_idx, pair_idx].unsqueeze(0)
                    cov = torch.matmul(centered_data.T, centered_data) / (
                        len(pair_data) - 1
                    )

                    # Add stability term
                    min_eig = torch.linalg.eigvals(cov).real.min()
                    if min_eig < 1e-6:
                        reg_term = (1e-6 - min_eig) + 1e-6
                        cov += torch.eye(2, device=self.device) * reg_term
                else:
                    # Single sample case - use identity matrix
                    cov = torch.eye(2, device=self.device)

                covs[class_idx, pair_idx] = cov

        return {
            "means": means,
            "covs": covs,
            "classes": unique_classes,
            "feature_pairs": self.feature_pairs,
        }

    def _compute_pairwise_likelihood_parallel_std(
        self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int
    ):
        """Optimized Gaussian parameter computation using full covariance matrices and vectorized pair extraction."""
        DEBUG.log("Starting optimized Gaussian likelihood computation")

        # Generate feature pairs if needed
        if self.feature_pairs is None:
            self.feature_pairs = self._generate_feature_combinations(
                feature_dims,
                self.config.get("likelihood_config", {}).get("feature_group_size", 2),
                self.config.get("likelihood_config", {}).get("max_combinations", None),
            )

        unique_classes = torch.unique(labels)
        n_classes = len(unique_classes)
        n_pairs = len(self.feature_pairs)
        n_features = dataset.size(1)

        # Convert feature pairs to tensor for advanced indexing
        pair_indices = torch.tensor(
            self.feature_pairs, device=self.device, dtype=torch.long
        )

        # Initialize storage tensors
        means = torch.zeros((n_classes, n_pairs, 2), device=self.device)
        covs = torch.zeros((n_classes, n_pairs, 2, 2), device=self.device)

        # Process each class with batched operations
        for class_idx, class_id in enumerate(unique_classes):
            class_mask = labels == class_id
            class_data = dataset[class_mask]

            if class_data.size(0) == 0:  # Handle empty classes
                covs[class_idx] += torch.eye(2, device=self.device) * 1e-6
                continue

            # Compute full feature means
            full_mean = torch.mean(class_data, dim=0)

            # Center data and compute full covariance matrix
            centered = class_data - full_mean.unsqueeze(0)
            cov_matrix = torch.matmul(centered.T, centered) / (
                centered.size(0) - 1 + 1e-6
            )
            cov_matrix += (
                torch.eye(n_features, device=self.device) * 1e-6
            )  # Regularization

            # Extract means for all pairs simultaneously
            means[class_idx] = full_mean[pair_indices]

            # Extract covariance components using vectorized indexing
            i, j = pair_indices[:, 0], pair_indices[:, 1]
            cov_ii = cov_matrix[i, i]
            cov_ij = cov_matrix[i, j]
            cov_jj = cov_matrix[j, j]

            # Build covariance matrices in parallel
            covs[class_idx, :, 0, 0] = cov_ii
            covs[class_idx, :, 0, 1] = cov_ij
            covs[class_idx, :, 1, 0] = cov_ij  # Symmetric
            covs[class_idx, :, 1, 1] = cov_jj

        # Store parameters with standardized structure
        self.gaussian_params = {
            "means": means.contiguous(),
            "covs": covs.contiguous(),
            "classes": unique_classes,
            "feature_pairs": self.feature_pairs,
        }

        return self.gaussian_params

    def _compute_batch_posterior_std(
        self, features: torch.Tensor, epsilon: float = 1e-10
    ):
        """Gaussian posterior computation focusing on relative class probabilities"""
        features = features.to(self.device)
        batch_size = len(features)
        n_classes = len(self.likelihood_params["classes"])
        n_pairs = len(self.feature_pairs)

        # Initialize log likelihoods
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        # Setup progress bars
        with tqdm(
            total=n_pairs, desc="Processing feature pairs", leave=False
        ) as pair_pbar:
            # Process each feature pair
            for pair_idx, pair in enumerate(self.feature_pairs):
                pair_data = features[:, pair]

                # Get weights for this pair
                pair_weights = [
                    self.weight_updater.get_gaussian_weights(class_idx, pair_idx)
                    for class_idx in range(n_classes)
                ]

                # Class processing progress bar
                with tqdm(
                    total=n_classes,
                    desc=f"Pair {pair_idx+1}/{n_pairs} classes",
                    leave=False,
                ) as class_pbar:
                    # Compute class contributions for this pair
                    for class_idx in range(n_classes):
                        mean = self.likelihood_params["means"][class_idx, pair_idx]
                        cov = self.likelihood_params["covs"][class_idx, pair_idx]
                        weight = pair_weights[class_idx]

                        # Center the data
                        centered = pair_data - mean.unsqueeze(0)

                        # Compute class likelihood
                        try:
                            # Add minimal regularization
                            reg_cov = cov + torch.eye(2, device=self.device) * 1e-6
                            prec = torch.inverse(reg_cov)

                            # Quadratic term
                            quad = torch.sum(
                                torch.matmul(centered.unsqueeze(1), prec).squeeze(1)
                                * centered,
                                dim=1,
                            )

                            # Log likelihood (excluding constant terms)
                            class_ll = -0.5 * quad + torch.log(weight + epsilon)

                        except RuntimeError:
                            # Handle numerical issues
                            class_ll = torch.full_like(quad, -1e10)

                        log_likelihoods[:, class_idx] += class_ll
                        class_pbar.update(1)  # Update class progress

                pair_pbar.update(1)  # Update pair progress

        # Convert to probabilities using softmax
        max_log_ll = torch.max(log_likelihoods, dim=1, keepdim=True)[0]
        exp_ll = torch.exp(log_likelihoods - max_log_ll)
        posteriors = exp_ll / (torch.sum(exp_ll, dim=1, keepdim=True) + epsilon)

        return posteriors, None

    def _initialize_bin_weights(self):
        """Initialize weights with config-consistent bin dimensions"""
        bin_sizes = self.config.get("likelihood_config", {}).get("bin_sizes", [128])
        n_bins_per_dim = bin_sizes[0]  # Match likelihood computation

        self.weight_updater = BinWeightUpdater(
            n_classes=len(self.label_encoder.classes_),
            feature_pairs=self.feature_pairs,
            n_bins_per_dim=n_bins_per_dim,  # Now matches likelihood bins
            batch_size=self.batch_size,
        )

    def _update_priors_parallel(self, failed_cases: List[Tuple], batch_size: int = 128):
        """Vectorized weight updates with proper error handling"""
        n_failed = len(failed_cases)
        if n_failed == 0:
            self.consecutive_successes += 1
            return

        self.consecutive_successes = 0
        self.learning_rate = max(self.learning_rate / 2, 1e-6)

        # Stack all features and convert classes at once
        features = torch.stack([case[0] for case in failed_cases]).to(self.device)
        true_classes = torch.tensor(
            [int(case[1]) for case in failed_cases], device=self.device
        )

        # Compute posteriors for all cases at once
        if self.model_type == "Histogram":
            posteriors, bin_indices = self._compute_batch_posterior(features)
        else:  # Gaussian model
            posteriors, _ = self._compute_batch_posterior_std(features)
            return  # Gaussian model doesn't need bin-based updates

        pred_classes = torch.argmax(posteriors, dim=1)

        # Compute adjustments for all cases at once
        true_posteriors = posteriors[torch.arange(n_failed), true_classes]
        pred_posteriors = posteriors[torch.arange(n_failed), pred_classes]
        adjustments = self.learning_rate * (1.0 - (true_posteriors / pred_posteriors))

        # Update weights for each feature group
        if (
            bin_indices is not None
        ):  # Only proceed if we have bin indices (Histogram model)
            for group_idx in bin_indices:
                bin_i, bin_j = bin_indices[group_idx]

                # Group updates by class for vectorization
                for class_id in range(self.weight_updater.n_classes):
                    class_mask = true_classes == class_id
                    if not class_mask.any():
                        continue

                    # Get relevant indices and adjustments for this class
                    class_bin_i = bin_i[class_mask]
                    class_bin_j = bin_j[class_mask]
                    class_adjustments = adjustments[class_mask]

                    # Update weights for this class
                    weights = self.weight_updater.histogram_weights[class_id][group_idx]
                    for idx in range(len(class_adjustments)):
                        i, j = class_bin_i[idx], class_bin_j[idx]
                        weights[i, j] += class_adjustments[idx]

    # ------------------------------------------Boost weights------------------------------------------

    def _compute_custom_bin_edges(
        self, data: torch.Tensor, bin_sizes: List[int]
    ) -> List[torch.Tensor]:
        """
        Compute bin edges based on custom bin sizes.
        Supports both uniform and non-uniform binning.

        Args:
            data: Input tensor of shape [n_samples, n_features]
            bin_sizes: List of integers specifying bin sizes for each dimension

        Returns:
            List of tensors containing bin edges for each dimension
        """
        n_dims = data.shape[1]
        bin_edges = []

        # If single bin size provided, use it for all dimensions
        if len(bin_sizes) == 1:
            bin_sizes = bin_sizes * n_dims

        # Ensure we have enough bin sizes
        if len(bin_sizes) < n_dims:
            raise ValueError(
                f"Not enough bin sizes provided. Need {n_dims}, got {len(bin_sizes)}"
            )

        for dim in range(n_dims):
            dim_data = data[:, dim]
            dim_min, dim_max = dim_data.min(), dim_data.max()
            padding = (dim_max - dim_min) * 0.01

            # Create edges based on specified bin size
            if bin_sizes[dim] <= 1:
                raise ValueError(f"Bin size must be > 1, got {bin_sizes[dim]}")

            edges = torch.linspace(
                dim_min - padding,
                dim_max + padding,
                bin_sizes[dim] + 1,
                device=self.device,
            )
            bin_edges.append(edges)

        return bin_edges

    # ---------------------------------------------------------Save Last data -------------------------
    def load_last_known_split(self):
        """Load the last known split using stored original indices"""
        dataset_name = self.dataset_name
        train_file = f"data/{dataset_name}/Last_training.csv"
        test_file = f"data/{dataset_name}/Last_testing.csv"

        if os.path.exists(train_file) and os.path.exists(test_file):
            try:
                # Load indices directly from index column
                train_df = pd.read_csv(train_file)
                test_df = pd.read_csv(test_file)

                # Validate index column exists
                if (
                    "original_index" not in train_df.columns
                    or "original_index" not in test_df.columns
                ):
                    print("\033[K" + "No original_index column found in split files")
                    return None, None

                # Get indices as lists
                train_indices = train_df["original_index"].tolist()
                test_indices = test_df["original_index"].tolist()

                # Validate indices against current data
                max_valid_index = len(self.data) - 1
                train_indices = [
                    idx for idx in train_indices if 0 <= idx <= max_valid_index
                ]
                test_indices = [
                    idx for idx in test_indices if 0 <= idx <= max_valid_index
                ]

                print(
                    f"\033[KLoaded previous split - Training: {len(train_indices)}, Testing: {len(test_indices)}"
                )
                return train_indices, test_indices

            except Exception as e:
                print(f"\033[KError loading previous split: {str(e)}")
                return None, None

        return None, None

    def save_last_split(self, train_indices: list, test_indices: list):
        """Save split with original dataset indices"""
        dataset_name = self.dataset_name
        os.makedirs(f"data/{dataset_name}", exist_ok=True)

        # Save training indices with original indexes
        train_df = self.data.iloc[train_indices].copy()
        train_df["original_index"] = train_indices
        train_df.to_csv(f"data/{dataset_name}/Last_training.csv", index=False)

        # Save testing indices with original indexes
        test_df = self.data.iloc[test_indices].copy()
        test_df["original_index"] = test_indices
        test_df.to_csv(f"data/{dataset_name}/Last_testing.csv", index=False)

        print(
            f"\033[KSaved split with {len(train_indices)} train, {len(test_indices)} test samples"
        )

    # ---------------Predcit New --------------------

    # -------------Predict New -----------------------
    def predict(
        self, X: Union[pd.DataFrame, torch.Tensor], batch_size: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions in batches with consistent NaN handling.
        Returns both predicted classes and class probabilities.
        Handles both DataFrame and Tensor inputs.

        Args:
            X: Input features (DataFrame or Tensor)
            batch_size: Batch size for prediction

        Returns:
            Tuple containing:
            - predictions: Tensor of predicted class indices
            - posteriors: Tensor of class probabilities for all classes
        """
        # Ensure we have a properly initialized label encoder
        if not hasattr(self.label_encoder, "classes_"):
            if hasattr(self, "data"):
                # If we have data, fit the encoder (shouldn't happen in prediction)
                self.label_encoder.fit(self.data[self.target_column])
            else:
                raise RuntimeError(
                    "Label encoder not initialized and no data available to fit it"
                )

        # Store current weights temporarily
        temp_W = self.current_W
        self.current_W = (
            self.best_W.clone() if self.best_W is not None else self.current_W
        )

        try:
            # Convert DataFrame to tensor if needed
            if isinstance(X, pd.DataFrame):
                # Preprocess with same NaN handling as training
                X_processed = self._preprocess_data(X, is_training=False)
                X_tensor = X_processed.to(self.device)
            else:
                # For tensor input, ensure proper NaN handling
                X_tensor = X.to(self.device)
                self.nan_mask = torch.isnan(X_tensor)
                X_tensor = torch.where(
                    self.nan_mask,
                    torch.tensor(-99999, device=X_tensor.device),
                    X_tensor,
                )

            n_batches = (len(X_tensor) + batch_size - 1) // batch_size
            all_predictions = []
            all_posteriors = []

            with tqdm(
                total=n_batches, desc="Prediction batches", leave=False
            ) as pred_pbar:
                for i in range(0, len(X_tensor), batch_size):
                    batch_X = X_tensor[i : min(i + batch_size, len(X_tensor))]
                    # Get posteriors (NaN handling happens inside these methods)
                    if self.model_type == "Histogram":
                        posteriors, _ = self._compute_batch_posterior(batch_X)
                    elif self.model_type == "Gaussian":
                        posteriors, _ = self._compute_batch_posterior_std(batch_X)
                    else:
                        raise ValueError(f"Invalid model type: {self.model_type}")

                    # Get predictions and store posteriors
                    batch_predictions = torch.argmax(posteriors, dim=1)
                    all_predictions.append(batch_predictions)
                    all_posteriors.append(posteriors)
                    pred_pbar.update(1)

            # Concatenate all batches
            predictions = torch.cat(all_predictions).cpu()
            posteriors = torch.cat(all_posteriors).cpu()

            return predictions, posteriors

        finally:
            # Restore original weights
            self.current_W = temp_W

    def _save_best_weights(self):
        """Save the best weights and corresponding training data to file"""
        if self.best_W is None:
            print("\033[KWarning: No best weights to save")
            return

        try:
            # Create directory for model components
            model_dir = os.path.join("Model")
            os.makedirs(model_dir, exist_ok=True)
            print(f"\033[KCreated model directory at {model_dir}")

            # Convert weights to numpy and then to list for JSON serialization
            # weights_np = self.best_W.cpu().numpy()
            weights_np = self.current_W.cpu().numpy()

            # Save weights
            weights_dict = {
                "version": 2,
                "weights": weights_np.tolist(),
                "shape": list(weights_np.shape),
                "dtype": str(weights_np.dtype),
            }

            weights_file = os.path.join(
                "Model", f"Best_{self.model_type}_{self.dataset_name}_weights.json"
            )
            print(f"\033[KAttempting to save weights to {weights_file}")

            # Use atomic write to prevent corruption
            temp_file = weights_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(weights_dict, f, indent=2)
                f.flush()  # Ensure the buffer is flushed to disk
                os.fsync(f.fileno())  # Force write to disk

            # Atomic rename
            os.replace(temp_file, weights_file)
            print(f"\033[KSaved weights to {weights_file}")

            # Save training data if available
            if hasattr(self, "train_indices") and hasattr(self, "data"):
                train_data = self.data.iloc[self.train_indices]
                train_data_file = os.path.join(model_dir, "best_training_data.csv")

                print(f"\033[KAttempting to save training data to {train_data_file}")
                train_data.to_csv(train_data_file, index=False)
                print(f"\033[KSaved training data to {train_data_file}")

        except Exception as e:
            print(f"\033[KError saving best weights: {str(e)}")
            traceback.print_exc()

    def _load_best_weights(self):
        """Load the best weights and corresponding training data from file"""
        model_dir = os.path.join("Model")
        weights_file = os.path.join(
            "Model", f"Best_{self.model_type}_{self.dataset_name}_weights.json"
        )

        if os.path.exists(weights_file):
            try:
                print(f"Attempting to load weights file {weights_file}")
                with open(weights_file) as f:
                    weights_dict = json.load(f)

                # Load weights
                weights_array = np.array(weights_dict["weights"])
                self.best_W = torch.tensor(
                    weights_array, dtype=torch.float32, device=self.device
                )
                self.current_W = torch.tensor(
                    weights_array, dtype=torch.float32, device=self.device
                )
                print(f"Weights file {weights_file} loaded succesfully")
                # Only try to load training data if self.data exists
                if hasattr(self, "data"):
                    train_data_file = os.path.join(model_dir, "best_training_data.csv")
                    if os.path.exists(train_data_file):
                        train_data = pd.read_csv(train_data_file)
                        # Find matching indices in current data
                        self.train_indices = []
                        current_data = self.data.drop(columns=[self.target_column])

                        for idx, row in train_data.drop(
                            columns=[self.target_column]
                        ).iterrows():
                            matches = (current_data == row).all(axis=1)
                            if matches.any():
                                self.train_indices.extend(
                                    matches[matches].index.tolist()
                                )

                        print(
                            f"\033[KLoaded {len(self.train_indices)} training samples from best model"
                        )

                print(f"\033[KLoaded best weights from {weights_file}")
            except Exception as e:
                print(
                    f"\033[KWarning: Could not load weights from {weights_file}: {str(e)}"
                )
                self.best_W = None

    def _init_keyboard_listener(self):
        """Initialize keyboard listener with shared display connection"""
        if not hasattr(self, "_display"):
            try:
                import Xlib.display

                self._display = Xlib.display.Display()
            except Exception as e:
                print("\033[K" + f"Warning: Could not initialize X display: {e}")
                return None

        try:
            from pynput import keyboard

            return keyboard.Listener(
                on_press=self._on_key_press,
                _display=self._display,  # Pass shared display connection
            )
        except Exception as e:
            print("\033[K" + f"Warning: Could not create keyboard listener: {e}")
            return None

    def _cleanup_keyboard(self):
        """Clean up keyboard resources"""
        if hasattr(self, "_display"):
            try:
                self._display.close()
                del self._display
            except:
                pass

    def print_colored_confusion_matrix(
        self, y_true, y_pred, class_labels=None, header=None
    ):
        # Decode numeric labels back to original alphanumeric labels
        class_accuracies = self._calculate_class_wise_accuracy(
            torch.as_tensor(y_true), torch.as_tensor(y_pred)
        )

        # Get string labels for display only
        y_true_labels = self.label_encoder.inverse_transform(y_true)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)

        # Use minimum class accuracy as the criterion
        self.best_combined_accuracy = sum(
            [v["accuracy"] for v in class_accuracies.values()]
        ) / len(class_accuracies)

        # Get unique classes from both true and predicted labels
        unique_true = np.unique(y_true_labels)
        unique_pred = np.unique(y_pred_labels)

        # Use provided class labels or get from label encoder
        if class_labels is None:
            class_labels = self.label_encoder.classes_

        # Ensure all classes are represented in confusion matrix
        all_classes = np.unique(
            np.concatenate([unique_true, unique_pred, class_labels])
        )
        n_classes = len(all_classes)

        # Create class index mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

        # Initialize confusion matrix with zeros
        cm = np.zeros((n_classes, n_classes), dtype=int)

        # Fill confusion matrix
        for t, p in zip(y_true_labels, y_pred_labels):
            if t in class_to_idx and p in class_to_idx:
                cm[class_to_idx[t], class_to_idx[p]] += 1

        # Calculate precision for each class
        precision = []
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precision.append(prec)

        # Print confusion matrix with colors
        print(
            "\033[K"
            + f"{Colors.BOLD}Confusion Matrix and Class-wise Metrics for [{header}]:{Colors.ENDC}"
        )
        print("\033[K" + f"{'Actual/Predicted':<15}", end="")
        for label in all_classes:
            print("\033[K" + f"{str(label):<8}", end="")
        print("\033[K" + "Accuracy  Precision")
        print("\033[K" + "-" * (15 + 8 * n_classes + 20))

        # Print matrix with colors
        for i in range(n_classes):
            # Print actual class label
            print(
                "\033[K" + f"{Colors.BOLD}{str(all_classes[i]):<15}{Colors.ENDC}",
                end="",
            )

            # Print confusion matrix row
            for j in range(n_classes):
                if i == j:
                    # Correct predictions in green
                    color = Colors.GREEN
                else:
                    # Incorrect predictions in red
                    color = Colors.RED
                print("\033[K" + f"{color}{cm[i, j]:<8}{Colors.ENDC}", end="")

            # Print class accuracy with color based on performance
            acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
            if acc >= 0.9:
                acc_color = Colors.GREEN
            elif acc >= 0.7:
                acc_color = Colors.YELLOW
            else:
                acc_color = Colors.BLUE

            # Print class precision with color based on performance
            prec = precision[i]
            if prec >= 0.9:
                prec_color = Colors.GREEN
            elif prec >= 0.7:
                prec_color = Colors.YELLOW
            else:
                prec_color = Colors.BLUE

            print(
                "\033[K"
                + f"{acc_color}{acc:>7.2%}{Colors.ENDC}  {prec_color}{prec:>8.2%}{Colors.ENDC}"
            )

        # Print precision row at the bottom
        print("\033[K" + f"{Colors.BOLD}{'Precision':<15}{Colors.ENDC}", end="")
        for j in range(n_classes):
            tp = cm[j, j]
            fp = cm[:, j].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            if prec >= 0.9:
                prec_color = Colors.GREEN
            elif prec >= 0.7:
                prec_color = Colors.YELLOW
            else:
                prec_color = Colors.BLUE

            print("\033[K" + f"{prec_color}{prec:>8.2%}{Colors.ENDC}", end="")
        print("\033[K" + "")  # New line after precision row

        # Print overall accuracy and precision
        total_correct = np.diag(cm).sum()
        total_samples = cm.sum()
        if total_samples > 0:
            overall_acc = total_correct / total_samples
            # Micro-averaged precision (same as accuracy in multi-class)
            overall_prec = total_correct / total_samples

            print("\033[K" + "-" * (15 + 8 * n_classes + 20))
            acc_color = (
                Colors.GREEN
                if overall_acc >= 0.9
                else Colors.YELLOW
                if overall_acc >= 0.7
                else Colors.BLUE
            )
            prec_color = (
                Colors.GREEN
                if overall_prec >= 0.9
                else Colors.YELLOW
                if overall_prec >= 0.7
                else Colors.BLUE
            )
            print(
                "\033[K"
                + f"{Colors.BOLD}Overall Accuracy:{Colors.ENDC} {acc_color}{overall_acc:.2%}{Colors.ENDC}"
            )
            print(
                "\033[K"
                + f"{Colors.BOLD}Overall Precision:{Colors.ENDC} {prec_color}{overall_prec:.2%}{Colors.ENDC}"
            )
            print(
                "\033[K"
                + f"Best Overall (Classwise) Accuracy till now is: {Colors.GREEN}{self.best_combined_accuracy:.2%}{Colors.ENDC}"
            )

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        batch_size: int = 128,
    ):
        """Training loop with proper weight handling and enhanced progress tracking"""
        print("\033[K" + "Starting training...", end="\r", flush=True)
        # Initialize best combined accuracy if not already set
        if not hasattr(self, "best_combined_accuracy"):
            self.best_combined_accuracy = 0.0

        # Initialize best model weights if not already set
        if not hasattr(self, "best_model_weights"):
            self.best_model_weights = None

        # Store initial conditions at the start of training
        if self.best_round_initial_conditions is None:
            self.best_round_initial_conditions = {
                "weights": self.current_W.clone(),
                "likelihood_params": self.likelihood_params,
                "feature_pairs": self.feature_pairs,
                "bin_edges": self.bin_edges,
                "gaussian_params": self.gaussian_params,
            }

        # Initialize progress bar for epochs
        epoch_pbar = tqdm(total=self.max_epochs, desc="Training epochs", leave=False)

        # Store current weights for prediction during training
        train_weights = self.current_W.clone() if self.current_W is not None else None

        # Pre-allocate tensors for batch processing
        n_samples = len(X_train)
        predictions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        batch_mask = torch.empty(batch_size, dtype=torch.bool, device=self.device)

        # Initialize tracking variables
        error_rates = []
        train_losses = []
        train_accuracies = []
        float("inf")
        patience_counter = 0
        best_train_accuracy = 0.0

        if self.in_adaptive_fit:
            patience = 5
        else:
            patience = Trials

        for epoch in range(self.max_epochs):
            # Save epoch data
            self.save_epoch_data(epoch, self.train_indices, self.test_indices)

            Trstart_time = time.time()
            failed_cases = []
            n_errors = 0

            # Process training data in batches
            n_batches = (len(X_train) + batch_size - 1) // batch_size
            batch_pbar = tqdm(
                total=n_batches, desc=f"Epoch {epoch+1} batches", leave=False
            )

            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                current_batch_size = batch_end - i

                batch_X = X_train[i:batch_end]
                batch_y = y_train[i:batch_end]

                # Compute posteriors for batch
                if self.model_type == "Histogram":
                    posteriors, bin_indices = self._compute_batch_posterior(batch_X)
                elif self.model_type == "Gaussian":
                    posteriors, comp_resp = self._compute_batch_posterior_std(batch_X)

                predictions[:current_batch_size] = torch.argmax(posteriors, dim=1)
                batch_mask[:current_batch_size] = (
                    predictions[:current_batch_size] != batch_y
                )

                n_errors += batch_mask[:current_batch_size].sum().item()

                if batch_mask[:current_batch_size].any():
                    failed_indices = torch.where(batch_mask[:current_batch_size])[0]
                    for idx in failed_indices:
                        failed_cases.append(
                            (
                                batch_X[idx],
                                batch_y[idx].item(),
                                posteriors[idx].cpu().numpy(),
                            )
                        )
                batch_pbar.update(1)

            batch_pbar.close()

            # Calculate training error rate
            train_error_rate = n_errors / n_samples
            error_rates.append(train_error_rate)

            # Calculate metrics using current weights
            with torch.no_grad():
                # Temporarily set current_W for training metrics
                orig_weights = self.current_W
                self.current_W = train_weights

                # Training metrics
                # print("\033[K" +f"{Colors.GREEN}Predctions on Training data{Colors.ENDC}", end="\r", flush=True)
                train_pred_classes, train_posteriors = self.predict(
                    X_train, batch_size=batch_size
                )
                train_accuracy = (train_pred_classes == y_train.cpu()).float().mean()
                train_loss = n_errors / n_samples

                # Restore original weights
                self.current_W = orig_weights

            # Update best accuracies
            best_train_accuracy = max(best_train_accuracy, train_accuracy)

            # Store metrics
            train_losses.append(train_loss)

            train_accuracies.append(train_accuracy)

            # Calculate training time
            Trend_time = time.time()
            Trend_time - Trstart_time

            # Update progress display
            epoch_pbar.update(1)
            epoch_pbar.set_postfix(
                {
                    "train_err": f"{train_error_rate:.4f} (best: {1-best_train_accuracy:.4f})",
                    "train_acc": f"{train_accuracy:.4f} (best: {best_train_accuracy:.4f})",
                }
            )

            # print("\033[K" +f"Epoch {epoch + 1}/{self.max_epochs}:", end="\r", flush=True)
            # print("\033[K" +f"Training time: {Colors.highlight_time(training_time)} seconds", end="\r", flush=True)
            # print("\033[K" +f"Train error rate: {Colors.color_value(train_error_rate, prev_train_error, False)} (best: {1-best_train_accuracy:.4f})", end="\r", flush=True)
            # print("\033[K" +f"Train accuracy: {Colors.color_value(train_accuracy, prev_train_accuracy, True)} (best: {Colors.GREEN}{best_train_accuracy:.4f}{Colors.ENDC})", end="\r", flush=True)

            # Update previous values for next iteration

            # Check if this is the best round so far
            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy
                self.best_round = epoch
                self.best_round_initial_conditions = {
                    "weights": self.current_W.clone(),
                    "likelihood_params": self.likelihood_params,
                    "feature_pairs": self.feature_pairs,
                    "bin_edges": self.bin_edges,
                    "gaussian_params": self.gaussian_params,
                }

            # Update best model if improved
            if train_error_rate <= self.best_error:
                improvement = self.best_error - train_error_rate
                self.best_error = train_error_rate
                self.best_W = self.current_W.clone()  # Save current weights as best
                # self._save_best_weights()   # move it to train_fit

                if improvement <= 0.001:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    self.learning_rate = LearningRate
            else:
                patience_counter += 1

            # Early stopping check
            if patience_counter >= patience or train_accuracy == 1.00:
                print("\033[K" + f"{Colors.YELLOW} Early stopping.{Colors.ENDC}")
                break

            # Update weights if there were failures
            if failed_cases:
                self._update_priors_parallel(failed_cases, batch_size)
            # Save reconstruction plots if enabled
            if self.save_plots:
                # Reconstruct features from predictions
                reconstructed_features = self.reconstruct_features(posteriors)
                save_path = f"data/{self.dataset_name}/plots/epoch_{epoch+1}"
                os.makedirs(save_path, exist_ok=True)
                self._save_reconstruction_plots(
                    original_features=X_train.cpu().numpy(),
                    reconstructed_features=reconstructed_features.cpu().numpy(),
                    true_labels=y_train.cpu().numpy(),
                    save_path=save_path,
                )

        # Training complete
        epoch_pbar.close()
        # self._save_model_components()
        return self.current_W.cpu(), error_rates

    # ---------------------------------Train InvertableDBNN on the fly ------------------------------------
    def load_inverse_model(self, custom_path: str = None) -> bool:
        try:
            load_dir = custom_path or os.path.join(
                "Model", f"Best_inverse_{self.forward_model.dataset_name}"
            )
            model_path = os.path.join(load_dir, "inverse_model.pt")
            config_path = os.path.join(load_dir, "inverse_config.json")

            if not (os.path.exists(model_path) and os.path.exists(config_path)):
                print("\033[K" + f"No saved inverse model found at {load_dir}")
                return False

            model_state = torch.load(
                model_path, map_location=self.device, weights_only=True
            )

            with open(config_path) as f:
                config = json.load(f)

            if (
                config["feature_dims"] != self.feature_dims
                or config["n_classes"] != self.n_classes
            ):
                raise ValueError("Model architecture mismatch")

            # Load parameters
            self.weight_linear.data = model_state["weight_linear"]
            self.weight_nonlinear.data = model_state["weight_nonlinear"]
            self.bias_linear.data = model_state["bias_linear"]
            self.bias_nonlinear.data = model_state["bias_nonlinear"]
            self.feature_attention.data = model_state["feature_attention"]
            self.layer_norm.load_state_dict(model_state["layer_norm"])

            # Safely update or register buffers
            for param in [
                "min_vals",
                "max_vals",
                "scale_factors",
                "inverse_feature_pairs",
            ]:
                if param in model_state:
                    buffer_data = model_state[param]
                    if buffer_data is not None:
                        if hasattr(self, param) and getattr(self, param) is not None:
                            getattr(self, param).copy_(buffer_data)
                        else:
                            self.register_buffer(param, buffer_data)

            # Restore other attributes
            self.metrics = model_state.get("metrics", {})
            self.reconstruction_weight = model_state.get("reconstruction_weight", 0.5)
            self.feedback_strength = model_state.get("feedback_strength", 0.3)

            print("\033[K" + f"Loaded inverse model from {load_dir}")
            return True

        except Exception as e:
            print("\033[K" + f"Error loading inverse model: {str(e)}")
            traceback.print_exc()
            return False

    def save_reconstruction_features(
        self,
        reconstructed_features: torch.Tensor,
        original_features: torch.Tensor,
        predictions: torch.Tensor,
        true_labels: torch.Tensor = None,
        class_probs: torch.Tensor = None,
    ) -> Dict:
        """Save reconstruction features and return JSON-compatible output.

        Args:
            reconstructed_features: Reconstructed feature tensor
            original_features: Original input feature tensor
            predictions: Model predictions tensor
            true_labels: True labels tensor (optional)
            class_probs: Class probabilities tensor (optional)

        Returns:
            Dict containing reconstruction data and paths
        """
        # Create reconstruction directory
        dataset_name = os.path.splitext(os.path.basename(self.dataset_name))[0]
        recon_dir = os.path.join("data", dataset_name, "reconstruction")
        os.makedirs(recon_dir, exist_ok=True)

        # Convert tensors to numpy arrays
        recon_np = reconstructed_features.cpu().numpy()
        orig_np = original_features.cpu().numpy()
        pred_np = predictions.cpu().numpy()

        # Create DataFrame with original and reconstructed features
        feature_cols = [f"feature_{i}" for i in range(orig_np.shape[1])]
        recon_cols = [f"reconstructed_{i}" for i in range(recon_np.shape[1])]

        df = pd.DataFrame(orig_np, columns=feature_cols)
        df = pd.concat([df, pd.DataFrame(recon_np, columns=recon_cols)], axis=1)

        # Add predictions
        df["predicted_class"] = self.label_encoder.inverse_transform(pred_np)

        # Add true labels if provided
        if true_labels is not None:
            true_np = true_labels.cpu().numpy()
            df["true_class"] = self.label_encoder.inverse_transform(true_np)

        # Add class probabilities if provided
        if class_probs is not None:
            probs_np = class_probs.cpu().numpy()
            for i, class_name in enumerate(self.label_encoder.classes_):
                df[f"prob_{class_name}"] = probs_np[:, i]

        # Add reconstruction error
        df["reconstruction_error"] = np.mean((orig_np - recon_np) ** 2, axis=1)

        # Save to CSV
        csv_path = os.path.join(
            recon_dir, f"data/{dataset_name}/{dataset_name}_reconstruction.csv"
        )
        df.to_csv(csv_path, index=False)

        # Create JSON-compatible output
        output = {
            "dataset": dataset_name,
            "reconstruction_path": csv_path,
            "feature_count": orig_np.shape[1],
            "sample_count": len(df),
            "mean_reconstruction_error": float(df["reconstruction_error"].mean()),
            "std_reconstruction_error": float(df["reconstruction_error"].std()),
            "features": {"original": feature_cols, "reconstructed": recon_cols},
            "class_mapping": dict(
                zip(
                    range(len(self.label_encoder.classes_)), self.label_encoder.classes_
                )
            ),
        }

        # Save metadata as JSON
        json_path = os.path.join(recon_dir, f"{dataset_name}_reconstruction_meta.json")
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)

        return output

    # ------------------------------End Train InvertableDBNN on the fly ------------------------------------

    def plot_training_metrics(
        self, train_loss, test_loss, train_acc, test_acc, save_path=None
    ):
        """Plot training and testing metrics over epochs"""
        plt.figure(figsize=(12, 8))

        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(train_loss, label="Train Loss", marker="o")
        plt.plot(test_loss, label="Test Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Over Epochs")
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(train_acc, label="Train Accuracy", marker="o")
        plt.plot(test_acc, label="Test Accuracy", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Over Epochs")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            # Also save metrics to CSV
            metrics_df = pd.DataFrame(
                {
                    "epoch": range(1, len(train_loss) + 1),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                }
            )
            metrics_df.to_csv(save_path.replace(".png", "_metrics.csv"), index=False)

        plt.close()

    def verify_classifications(
        self, X: pd.DataFrame, true_labels: pd.Series, predictions: torch.Tensor
    ) -> None:
        """
        Verify classification accuracy computation with proper index handling
        """
        # Convert predictions to numpy array if it's a tensor
        pred_labels = self.label_encoder.inverse_transform(
            predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        )

        # Ensure true_labels is a numpy array
        true_labels_array = (
            true_labels.to_numpy()
            if isinstance(true_labels, pd.Series)
            else true_labels
        )

        # Calculate accuracy metrics
        n_total = len(true_labels_array)
        correct_mask = true_labels_array == pred_labels
        n_correct = correct_mask.sum()

        # Print basic metrics with colors
        print(
            "\033[K"
            + f"{Colors.BOLD}{Colors.BLUE}Detailed Classification Analysis:{Colors.ENDC}"
        )
        print(
            "\033[K"
            + f"{Colors.BOLD}Total samples:{Colors.ENDC} {Colors.YELLOW}{n_total:,}{Colors.ENDC}"
        )

        # Color code for correctly classified
        correct_color = (
            Colors.GREEN
            if (n_correct / n_total) >= 0.9
            else Colors.YELLOW
            if (n_correct / n_total) >= 0.7
            else Colors.BLUE
        )
        print(
            "\033[K"
            + f"{Colors.BOLD}Correctly classified:{Colors.ENDC} {correct_color}{n_correct:,}{Colors.ENDC}"
        )

        # Color code for incorrectly classified
        incorrect = n_total - n_correct
        incorrect_color = (
            Colors.GREEN
            if (incorrect / n_total) <= 0.1
            else Colors.YELLOW
            if (incorrect / n_total) <= 0.3
            else Colors.RED
        )
        print(
            "\033[K"
            + f"{Colors.BOLD}Incorrectly classified:{Colors.ENDC} {incorrect_color}{incorrect:,}{Colors.ENDC}"
        )

        # Color code for raw accuracy
        accuracy = n_correct / n_total
        accuracy_color = (
            Colors.GREEN
            if accuracy >= 0.9
            else Colors.YELLOW
            if accuracy >= 0.7
            else Colors.BLUE
        )
        print(
            "\033[K"
            + f"{Colors.BOLD}Raw accuracy:{Colors.ENDC} {accuracy_color}{accuracy:.4%}{Colors.ENDC}\n"
        )

        # Print confusion matrix with colors
        self.print_colored_confusion_matrix(
            true_labels_array, pred_labels, header="Test data"
        )

        # Save detailed analysis to file
        analysis_file = f"classification_analysis_{self.dataset_name}.txt"
        with open(analysis_file, "w") as f:
            f.write("Per-class breakdown:\n")
            for cls in np.unique(true_labels_array):
                cls_mask = true_labels_array == cls
                n_cls = cls_mask.sum()
                n_correct_cls = (correct_mask & cls_mask).sum()

                f.write(f"\nClass {cls}:\n")
                f.write(f"Total samples: {n_cls}\n")
                f.write(f"Correctly classified: {n_correct_cls}\n")
                f.write(f"Class accuracy: {n_correct_cls/n_cls:.4f}\n")

                if n_cls - n_correct_cls > 0:
                    # Find misclassified examples
                    misclassified_mask = ~correct_mask & cls_mask
                    mis_predictions = pred_labels[misclassified_mask]
                    unique_mispred, counts = np.unique(
                        mis_predictions, return_counts=True
                    )

                    f.write("\nMisclassified as:\n")
                    for pred_cls, count in zip(unique_mispred, counts):
                        f.write(f"{pred_cls}: {count}\n")

                    # Save examples of misclassified instances
                    f.write("\nSample misclassified instances:\n")
                    misclassified_indices = np.where(misclassified_mask)[0]
                    for idx in misclassified_indices[:5]:  # Show up to 5 examples
                        f.write(f"\nInstance {idx}:\n")
                        f.write(f"True class: {true_labels_array[idx]}\n")
                        f.write(f"Predicted class: {pred_labels[idx]}\n")
                        f.write("Feature values:\n")
                        for col, val in X.iloc[idx].items():
                            f.write(f"{col}: {val}\n")

        print("\033[K" + f"Detailed analysis saved to {analysis_file}")

    # ------------------------------------------------------------End of PP code ---------------------------------------------------
    def _compute_pairwise_likelihood(self, dataset, labels, feature_dims):
        """Compute pairwise likelihood PDFs"""
        unique_classes = torch.unique(labels)
        feature_pairs = list(combinations(range(feature_dims), 2))
        likelihood_pdfs = {}

        for class_id in unique_classes:
            class_mask = labels == class_id
            class_data = dataset[class_mask]
            likelihood_pdfs[class_id.item()] = {}

            for feat_i, feat_j in feature_pairs:
                pair_data = torch.stack(
                    [class_data[:, feat_i], class_data[:, feat_j]], dim=1
                )

                mean = torch.mean(pair_data, dim=0)
                centered_data = pair_data - mean
                cov = torch.mm(centered_data.T, centered_data) / (len(pair_data) - 1)
                cov = cov + torch.eye(2) * 1e-6

                likelihood_pdfs[class_id.item()][(feat_i, feat_j)] = {
                    "mean": mean,
                    "cov": cov,
                }

        return likelihood_pdfs

    def _get_weights_filename(self):
        """Get the filename for saving/loading weights"""
        return os.path.join(
            "Model", f"Best_{self.model_type}_{self.dataset_name}_weights.json"
        )

    def _get_encoders_filename(self):
        """Get the filename for saving/loading categorical encoders"""
        return os.path.join(
            "Model", f"Best_{self.model_type}_{self.dataset_name}_encoders.json"
        )

    def _remove_high_cardinality_columns(
        self, df: pd.DataFrame, threshold: float = 0.8
    ) -> pd.DataFrame:
        """Remove high cardinality columns and round features with detailed debugging"""
        DEBUG.log(f" Starting high cardinality removal with threshold {threshold}")
        DEBUG.log(f" Initial columns: {df.columns.tolist()}")

        # Round all features first if cardinality_tolerance is not -1
        if self.cardinality_tolerance != -1:
            DEBUG.log(f" Rounding features with tolerance {self.cardinality_tolerance}")
            df_rounded = self._round_features(df)
        else:
            df_rounded = df.copy()

        df_filtered = df_rounded.copy()
        columns_to_drop = []

        for column in df.columns:
            if column == self.target_column:
                continue

            # Use rounded data for cardinality check
            unique_count = len(df_rounded[column].unique())
            unique_ratio = unique_count / len(df)
            DEBUG.log(
                f" Column {column}: {unique_count} unique values, ratio {unique_ratio:.4f}"
            )

            if unique_ratio > threshold:
                columns_to_drop.append(column)
                DEBUG.log(
                    f" Marking {column} for removal (ratio {unique_ratio:.4f} > {threshold})"
                )

        if columns_to_drop:
            DEBUG.log(f" Dropping columns: {columns_to_drop}")
            df_filtered = df_filtered.drop(columns=columns_to_drop)

        DEBUG.log(f" Columns after filtering: {df_filtered.columns.tolist()}")
        DEBUG.log(f" Remaining features: {len(df_filtered.columns)}")

        if len(df_filtered.columns) == 0:
            print(
                "\033[K"
                + "[WARNING] All features were removed! Reverting to original features with warnings."
            )
            return df.copy()

        return df_filtered

    def _detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect categorical columns with improved debugging"""
        DEBUG.log(" Starting categorical column detection")
        categorical_columns = []

        for column in df.columns:
            if column != self.target_column:
                dtype = df[column].dtype
                unique_count = len(df[column].unique())
                DEBUG.log(
                    f" Column {column}: dtype={dtype}, unique values={unique_count}"
                )

                if dtype == "object" or dtype.name == "category":
                    DEBUG.log(f" Adding {column} as categorical (object/category type)")
                    categorical_columns.append(column)
                elif dtype in ["int64", "float64"]:
                    threshold = min(50, len(df) * 0.05)
                    if unique_count < threshold:
                        DEBUG.log(
                            f" Adding {column} as categorical (numeric with few unique values: {unique_count} < {threshold})"
                        )
                        categorical_columns.append(column)
                    else:
                        DEBUG.log(
                            f" Keeping {column} as numeric (unique values: {unique_count} >= {threshold})"
                        )

        DEBUG.log(f" Detected categorical columns: {categorical_columns}")
        return categorical_columns

    def _get_train_test_split(self, X_tensor, y_tensor):
        """Get or create consistent train-test split using smaller chunks to avoid memory issues."""
        dataset_folder = os.path.splitext(os.path.basename(self.dataset_name))[0]
        base_path = self.config.get("training_params", {}).get(
            "training_save_path", "training_data"
        )
        split_path = os.path.join(base_path, dataset_folder, "train_test_split.pkl")

        if os.path.exists(split_path):
            with open(split_path, "rb") as f:
                split_indices = pickle.load(f)
                train_idx, test_idx = split_indices["train"], split_indices["test"]
                return (
                    X_tensor[train_idx],
                    X_tensor[test_idx],
                    y_tensor[train_idx],
                    y_tensor[test_idx],
                )

        # If no saved split exists, create one using smaller chunks
        X_cpu = X_tensor.cpu().numpy()  # Move data to CPU for sklearn compatibility
        y_cpu = y_tensor.cpu().numpy()

        # Perform train-test split on CPU in smaller chunks
        X_train, X_test, y_train, y_test = train_test_split(
            X_cpu,  # Use NumPy array for sklearn compatibility
            y_cpu,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=(self.shuffle_state != -1),
        )

        # Convert back to PyTorch tensors and move to the appropriate device
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(self.device)

        # Save split indices to avoid recomputing
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        split_indices = {
            "train": torch.arange(len(X_train)),  # Indices for training set
            "test": torch.arange(
                len(X_train), len(X_train) + len(X_test)
            ),  # Indices for test set
        }
        with open(split_path, "wb") as f:
            pickle.dump(split_indices, f)

        # Store test indices for later use
        self.test_indices = split_indices["test"]

        return X_train, X_test, y_train, y_test

    def _train_test_split_tensor(self, X, y, test_size, random_state):
        """Split data consistently using fixed indices"""
        num_samples = len(X)

        # Generate fixed permutation
        if random_state == -1:
            # Use numpy's random permutation directly
            indices = torch.from_numpy(np.random.permutation(num_samples))
        else:
            rng = np.random.RandomState(random_state)
            indices = torch.from_numpy(rng.permutation(num_samples))

        split = int(num_samples * (1 - test_size))
        train_idx = indices[:split]
        test_idx = indices[split:]

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def _multivariate_normal_pdf(self, x, mean, cov):
        """Compute multivariate normal PDF"""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        dim = 2
        centered_x = x - mean.unsqueeze(0)
        inv_cov = torch.inverse(cov)
        det = torch.det(cov)
        quad_form = torch.sum(torch.mm(centered_x, inv_cov) * centered_x, dim=1)
        norm_const = 1.0 / (torch.sqrt((2 * torch.tensor(np.pi)) ** dim * det))
        return norm_const * torch.exp(-0.5 * quad_form)

    def _initialize_priors(self):
        """Initialize weights"""
        if self.best_W is not None:
            return self.best_W

        W = {}
        for class_id in self.likelihood_pdfs.keys():
            W[class_id] = {}
            for feature_pair in self.likelihood_pdfs[class_id].keys():
                W[class_id][feature_pair] = torch.tensor(0.1, dtype=torch.float32)
        return W

    def _calculate_class_wise_accuracy(self, y_true, y_pred):
        """Calculate class-wise accuracy metrics with device handling"""
        # Ensure both tensors are on the same device (preferably CPU for this operation)
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

        unique_classes = torch.unique(y_true)
        class_accuracies = {}

        for class_id in unique_classes:
            class_mask = y_true == class_id
            n_class_samples = class_mask.sum().item()
            if n_class_samples == 0:
                continue

            correct = (y_pred[class_mask] == y_true[class_mask]).sum().item()
            class_acc = correct / n_class_samples
            class_accuracies[class_id.item()] = {
                "accuracy": class_acc,
                "n_samples": n_class_samples,
                "correct": correct,
            }

        return class_accuracies

    def fit_predict(self, batch_size: int = 128, save_path: str = None):
        """Full training and prediction pipeline with GPU optimization and optional prediction saving"""
        try:
            # Set a flag to indicate we're printing metrics
            self._last_metrics_printed = True
            class_preference = self.config.get("training_params", {}).get(
                "class_preference", True
            )

            # If this is a fresh training round, reset to the best round's initial conditions
            if self.best_round_initial_conditions is not None:
                print(
                    "\033[K"
                    + "Starting fresh training with best round's initial conditions",
                    end="\r",
                    flush=True,
                )
                self.current_W = self.best_round_initial_conditions["weights"].clone()
                self.likelihood_params = self.best_round_initial_conditions[
                    "likelihood_params"
                ]
                self.feature_pairs = self.best_round_initial_conditions["feature_pairs"]
                self.bin_edges = self.best_round_initial_conditions["bin_edges"]
                self.gaussian_params = self.best_round_initial_conditions[
                    "gaussian_params"
                ]

            # Handle data preparation based on whether we're in adaptive training or final evaluation
            if self.in_adaptive_fit:
                if not hasattr(self, "X_tensor") or not hasattr(self, "y_tensor"):
                    raise ValueError(
                        "X_tensor or y_tensor not found. Initialize them in adaptive_fit_predict first."
                    )

                if not hasattr(self, "train_indices") or not hasattr(
                    self, "test_indices"
                ):
                    raise ValueError("train_indices or test_indices not found")

                # Use stored tensors and indices, but verify sizes match
                try:
                    X_train = self.X_tensor[self.train_indices]
                    X_test = self.X_tensor[self.test_indices]
                    y_train = self.y_tensor[self.train_indices]
                    y_test = self.y_tensor[self.test_indices]
                except Exception as e:
                    # If there's any issue with indices, fall back to regular training path
                    DEBUG.log(
                        f"Error using stored indices: {str(e)}. Falling back to regular training."
                    )
                    self.in_adaptive_fit = False
                    # Reset indices and proceed with regular path
                    self.train_indices = None
                    self.test_indices = None
                    return self.fit_predict(batch_size=batch_size, save_path=save_path)

            else:
                # Regular training path
                X = self.data.drop(columns=[self.target_column])
                y = self.data[self.target_column]

                # Check if label encoder is already fitted
                if not hasattr(self.label_encoder, "classes_"):
                    y_encoded = self.label_encoder.fit_transform(y)
                else:
                    y_encoded = self.label_encoder.transform(y)

                # Preprocess features including categorical encoding
                X_processed = self._preprocess_data(X, is_training=True)

                # Convert to tensors and move to device
                X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(
                    self.device
                )
                y_tensor = torch.LongTensor(y_encoded).to(self.device)

                # Split data
                # Get consistent train-test split
                X_train, X_test, y_train, y_test = self._get_train_test_split(
                    X_tensor, y_tensor
                )

                # Convert split data back to tensors
                X_train = torch.from_numpy(X_train).to(self.device, dtype=torch.float32)
                X_test = torch.from_numpy(X_test).to(self.device, dtype=torch.float32)
                y_train = torch.from_numpy(y_train).to(self.device, dtype=torch.long)
                y_test = torch.from_numpy(y_test).to(self.device, dtype=torch.long)

            # Verify tensor sizes match before training
            if X_train.size(0) != y_train.size(0) or X_test.size(0) != y_test.size(0):
                raise ValueError(
                    f"Tensor size mismatch. X_train: {X_train.size(0)}, y_train: {y_train.size(0)}, "
                    f"X_test: {X_test.size(0)}, y_test: {y_test.size(0)}"
                )

            # Train model
            final_W, error_rates = self.train(
                X_train, y_train, X_test, y_test, batch_size=batch_size
            )

            # Save categorical encoders
            self._save_categorical_encoders()

            # Make predictions on the entire dataset
            print(
                "\033[K"
                + f"{Colors.YELLOW}Generating predictions for the entire dataset{Colors.ENDC}",
                end="\r",
                flush=True,
            )
            X_all = torch.cat([X_train, X_test], dim=0)
            y_all = torch.cat([y_train, y_test], dim=0)

            all_pred_classes, all_posteriors = self.predict(
                X_all, batch_size=batch_size
            )

            # Move tensors to CPU for accuracy calculation
            y_all_cpu = y_all.cpu()
            all_pred_classes_cpu = all_pred_classes.cpu()

            # Calculate accuracy metrics
            if class_preference:
                # Calculate class-wise accuracy on CPU
                class_accuracies = self._calculate_class_wise_accuracy(
                    y_all_cpu, all_pred_classes_cpu
                )

                # Use minimum class accuracy as the criterion
                current_metric = sum(
                    [v["accuracy"] for v in class_accuracies.values()]
                ) / len(class_accuracies)
                best_metric = self.best_combined_accuracy

                # Print class-wise metrics
                print("\033[K" + f"{Colors.GREEN}Class-wise accuracies:{Colors.ENDC}")
                for class_id, metrics in class_accuracies.items():
                    class_name = self.label_encoder.inverse_transform([class_id])[0]
                    print(
                        f"\033[K  {class_name}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['n_samples']})"
                    )
                print(f"\033[KClass-wise accuracy: {current_metric:.2%}")
            else:
                # Original behavior - overall accuracy calculated on CPU
                current_metric = (
                    (y_all_cpu == all_pred_classes_cpu).float().mean().item()
                )
                best_metric = self.best_combined_accuracy

            # Update best model if improved
            if current_metric > best_metric:
                print(
                    "\033[K"
                    + f"{Colors.RED}---------------------------------------------------------------------------------------{Colors.ENDC}"
                )
                if class_preference:
                    print(
                        "\033[K"
                        + f"{Colors.GREEN}Best minimum class accuracy improved from {best_metric:.2%} to {current_metric:.2%}{Colors.ENDC}"
                    )
                    # Store the mean class accuracy as best_combined_accuracy
                    mean_class_acc = sum(
                        [v["accuracy"] for v in class_accuracies.values()]
                    ) / len(class_accuracies)
                    self.best_combined_accuracy = (
                        current_metric  # maximise the classwise accuracies
                    )
                    print(f"The mean class accuracy is {mean_class_acc}")
                else:
                    print(
                        "\033[K"
                        + f"{Colors.GREEN}Best classwise accuracy improved from {best_metric:.2%} to {current_metric:.2%}{Colors.ENDC}"
                    )
                    self.best_combined_accuracy = current_metric
                print(
                    "\033[K"
                    + f"{Colors.RED}---------------------------------------------------------------------------------------{Colors.ENDC}"
                )

                self._save_model_components()
            else:
                print(
                    "\033[K"
                    + f"{Colors.RED}Current classwise accuracy declined from {best_metric:.2%} to {current_metric:.2%}{Colors.ENDC}"
                )
            self.reset_to_initial_state()  # After saving the weights, reset to initial state for next round.

            # Extract predictions for training and test data using stored indices
            y_train_pred = all_pred_classes[
                : len(y_train)
            ]  # Predictions for training data
            y_test_pred = all_pred_classes[len(y_train) :]  # Predictions for test data

            # Generate detailed predictions for the entire dataset
            print(
                "\033[K" + "Computing detailed predictions for the whole data",
                end="\r",
                flush=True,
            )
            all_results = self._generate_detailed_predictions(
                self.X_Orig, all_pred_classes, y_all, all_posteriors
            )
            train_results = all_results.iloc[self.train_indices]
            test_results = all_results.iloc[self.test_indices]
            # Filter failed examples (where predicted class != true class)
            failed_examples = all_results[
                all_results["predicted_class"] != all_results["true_class"]
            ]
            # Filter passed examples (where predicted class == true class)
            passed_examples = all_results[
                all_results["predicted_class"] == all_results["true_class"]
            ]

            # Save results if path is provided
            if save_path:
                # Save training predictions
                print("\033[K" + "Saving Train predictions", end="\r", flush=True)
                train_results.to_csv(f"{save_path}/train_predictions.csv", index=False)

                # Save test predictions
                print("\033[K" + "Saving Test predictions", end="\r", flush=True)
                test_results.to_csv(f"{save_path}/test_predictions.csv", index=False)

                # Save all predictions
                print("\033[K" + "Saving Combined predictions", end="\r", flush=True)
                all_results.to_csv(f"{save_path}/combined_predictions.csv", index=False)

                # Save failed examples
                print("\033[K" + "Saving Failed examples", end="\r", flush=True)
                failed_examples.to_csv(f"{save_path}/failed_examples.csv", index=False)

                # Save passed examples
                print("\033[K" + "Saving Passed examples", end="\r", flush=True)
                passed_examples.to_csv(f"{save_path}/passed_examples.csv", index=False)

                # Save metadata
                print("\033[K" + "Saving Metadata", end="\r", flush=True)
                metadata = {
                    "rejected_columns": self.high_cardinality_columns,
                    "feature_columns": self.feature_columns,
                    "target_column": self.target_column,
                    "preprocessing_details": {
                        "cardinality_threshold": self.cardinality_threshold,
                        "cardinality_tolerance": self.cardinality_tolerance,
                        "categorical_columns": list(self.categorical_encoders.keys()),
                    },
                }
                with open(f"{save_path}/metadata.json", "w") as f:
                    json.dump(metadata, f, indent=4)

            # Calculate metrics for test set
            y_test_cpu = y_test.cpu().numpy()
            y_train_cpu = y_train.cpu().numpy()

            # Convert numerical labels back to original classes
            y_test_labels = self.label_encoder.inverse_transform(y_test_cpu)
            self.label_encoder.inverse_transform(y_train_cpu)
            y_test_pred_labels = self.label_encoder.inverse_transform(y_test_pred)
            self.label_encoder.inverse_transform(y_train_pred)

            # Prepare results
            print("\033[K" + "Preparing results of training", end="\r", flush=True)
            results = {
                "all_predictions": all_results,
                "train_predictions": train_results,
                "test_predictions": test_results,
                "metadata": metadata,
                "classification_report": classification_report(
                    y_test_labels, y_test_pred_labels
                ),
                "confusion_matrix": confusion_matrix(y_test_labels, y_test_pred_labels),
                "error_rates": error_rates,
                "test_accuracy": (np.asarray(y_test_pred) == np.asarray(y_test_cpu))
                .astype(float)
                .mean()
                .item(),  # Convert to float before mean
                "train_accuracy": (np.asarray(y_train_pred) == np.asarray(y_train_cpu))
                .astype(float)
                .mean()
                .item(),  # Convert to float before mean
            }

            # Generate point-colored confusion matrices for train, test, and combined data
            print(
                "\033[K" + f"{Colors.BOLD}Generating Confusion Matrices:{Colors.ENDC}",
                end="\r",
                flush=True,
            )

            # Confusion matrix for training data
            self.print_colored_confusion_matrix(
                y_train_cpu, y_train_pred, header="Training Data"
            )

            # Confusion matrix for test data
            self.print_colored_confusion_matrix(
                y_test_cpu, y_test_pred, header="Test Data"
            )

            # Confusion matrix for combined data
            y_all_cpu = y_all.cpu().numpy()
            self.print_colored_confusion_matrix(
                y_all_cpu, all_pred_classes.cpu().numpy(), header="Combined Data"
            )

            return results

        except Exception as e:
            DEBUG.log(f"Error in fit_predict: {str(e)}")
            DEBUG.log(f"Traceback: {traceback.format_exc()}")
            raise

    def _get_model_components_filename(self):
        """Get filename for model components"""
        return os.path.join(
            "Model", f"Best_{self.model_type}_{self.dataset_name}_components.pkl"
        )

    # ----------------Handling categorical variables across sessions -------------------------
    def _save_categorical_encoders(self):
        """Save categorical feature encoders"""
        if self.categorical_encoders:
            # Create a serializable dictionary structure
            encoders_dict = {
                "encoders": {
                    column: {str(k): v for k, v in mapping.items()}
                    for column, mapping in self.categorical_encoders.items()
                }
            }

            # Add metadata
            if hasattr(self, "original_columns"):
                if isinstance(self.original_columns, list):
                    column_types = {
                        col: str(self.data[col].dtype)
                        for col in self.original_columns
                        if col in self.data.columns
                    }
                else:
                    column_types = {
                        col: str(dtype) for col, dtype in self.original_columns.items()
                    }

                encoders_dict["metadata"] = {
                    "column_types": column_types,
                    "timestamp": pd.Timestamp.now().isoformat(),
                }

            with open(self._get_encoders_filename(), "w") as f:
                json.dump(encoders_dict, f, indent=2)

    def _load_categorical_encoders(self):
        """Load categorical feature encoders from file"""
        encoders_file = self._get_encoders_filename()
        if os.path.exists(encoders_file):
            try:
                with open(encoders_file) as f:
                    data = json.load(f)

                # Extract encoders from the loaded data
                if "encoders" in data:
                    self.categorical_encoders = {
                        column: {
                            k: int(v) if isinstance(v, (str, int, float)) else v
                            for k, v in mapping.items()
                        }
                        for column, mapping in data["encoders"].items()
                    }
                else:
                    # Handle legacy format where encoders were at top level
                    self.categorical_encoders = {
                        column: {
                            k: int(v) if isinstance(v, (str, int, float)) else v
                            for k, v in mapping.items()
                        }
                        for column, mapping in data.items()
                    }

                print(
                    "\033[K" + f"Loaded categorical encoders from {encoders_file}",
                    end="\r",
                    flush=True,
                )
            except Exception as e:
                print(
                    "\033[K" + f"Warning: Failed to load categorical encoders: {str(e)}"
                )
                self.categorical_encoders = {}

    def _encode_categorical_features(
        self, df: pd.DataFrame, is_training: bool = False
    ) -> pd.DataFrame:
        """Encode categorical features with proper type handling"""
        df_encoded = df.copy()

        for column in df.columns:
            if column in self.categorical_encoders or (
                df[column].dtype == "object" or df[column].dtype.name == "category"
            ):
                # Store original dtype
                original_dtype = df[column].dtype

                # Convert to string for consistent encoding
                col_data = df[column].astype(str)

                if is_training or column not in self.categorical_encoders:
                    # Training mode or new column - create new mapping
                    unique_values = col_data.unique()
                    mapping = {v: i for i, v in enumerate(unique_values)}
                    self.categorical_encoders[column] = mapping
                else:
                    # Prediction mode - use existing mapping
                    mapping = self.categorical_encoders[column]

                # Encode values
                encoded = col_data.map(mapping)

                # Handle unseen values during prediction
                if not is_training:
                    unmapped = encoded.isna()
                    if unmapped.any():
                        # Use mean value of known encodings for unseen values
                        mean_value = np.mean(list(mapping.values()))
                        # Convert mean to original dtype before assignment
                        if np.issubdtype(original_dtype, np.integer):
                            mean_value = int(round(mean_value))

                        # Create new series with correct dtype
                        fill_values = pd.Series(
                            [mean_value] * len(unmapped), index=unmapped[unmapped].index
                        )
                        fill_values = fill_values.astype(original_dtype)

                        # Assign with loc to avoid dtype warnings
                        encoded = encoded.astype(float)  # Intermediate float type
                        encoded.loc[unmapped] = fill_values

                # Convert back to original dtype
                df_encoded[column] = encoded.astype(original_dtype)

        return df_encoded

    def save_predictions(
        self,
        X: pd.DataFrame,
        predictions: torch.Tensor,
        output_file: str,
        true_labels: pd.Series = None,
    ):
        """Save predictions with proper class handling and probability computation"""
        predictions = predictions.cpu()

        # Create a copy of the original dataset to preserve all columns
        result_df = X.copy()

        # Convert predictions to original class labels
        pred_labels = self.label_encoder.inverse_transform(predictions.numpy())
        result_df["predicted_class"] = pred_labels

        if true_labels is not None:
            result_df["true_class"] = true_labels

        # Get preprocessed features for probability computation
        X_processed = self._preprocess_data(X, is_training=False)
        if isinstance(X_processed, torch.Tensor):
            X_tensor = X_processed.clone().detach().to(self.device)
        else:
            X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(self.device)

        # Compute probabilities in batches
        batch_size = 128
        all_probabilities = []

        for i in range(0, len(X_tensor), batch_size):
            batch_end = min(i + batch_size, len(X_tensor))
            batch_X = X_tensor[i:batch_end]

            try:
                if self.model_type == "Histogram":
                    batch_probs, _ = self._compute_batch_posterior(batch_X)
                elif self.model_type == "Gaussian":
                    batch_probs, _ = self._compute_batch_posterior_std(batch_X)
                else:
                    raise ValueError(f"{self.model_type} is invalid")

                all_probabilities.append(batch_probs.cpu().numpy())

            except Exception as e:
                print(
                    "\033[K" + f"Error computing probabilities for batch {i}: {str(e)}"
                )
                return None

        if all_probabilities:
            all_probabilities = np.vstack(all_probabilities)
        else:
            print(
                "\033[K" + "No probabilities were computed successfully",
                end="\r",
                flush=True,
            )
            return None

        # Ensure we're only using valid class indices
        valid_classes = self.label_encoder.classes_
        n_classes = len(valid_classes)

        # Verify probability array shape matches number of classes
        if all_probabilities.shape[1] != n_classes:
            print(
                "\033[K"
                + f"Warning: Probability array shape ({all_probabilities.shape}) doesn't match number of classes ({n_classes})"
            )
            # Adjust probabilities array if necessary
            if all_probabilities.shape[1] > n_classes:
                all_probabilities = all_probabilities[:, :n_classes]
            else:
                # Pad with zeros if needed
                pad_width = ((0, 0), (0, n_classes - all_probabilities.shape[1]))
                all_probabilities = np.pad(
                    all_probabilities, pad_width, mode="constant"
                )

        # Add probability columns for each valid class
        for i, class_name in enumerate(valid_classes):
            if i < all_probabilities.shape[1]:  # Safety check
                result_df[f"prob_{class_name}"] = all_probabilities[:, i]

        # Add maximum probability
        result_df["max_probability"] = all_probabilities.max(axis=1)

        # Create the output directory if it doesn't exist
        dataset_name = os.path.splitext(os.path.basename(self.dataset_name))[0]
        output_dir = os.path.join("data", dataset_name, "Predictions")
        os.makedirs(output_dir, exist_ok=True)

        # Save the predictions file in the new directory
        output_path = os.path.join(output_dir, output_file)
        result_df.to_csv(output_path, index=False)
        print(
            "\033[K" + f"{Colors.GREEN}Saved predictions to {output_path}{Colors.ENDC}",
            end="\r",
            flush=True,
        )

        if true_labels is not None:
            # Verification analysis
            self.verify_classifications(X, true_labels, predictions)

        return result_df

    # --------------------------------------------------------------------------------------------------------------

    def _save_model_components(self):
        """Enhanced model component saving with validation and atomic writes"""
        try:
            # Validate critical components exist before saving
            required_components = {
                "scaler": self.scaler,
                "label_encoder": self.label_encoder,
                "feature_pairs": self.feature_pairs,
                "model_type": self.model_type,
                "target_column": self.target_column,
                "n_bins_per_dim": self.n_bins_per_dim,
            }

            for name, component in required_components.items():
                if component is None:
                    raise ValueError(f"Cannot save model: {name} is None")

            # Validate label encoder state
            if not hasattr(self.label_encoder, "classes_"):
                raise ValueError("Label encoder not properly fitted")

            # Prepare components dictionary with validation
            components = {
                "version": 3,  # Version identifier for compatibility
                "scaler": self.scaler,
                "label_encoder": {
                    "classes_": self.label_encoder.classes_.tolist(),
                    "fitted": hasattr(self.label_encoder, "classes_"),
                },
                "model_type": self.model_type,
                "feature_pairs": self.feature_pairs.cpu().tolist()
                if torch.is_tensor(self.feature_pairs)
                else self.feature_pairs,
                "global_mean": self.global_mean,
                "global_std": self.global_std,
                "categorical_encoders": self.categorical_encoders,
                "feature_columns": self.feature_columns,
                "original_columns": getattr(self, "original_columns", None),
                "target_column": self.target_column,
                "config": self.config,
                "high_cardinality_columns": getattr(
                    self, "high_cardinality_columns", []
                ),
                "best_error": self.best_error,
                "weight_updater": self.weight_updater,
                "n_bins_per_dim": self.n_bins_per_dim,
                "bin_edges": [
                    edge.cpu().tolist() if torch.is_tensor(edge) else edge
                    for edge in self.bin_edges
                ]
                if hasattr(self, "bin_edges")
                else None,
                "gaussian_params": {
                    "means": self.gaussian_params["means"].cpu().tolist()
                    if torch.is_tensor(self.gaussian_params["means"])
                    else self.gaussian_params["means"],
                    "covs": self.gaussian_params["covs"].cpu().tolist()
                    if torch.is_tensor(self.gaussian_params["covs"])
                    else self.gaussian_params["covs"],
                    "classes": self.gaussian_params["classes"].cpu().tolist()
                    if torch.is_tensor(self.gaussian_params["classes"])
                    else self.gaussian_params["classes"],
                }
                if hasattr(self, "gaussian_params") and self.gaussian_params is not None
                else None,
            }

            # Add model-specific components with validation
            if self.model_type == "Histogram":
                if not all(
                    k in self.likelihood_params
                    for k in ["bin_probs", "bin_edges", "classes"]
                ):
                    raise ValueError("Incomplete Histogram model parameters")
                components.update(
                    {
                        "likelihood_params": {
                            "bin_probs": [
                                prob.cpu().tolist() if torch.is_tensor(prob) else prob
                                for prob in self.likelihood_params["bin_probs"]
                            ],
                            "bin_edges": [
                                [
                                    edge.cpu().tolist()
                                    if torch.is_tensor(edge)
                                    else edge
                                    for edge in pair
                                ]
                                for pair in self.likelihood_params["bin_edges"]
                            ],
                            "classes": self.likelihood_params["classes"].cpu().tolist()
                            if torch.is_tensor(self.likelihood_params["classes"])
                            else self.likelihood_params["classes"],
                            "feature_pairs": self.likelihood_params["feature_pairs"]
                            .cpu()
                            .tolist()
                            if torch.is_tensor(self.likelihood_params["feature_pairs"])
                            else self.likelihood_params["feature_pairs"],
                        }
                    }
                )
            elif self.model_type == "Gaussian":
                if not all(
                    k in self.likelihood_params for k in ["means", "covs", "classes"]
                ):
                    raise ValueError("Incomplete Gaussian model parameters")
                components.update(
                    {
                        "likelihood_params": {
                            "means": self.likelihood_params["means"].cpu().tolist()
                            if torch.is_tensor(self.likelihood_params["means"])
                            else self.likelihood_params["means"],
                            "covs": self.likelihood_params["covs"].cpu().tolist()
                            if torch.is_tensor(self.likelihood_params["covs"])
                            else self.likelihood_params["covs"],
                            "classes": self.likelihood_params["classes"].cpu().tolist()
                            if torch.is_tensor(self.likelihood_params["classes"])
                            else self.likelihood_params["classes"],
                            "feature_pairs": self.likelihood_params["feature_pairs"]
                            .cpu()
                            .tolist()
                            if torch.is_tensor(self.likelihood_params["feature_pairs"])
                            else self.likelihood_params["feature_pairs"],
                        }
                    }
                )

            # Get filename and ensure directory exists
            components_file = self._get_model_components_filename()
            os.makedirs(os.path.dirname(components_file), exist_ok=True)

            # Atomic save operation
            temp_file = components_file + ".tmp"
            with open(temp_file, "wb") as f:
                pickle.dump(components, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            os.replace(temp_file, components_file)

            print(
                f"\033[K[SUCCESS] Saved model components to {components_file} (Size: {os.path.getsize(components_file)/1024:.2f} KB)"
            )
            return True

        except Exception as e:
            print(f"\033[K[ERROR] Failed to save model components: {str(e)}")
            traceback.print_exc()
            # Clean up temporary file if it exists
            if "temp_file" in locals() and os.path.exists(temp_file):
                os.remove(temp_file)
            return False

    def _load_model_components(self):
        """Enhanced model component loading with comprehensive validation"""
        components_file = self._get_model_components_filename()

        if not os.path.exists(components_file):
            print(f"\033[K[ERROR] Model components file not found: {components_file}")
            return False

        try:
            print(
                f"\033[K[INFO] Loading model components from {components_file} (Size: {os.path.getsize(components_file)/1024:.2f} KB)"
            )

            with open(components_file, "rb") as f:
                components = pickle.load(f)

            # Validate file version and basic structure
            if not isinstance(components, dict) or "version" not in components:
                raise ValueError("Invalid components file format")

            # Version-specific validation
            if components["version"] < 2:
                raise ValueError(
                    f"Unsupported components version: {components['version']}"
                )

            # Load and validate label encoder
            if "label_encoder" not in components or not components["label_encoder"].get(
                "fitted", False
            ):
                raise ValueError("Label encoder not properly saved or not fitted")

            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(
                components["label_encoder"]["classes_"]
            )

            # Validate and load core components
            required_components = [
                "scaler",
                "model_type",
                "feature_pairs",
                "target_column",
                "n_bins_per_dim",
            ]
            for comp in required_components:
                if comp not in components:
                    raise ValueError(f"Missing required component: {comp}")
                setattr(self, comp, components[comp])

            # Load likelihood parameters with model-specific validation
            if "likelihood_params" not in components:
                raise ValueError("Missing likelihood parameters")

            self.likelihood_params = components["likelihood_params"]

            # Convert back to tensors using proper methods
            def safe_to_tensor(data, device=None):
                """Convert data to tensor using proper method based on input type"""
                if isinstance(data, torch.Tensor):
                    return data.to(device) if device else data.clone().detach()
                return (
                    torch.tensor(data, device=device) if device else torch.tensor(data)
                )

            if self.model_type == "Histogram":
                self.likelihood_params["bin_probs"] = [
                    safe_to_tensor(prob, self.device)
                    for prob in self.likelihood_params["bin_probs"]
                ]
                self.likelihood_params["bin_edges"] = [
                    [safe_to_tensor(edge, self.device) for edge in pair]
                    for pair in self.likelihood_params["bin_edges"]
                ]
                self.likelihood_params["classes"] = safe_to_tensor(
                    self.likelihood_params["classes"], self.device
                )
                self.likelihood_params["feature_pairs"] = safe_to_tensor(
                    self.likelihood_params["feature_pairs"], self.device
                )
            elif self.model_type == "Gaussian":
                self.likelihood_params["means"] = safe_to_tensor(
                    self.likelihood_params["means"], self.device
                )
                self.likelihood_params["covs"] = safe_to_tensor(
                    self.likelihood_params["covs"], self.device
                )
                self.likelihood_params["classes"] = safe_to_tensor(
                    self.likelihood_params["classes"], self.device
                )
                self.likelihood_params["feature_pairs"] = safe_to_tensor(
                    self.likelihood_params["feature_pairs"], self.device
                )

            # Load optional components
            optional_components = [
                "global_mean",
                "global_std",
                "categorical_encoders",
                "feature_columns",
                "original_columns",
                "high_cardinality_columns",
                "best_error",
                "weight_updater",
                "bin_edges",
                "gaussian_params",
            ]
            for comp in optional_components:
                if comp in components:
                    # Convert tensors if needed
                    if comp == "bin_edges" and components[comp] is not None:
                        setattr(
                            self,
                            comp,
                            [
                                [edge.clone().detach().to(self.device) for edge in pair]
                                for pair in components[comp]
                            ],
                        )
                    elif comp == "gaussian_params" and components[comp] is not None:
                        gaussian_params = {
                            "means": torch.tensor(
                                components[comp]["means"], device=self.device
                            ),
                            "covs": torch.tensor(
                                components[comp]["covs"], device=self.device
                            ),
                            "classes": torch.tensor(
                                components[comp]["classes"], device=self.device
                            ),
                        }
                        setattr(self, comp, gaussian_params)
                    else:
                        setattr(self, comp, components[comp])

            print(f"\033[K[SUCCESS] Loaded model components from {components_file}")
            return True

        except Exception as e:
            print(f"\033[K[ERROR] Failed to load model components: {str(e)}")
            traceback.print_exc()
            # Reset critical components to prevent partial state
            self.label_encoder = LabelEncoder()
            self.scaler = StandardScaler()
            self.feature_pairs = None
            return False

    # --------------------------------------------------Class Ends ----------------------------------------------------------
    # DBNN class to handle prediction functionality
    def _validate_target_column(self, y: pd.Series) -> bool:
        """Check if target column values match label encoder classes"""
        # Handle case where label encoder isn't initialized
        if not hasattr(self.label_encoder, "classes_"):
            print(f"{Colors.RED}Label encoder not initialized!{Colors.ENDC}")
            return False  # Or raise ValueError if critical

        # Handle empty classes
        if len(self.label_encoder.classes_) == 0:
            print(f"{Colors.RED}Label encoder has no classes!{Colors.ENDC}")
            return False

        # Check value alignment
        unique_values = set(y.unique())
        encoder_classes = set(self.label_encoder.classes_)

        # Allow validation in prediction mode (target column may not exist)
        if self.mode == "predict" and self.target_column not in y.name:
            return True

        return unique_values.issubset(encoder_classes)

    def predict_from_file(
        self,
        input_csv: str,
        output_path: str = None,
        model_type=None,
        image_dir: str = None,
        batch_size: int = 128,
    ) -> Dict:
        """
        Make predictions from CSV file with comprehensive output handling, including
        failure/success analysis and PDF mosaics.

        Args:
            input_csv: Path to input CSV file
            output_path: Directory to save prediction results
            image_dir: Optional directory containing images for mosaics
            batch_size: Batch size for prediction

        Returns:
            Dictionary containing prediction results and metrics
        """
        # Create output directory if needed
        os.makedirs(output_path, exist_ok=True)
        try:
            # Load data
            df = pd.read_csv(input_csv)
            if self.target_column in df.columns:
                df[self.target_column] = df[self.target_column].apply(
                    str
                )  # Force string

            print(
                f"\n{Colors.BLUE}Processing predictions for: {input_csv}{Colors.ENDC}"
            )
            predict_mode = True if self.mode == "predict" else False
            self.model_type = model_type
            # Handle target column validation
            if predict_mode and self.target_column in df.columns:
                if not self._validate_target_column(df[self.target_column]):
                    print(
                        f"\033[K"
                        + f"{Colors.RED}The predict mode is {predict_mode} and target column is invalid. We will ignore it{Colors.ENDC}"
                    )
                    # Get the current column names
                    column_names = df.columns.tolist()
                    # Find the index of the target column
                    try:
                        index = column_names.index(self.target_column)
                        # Update the name
                        column_names[index] = "dummy_target"
                        # Assign the updated list back to columns
                        df.columns = column_names
                        # Update the target_column reference
                        self.target_column = None
                    except ValueError:
                        print(
                            f"\033[K"
                            + f"Warning: Target column '{self.target_column}' not found in dataset columns: {column_names}"
                        )
                        # If target column isn't found, just proceed without renaming

            # Store original data
            self.X_orig = df.copy()

            # Handle output directory
            if output_path:
                # Handle existing output path
                if os.path.exists(output_path):
                    print(
                        f"{Colors.BLUE}Output directory exists: {output_path}{Colors.ENDC}"
                    )
                    print(f"{Colors.BOLD}Choose an action:{Colors.ENDC}")
                    print("1. Overwrite existing content")
                    print("2. Create new version (append timestamp)")
                    print("3. Specify different output directory")
                    print("q. Quit")

                    while True:
                        choice = (
                            input(f"{Colors.YELLOW}Your choice (1-3/q): {Colors.ENDC}")
                            .strip()
                            .lower()
                        )

                        if choice == "1":  # Overwrite
                            print(
                                f"{Colors.YELLOW}Existing files will be overwritten{Colors.ENDC}"
                            )
                            # Clear existing predictions file if it exists
                            predictions_path = os.path.join(
                                output_path, "predictions.csv"
                            )
                            if os.path.exists(predictions_path):
                                os.remove(predictions_path)
                            # Clear analysis directories if they exist
                            for analysis_type in [
                                "mosaics",
                                "failed_analysis",
                                "correct_analysis",
                            ]:
                                analysis_dir = os.path.join(output_path, analysis_type)
                                if os.path.exists(analysis_dir):
                                    shutil.rmtree(analysis_dir)
                            break

                        elif choice == "2":  # New version
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_path = f"{output_path}_{timestamp}"
                            os.makedirs(output_path, exist_ok=True)
                            print(
                                f"{Colors.GREEN}Creating new version at: {output_path}{Colors.ENDC}"
                            )
                            break

                        elif choice == "3":  # Different directory
                            new_dir = input(
                                f"{Colors.YELLOW}Enter new directory path: {Colors.ENDC}"
                            ).strip()
                            if new_dir:
                                output_path = new_dir
                                os.makedirs(output_path, exist_ok=True)
                                print(
                                    f"{Colors.GREEN}Using new directory: {output_path}{Colors.ENDC}"
                                )
                                break
                            else:
                                print(
                                    f"{Colors.RED}Invalid path. Please try again.{Colors.ENDC}"
                                )

                        elif choice == "q":  # Quit
                            return None

                        else:
                            print(
                                f"{Colors.RED}Invalid option. Please choose 1-3 or q.{Colors.ENDC}"
                            )
                else:
                    os.makedirs(output_path, exist_ok=True)

            # Handle true labels if target column exists
            if hasattr(self, "target_column") and self.target_column in df.columns:
                y_true_str = df[self.target_column]
                try:
                    if hasattr(self.label_encoder, "classes_"):
                        y_true = self.label_encoder.transform(y_true_str)
                    else:
                        print(
                            f"{Colors.YELLOW}Warning: Label encoder not fitted, using raw labels{Colors.ENDC}"
                        )
                        y_true = y_true_str
                except ValueError as e:
                    print(
                        f"{Colors.RED}Error encoding true labels: {str(e)}{Colors.ENDC}"
                    )
                    print(f"Encoder knows: {self.label_encoder.classes_}")
                    print(f"Data contains: {np.unique(y_true_str)}")
                    raise
            else:
                y_true_str = None
                y_true = None

            # Get features (drop target column if exists)
            if self.target_column in df.columns:
                X = df.drop(columns=[self.target_column])
            else:
                X = df.copy()
                DEBUG.log("No target column found - running in pure prediction mode")

            # Generate predictions
            self._load_model_components()
            print(f"{Colors.BLUE}Generating predictions...{Colors.ENDC}")
            y_pred, posteriors = self.predict(X, batch_size=batch_size)
            pred_classes = self.label_encoder.inverse_transform(y_pred.cpu().numpy())
            confidences = posteriors[np.arange(len(y_pred)), y_pred].cpu().numpy()

            # Generate detailed results
            print(f"{Colors.BLUE}Generating detailed predictions...{Colors.ENDC}")
            results = self._generate_detailed_predictions(
                X_orig=self.X_orig,
                predictions=y_pred,
                true_labels=(y_true_str if y_true_str is not None else None),
                posteriors=posteriors,
            )

            # Save results if output path specified
            if output_path:
                # Standard paths
                predictions_path = os.path.join(output_path, "predictions.csv")
                metrics_path = os.path.join(output_path, "metrics.txt")

                # Ensure the predictions directory exists
                os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
                os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

                # Save predictions
                results["predicted_class"] = pred_classes
                results["confidence"] = confidences
                results.to_csv(predictions_path, index=False)
                print(
                    f"{Colors.GREEN}Predictions saved to {predictions_path}{Colors.ENDC}"
                )

                # Save metadata
                metadata = {
                    "dataset": os.path.basename(input_csv),
                    "model_type": self.model_type,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "feature_columns": list(X.columns),
                    "target_column": self.target_column
                    if hasattr(self, "target_column")
                    else None,
                    "label_encoder_classes": (
                        self.label_encoder.classes_.tolist()
                        if hasattr(self.label_encoder, "classes_")
                        else None
                    ),
                }
                with open(os.path.join(output_path, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

                # --- Enhanced Mosaic Generation ---
                if "filepath" in results.columns:
                    # Get mosaic layout parameters
                    columns = (
                        input(
                            "Please specify the number of columns of images per page (default 10): "
                        )
                        or 10
                    )
                    rows = (
                        input(
                            "Please specify the number of rows of images per page (default 10): "
                        )
                        or 10
                    )

                    try:
                        columns = int(columns)
                        rows = int(rows)
                    except ValueError:
                        print(
                            f"{Colors.RED}Invalid input. Using default 10x10 grid{Colors.ENDC}"
                        )
                        columns = 10
                        rows = 10

                    # Create main mosaics directory
                    mosaic_dir = os.path.join(output_path, "mosaics")
                    os.makedirs(mosaic_dir, exist_ok=True)

                    # Generate class-wise mosaics
                    for class_name, group in results.groupby("predicted_class"):
                        valid_images = []
                        for _, row in group.iterrows():
                            img_path = row["filepath"]
                            if os.path.exists(img_path):
                                valid_images.append(row)

                        if valid_images:
                            class_df = pd.DataFrame(valid_images)
                            self.generate_class_pdf_mosaics(
                                predictions_df=class_df,
                                output_dir=mosaic_dir,
                                columns=columns,
                                rows=rows,
                            )

                    # --- Failure/Success Analysis ---
                    if y_true_str is not None and "true_class" in results.columns:
                        # Create analysis directories
                        failed_dir = os.path.join(output_path, "failed_analysis")
                        correct_dir = os.path.join(output_path, "correct_analysis")
                        os.makedirs(failed_dir, exist_ok=True)
                        os.makedirs(correct_dir, exist_ok=True)

                        # Split into failed and correct predictions
                        failed_predictions = results[
                            results["predicted_class"] != results["true_class"]
                        ]
                        correct_predictions = results[
                            results["predicted_class"] == results["true_class"]
                        ]

                        # Save analysis CSVs
                        failed_predictions.to_csv(
                            os.path.join(failed_dir, "failed_predictions.csv"),
                            index=False,
                        )
                        correct_predictions.to_csv(
                            os.path.join(correct_dir, "correct_predictions.csv"),
                            index=False,
                        )

                        # Generate failure analysis mosaics
                        if not failed_predictions.empty:
                            print(
                                f"{Colors.BLUE}Generating failure analysis mosaics...{Colors.ENDC}"
                            )
                            for true_class, group in failed_predictions.groupby(
                                "true_class"
                            ):
                                valid_images = []
                                for _, row in group.iterrows():
                                    img_path = row["filepath"]
                                    if os.path.exists(img_path):
                                        valid_images.append(row)

                                if valid_images:
                                    class_df = pd.DataFrame(valid_images)
                                    self.generate_class_pdf_mosaics(
                                        predictions_df=class_df,
                                        output_dir=failed_dir,
                                        columns=columns,
                                        rows=rows,
                                    )

                        # Generate success analysis mosaics
                        if not correct_predictions.empty:
                            print(
                                f"{Colors.BLUE}Generating success analysis mosaics...{Colors.ENDC}"
                            )
                            for pred_class, group in correct_predictions.groupby(
                                "predicted_class"
                            ):
                                valid_images = []
                                for _, row in group.iterrows():
                                    img_path = row["filepath"]
                                    if os.path.exists(img_path):
                                        valid_images.append(row)

                                if valid_images:
                                    class_df = pd.DataFrame(valid_images)
                                    self.generate_class_pdf_mosaics(
                                        predictions_df=class_df,
                                        output_dir=correct_dir,
                                        columns=columns,
                                        rows=rows,
                                    )

            # Compute and return metrics if we have true labels
            metrics = {}
            if y_true is not None and y_pred is not None:
                print(f"\n{Colors.BLUE}Computing evaluation metrics...{Colors.ENDC}")

                # Ensure we have numpy arrays for sklearn metrics
                y_true_np = (
                    y_true
                    if isinstance(y_true, (np.ndarray, list))
                    else y_true.cpu().numpy()
                )
                y_pred_np = (
                    y_pred
                    if isinstance(y_pred, (np.ndarray, list))
                    else y_pred.cpu().numpy()
                )

                # Calculate metrics
                metrics["accuracy"] = accuracy_score(y_true_np, y_pred_np)
                metrics["classification_report"] = classification_report(
                    y_true,
                    y_pred,
                    output_dict=True,
                    target_names=[str(cls) for cls in self.label_encoder.classes_],
                )
                metrics["confusion_matrix"] = confusion_matrix(
                    y_true_np, y_pred_np
                ).tolist()
                # For saving as string
                metrics["classification_report_str"] = classification_report(
                    y_true,
                    y_pred,
                    target_names=[str(cls) for cls in self.label_encoder.classes_],
                )
                # Print colored confusion matrix
                if hasattr(self.label_encoder, "classes_"):
                    self.print_colored_confusion_matrix(
                        y_true_np,
                        y_pred_np,
                        class_labels=self.label_encoder.classes_,
                        header="Prediction Results",
                    )
                else:
                    print(
                        f"{Colors.YELLOW}Warning: No class labels available for confusion matrix{Colors.ENDC}"
                    )

                # Save metrics if output path exists
                if output_path:
                    with open(metrics_path, "w") as f:
                        f.write(metrics["classification_report_str"])
                    print(f"{Colors.GREEN}Metrics saved to {metrics_path}{Colors.ENDC}")

            return {
                "predictions": results,
                "metrics": metrics if metrics else None,
                "metadata": metadata if output_path else None,
                "analysis_files": {
                    "failed_predictions": os.path.join(output_path, "failed_analysis")
                    if y_true_str is not None
                    else None,
                    "correct_predictions": os.path.join(output_path, "correct_analysis")
                    if y_true_str is not None
                    else None,
                }
                if output_path
                else None,
            }

        except Exception as e:
            print(f"{Colors.RED}Prediction failed: {str(e)}{Colors.ENDC}")
            traceback.print_exc()
            raise

    def predict_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data using the trained model.
        Handles preprocessing and returns results with probabilities.

        Args:
            new_data: DataFrame containing new data to predict

        Returns:
            DataFrame with predictions and probabilities
        """
        if not hasattr(self, "feature_columns"):
            raise RuntimeError("Model not properly trained - missing feature columns")

        # Make a copy of the input data to preserve original
        df = new_data.copy()

        # Store original index for results
        df.index

        # Check if target column exists in input data
        target_in_data = self.target_column in df.columns

        # Remove target column if present (but keep track of it)
        if target_in_data:
            y_true = df[self.target_column]
            df = df.drop(columns=[self.target_column])

        # Ensure we have all required features
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Input data missing required features: {missing_features}"
            )

        # Reorder columns to match training order
        df = df[self.feature_columns]

        # Preprocess the data
        X_processed = self._preprocess_data(df, is_training=False)

        # Make predictions
        predictions = self.predict(X_processed)
        pred_labels = self.label_encoder.inverse_transform(predictions.cpu().numpy())

        # Compute probabilities
        probabilities = self._compute_probabilities(X_processed)

        # Create results DataFrame
        results = df.copy()
        results["predicted_class"] = pred_labels

        # Add true values if they existed in input
        if target_in_data:
            results[self.target_column] = y_true

        # Add probability columns
        for i, class_name in enumerate(self.label_encoder.classes_):
            results[f"prob_{class_name}"] = probabilities[:, i]

        results["max_probability"] = probabilities.max(axis=1)

        return results

    def _compute_probabilities(self, X: torch.Tensor) -> np.ndarray:
        """
        Compute class probabilities for input data.

        Args:
            X: Input tensor (already preprocessed)

        Returns:
            Array of class probabilities [n_samples, n_classes]
        """
        # Store current weights
        orig_weights = self.current_W
        self.current_W = self.best_W if self.best_W is not None else self.current_W

        try:
            # Compute in batches to handle large datasets
            batch_size = 128
            all_probs = []

            for i in range(0, len(X), batch_size):
                batch_X = X[i : i + batch_size]

                if self.model_type == "Histogram":
                    probs, _ = self._compute_batch_posterior(batch_X)
                elif self.model_type == "Gaussian":
                    probs, _ = self._compute_batch_posterior_std(batch_X)

                all_probs.append(probs.cpu().numpy())

            return np.vstack(all_probs)

        finally:
            # Restore original weights
            self.current_W = orig_weights

    def load_model_for_prediction(self, dataset_name: str) -> bool:
        """
        Load a trained model for prediction only.

        Args:
            dataset_name: Name of the dataset to load model for

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Load configuration
            self.config = DatasetConfig.load_config(dataset_name)
            if self.config is None:
                raise ValueError(
                    f"Failed to load configuration for dataset: {dataset_name}"
                )

            self.target_column = self.config["target_column"]
            self.model_type = self.config.get("modelType", "Histogram")

            # Load model components
            # self.label_encoder =load_label_encoder(dataset_name)
            self._load_model_components()

            print(f"Successfully loaded model for dataset: {dataset_name}")
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
            return False


# ---------------------------------------------------DBNN Prediction Functions  Ends--------------------------------


def create_prediction_mosaic(
    image_dir: list,  # List of dictionary rows (with filepath and original_filename)
    csv_path: Optional[str] = None,  # Not used but kept for compatibility
    output_path: str = "mosaic.jpg",
    mosaic_size: tuple = (2000, 2000),
    tile_size: tuple = (200, 200),
    max_images: int = 100,
    font_path: Optional[str] = None,
):
    """
    Create a prediction mosaic from a list of image rows (containing filepath and metadata).

    Args:
        image_dir: List of dictionary rows (must contain 'filepath' and 'original_filename')
        csv_path: (Deprecated) Kept for compatibility
        output_path: Path to save the mosaic image
        mosaic_size: Total size of the output mosaic (width, height)
        tile_size: Size of each individual tile (width, height)
        max_images: Maximum number of images to include
        font_path: Optional path to font file for labels
    """
    try:
        # Validate inputs
        if not image_dir:
            raise ValueError("No images provided in image_dir")

        # Convert to DataFrame if needed
        if not isinstance(image_dir, pd.DataFrame):
            image_df = pd.DataFrame(image_dir)
        else:
            image_df = image_dir.copy()

        # Validate required columns
        if not all(
            col in image_df.columns for col in ["filepath", "original_filename"]
        ):
            raise ValueError(
                "Input must contain 'filepath' and 'original_filename' columns"
            )

        # Limit number of images
        if len(image_df) > max_images:
            image_df = image_df.sample(max_images, random_state=42)
            DEBUG.log(f"Using random sample of {max_images} images for mosaic")

        # Calculate grid dimensions
        tiles_per_row = mosaic_size[0] // tile_size[0]
        tiles_per_col = mosaic_size[1] // tile_size[1]
        max_tiles = tiles_per_row * tiles_per_col
        image_df = image_df.head(max_tiles)  # Ensure we don't exceed mosaic capacity

        # Create blank mosaic
        mosaic = Image.new("RGB", mosaic_size, (0, 0, 0))
        draw = ImageDraw.Draw(mosaic)

        # Try to load font (fallback to default if not specified)
        try:
            font = (
                ImageFont.truetype(font_path, 12)
                if font_path
                else ImageFont.load_default()
            )
        except:
            font = ImageFont.load_default()

        # Process and place each image
        for i, (_, row) in enumerate(image_df.iterrows()):
            try:
                img_path = row["filepath"]
                if not os.path.exists(img_path):
                    DEBUG.log(f"Image not found: {img_path}")
                    continue

                with Image.open(img_path) as img:
                    # Convert to RGB and resize
                    img = img.convert("RGB")
                    img.thumbnail(tile_size, Image.ANTIALIAS)

                    # Calculate position
                    row_pos = i // tiles_per_row
                    col_pos = i % tiles_per_row
                    x = col_pos * tile_size[0]
                    y = row_pos * tile_size[1]

                    # Center the image
                    offset_x = (tile_size[0] - img.size[0]) // 2
                    offset_y = (tile_size[1] - img.size[1]) // 2
                    mosaic.paste(img, (x + offset_x, y + offset_y))

                    # Add filename label (truncated if too long)
                    label = os.path.splitext(row["original_filename"])[0][:15]
                    draw.text(
                        (x + 5, y + tile_size[1] - 20),
                        label,
                        font=font,
                        fill=(255, 255, 255),
                    )

            except Exception as img_error:
                DEBUG.log(f"Error processing {img_path}: {str(img_error)}")
                continue

        # Save the mosaic
        mosaic.save(output_path, quality=95)
        DEBUG.log(
            f"Saved prediction mosaic to {output_path} "
            f"({len(image_df)} images, {tiles_per_row}x{tiles_per_col} grid)"
        )

    except Exception as e:
        DEBUG.log(f"Error in create_prediction_mosaic: {str(e)}")
        raise RuntimeError(f"Prediction mosaic creation failed: {str(e)}")


def plot_training_progress(error_rates: List[float], dataset_name: str):
    """Plot training error rates over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(error_rates)
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.title(f"Training Progress - {dataset_name.capitalize()} Dataset")
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(
    confusion_mat: np.ndarray, class_names: np.ndarray, dataset_name: str
):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {dataset_name.capitalize()} Dataset")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def load_label_encoder(dataset_name, save_dir="Model", model_type="Histogram"):
    encoder_path = os.path.join(
        save_dir, f"Best_{model_type}_{dataset_name}_label_encoder.pkl"
    )
    if os.path.exists(encoder_path):
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        print(
            "\033[K" + f"Label encoder loaded from {encoder_path}", end="\r", flush=True
        )
        return label_encoder
    else:
        raise FileNotFoundError(f"Label encoder file not found at {encoder_path}")


def generate_test_datasets():
    """Generate XOR and 3D XOR test datasets"""
    # Generate 2D XOR
    with open("xor.csv", "w") as f:
        f.write("x1,x2,target\n")
        f.write("0,0,0\n")
        f.write("0,1,1\n")
        f.write("1,0,1\n")
        f.write("1,1,0\n")
        f.write("0,0,0\n")
        f.write("0,1,1\n")
        f.write("1,0,1\n")
        f.write("1,1,0\n")
        f.write("0,0,0\n")
        f.write("0,1,1\n")
        f.write("1,0,1\n")
        f.write("1,1,0\n")

    # Generate 3D XOR
    with open("xor3d.csv", "w") as f:
        f.write("x1,x2,x3,target\n")
        f.write("0,0,0,0\n")
        f.write("0,0,1,1\n")
        f.write("0,1,0,1\n")
        f.write("0,1,1,1\n")
        f.write("1,0,0,1\n")
        f.write("1,0,1,1\n")
        f.write("1,1,0,1\n")
        f.write("1,1,1,0\n")
        f.write("0,0,0,0\n")
        f.write("0,0,1,1\n")
        f.write("0,1,0,1\n")
        f.write("0,1,1,1\n")
        f.write("1,0,0,1\n")
        f.write("1,0,1,1\n")
        f.write("1,1,0,1\n")
        f.write("1,1,1,0\n")
        f.write("0,0,0,0\n")
        f.write("0,0,1,1\n")
        f.write("0,1,0,1\n")
        f.write("0,1,1,1\n")
        f.write("1,0,0,1\n")
        f.write("1,0,1,1\n")
        f.write("1,1,0,1\n")
        f.write("1,1,1,0\n")


class DebugLogger:
    def __init__(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def log(self, msg, force=False):
        """Only print if enabled or forced"""
        if self.enabled or force:
            print(msg)


# Create single global instance
DEBUG = DebugLogger()


def configure_debug(config):
    """Configure debug state from config"""
    debug_enabled = config.get("training_params", {}).get("debug_enabled", False)
    if debug_enabled:
        DEBUG.enable()
    else:
        DEBUG.disable()


import json

# -------------------------------------------------------unit test ----------------------------------
import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd


def load_or_create_config(config_path: str) -> dict:
    """
    Load the configuration file if it exists, or create a default one if it doesn't.
    Update global variables based on the configuration file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary containing the configuration.
    """
    default_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "trials": 100,
        "cardinality_threshold": 0.9,
        "cardinality_tolerance": 8,
        "learning_rate": 0.1,
        "random_seed": 42,
        "epochs": 1000,
        "test_fraction": 0.2,
        "train": True,
        "train_only": False,
        "predict": True,
        "gen_samples": False,
        "enable_adaptive": True,
        "nokbd": False,
        "display": None,
        "training_params": {
            "batch_size": None,  # Batch size will be dynamically calculated if not provided
            "n_bins_per_dim": 128,
            "minimum_training_accuracy": 0.95,
            "invert_DBNN": True,
            "reconstruction_weight": 0.5,
            "feedback_strength": 0.3,
            "inverse_learning_rate": 0.1,
            "Save_training_epochs": False,
            "training_save_path": "data",
            "enable_vectorized": False,
            "vectorization_warning_acknowledged": False,
            "compute_device": "auto",
            "use_interactive_kbd": False,
            "modelType": "Histogram",
            "class_preference": True,
        },
        "active_learning": {
            "tolerance": 1.0,
            "cardinality_threshold_percentile": 95,
            "strong_margin_threshold": 0.01,  # Consider only a margin of 1% of the max for divergence computation and sample selection.
            "marginal_margin_threshold": 0.01,
            "min_divergence": 0.1,
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

    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_path}")
    else:
        config = default_config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"Created default configuration file at {config_path}")
        print(config)
        input("Press Enter Key")

    # Update global variables based on the configuration file
    global predict_mode, Train_device, bin_sizes, n_bins_per_dim, Trials, cardinality_threshold, cardinality_tolerance, LearningRate, TrainingRandomSeed, Epochs, TestFraction, Train, Train_only, Predict, Gen_Samples, EnableAdaptive, nokbd, display

    # Update Train_device based on the compute_device setting in the configuration file
    Train_device = config.get(
        "compute_device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    Trials = config.get("trials", 100)
    cardinality_threshold = config.get("cardinality_threshold", 0.9)
    cardinality_tolerance = config.get("cardinality_tolerance", 8)
    bin_sizes = config.get("bin_sizes", 128)
    n_bins_per_dim = config.get("n_bins_per_dim", 128)
    LearningRate = config.get("learning_rate", 0.1)
    TrainingRandomSeed = config.get("random_seed", 42)
    Epochs = config.get("epochs", 1000)
    TestFraction = config.get("test_fraction", 0.2)
    Train = config.get("train", True)
    Train_only = config.get("train_only", False)
    Predict = config.get("predict", True)
    Gen_Samples = config.get("gen_samples", False)
    EnableAdaptive = config.get("enable_adaptive", True)
    nokbd = config.get("nokbd", False)
    display = config.get("display", None)

    # Dynamically calculate batch size if not provided in the training_params section
    if "training_params" in config:
        if (
            "batch_size" not in config["training_params"]
            or config["training_params"]["batch_size"] is None
        ):
            # Calculate optimal batch size dynamically
            sample_tensor_size = (
                4 * 1024 * 1024
            )  # Example: 4MB per sample (adjust based on your dataset)
            config["training_params"]["batch_size"] = _calculate_optimal_batch_size(
                sample_tensor_size
            )

    return config


def find_dataset_pairs(data_dir: str = "data") -> List[Tuple[str, str, str]]:
    """
    Recursively find all matching .conf and .csv files in the data directory structure.

    Args:
        data_dir: Root directory to search for datasets.

    Returns:
        List of tuples (basename, conf_path, csv_path).
    """
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        print(
            "\033[K" + f"No '{data_dir}' directory found. Creating one...",
            end="\r",
            flush=True,
        )
        os.makedirs(data_dir)
        return []

    dataset_pairs = []
    processed_datasets = set()

    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_dir):
        # Process configuration files
        conf_files = [
            f for f in files if f.endswith(".conf") and f != "adaptive_dbnn.conf"
        ]

        for conf_file in conf_files:
            basename = os.path.splitext(conf_file)[0]
            if basename in processed_datasets:
                continue

            conf_path = os.path.join(root, conf_file)

            # Look for matching CSV in the same directory and dataset-specific subdirectory
            csv_file = f"{basename}.csv"
            csv_paths = [
                os.path.join(root, csv_file),
                os.path.join(root, basename, csv_file),
            ]

            # Load adaptive_dbnn.conf if it exists
            adaptive_conf_path = os.path.join(data_dir, basename, "adaptive_dbnn.conf")
            if os.path.exists(adaptive_conf_path):
                try:
                    with open(adaptive_conf_path) as f:
                        json.load(f)
                    print(
                        "\033[K"
                        + f"Loaded adaptive configuration from {adaptive_conf_path}",
                        end="\r",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not load adaptive configuration from {adaptive_conf_path}: {str(e)}"
                    )
            else:
                print(
                    "\033[K"
                    + f"No adaptive_dbnn.conf found in working directory {adaptive_conf_path}",
                    end="\r",
                    flush=True,
                )

            # Find the CSV file
            csv_path = None
            for path in csv_paths:
                if os.path.exists(path):
                    csv_path = path
                    break

            if csv_path:
                # Load the dataset configuration
                try:
                    with open(conf_path) as f:
                        json.load(f)
                except Exception as e:
                    print(
                        f"Warning: Could not update configuration for {basename}: {str(e)}"
                    )

                dataset_pairs.append((basename, conf_path, csv_path))
                processed_datasets.add(basename)
                print("\033[K" + f"Found dataset pair:", end="\r", flush=True)
                print("\033[K" + f"  Config: {conf_path}", end="\r", flush=True)
                print("\033[K" + f"  Data  : {csv_path}", end="\r", flush=True)
            else:
                print(
                    "\033[K"
                    + f"Warning: Config file {conf_file} exists but no matching CSV found"
                )
                print("\033[K" + f"Looked in:")
                for path in csv_paths:
                    print("\033[K" + f"  - {path}")

    if not dataset_pairs:
        print("\033[K" + "No matching .conf and .csv file pairs found.")
        print(
            "\033[K"
            + "Each dataset should have both a .conf configuration file and a matching .csv data file."
        )
        print("\033[K" + "Example: 'dataset1.conf' and 'dataset1.csv'")

    return dataset_pairs


def validate_config(config: dict) -> dict:
    """
    Validate the configuration parameters, ensuring that only the batch size is dynamically updated.

    Args:
        config: Configuration dictionary.

    Returns:
        dict: Validated configuration dictionary.
    """
    validated_config = config.copy()

    # Validate other parameters (only replace with defaults if missing or invalid)
    if "device" not in validated_config or validated_config["device"] not in [
        "cuda",
        "cpu",
    ]:
        validated_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    if "trials" not in validated_config or not isinstance(
        validated_config["trials"], int
    ):
        validated_config["trials"] = 100

    if "cardinality_threshold" not in validated_config or not isinstance(
        validated_config["cardinality_threshold"], float
    ):
        validated_config["cardinality_threshold"] = 0.9

    if "cardinality_tolerance" not in validated_config or not isinstance(
        validated_config["cardinality_tolerance"], int
    ):
        validated_config["cardinality_tolerance"] = 8

    if "learning_rate" not in validated_config or not isinstance(
        validated_config["learning_rate"], float
    ):
        validated_config["learning_rate"] = 0.1

    if "random_seed" not in validated_config or not isinstance(
        validated_config["random_seed"], int
    ):
        validated_config["random_seed"] = 42

    if "epochs" not in validated_config or not isinstance(
        validated_config["epochs"], int
    ):
        validated_config["epochs"] = 1000

    if "test_fraction" not in validated_config or not isinstance(
        validated_config["test_fraction"], float
    ):
        validated_config["test_fraction"] = 0.2

    if "train" not in validated_config or not isinstance(
        validated_config["train"], bool
    ):
        validated_config["train"] = True

    if "train_only" not in validated_config or not isinstance(
        validated_config["train_only"], bool
    ):
        validated_config["train_only"] = False

    if "predict" not in validated_config or not isinstance(
        validated_config["predict"], bool
    ):
        validated_config["predict"] = True

    if "gen_samples" not in validated_config or not isinstance(
        validated_config["gen_samples"], bool
    ):
        validated_config["gen_samples"] = False

    if "enable_adaptive" not in validated_config or not isinstance(
        validated_config["enable_adaptive"], bool
    ):
        validated_config["enable_adaptive"] = True

    if "nokbd" not in validated_config or not isinstance(
        validated_config["nokbd"], bool
    ):
        validated_config["nokbd"] = False

    if "display" not in validated_config:
        validated_config["display"] = None

    return validated_config


def main():
    parser = argparse.ArgumentParser(description="Process ML datasets")
    parser.add_argument(
        "--file_path", nargs="?", help="Path to dataset CSV file in data folder"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "train_predict", "invertDBNN", "predict"],
        required=False,
        help="Mode to run the network: train, train_predict, predict, or invertDBNN.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode to modify settings.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["Histogram", "Gaussian"],
        help="Override model type (Histogram/Gaussian)",
    )
    args = parser.parse_args()
    DatasetProcessor()

    def validate_config(config):
        """Ensure required configuration parameters exist"""
        required_params = {
            "training_params": ["learning_rate", "epochs", "test_fraction"],
            "modelType": None,
            "target_column": None,
        }

        for section, params in required_params.items():
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config")
            if params:
                for param in params:
                    if param not in config[section]:
                        raise ValueError(
                            f"Missing required parameter '{param}' in section '{section}'"
                        )

    def load_or_create_config(config_path):
        """Load or create configuration with validation"""
        try:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                validate_config(config)
                return config
            else:
                # Create default config
                config = DatasetConfig.create_default_config(
                    os.path.basename(config_path).split(".")[0]
                )
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                return config
        except Exception as e:
            print(
                f"\033[K{Colors.RED}Error loading/creating config: {str(e)}{Colors.ENDC}"
            )
            raise

    def find_dataset_pairs():
        """Find matching dataset/config pairs in data folder"""
        dataset_pairs = []
        data_dir = "data"

        if os.path.exists(data_dir):
            for dataset_name in os.listdir(data_dir):
                dataset_dir = os.path.join(data_dir, dataset_name)
                if os.path.isdir(dataset_dir):
                    conf_path = os.path.join(dataset_dir, f"{dataset_name}.conf")
                    csv_path = os.path.join(dataset_dir, f"{dataset_name}.csv")

                    if os.path.exists(conf_path) and os.path.exists(csv_path):
                        dataset_pairs.append((dataset_name, conf_path, csv_path))

        return dataset_pairs

    def process_datasets():
        """Process all found datasets"""
        dataset_pairs = find_dataset_pairs()

        if not dataset_pairs:
            print(
                "\033[K"
                + f"{Colors.RED}No valid dataset/config pairs found in data folder.{Colors.ENDC}"
            )
            return

        for i, (dataset_name, conf_path, csv_path) in enumerate(dataset_pairs):
            print(
                f"\033[K{Colors.BOLD}{i+1}. Found dataset: {dataset_name}{Colors.ENDC}"
            )

        choice = input(
            "\033[K"
            + f"{Colors.BOLD}Select dataset to process (1-{len(dataset_pairs)} or 'all'): {Colors.ENDC}"
        ).strip()

        if choice.lower() == "all":
            for dataset_name, conf_path, csv_path in dataset_pairs:
                process_single_dataset(dataset_name, conf_path, csv_path)
        elif choice.isdigit() and 1 <= int(choice) <= len(dataset_pairs):
            dataset_name, conf_path, csv_path = dataset_pairs[int(choice) - 1]
            process_single_dataset(dataset_name, conf_path, csv_path)
        else:
            print("\033[K" + f"{Colors.RED}Invalid selection.{Colors.ENDC}")

    def process_single_dataset(
        dataset_name, conf_path, csv_path, mode=None, model_type="Histogram"
    ):
        """Process a single dataset with given mode"""
        try:
            # Load config
            config = load_or_create_config(conf_path)

            # Determine mode if not provided
            if not mode:
                mode = (
                    "train_predict"
                    if config.get("train", True) and config.get("predict", True)
                    else "train"
                    if config.get("train", True)
                    else "predict"
                )

            print(
                f"\033[K{Colors.BOLD}Processing {dataset_name} in {mode} mode{Colors.ENDC}"
            )

            # Create DBNN instance
            if mode == "train_predict":
                model = DBNN(
                    dataset_name=dataset_name, mode="train", model_type=model_type
                )
            else:
                model = DBNN(
                    dataset_name=dataset_name, mode=mode, model_type=model_type
                )

            if mode in ["train", "train_predict"]:
                # Training phase
                start_time = datetime.now()

                if config.get("enable_adaptive", True):
                    results = model.adaptive_fit_predict()
                else:
                    results = model.fit_predict()

                end_time = datetime.now()

                # Print results
                print("\033[K" + "Training complete!")
                print(
                    "\033[K"
                    + f"Time taken: {(end_time - start_time).total_seconds():.1f} seconds"
                )

                if "results_path" in results:
                    print("\033[K" + f"Results saved to: {results['results_path']}")
                if "log_path" in results:
                    print("\033[K" + f"Training log saved to: {results['log_path']}")

                # Save model components
                # model._save_model_components()
                # model._save_best_weights()
                # save_label_encoder(model.label_encoder, dataset_name)
                model._load_model_components()
                # model.label_encoder = load_label_encoder(dataset_name)
            if mode in ["predict", "train_predict"]:
                # Prediction phase
                print("\033[K" + f"{Colors.BOLD}Starting prediction...{Colors.ENDC}")

                dataset_name = get_dataset_name_from_path(args.file_path)
                predictor = DBNN(
                    dataset_name=dataset_name, mode="predict", model_type=model_type
                )
                print(f"Processing {dataset_name} in predict mode")

                if predictor.load_model_for_prediction(dataset_name):
                    # Use either the provided CSV or default dataset CSV
                    input_csv = (
                        args.file_path
                        if args.file_path and mode == "predict"
                        else csv_path
                    )
                    output_dir = os.path.join("data", dataset_name, "Predictions")
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{dataset_name}")

                    results = predictor.predict_from_file(
                        input_csv, output_dir, model_type=model_type
                    )
                    print("\033[K" + f"Predictions saved to: {output_path}")

            if mode == "invertDBNN":
                # Invert DBNN mode
                model._load_model_components()
                # model.label_encoder = load_label_encoder(dataset_name)

                print("\033[K" + "DEBUG: Inverse DBNN Settings:")
                for param in [
                    "reconstruction_weight",
                    "feedback_strength",
                    "inverse_learning_rate",
                ]:
                    value = config.get("training_params", {}).get(param, 0.1)
                    print("\033[K" + f"- {param}: {value}")

                inverse_model = InvertibleDBNN(
                    forward_model=model,
                    feature_dims=model.data.shape[1] - 1,
                    reconstruction_weight=config["training_params"].get(
                        "reconstruction_weight", 0.5
                    ),
                    feedback_strength=config["training_params"].get(
                        "feedback_strength", 0.3
                    ),
                )

                # Reconstruct features
                X_test = model.data.drop(columns=[model.target_column])
                test_probs = model._get_test_probabilities(X_test)
                reconstruction_features = inverse_model.reconstruct_features(test_probs)

                # Save reconstructed features
                output_dir = os.path.join("data", dataset_name, "Predicted_features")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{dataset_name}.csv")

                feature_columns = model.data.drop(columns=[model.target_column]).columns
                reconstructed_df = pd.DataFrame(
                    reconstruction_features.cpu().numpy(), columns=feature_columns
                )
                reconstructed_df.to_csv(output_file, index=False)

                print("\033[K" + f"Reconstructed features saved to {output_file}")

        except Exception as e:
            print(
                f"\033[K{Colors.RED}Error processing dataset {dataset_name}: {str(e)}{Colors.ENDC}"
            )
            traceback.print_exc()

    if args.interactive:
        # Interactive mode
        print("\033[K" + f"{Colors.BOLD}{Colors.BLUE}Interactive Mode{Colors.ENDC}")
        dataset_name = (
            input(
                "\033[K" + f"{Colors.BOLD}Enter the name of the database:{Colors.ENDC}"
            )
            .strip()
            .lower()
        )
        conf_path = f"data/{dataset_name}/{dataset_name}.conf"

        # Load or create config
        config = load_or_create_config(conf_path)

        # Display current configuration
        print("\033[K" + f"{Colors.BOLD}Current Configuration:{Colors.ENDC}")
        print(
            "\033[K"
            + f"- Device: {config.get('compute_device', 'cuda' if torch.cuda.is_available() else 'cpu')}"
        )
        print(
            "\033[K" + f"- Mode: {'Train' if config.get('train', True) else 'Predict'}"
        )
        print(
            "\033[K"
            + f"- Learning Rate: {config.get('training_params', {}).get('learning_rate', 0.1)}"
        )
        print(
            "\033[K"
            + f"- Epochs: {config.get('training_params', {}).get('epochs', 1000)}"
        )
        print(
            "\033[K"
            + f"- Test Fraction: {config.get('training_params', {}).get('test_fraction', 0.2)}"
        )
        print(
            "\033[K"
            + f"- Enable Adaptive: {config.get('training_params', {}).get('enable_adaptive', True)}"
        )
        print(
            "\033[K"
            + f"- Enable class-wise preference in training: {config.get('training_params', {}).get('class_preference',True )}"
        )

        # Get mode
        mode = (
            input(
                "\033[K"
                + f"{Colors.BOLD}Enter mode (train/train_predict/predict/invertDBNN): {Colors.ENDC}"
            )
            .strip()
            .lower()
        )
        while mode not in ["train", "train_predict", "predict", "invertDBNN"]:
            print(
                "\033[K"
                + f"{Colors.RED}Invalid mode. Please enter 'train', 'train_predict', 'predict', or 'invertDBNN'.{Colors.ENDC}"
            )
            mode = (
                input("\033[K" + f"{Colors.BOLD}Enter mode: {Colors.ENDC}")
                .strip()
                .lower()
            )

        # Update config based on mode
        config["train"] = mode in ["train", "train_predict"]
        config["predict"] = mode in ["predict", "train_predict"]

        # For predict mode, get input file
        input_csv = None
        if mode == "predict":
            input_csv = input(
                "\033[K"
                + f"{Colors.BOLD}Enter path to input CSV file (or press Enter to use default): {Colors.ENDC}"
            ).strip()
            if not input_csv:
                dataset_pairs = find_dataset_pairs()
                if dataset_pairs:
                    input_csv = dataset_pairs[0][2]  # Use first found CSV
                    print(
                        "\033[K"
                        + f"{Colors.YELLOW}Using default CSV file: {input_csv}{Colors.ENDC}"
                    )
                else:
                    print(
                        "\033[K"
                        + f"{Colors.RED}No default CSV file found.{Colors.ENDC}"
                    )
                    return

        # Save updated config
        with open(conf_path, "w") as f:
            json.dump(config, f, indent=2)

        print("\033[K" + f"{Colors.GREEN}Configuration updated.{Colors.ENDC}")

        # Process the dataset
        csv_path = input_csv if input_csv else f"data/{dataset_name}/{dataset_name}.csv"
        process_single_dataset(dataset_name, conf_path, csv_path, mode)

    elif not args.file_path and not args.mode:
        # No arguments provided - search for datasets
        parser.print_help()
        input(
            "\nPress any key to search data folder for datasets (or Ctrl-C to exit)..."
        )
        process_datasets()

    elif args.mode:
        # Specific mode requested
        if args.mode == "invertDBNN":
            if not args.file_path:
                dataset_pairs = find_dataset_pairs()
                if dataset_pairs:
                    args.file_path = dataset_pairs[0][2]  # Use first found CSV
                    print(
                        "\033[K"
                        + f"{Colors.YELLOW}Using default CSV file: {args.file_path}{Colors.ENDC}"
                    )
                else:
                    print(
                        "\033[K"
                        + f"{Colors.RED}No datasets found for inversion.{Colors.ENDC}"
                    )
                    return

            basename = os.path.splitext(os.path.basename(args.file_path))[0]
            conf_path = os.path.join("data", basename, f"{basename}.conf")
            csv_path = os.path.join("data", basename, f"{basename}.csv")
            process_single_dataset(
                basename, conf_path, csv_path, "invertDBNN", model_type=args.model_type
            )

        elif args.mode in ["train", "train_predict", "predict"]:
            if args.file_path:
                basename = get_dataset_name_from_path(args.file_path)
                workfile = os.path.splitext(os.path.basename(args.file_path))[0]
                conf_path = os.path.join("data", basename, f"{basename}.conf")
                csv_path = os.path.join("data", basename, f"{workfile}.csv")
                process_single_dataset(
                    basename, conf_path, csv_path, args.mode, model_type=args.model_type
                )
            else:
                dataset_pairs = find_dataset_pairs()
                if dataset_pairs:
                    basename, conf_path, csv_path = dataset_pairs[0]
                    print(
                        "\033[K"
                        + f"{Colors.YELLOW}Using default dataset: {basename}{Colors.ENDC}"
                    )
                    process_single_dataset(
                        basename,
                        conf_path,
                        csv_path,
                        args.mode,
                        model_type=args.model_type,
                    )
                else:
                    print("\033[K" + f"{Colors.RED}No datasets found.{Colors.ENDC}")

    else:
        parser.print_help()


if __name__ == "__main__":
    print("\033[K" + "DBNN Dataset Processor")
    print("\033[K" + "=" * 40)
    main()
