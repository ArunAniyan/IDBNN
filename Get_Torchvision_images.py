import argparse
import json
import os
import shutil
import sys
import tarfile
import zipfile  # Add this import at the top with other imports
from typing import Any, Dict, List, Tuple

import numpy as np
import torchvision
from PIL import Image


def create_zip_archive(source_dir: str, output_zip: str) -> None:
    """
    Create a zip archive of the source directory.

    Args:
        source_dir: Path to directory to be zipped
        output_zip: Path to output zip file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_zip), exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to zip with relative path
                arcname = os.path.relpath(file_path, start=source_dir)
                zipf.write(file_path, arcname)
    print(f"Created zip archive: {output_zip}")


def get_image_properties(
    dataset_path: str,
) -> Tuple[int, Tuple[int, int], List[float], List[float]]:
    """
    Analyze images in the dataset to determine:
    - Number of channels (1 for grayscale, 3 for RGB)
    - Image size (width, height)
    - Mean pixel values per channel
    - Standard deviation per channel

    Args:
        dataset_path: Path to the dataset's train directory (should be the actual directory containing class folders)

    Returns:
        Tuple of (channels, (width, height), mean_values, std_values)
    """
    # dataset_path should already be the train directory in this case
    if not os.path.exists(dataset_path):
        raise ValueError(f"Directory not found: {dataset_path}")

    # Find first image in the directory
    for root, _, files in os.walk(dataset_path):
        if files:
            first_image = os.path.join(root, files[0])
            break
    else:
        raise ValueError(f"No images found in directory: {dataset_path}")

    img = Image.open(first_image)
    img_array = np.array(img)

    # Special handling for MNIST which might be saved as L mode (8-bit pixels, black and white)
    if img.mode == "L":
        channels = 1
        height, width = img_array.shape
    elif len(img_array.shape) == 2:  # Grayscale
        channels = 1
        height, width = img_array.shape
    else:  # Color
        height, width, channels = img_array.shape

    means = []
    stds = []
    sample_count = 0
    max_samples = 100

    for root, _, files in os.walk(dataset_path):  # Walk the provided path directly
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                img = Image.open(os.path.join(root, file))
                img_array = np.array(img)

                if channels == 1:
                    if len(img_array.shape) == 3:
                        img_array = np.mean(img_array, axis=2)
                    means.append(np.mean(img_array))
                    stds.append(np.std(img_array))
                else:
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
                    means.append(np.mean(img_array, axis=(0, 1)))
                    stds.append(np.std(img_array, axis=(0, 1)))

                sample_count += 1
                if sample_count >= max_samples:
                    break
        if sample_count >= max_samples:
            break

    if channels == 1:
        mean_val = [float(np.mean(means))]
        std_val = [float(np.mean(stds))]
    else:
        mean_val = [float(x) for x in np.mean(means, axis=0)]
        std_val = [float(x) for x in np.mean(stds, axis=0)]

    return channels, (width, height), mean_val, std_val


def create_config_file(
    dataset_name: str,
    dataset_path: str,
    class_names: List[str],
    input_size: Tuple[int, int] = (32, 32),
    in_channels: int = 3,
) -> None:
    """
    Create a JSON configuration file for the dataset based on the template.
    """
    config_path = os.path.join(dataset_path, f"{dataset_name.lower()}.json")
    # Pass the train directory path directly
    train_path = os.path.join(dataset_path, "train")
    channels, (width, height), mean_val, std_val = get_image_properties(train_path)

    config: Dict[str, Any] = {
        "dataset": {
            "name": dataset_name.lower(),
            "type": "torchvision",
            "in_channels": channels,
            "num_classes": len(class_names),
            "input_size": [width, height],
            "mean": mean_val,
            "std": std_val,
            "resize_images": True,
            "train_dir": train_path,
            "test_dir": os.path.join(dataset_path, "test")
            if os.path.exists(os.path.join(dataset_path, "test"))
            else "",
        },
        # Rest of the configuration can be copied from template
        "model": {
            "encoder_type": "autoenc",
            "enable_adaptive": True,
            "feature_dims": 128,
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
                "convergence_threshold": 0.001,
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
            "epochs": 20,
            "num_workers": 4,
            "checkpoint_dir": "data/mnist/checkpoints",
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
            "normalize": {"enabled": True, "mean": [0.5], "std": [0.5]},
        },
        "execution_flags": {
            "mode": "train_and_predict",
            "use_gpu": False,
            "mixed_precision": True,
            "distributed_training": False,
            "debug_mode": False,
            "use_previous_model": True,
            "fresh_start": False,
        },
        "output": {
            "features_file": "data/mnist/mnist.csv",
            "model_dir": "data/mnist/models",
            "visualization_dir": "data/mnist/visualizations",
        },
    }

    # Automatically disable color jitter for grayscale images
    if channels == 1:
        config["augmentation"]["color_jitter"]["enabled"] = False

    # Save the configuration file
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Created configuration file: {config_path}")


def list_available_datasets() -> List[str]:
    """List all available datasets in torchvision.datasets"""
    dataset_classes = []
    for name in dir(torchvision.datasets):
        if name[0].isupper() and name not in [
            "VisionDataset",
            "DatasetFolder",
            "ImageFolder",
        ]:
            dataset_classes.append(name)
    return sorted(dataset_classes)


def get_dataset_info(dataset_name: str) -> Dict:
    """Get information about a specific dataset"""
    try:
        dataset_class = getattr(torchvision.datasets, dataset_name)
        return {
            "name": dataset_name,
            "has_train_test_split": hasattr(dataset_class, "train")
            and hasattr(dataset_class, "test"),
            "default_root": os.path.join("data", dataset_name.lower()),
        }
    except AttributeError:
        raise ValueError(f"Dataset {dataset_name} not found in torchvision.datasets")


def ensure_directory_exists(path: str) -> None:
    """Ensure a directory exists, create if it doesn't"""
    os.makedirs(path, exist_ok=True)


def extract_tar_file(tar_path: str, extract_to: str) -> None:
    """Extract a tar file to the specified directory"""
    ensure_directory_exists(extract_to)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_to)


def move_all_to_train(dataset_root: str, train_path: str) -> List[str]:
    """
    Move all image class subfolders to the train directory.
    Cleans the target directory first if it exists.
    Handles cases where:
    - There are train/test folders (move their contents to train)
    - Only raw class folders exist (move them to train)
    - There's a containing folder like 256_ObjectCategories (move its contents to train)
    """
    # Clean the train directory if it exists
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    os.makedirs(train_path, exist_ok=True)

    class_names = []

    # First check for train/test folders
    for split in ["train", "test"]:
        split_path = os.path.join(dataset_root, split)
        if os.path.exists(split_path):
            # Move all class folders from this split to train
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    dest_path = os.path.join(train_path, class_name)

                    # If class already exists in train (from previous split), merge the contents
                    if os.path.exists(dest_path):
                        for item in os.listdir(class_path):
                            src_item = os.path.join(class_path, item)
                            if os.path.isdir(src_item):
                                # For nested directories, merge recursively
                                dest_item = os.path.join(dest_path, item)
                                if os.path.exists(dest_item):
                                    shutil.rmtree(dest_item)
                                shutil.move(src_item, dest_path)
                            else:
                                # For files, overwrite if they exist
                                dest_file = os.path.join(dest_path, item)
                                if os.path.exists(dest_file):
                                    os.remove(dest_file)
                                shutil.move(src_item, dest_path)
                    else:
                        shutil.move(class_path, dest_path)

                    if class_name not in class_names:
                        class_names.append(class_name)

            # Remove the now-empty split directory
            shutil.rmtree(split_path)

    # If no train/test folders found, look for direct class folders or containing folders
    if not class_names:
        for item in os.listdir(dataset_root):
            item_path = os.path.join(dataset_root, item)

            # Skip hidden files and non-directories
            if not os.path.isdir(item_path) or item.startswith("."):
                continue

            # If this is a directory that contains class folders (like 256_ObjectCategories)
            if any(
                os.path.isdir(os.path.join(item_path, subitem))
                for subitem in os.listdir(item_path)
            ):
                # Move all its subdirectories to train
                for class_name in os.listdir(item_path):
                    class_path = os.path.join(item_path, class_name)
                    if os.path.isdir(class_path):
                        dest_path = os.path.join(train_path, class_name)
                        if os.path.exists(dest_path):
                            shutil.rmtree(dest_path)
                        shutil.move(class_path, dest_path)
                        class_names.append(class_name)
                # Remove the now-empty containing directory
                shutil.rmtree(item_path)
            else:
                # It's a class folder itself, move it to train
                dest_path = os.path.join(train_path, item)
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.move(item_path, dest_path)
                class_names.append(item)

    return sorted(class_names)


def download_dataset(
    dataset_name: str, root: str, merge_train_test: bool = True, **kwargs
) -> Tuple[str, List[str]]:
    """
    Download and organize a torchvision dataset.
    Always creates data/<dataset>/train/<class_folders> structure.

    Args:
        dataset_name: Name of the dataset to download
        root: Root directory for downloaded datasets
        merge_train_test: Whether to merge train and test sets
        **kwargs: Additional arguments to pass to dataset constructor

    Returns:
        Tuple of (path to train directory, list of class names)
    """
    dataset_class = getattr(torchvision.datasets, dataset_name)
    dataset_path = os.path.join(root, dataset_name.lower())
    train_path = os.path.join(dataset_path, "train")

    # Clean up existing directory structure completely before processing
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    ensure_directory_exists(train_path)

    try:
        # Download the dataset (this may download a tar file)
        dataset = dataset_class(root=root, download=True, **kwargs)

        # Special handling for MNIST/FashionMNIST/KMNIST/EMNIST datasets
        if dataset_name.lower() in ["mnist", "fashionmnist", "kmnist", "emnist"]:
            # These datasets have different structure, we need to convert to image files
            train_img_path = os.path.join(dataset_path, "train")
            test_img_path = os.path.join(dataset_path, "test")

            # Create fresh directories
            os.makedirs(train_img_path, exist_ok=True)
            os.makedirs(test_img_path, exist_ok=True)

            # Process training set
            train_set = dataset_class(root=root, train=True, download=True)
            for i in range(len(train_set)):
                img, label = train_set[i]
                class_dir = os.path.join(train_img_path, str(label))
                os.makedirs(class_dir, exist_ok=True)
                img_path = os.path.join(class_dir, f"{i}.png")
                if not os.path.exists(img_path):  # Only save if file doesn't exist
                    img.save(img_path)

            # Process test set
            test_set = dataset_class(root=root, train=False, download=True)
            for i in range(len(test_set)):
                img, label = test_set[i]
                class_dir = os.path.join(test_img_path, str(label))
                os.makedirs(class_dir, exist_ok=True)
                img_path = os.path.join(class_dir, f"{i}.png")
                if not os.path.exists(img_path):  # Only save if file doesn't exist
                    img.save(img_path)

            # Get class names
            if hasattr(dataset, "classes") and dataset.classes:
                class_names = dataset.classes
            else:
                class_names = [str(i) for i in range(10)]  # Default for digit datasets

            # Clean up raw files
            raw_dir = os.path.join(root, dataset_name.upper(), "raw")
            if os.path.exists(raw_dir):
                shutil.rmtree(raw_dir)

            # Move test to train if merge_train_test is True
            if merge_train_test:
                for class_name in class_names:
                    src = os.path.join(test_img_path, str(class_name))
                    dst = os.path.join(train_img_path, str(class_name))
                    if os.path.exists(src):
                        if os.path.exists(dst):
                            # Merge contents without overwriting
                            for file in os.listdir(src):
                                src_file = os.path.join(src, file)
                                dst_file = os.path.join(dst, file)
                                if not os.path.exists(dst_file):
                                    shutil.move(src_file, dst)
                        else:
                            shutil.move(src, dst)
                if os.path.exists(test_img_path):
                    shutil.rmtree(test_img_path)

            return train_img_path, class_names

        # Special handling for Caltech256 which comes as a tar file
        if dataset_name.lower() == "caltech256":
            tar_path = os.path.join(root, "256_ObjectCategories.tar")
            if os.path.exists(tar_path):
                extract_tar_file(tar_path, dataset_path)
                os.remove(tar_path)

        # Handle all other standard datasets
        class_names = move_all_to_train(dataset_path, train_path)

        # Get class names from dataset if available
        if hasattr(dataset, "classes") and dataset.classes:
            return train_path, dataset.classes
        return train_path, class_names

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        raise


def interactive_mode():
    """Interactive mode for dataset selection and downloading"""
    available_datasets = list_available_datasets()

    print("Available datasets:")
    for i, dataset in enumerate(available_datasets, 1):
        print(f"{i}. {dataset}")

    dataset_name = input(
        "\nEnter dataset name (press Enter to process all datasets): "
    ).strip()

    if not dataset_name:
        # Process all datasets
        for dataset in available_datasets:
            process_dataset(dataset)
    else:
        # Process single dataset
        process_dataset(dataset_name)


def process_dataset(
    dataset_name: str, root: str = "data", merge_train_test: bool = True
) -> None:
    """Process a single dataset including download, config creation, and zip archive"""
    if dataset_name not in list_available_datasets():
        print(f"Dataset '{dataset_name}' not found.")
        return

    print(f"\nProcessing dataset: {dataset_name}")
    try:
        get_dataset_info(dataset_name)

        # Download and organize dataset
        final_path, class_names = download_dataset(
            dataset_name=dataset_name, root=root, merge_train_test=merge_train_test
        )

        # Create configuration file with actual image properties
        create_config_file(
            dataset_name=dataset_name,
            dataset_path=os.path.join(root, dataset_name.lower()),
            class_names=class_names,
        )

        # Create zip archive of the training data
        train_dir = os.path.join(root, dataset_name.lower(), "train")
        output_zip = os.path.join("Data", f"{dataset_name.lower()}.zip")
        create_zip_archive(train_dir, output_zip)

        print(f"\nSuccessfully processed dataset:")
        print(f"- Files saved to: {final_path}")
        print(f"- Found {len(class_names)} classes")
        print(
            f"- Configuration file created: {os.path.join(root, dataset_name.lower(), f'{dataset_name.lower()}.json')}"
        )
        print(f"- Training data archived to: {output_zip}")

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")


def main():
    if len(sys.argv) > 1:
        # Command line mode
        parser = argparse.ArgumentParser(
            description="Download and organize torchvision image datasets"
        )
        parser.add_argument(
            "--dataset", type=str, default="", help="Name of dataset to download"
        )
        parser.add_argument(
            "--root",
            type=str,
            default="data",
            help="Root directory for downloaded datasets",
        )
        parser.add_argument(
            "--merge",
            action="store_true",
            help="Merge train and test sets (if available)",
        )
        parser.add_argument(
            "--all", action="store_true", help="Download all available datasets"
        )
        args = parser.parse_args()

        if args.dataset or args.all:
            available_datasets = list_available_datasets()

            if args.all:
                datasets_to_process = available_datasets
            else:
                if args.dataset not in available_datasets:
                    print(f"Dataset '{args.dataset}' not found.")
                    exit()
                datasets_to_process = [args.dataset]

            for dataset_name in datasets_to_process:
                process_dataset(dataset_name, args.root, args.merge)
        else:
            interactive_mode()
    else:
        # Interactive mode
        interactive_mode()


def interactive_mode():
    """Interactive mode for dataset selection and downloading"""
    available_datasets = list_available_datasets()

    print("Available datasets:")
    for i, dataset in enumerate(available_datasets, 1):
        print(f"{i}. {dataset}")

    dataset_name = input(
        "\nEnter dataset name (press Enter to process all datasets): "
    ).strip()

    if not dataset_name:
        # Process all datasets
        for dataset in available_datasets:
            process_dataset(dataset)
    else:
        # Process single dataset
        process_dataset(dataset_name)


if __name__ == "__main__":
    main()
