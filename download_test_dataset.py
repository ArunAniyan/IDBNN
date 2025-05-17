"""
Download a small test dataset in COCO format for testing the object detection model
Author: Cascade AI Assistant
Date: May 17, 2025
"""

import argparse
import io
import json
import logging
import os
import random
import shutil

import requests
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define categories for our test dataset (subset of COCO)
CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
    {"id": 3, "name": "car", "supercategory": "vehicle"},
    {"id": 4, "name": "motorcycle", "supercategory": "vehicle"},
    {"id": 5, "name": "airplane", "supercategory": "vehicle"},
    {"id": 6, "name": "bus", "supercategory": "vehicle"},
    {"id": 7, "name": "train", "supercategory": "vehicle"},
    {"id": 8, "name": "truck", "supercategory": "vehicle"},
    {"id": 9, "name": "boat", "supercategory": "vehicle"},
]

# Base URLs for sample images from Pexels (public domain or CC0 license)
BASE_IMAGE_URLS = [
    "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg",
    "https://images.pexels.com/photos/1108101/pexels-photo-1108101.jpeg",
    "https://images.pexels.com/photos/1629781/pexels-photo-1629781.jpeg",
    "https://images.pexels.com/photos/1154723/pexels-photo-1154723.jpeg",
    "https://images.pexels.com/photos/1149137/pexels-photo-1149137.jpeg",
    "https://images.pexels.com/photos/2252584/pexels-photo-2252584.jpeg",
    "https://images.pexels.com/photos/1123567/pexels-photo-1123567.jpeg",
    "https://images.pexels.com/photos/1108117/pexels-photo-1108117.jpeg",
    "https://images.pexels.com/photos/1108141/pexels-photo-1108141.jpeg",
    "https://images.pexels.com/photos/1108142/pexels-photo-1108142.jpeg",
    # Additional Pexels URLs for more variety
    "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg",
    "https://images.pexels.com/photos/1170986/pexels-photo-1170986.jpeg",
    "https://images.pexels.com/photos/33109/fall-autumn-red-season.jpg",
    "https://images.pexels.com/photos/164634/pexels-photo-164634.jpeg",
    "https://images.pexels.com/photos/1643456/pexels-photo-1643456.jpeg",
    "https://images.pexels.com/photos/414612/pexels-photo-414612.jpeg",
    "https://images.pexels.com/photos/67636/rose-blue-flower-rose-blooms-67636.jpeg",
    "https://images.pexels.com/photos/248797/pexels-photo-248797.jpeg",
    "https://images.pexels.com/photos/46523/swiss-shepherd-dog-dog-white-animal-46523.jpeg",
    "https://images.pexels.com/photos/39317/chihuahua-dog-puppy-cute-39317.jpeg",
]


def generate_image_urls(num_images):
    """Generate image URLs for the test dataset"""
    # Generate URLs
    urls = []

    # First add all base URLs
    urls.extend(BASE_IMAGE_URLS)

    # If we need more URLs than we have in our base set, just duplicate them with different IDs
    if num_images > len(urls):
        # Calculate how many complete sets we need
        num_sets = (num_images // len(BASE_IMAGE_URLS)) + 1

        # Create duplicated list with enough URLs
        extended_urls = []
        for i in range(num_sets):
            for url in BASE_IMAGE_URLS:
                # Add a different query parameter to avoid caching issues
                if "?" in url:
                    extended_urls.append(f"{url}&set={i}")
                else:
                    extended_urls.append(f"{url}?set={i}")

        urls = extended_urls

    # Return the requested number of URLs
    return urls[:num_images]


def download_file(url, save_path):
    """
    Download a file from a URL with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(save_path, "wb") as f, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

    return save_path


def create_test_dataset(
    output_dir, num_images=10, image_size=(640, 480), val_split=0.2
):
    """
    Create a small test dataset in COCO format

    Args:
        output_dir: Directory to save the dataset
        num_images: Number of images to download (max 10)
        image_size: Size to resize images to
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Generate image URLs based on the requested number
    image_urls = generate_image_urls(num_images)

    # Initialize COCO annotations structure
    coco_data = {
        "info": {
            "description": "Test dataset for object detection",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "Cascade AI Assistant",
            "date_created": "2025-05-17",
        },
        "licenses": [
            {
                "id": 1,
                "name": "Public Domain",
                "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            }
        ],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    # Download images and create annotations
    annotation_id = 1

    for image_id, url in enumerate(image_urls, 1):
        # Download image
        image_filename = f"image_{image_id:04d}.jpg"

        # Decide if this image goes to train or validation set
        is_val = image_id <= int(num_images * val_split)
        target_dir = val_dir if is_val else train_dir
        image_path = os.path.join(target_dir, image_filename)

        try:
            # Download and process image
            response = requests.get(url)
            img = Image.open(io.BytesIO(response.content))

            # Resize image
            img = img.convert("RGB")
            img = img.resize(image_size)
            img.save(image_path)

            # Get image dimensions
            width, height = img.size

            # Add image to COCO data
            coco_data["images"].append(
                {
                    "id": image_id,
                    "license": 1,
                    "file_name": image_filename,
                    "height": height,
                    "width": width,
                    "date_captured": "2025-05-17",
                }
            )

            # Create synthetic annotations
            # For simplicity, we'll create 1-3 random boxes per image
            num_boxes = random.randint(1, 3)

            for _ in range(num_boxes):
                # Random box dimensions (between 10% and 50% of image size)
                box_width = random.uniform(0.1, 0.5) * width
                box_height = random.uniform(0.1, 0.5) * height

                # Random box position
                x = random.uniform(0, width - box_width)
                y = random.uniform(0, height - box_height)

                # Random category
                category_id = random.choice([cat["id"] for cat in CATEGORIES])

                # Add annotation
                coco_data["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x, y, box_width, box_height],
                        "area": box_width * box_height,
                        "segmentation": [],
                        "iscrowd": 0,
                    }
                )

                annotation_id += 1

            logger.info(f"Processed image {image_id}/{num_images}: {image_filename}")

        except Exception as e:
            logger.error(f"Error processing {url}: {e}")

    # Split annotations into train and validation sets
    train_annotations = {
        "info": coco_data["info"],
        "licenses": coco_data["licenses"],
        "images": [],
        "annotations": [],
        "categories": coco_data["categories"],
    }

    val_annotations = {
        "info": coco_data["info"],
        "licenses": coco_data["licenses"],
        "images": [],
        "annotations": [],
        "categories": coco_data["categories"],
    }

    # Split images and annotations
    for image in coco_data["images"]:
        image_id = image["id"]
        is_val = image_id <= int(num_images * val_split)

        if is_val:
            val_annotations["images"].append(image)
        else:
            train_annotations["images"].append(image)

    # Split annotations
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        is_val = image_id <= int(num_images * val_split)

        if is_val:
            val_annotations["annotations"].append(ann)
        else:
            train_annotations["annotations"].append(ann)

    # Save train annotations
    train_annotations_path = os.path.join(output_dir, "train_annotations.json")
    with open(train_annotations_path, "w") as f:
        json.dump(train_annotations, f, indent=2)

    # Save validation annotations
    val_annotations_path = os.path.join(output_dir, "val_annotations.json")
    with open(val_annotations_path, "w") as f:
        json.dump(val_annotations, f, indent=2)

    # Also save combined annotations for compatibility
    annotations_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    logger.info(
        f"Created dataset with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations"
    )
    logger.info(
        f"Training set: {len(train_annotations['images'])} images, {len(train_annotations['annotations'])} annotations"
    )
    logger.info(
        f"Validation set: {len(val_annotations['images'])} images, {len(val_annotations['annotations'])} annotations"
    )
    logger.info(f"Dataset saved to {output_dir}")

    return output_dir


def download_coco_mini(output_dir, num_images=50, val_split=0.2, categories=None):
    """
    Download a subset of the COCO dataset

    Args:
        output_dir: Directory to save the dataset
        num_images: Number of images to download
        val_split: Fraction of data to use for validation
        categories: List of category names to filter by

    Returns:
        Path to the dataset
    """
    try:
        import torchvision
        from pycocotools.coco import COCO
    except ImportError:
        logger.error(
            "torchvision and pycocotools are required to download COCO dataset"
        )
        logger.error("Install with: pip install torchvision pycocotools")
        raise

    logger.info(f"Downloading COCO mini dataset to {output_dir}")

    # Create directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    # Download COCO val2017 dataset (smaller than train2017)
    try:
        # First try to download directly via torchvision
        logger.info("Downloading COCO dataset via torchvision...")
        try:
            coco_dataset = torchvision.datasets.CocoDetection(
                root=output_dir,
                annFile=os.path.join(output_dir, "annotations/instances_val2017.json"),
                transforms=None,
                download=True,
            )
            logger.info("Successfully downloaded COCO dataset")
        except Exception as e:
            logger.warning(f"Torchvision download failed: {e}")
            logger.info("Trying alternative download method...")

            # Alternative: Download manually
            ann_dir = os.path.join(output_dir, "annotations")
            os.makedirs(ann_dir, exist_ok=True)

            # Download annotations
            ann_url = (
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            )
            ann_file = os.path.join(output_dir, "annotations.zip")
            logger.info(f"Downloading annotations from {ann_url}")

            import requests

            response = requests.get(ann_url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024
            with open(ann_file, "wb") as f:
                for data in tqdm(
                    response.iter_content(block_size),
                    total=total_size // block_size,
                    desc="Downloading annotations",
                ):
                    f.write(data)

            # Extract annotations
            logger.info("Extracting annotations...")
            import zipfile

            with zipfile.ZipFile(ann_file, "r") as zip_ref:
                zip_ref.extractall(output_dir)

            # Download val2017 images
            img_url = "http://images.cocodataset.org/zips/val2017.zip"
            img_file = os.path.join(output_dir, "val2017.zip")
            logger.info(f"Downloading val2017 images from {img_url}")

            response = requests.get(img_url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            with open(img_file, "wb") as f:
                for data in tqdm(
                    response.iter_content(block_size),
                    total=total_size // block_size,
                    desc="Downloading images",
                ):
                    f.write(data)

            # Extract images
            logger.info("Extracting images...")
            with zipfile.ZipFile(img_file, "r") as zip_ref:
                zip_ref.extractall(output_dir)

            # Clean up zip files
            os.remove(ann_file)
            os.remove(img_file)
    except Exception as e:
        logger.error(f"Error downloading COCO dataset: {e}")
        raise

    try:
        # Load COCO annotations
        ann_file = os.path.join(output_dir, "annotations/instances_val2017.json")
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        logger.info(f"Loading COCO annotations from {ann_file}")
        coco = COCO(ann_file)

        # Filter by categories if specified
        if categories:
            logger.info(f"Filtering for categories: {categories}")
            # Get category IDs
            cat_ids = []
            for category in categories:
                # Try exact match first
                cat_id = coco.getCatIds(catNms=[category])
                if cat_id:
                    cat_ids.extend(cat_id)
                    logger.info(
                        f"Found exact category match: {category} (ID: {cat_id})"
                    )
                else:
                    # Try partial match
                    all_cats = coco.loadCats(coco.getCatIds())
                    for cat in all_cats:
                        if category.lower() in cat["name"].lower():
                            cat_ids.append(cat["id"])
                            logger.info(
                                f"Found partial category match: {cat['name']} (ID: {cat['id']})"
                            )

            if not cat_ids:
                logger.warning(f"No matching categories found. Using all categories.")
                img_ids = list(coco.imgs.keys())
            else:
                # Get image IDs for the selected categories
                img_ids = []
                for cat_id in cat_ids:
                    img_ids.extend(coco.getImgIds(catIds=[cat_id]))
                img_ids = list(set(img_ids))  # Remove duplicates
                logger.info(f"Found {len(img_ids)} images with specified categories")
        else:
            # Get all image IDs
            img_ids = list(coco.imgs.keys())
            logger.info(f"Using all categories, found {len(img_ids)} images")

        # Ensure we don't exceed available images
        if num_images > len(img_ids):
            logger.warning(
                f"Requested {num_images} images, but only {len(img_ids)} are available"
            )
            num_images = len(img_ids)

        # Shuffle and select subset
        random.shuffle(img_ids)
        selected_img_ids = img_ids[:num_images]

        # Split into train and validation sets
        num_val = max(
            1, int(num_images * val_split)
        )  # Ensure at least 1 validation image
        num_train = num_images - num_val
        train_img_ids = selected_img_ids[:num_train]
        val_img_ids = selected_img_ids[num_train:]

        logger.info(
            f"Creating dataset with {num_train} training and {num_val} validation images"
        )

        # Create new COCO annotations
        mini_coco = {
            "info": coco.dataset["info"],
            "licenses": coco.dataset["licenses"],
            "images": [],
            "annotations": [],
            "categories": coco.dataset["categories"],
        }

        train_coco = copy.deepcopy(mini_coco)
        val_coco = copy.deepcopy(mini_coco)

        # Process training images
        for img_id in tqdm(train_img_ids, desc="Creating training dataset"):
            # Get image info
            img_info = coco.imgs[img_id]
            mini_coco["images"].append(img_info)
            train_coco["images"].append(img_info)

            # Copy image
            src_path = os.path.join(output_dir, "val2017", img_info["file_name"])
            dst_path = os.path.join(train_dir, img_info["file_name"])
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                logger.warning(f"Source image not found: {src_path}")

            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            mini_coco["annotations"].extend(anns)
            train_coco["annotations"].extend(anns)

        # Process validation images
        for img_id in tqdm(val_img_ids, desc="Creating validation dataset"):
            # Get image info
            img_info = coco.imgs[img_id]
            mini_coco["images"].append(img_info)
            val_coco["images"].append(img_info)

            # Copy image
            src_path = os.path.join(output_dir, "val2017", img_info["file_name"])
            dst_path = os.path.join(val_dir, img_info["file_name"])
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                logger.warning(f"Source image not found: {src_path}")

            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            mini_coco["annotations"].extend(anns)
            val_coco["annotations"].extend(anns)

        # Save all annotation files
        logger.info("Saving annotation files...")
        annotations_path = os.path.join(output_dir, "annotations.json")
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(mini_coco, f)

        train_annotations_path = os.path.join(output_dir, "train_annotations.json")
        with open(train_annotations_path, "w", encoding="utf-8") as f:
            json.dump(train_coco, f)

        val_annotations_path = os.path.join(output_dir, "val_annotations.json")
        with open(val_annotations_path, "w", encoding="utf-8") as f:
            json.dump(val_coco, f)

        # Log dataset statistics
        logger.info(f"Created mini COCO dataset with {len(mini_coco['images'])} images")
        logger.info(
            f"Training set: {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations"
        )
        logger.info(
            f"Validation set: {len(val_coco['images'])} images, {len(val_coco['annotations'])} annotations"
        )

        return output_dir

    except Exception as e:
        logger.error(f"Error creating COCO mini dataset: {e}")
        logger.exception("Detailed error information:")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download a test dataset for object detection"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_dataset",
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--num_images", type=int, default=50, help="Number of images to download"
    )
    parser.add_argument(
        "--use_coco",
        action="store_true",
        default=True,
        help="Use COCO dataset (default: True)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Create synthetic dataset instead of downloading COCO subset",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[640, 480],
        help="Size to resize images to (width height)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (default: 0.2)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        help="Specific COCO categories to include (e.g., person car)",
    )

    args = parser.parse_args()

    # Print information about the dataset creation
    print(f"\n{'='*50}")
    print(f"Downloading dataset to: {args.output_dir}")
    print(f"Number of images: {args.num_images}")
    print(f"Validation split: {args.val_split * 100:.1f}%")

    # Create dataset
    if args.synthetic:
        print(f"Creating synthetic dataset...")
        create_test_dataset(
            args.output_dir, args.num_images, tuple(args.image_size), args.val_split
        )
    else:
        print(f"Downloading COCO subset...")
        try:
            download_coco_mini(
                args.output_dir,
                args.num_images,
                args.val_split,
                categories=args.categories,
            )
            print(f"\nSuccess! Dataset downloaded to {args.output_dir}")
        except Exception as e:
            print(f"\nError downloading COCO dataset: {e}")
            print("Falling back to synthetic dataset...")
            create_test_dataset(
                args.output_dir, args.num_images, tuple(args.image_size), args.val_split
            )

    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
