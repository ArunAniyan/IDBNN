"""
Inference script for Object Detection using CDBNN Feature Extraction and DBNN Logic
Author: Cascade AI Assistant
Date: May 17, 2025
"""

import argparse
import glob
import json
import logging
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Import our object detector
from object_detector import ObjectDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("object_detection_inference.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_model(model_path, device):
    """
    Load a trained object detection model

    Args:
        model_path: Path to the saved model
        device: Device to load the model on

    Returns:
        Loaded model
    """
    # Load configuration
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Initialize model with config
    model = ObjectDetector(
        num_classes=config["num_classes"],
        confidence_threshold=config["confidence_threshold"],
        nms_threshold=config["nms_threshold"],
        device=device,
    )

    # Load model weights
    model.load_model(model_path)
    model.to(device)
    model.eval()

    return model


def process_image(model, image_path, output_path=None, show=True):
    """
    Process a single image for object detection

    Args:
        model: Trained object detection model
        image_path: Path to the input image
        output_path: Path to save the output image
        show: Whether to display the result

    Returns:
        Detection results
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Apply transforms
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Move to device
    input_tensor = input_tensor.to(model.device)

    # Perform inference
    with torch.no_grad():
        detections = model(input_tensor)[0]  # Get first item from batch

    # Visualize detections
    if output_path or show:
        model.visualize_detections(
            image, detections, output_path=output_path, show=show
        )

    return detections


def process_directory(model, input_dir, output_dir, extensions=None):
    """
    Process all images in a directory

    Args:
        model: Trained object detection model
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        extensions: List of file extensions to process

    Returns:
        Dictionary of detection results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Default extensions if none provided
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]

    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))

    logger.info(f"Found {len(image_files)} images to process")

    # Process each image
    results = {}
    for image_path in tqdm(image_files, desc="Processing images"):
        # Get output path
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)

        # Process image
        detections = process_image(model, image_path, output_path, show=False)

        # Store results
        results[filename] = {
            "boxes": detections["boxes"].cpu().numpy().tolist(),
            "scores": detections["scores"].cpu().numpy().tolist(),
            "labels": detections["labels"].cpu().numpy().tolist(),
        }

    # Save results to JSON
    results_path = os.path.join(output_dir, "detection_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Saved detection results to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Object Detection Inference with CDBNN and DBNN"
    )

    # Input parameters
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input image or directory"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to output image or directory"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Process input as a directory of images"
    )
    parser.add_argument(
        "--confidence", type=float, default=None, help="Override confidence threshold"
    )
    parser.add_argument(
        "--nms", type=float, default=None, help="Override NMS threshold"
    )
    parser.add_argument("--show", action="store_true", help="Show detection results")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device)

    # Override thresholds if provided
    if args.confidence is not None:
        model.confidence_threshold = args.confidence
        logger.info(f"Overriding confidence threshold to {args.confidence}")

    if args.nms is not None:
        model.nms_threshold = args.nms
        logger.info(f"Overriding NMS threshold to {args.nms}")

    # Process input
    if args.batch:
        # Process directory
        if args.output is None:
            args.output = os.path.join(os.path.dirname(args.input), "detections")

        logger.info(f"Processing directory: {args.input}")
        results = process_directory(model, args.input, args.output)
        logger.info(f"Processed {len(results)} images")
    else:
        # Process single image
        logger.info(f"Processing image: {args.input}")
        detections = process_image(model, args.input, args.output, show=args.show)

        # Print detection results
        num_detections = len(detections["boxes"])
        logger.info(f"Found {num_detections} objects")

        for i in range(num_detections):
            box = detections["boxes"][i].cpu().numpy()
            score = detections["scores"][i].item()
            label = detections["labels"][i].item()
            class_name = model.class_mapping.get(label, f"class_{label}")

            logger.info(f"  Object {i+1}: {class_name}, Score: {score:.4f}, Box: {box}")

    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
