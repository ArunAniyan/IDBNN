import os

import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes

from data_utils import DetectionDataset


def visualize_augmentations(
    dataset, num_samples=5, save_dir="augmentation_visualization"
):
    os.makedirs(save_dir, exist_ok=True)

    for i in range(min(num_samples, len(dataset))):
        image, target = dataset[i]

        # Skip if no boxes
        if len(target["boxes"]) == 0:
            continue

        # Convert tensor to PIL for visualization
        inv_normalize = T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        image_vis = inv_normalize(image) if image.min() < 0 else image
        image_vis = (image_vis * 255).byte()

        # Draw bounding boxes
        img_with_boxes = draw_bounding_boxes(
            image_vis,
            boxes=target["boxes"],
            labels=[f"Class {l}" for l in target["labels"].tolist()],
            width=2,
        )

        # Convert to HWC for matplotlib
        img_np = img_with_boxes.permute(1, 2, 0).numpy()

        # Plot
        plt.figure(figsize=(12, 8))
        plt.imshow(img_np)
        plt.axis("off")
        plt.title(f"Sample {i+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"aug_{i+1}.png"), bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    dataset = DetectionDataset(
        root_dir="path/to/your_dataset", split="train", augment=True
    )
    visualize_augmentations(dataset, num_samples=5)
