import math
import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance, ImageOps


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL Image and numpy.ndarray to torch.Tensor"""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """Normalize image with mean and standard deviation"""

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, image, target):
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return image, target


class RandomHorizontalFlip:
    """Randomly flip the image and bboxes horizontally with a probability of 0.5"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            width = image.width
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

            if "boxes" in target:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]  # Flip x-coordinates
                target["boxes"] = boxes

        return image, target


class RandomVerticalFlip:
    """Randomly flip the image and bboxes vertically with a probability of 0.5"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            height = image.height
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

            if "boxes" in target:
                boxes = target["boxes"].clone()
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]  # Flip y-coordinates
                target["boxes"] = boxes

        return image, target


class RandomRotate:
    """Randomly rotate the image and bboxes by a random angle between -angle and angle"""

    def __init__(self, angle=10, p=0.5):
        self.angle = angle
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            angle = random.uniform(-self.angle, self.angle)
            width, height = image.size
            center = (width // 2, height // 2)

            # Rotate image
            image = image.rotate(angle, resample=Image.BILINEAR, expand=False)

            if "boxes" in target:
                # Convert boxes to points
                boxes = target["boxes"].numpy()
                points = np.zeros((len(boxes) * 2, 2))

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    points[2 * i] = [x1, y1]
                    points[2 * i + 1] = [x2, y2]

                # Rotate points
                rad = math.radians(angle)
                cos_a = math.cos(rad)
                sin_a = math.sin(rad)
                cx, cy = center

                # Translate points to origin
                points[:, 0] -= cx
                points[:, 1] -= cy

                # Rotate points
                x = points[:, 0] * cos_a - points[:, 1] * sin_a
                y = points[:, 0] * sin_a + points[:, 1] * cos_a

                # Translate points back
                x += cx
                y += cy

                # Update boxes
                new_boxes = []
                for i in range(0, len(points), 2):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    new_boxes.append(
                        [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    )

                target["boxes"] = torch.tensor(new_boxes, dtype=torch.float32)

        return image, target


class RandomBrightness:
    """Randomly adjust image brightness"""

    def __init__(self, brightness=0.2, p=0.5):
        self.brightness = brightness
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            enhancer = ImageEnhance.Brightness(image)
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = enhancer.enhance(factor)
        return image, target


class RandomContrast:
    """Randomly adjust image contrast"""

    def __init__(self, contrast=0.2, p=0.5):
        self.contrast = contrast
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            enhancer = ImageEnhance.Contrast(image)
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = enhancer.enhance(factor)
        return image, target


class RandomSaturation:
    """Randomly adjust image saturation"""

    def __init__(self, saturation=0.2, p=0.5):
        self.saturation = saturation
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            enhancer = ImageEnhance.Color(image)
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            image = enhancer.enhance(factor)
        return image, target


class RandomHue:
    """Randomly adjust image hue"""

    def __init__(self, hue=0.1, p=0.5):
        self.hue = hue
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p and self.hue > 0:
            # Convert to HSV
            image = image.convert("HSV")
            h, s, v = image.split()

            # Adjust hue
            h = h.point(lambda x: (x + random.uniform(-self.hue, self.hue) * 255) % 255)

            # Convert back to RGB
            image = Image.merge("HSV", (h, s, v)).convert("RGB")

        return image, target


class RandomGrayscale:
    """Randomly convert image to grayscale"""

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = ImageOps.grayscale(image)
            image = ImageOps.colorize(image, black="black", white="white")
        return image, target


class RandomCrop:
    """Randomly crop the image and adjust bboxes accordingly"""

    def __init__(self, min_scale=0.6, p=0.5):
        self.min_scale = min_scale
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p and "boxes" in target and len(target["boxes"]) > 0:
            width, height = image.size
            boxes = target["boxes"].numpy()

            # Find a random crop that includes at least one box
            for _ in range(10):  # Try 10 times
                # Random scale
                scale = random.uniform(self.min_scale, 1.0)
                new_width = int(width * scale)
                new_height = int(height * scale)

                # Random position
                x = random.randint(0, width - new_width)
                y = random.randint(0, height - new_height)

                # Check if any box is included
                for box in boxes:
                    x1, y1, x2, y2 = box
                    if x1 < x + new_width and x2 > x and y1 < y + new_height and y2 > y:
                        # Found a valid crop
                        image = image.crop((x, y, x + new_width, y + new_height))

                        # Update boxes
                        new_boxes = []
                        for box in boxes:
                            x1, y1, x2, y2 = box
                            # Clip boxes to crop
                            x1 = max(0, x1 - x)
                            y1 = max(0, y1 - y)
                            x2 = min(new_width, x2 - x)
                            y2 = min(new_height, y2 - y)

                            # Only keep boxes with valid dimensions
                            if x1 < x2 and y1 < y2:
                                new_boxes.append([x1, y1, x2, y2])

                        if new_boxes:
                            target["boxes"] = torch.tensor(
                                new_boxes, dtype=torch.float32
                            )
                            return image, target

                        break

        return image, target


class RandomResize:
    """Randomly resize the image and adjust bboxes accordingly"""

    def __init__(self, min_size=600, max_size=1000, p=0.5):
        self.min_size = min_size
        self.max_size = max_size
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            # Randomly select a size
            size = random.randint(self.min_size, self.max_size)

            # Calculate scaling factor
            width, height = image.size
            scale = size / min(width, height)

            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Resize image
            image = image.resize((new_width, new_height), Image.BILINEAR)

            # Scale boxes if they exist
            if "boxes" in target:
                boxes = target["boxes"].numpy()
                boxes = boxes * scale
                target["boxes"] = torch.tensor(boxes, dtype=torch.float32)

        return image, target


class RandomNoise:
    """Add random noise to the image"""

    def __init__(self, std=0.05, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            img_array = np.array(image)
            noise = np.random.normal(0, self.std, img_array.shape).astype(np.float32)
            img_array = img_array.astype(np.float32) / 255.0
            img_array = np.clip(img_array + noise, 0, 1)
            img_array = (img_array * 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        return image, target


def get_augmentation_pipeline(train=True):
    """Get data augmentation pipeline for training or validation"""
    if train:
        return Compose(
            [
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.2),
                RandomRotate(angle=15, p=0.3),
                RandomBrightness(brightness=0.2, p=0.3),
                RandomContrast(contrast=0.2, p=0.3),
                RandomSaturation(saturation=0.2, p=0.3),
                RandomHue(hue=0.1, p=0.3),
                RandomGrayscale(p=0.1),
                RandomCrop(min_scale=0.7, p=0.5),
                RandomResize(min_size=600, max_size=1000, p=0.5),
                RandomNoise(std=0.05, p=0.2),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return Compose(
            [
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
