import os
import unittest

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from cdbnn import (  # Assuming the module is named cdbnn
    BaseAutoencoder,
    CustomImageDataset,
)


class TestFilenameTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small dataset with known filenames and features
        cls.dataset_dir = "test_dataset"
        cls.image_dir = os.path.join(cls.dataset_dir, "train")
        os.makedirs(cls.image_dir, exist_ok=True)

        # Create dummy images and labels
        cls.filenames = ["image1.png", "image2.png", "image3.png"]
        cls.labels = [0, 1, 0]  # Class labels for the images

        # Create class subdirectories and save dummy images
        for filename, label in zip(cls.filenames, cls.labels):
            class_dir = os.path.join(cls.image_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            img = Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            )
            img.save(os.path.join(class_dir, filename))

        # Create a valid config dictionary for BaseAutoencoder
        cls.config = {
            "execution_flags": {"use_gpu": False},  # Use CPU for testing
            "training": {"checkpoint_dir": "checkpoints"},
            "dataset": {
                "name": "test_dataset",
                "in_channels": 3,
                "input_size": [32, 32],
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            "model": {
                "autoencoder_config": {
                    "enhancements": {
                        "use_kl_divergence": False,
                        "use_class_encoding": False,
                    }
                }
            },
        }

        # Create a dummy model (BaseAutoencoder) for feature extraction
        cls.model = BaseAutoencoder(
            input_shape=(3, 32, 32), feature_dims=128, config=cls.config
        )

    def test_filename_tracking(self):
        # Load the dataset
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = CustomImageDataset(self.image_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Extract features
        feature_dict = self.model.extract_features(dataloader)

        # Verify that filenames are included in the feature_dict
        self.assertIn("filenames", feature_dict, "Filenames not found in feature_dict")
        self.assertEqual(
            feature_dict["filenames"],
            self.filenames,
            "Filenames do not match expected order",
        )

        # Save features to CSV
        output_csv = os.path.join(self.dataset_dir, "features.csv")
        self.model.save_features(
            feature_dict, output_csv, image_names=feature_dict["filenames"]
        )

        # Load the CSV file
        df = pd.read_csv(output_csv)

        # Verify that the filenames in the CSV match the expected order
        self.assertEqual(list(df["image_name"]), self.filenames)

        # Verify that the features correspond to the correct filenames
        for i, filename in enumerate(self.filenames):
            # Check if the features in the CSV match the extracted features
            csv_features = df.iloc[i][
                df.columns.difference(["image_name", "target"])
            ].values
            extracted_features = feature_dict["embeddings"][i].cpu().numpy()
            self.assertTrue(
                torch.allclose(
                    torch.tensor(csv_features),
                    torch.tensor(extracted_features),
                    atol=1e-5,
                ),
                f"Features for {filename} do not match",
            )

    @classmethod
    def tearDownClass(cls):
        # Clean up the test dataset
        for label in set(cls.labels):
            class_dir = os.path.join(cls.image_dir, str(label))
            for filename in cls.filenames:
                if os.path.exists(os.path.join(class_dir, filename)):
                    os.remove(os.path.join(class_dir, filename))
            os.rmdir(class_dir)
        os.rmdir(cls.image_dir)
        os.rmdir(cls.dataset_dir)


if __name__ == "__main__":
    unittest.main()
