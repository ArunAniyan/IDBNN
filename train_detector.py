# Main script

"""
your_dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── val/
│       ├── img_101.jpg
│       ├── img_102.jpg
│       └── ...
└── annotations/
    ├── annotations_train.json
    └── annotations_val.json

checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1

"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet50

from data_utils import get_dataloaders
from object_detector import ObjectDetector
from train import ObjectDetectorTrainer


def main():
    # Configuration
    config = {
        "root_dir": "path/to/your_dataset",
        "num_classes": 21,  # Include background class
        "batch_size": 4,
        "num_workers": 4,
        "num_epochs": 50,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "step_size": 10,
        "gamma": 0.1,
        "output_dir": "checkpoints",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Initialize model
    backbone = resnet50(pretrained=True)
    model = ObjectDetector(backbone, num_classes=config["num_classes"], use_dbpn=True)

    # Data loaders
    dataloaders = get_dataloaders(
        root_dir=config["root_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    # Initialize trainer
    trainer = ObjectDetectorTrainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        device=config["device"],
        num_classes=config["num_classes"],
        output_dir=config["output_dir"],
    )

    # Train the model
    for epoch in range(1, config["num_epochs"] + 1):
        # Train for one epoch
        trainer.train_epoch(epoch)

        # Validate
        if epoch % 5 == 0:  # Validate every 5 epochs
            val_metrics = trainer.validate()

            # Save best model
            if val_metrics["mAP"] > best_map:
                val_metrics["mAP"]
                trainer.save_checkpoint("best_model.pth")

        # Step the scheduler
        scheduler.step()


if __name__ == "__main__":
    main()
