import argparse
import torch
from object_detector import ObjectDetector
from data_utils import get_dataloaders
from train import ObjectDetectorTrainer

class DynamicCNN(torch.nn.Module):
    """Example backbone network"""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    parser = argparse.ArgumentParser(description='Train Object Detector')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--num-classes', type=int, required=True, help='Number of object classes')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    args = parser.parse_args()

    # Create model
    backbone = DynamicCNN(in_channels=3, num_classes=args.num_classes)
    model = ObjectDetector(backbone, num_classes=args.num_classes)
    
    # Create dataloaders
    dataloaders = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Create trainer
    trainer = ObjectDetectorTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        device=args.device,
        num_classes=args.num_classes,
        output_dir=args.output_dir
    )
    
    # Train the model
    trainer.train(num_epochs=args.epochs)

if __name__ == '__main__':
    main()