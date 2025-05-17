import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class DBNN(nn.Module):
    """
    Difference Boosting Neural Network for region classification.
    This module takes ROI features and refines the classification scores.
    """
    def __init__(self, in_channels: int, num_classes: int, num_boosters: int = 3):
        """
        Args:
            in_channels: Number of input channels from ROI features
            num_classes: Number of object classes
            num_boosters: Number of boosting iterations
        """
        super().__init__()
        self.num_boosters = num_boosters
        self.num_classes = num_classes
        
        # Base classifier (initial prediction)
        self.base_classifier = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        # Boosting classifiers
        self.boosters = nn.ModuleList()
        for _ in range(num_boosters):
            booster = nn.Sequential(
                nn.Linear(in_channels + num_classes, 512),  # Takes both features and previous predictions
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            self.boosters.append(booster)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (N, in_channels)
            
        Returns:
            Final class logits of shape (N, num_classes)
        """
        # Initial prediction
        base_logits = self.base_classifier(x)
        current_pred = base_logits
        
        # Iterative boosting
        for booster in self.boosters:
            # Concatenate features with current predictions
            booster_input = torch.cat([x, current_pred.detach()], dim=1)
            
            # Get residual
            residual = booster(booster_input)
            
            # Update predictions
            current_pred = current_pred + residual
        
        return current_pred

class DBNNWithROI(nn.Module):
    """
    Wrapper that combines ROI pooling with DBNN classifier
    """
    def __init__(self, feature_extractor, dbnn, roi_size: int = 7):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.roi_pool = torch.ops.torchvision.roi_align
        self.roi_size = roi_size
        self.spatial_scale = 1.0 / 16.0  # Adjust based on your feature extractor's downsampling
        self.dbnn = dbnn
    
    def forward(self, images: List[torch.Tensor], 
               boxes: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            images: List of input images, each of shape (C, H, W)
            boxes: List of proposal boxes for each image, each of shape (num_boxes, 4)
                  in (x1, y1, x2, y2) format
                  
        Returns:
            Class logits for each proposal, shape (total_boxes, num_classes)
        """
        # Extract features
        features = self.feature_extractor(torch.stack(images))
        
        # ROI pooling
        rois = []
        for i, img_boxes in enumerate(boxes):
            if len(img_boxes) > 0:
                # Add batch index
                batch_indices = torch.full((len(img_boxes), 1), i, 
                                         dtype=img_boxes.dtype, 
                                         device=img_boxes.device)
                rois.append(torch.cat([batch_indices, img_boxes], dim=1))
        
        if not rois:  # No proposals
            return torch.zeros((0, self.dbnn.num_classes), 
                             device=features.device)
        
        rois = torch.cat(rois, dim=0)
        pooled_features = self.roi_pool(
            features, rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=self.spatial_scale,
            sampling_ratio=2
        )
        
        # Flatten features
        batch_size = pooled_features.size(0)
        flattened = pooled_features.view(batch_size, -1)
        
        # DBNN classification
        logits = self.dbnn(flattened)
        return logits