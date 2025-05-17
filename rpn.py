import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional

class RPN(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int = 9):
        super().__init__()
        self.num_anchors = num_anchors
        
        # RPN head
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Objectness score (foreground/background)
        self.objectness = nn.Conv2d(512, num_anchors, kernel_size=1)
        
        # Bounding box regression
        self.bbox_reg = nn.Conv2d(512, num_anchors * 4, kernel_size=1)
        
        # Initialize weights
        for module in [self.conv, self.objectness, self.bbox_reg]:
            nn.init.normal_(module.weight, std=0.01)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.conv(features))
        objectness_scores = self.objectness(x)  # (N, A, H, W)
        pred_bbox_deltas = self.bbox_reg(x)     # (N, A*4, H, W)
        return objectness_scores, pred_bbox_deltas

    def generate_anchors(self, feature_map_size: Tuple[int, int], 
                        image_size: Tuple[int, int],
                        scales: List[float] = [128, 256, 512],
                        ratios: List[float] = [0.5, 1, 2.0]) -> torch.Tensor:
        stride_h = image_size[0] / feature_map_size[0]
        stride_w = image_size[1] / feature_map_size[1]
        
        # Generate base anchors
        base_anchors = []
        for scale in scales:
            for ratio in ratios:
                w = scale * np.sqrt(ratio)
                h = scale / np.sqrt(ratio)
                base_anchors.append([-w/2, -h/2, w/2, h/2])
        
        base_anchors = torch.tensor(base_anchors, dtype=torch.float32)
        
        # Generate grid of centers
        grid_h, grid_w = feature_map_size
        shift_x = (torch.arange(0, grid_w) + 0.5) * stride_w
        shift_y = (torch.arange(0, grid_h) + 0.5) * stride_h
        
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1)
        shifts = shifts.reshape(-1, 1, 4)
        
        # Add shifts to base anchors
        anchors = base_anchors.view(1, -1, 4) + shifts
        anchors = anchors.reshape(-1, 4)
        
        # Clip to image boundaries
        anchors = torch.cat([
            torch.clamp(anchors[..., :2], min=0),
            torch.clamp(anchors[..., 2:], max=torch.tensor(image_size[::-1]))
        ], dim=-1)
        
        return anchors