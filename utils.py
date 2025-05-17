import torch
from typing import List, Tuple, Dict, Optional
import numpy as np

def generate_anchors(feature_map_size: Tuple[int, int], 
                    image_size: Tuple[int, int],
                    scales: List[float] = [128, 256, 512],
                    ratios: List[float] = [0.5, 1, 2.0]) -> torch.Tensor:
    """Generate anchor boxes for a feature map."""
    stride_h = image_size[0] / feature_map_size[0]
    stride_w = image_size[1] / feature_map_size[1]
    
    base_anchors = []
    for scale in scales:
        for ratio in ratios:
            w = scale * np.sqrt(ratio)
            h = scale / np.sqrt(ratio)
            base_anchors.append([-w/2, -h/2, w/2, h/2])
    
    base_anchors = torch.tensor(base_anchors, dtype=torch.float32)
    
    # Generate grid
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
        torch.clamp(anchors[..., 2:], max=torch.tensor([image_size[1], image_size[0]]))
    ], dim=-1)
    
    return anchors

def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou