import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
from typing import Dict, List, Tuple
from .object_detector import ObjectDetector
from .data_utils import get_dataloaders
from .rpn import RPN

# Add these imports at the top
from .dbnn import DBNN

class ObjectDetectorTrainer:
    def __init__(self, model, train_loader, val_loader, device, num_classes, output_dir='checkpoints'):
        # ... existing code ...
        
        # Add DBNN loss
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.SmoothL1Loss()
        self.dbpn_loss = nn.CrossEntropyLoss()  # For DBNN classification
        
    # In train.py - Update the train_epoch method to handle the augmented data:
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = cls_loss_total = reg_loss_total = dbpn_loss_total = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for images, targets in progress_bar:
            # Move data to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = self.model(images, targets)
            
            # Compute losses
            cls_loss = self.compute_cls_loss(outputs['rpn_objectness'], targets)
            reg_loss = self.compute_reg_loss(outputs['rpn_bbox'], targets)
            
            # Compute DBNN loss if available
            dbpn_loss = torch.tensor(0.0, device=self.device)
            if 'class_logits' in outputs and hasattr(self.model, 'use_dbpn') and self.model.use_dbpn:
                gt_labels = self._get_roi_labels(targets)
                dbpn_loss = self.dbpn_loss(outputs['class_logits'], gt_labels)
            
            # Total loss
            loss = cls_loss + reg_loss + dbpn_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients (optional but recommended for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            cls_loss_total += cls_loss.item()
            reg_loss_total += reg_loss.item()
            dbpn_loss_total += dbpn_loss.item() if hasattr(self, 'dbpn_loss') else 0.0
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / len(progress_bar),
                'cls': cls_loss_total / len(progress_bar),
                'reg': reg_loss_total / len(progress_bar),
                'dbpn': dbpn_loss_total / len(progress_bar) if hasattr(self, 'dbpn_loss') else 0.0
            })
        
            return {
                'loss': total_loss / len(self.train_loader),
                'cls_loss': cls_loss_total / len(self.train_loader),
                'reg_loss': reg_loss_total / len(self.train_loader),
                'dbpn_loss': dbpn_loss_total / len(self.train_loader) if hasattr(self, 'dbpn_loss') else 0.0
            }
        # ... rest of the method ...
    
    def _get_roi_labels(self, targets):
        # TODO: Implement this to get ground truth labels for sampled ROIs
        # This should match the ROIs generated in _sample_rois
        pass


class ObjectDetectorTrainer:
    def __init__(self, model, train_loader, val_loader, device, num_classes, output_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )
        
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.1)
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.SmoothL1Loss()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = cls_loss_total = reg_loss_total = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for images, targets in progress_bar:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            objectness_scores, pred_bbox_deltas = self.model(images)
            
            # Compute losses
            cls_loss = self.compute_cls_loss(objectness_scores, targets)
            reg_loss = self.compute_reg_loss(pred_bbox_deltas, targets)
            loss = cls_loss + reg_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            cls_loss_total += cls_loss.item()
            reg_loss_total += reg_loss.item()
            
            progress_bar.set_postfix({
                'loss': total_loss / len(progress_bar),
                'cls': cls_loss_total / len(progress_bar),
                'reg': reg_loss_total / len(progress_bar)
            })
        
        return {
            'loss': total_loss / len(self.train_loader),
            'cls_loss': cls_loss_total / len(self.train_loader),
            'reg_loss': reg_loss_total / len(self.train_loader)
        }
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_loss = cls_loss_total = reg_loss_total = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                objectness_scores, pred_bbox_deltas = self.model(images)
                
                cls_loss = self.compute_cls_loss(objectness_scores, targets)
                reg_loss = self.compute_reg_loss(pred_bbox_deltas, targets)
                val_loss += (cls_loss + reg_loss).item()
                cls_loss_total += cls_loss.item()
                reg_loss_total += reg_loss.item()
        
        return {
            'val_loss': val_loss / len(self.val_loader),
            'val_cls_loss': cls_loss_total / len(self.val_loader),
            'val_reg_loss': reg_loss_total / len(self.val_loader)
        }
    
    def compute_cls_loss(self, objectness_scores, targets):
        # TODO: Implement proper anchor matching
        return torch.tensor(0.0, device=self.device)
    
    def compute_reg_loss(self, pred_bbox_deltas, targets):
        # TODO: Implement proper regression target computation
        return torch.tensor(0.0, device=self.device)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        
        filename = os.path.join(self.output_dir, f'checkpoint_epoch{epoch}.pth')
        torch.save(state, filename)
        
        if is_best:
            best_filename = os.path.join(self.output_dir, 'model_best.pth')
            torch.save(state, best_filename)
    
    def train(self, num_epochs: int):
        best_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()
            
            is_best = val_metrics['val_loss'] < best_loss
            if is_best:
                best_loss = val_metrics['val_loss']
            
            self.save_checkpoint(epoch, is_best)
            
            print(f'\nEpoch {epoch}:')
            print(f'Train Loss: {train_metrics["loss"]:.4f} '
                  f'(Cls: {train_metrics["cls_loss"]:.4f}, '
                  f'Reg: {train_metrics["reg_loss"]:.4f})')
            print(f'Val Loss: {val_metrics["val_loss"]:.4f} '
                  f'(Cls: {val_metrics["val_cls_loss"]:.4f}, '
                  f'Reg: {val_metrics["val_reg_loss"]:.4f})')