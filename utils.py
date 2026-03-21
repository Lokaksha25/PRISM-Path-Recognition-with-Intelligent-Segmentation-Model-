"""
utils.py — Utility functions for LiteSeg drivable space segmentation.

Contains:
- Loss functions (ComboLoss, BoundaryAwareLoss)
- Metrics (IoU, mIoU, confusion matrix)
- Post-processing (boundary refinement, TTA)
- Visualization helpers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation — handles class imbalance.
    
    Dice = 2 * |pred ∩ gt| / (|pred| + |gt|)
    Loss = 1 - Dice
    
    Smooth term prevents division by zero.
    """
    
    def __init__(self, smooth=1.0):
        """Initialize DiceLoss.
        
        Args:
            smooth: Smoothing factor to prevent division by zero.
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """Compute Dice loss.
        
        Args:
            pred: Predicted probabilities, shape (N, 1, H, W).
            target: Ground truth binary mask, shape (N, 1, H, W).
            
        Returns:
            Scalar dice loss.
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice


class ComboLoss(nn.Module):
    """Combination of Binary Cross-Entropy and Dice Loss.
    
    ComboLoss = alpha * BCE + (1 - alpha) * Dice
    
    BCE handles per-pixel classification accuracy.
    Dice handles global overlap / class imbalance.
    """
    
    def __init__(self, alpha=0.5, smooth=1.0):
        """Initialize ComboLoss.
        
        Args:
            alpha: Weight for BCE (1-alpha for Dice). Default 0.5.
            smooth: Smoothing factor for Dice loss.
        """
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, pred, target):
        """Compute combo loss.
        
        Args:
            pred: Predicted probabilities, shape (N, 1, H, W).
            target: Ground truth binary mask, shape (N, 1, H, W).
            
        Returns:
            Scalar combo loss.
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss


class BoundaryAwareLoss(nn.Module):
    """Boundary-aware loss that applies extra weighting near drivable boundaries.
    
    Detects boundary pixels using morphological gradient, then applies
    higher loss weight at boundary regions to improve edge quality.
    """
    
    def __init__(self, alpha=0.5, boundary_weight=2.0, kernel_size=5):
        """Initialize BoundaryAwareLoss.
        
        Args:
            alpha: Weight for BCE vs Dice in base combo loss.
            boundary_weight: Extra weight multiplier for boundary pixels.
            kernel_size: Kernel size for boundary detection.
        """
        super().__init__()
        self.alpha = alpha
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size
        self.dice = DiceLoss()
    
    def _get_boundary_mask(self, target):
        """Extract boundary pixels using morphological gradient.
        
        Args:
            target: Binary mask tensor, shape (N, 1, H, W).
            
        Returns:
            Boundary mask tensor, same shape as target.
        """
        # Use max pooling to dilate and -min pooling (via max of negated) to erode
        kernel = self.kernel_size
        padding = kernel // 2
        dilated = F.max_pool2d(target, kernel, stride=1, padding=padding)
        eroded = -F.max_pool2d(-target, kernel, stride=1, padding=padding)
        boundary = dilated - eroded
        return boundary
    
    def forward(self, pred, target):
        """Compute boundary-aware loss.
        
        Args:
            pred: Predicted probabilities, shape (N, 1, H, W).
            target: Ground truth binary mask, shape (N, 1, H, W).
            
        Returns:
            Scalar boundary-aware loss.
        """
        # Get boundary mask for extra weighting
        boundary = self._get_boundary_mask(target)
        
        # Weight map: 1.0 everywhere + extra weight at boundaries
        weight_map = 1.0 + boundary * (self.boundary_weight - 1.0)
        
        # Weighted BCE
        bce = F.binary_cross_entropy(pred, target, weight=weight_map)
        
        # Standard Dice (global, not pixel-weighted)
        dice_loss = self.dice(pred, target)
        
        return self.alpha * bce + (1.0 - self.alpha) * dice_loss


class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining task loss and soft-target loss.
    
    Loss = alpha * task_loss(student, gt) + (1-alpha) * KD_loss(student, teacher)
    
    KD loss uses soft probabilities from the teacher to guide the student.
    """
    
    def __init__(self, alpha=0.5, temperature=3.0):
        """Initialize DistillationLoss.
        
        Args:
            alpha: Weight for ground-truth task loss vs distillation loss.
            temperature: Temperature for softening probabilities.
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.task_loss = ComboLoss()
    
    def forward(self, student_pred, teacher_pred, target):
        """Compute distillation loss.
        
        Args:
            student_pred: Student model output probabilities (N, 1, H, W).
            teacher_pred: Teacher model output probabilities (N, 1, H, W).
            target: Ground truth binary mask (N, 1, H, W).
            
        Returns:
            Scalar distillation loss.
        """
        # Task loss against ground truth
        task = self.task_loss(student_pred, target)
        
        # Soft target loss — MSE between student and teacher soft predictions
        # Apply temperature scaling in logit space
        student_logits = torch.logit(student_pred.clamp(1e-6, 1 - 1e-6))
        teacher_logits = torch.logit(teacher_pred.clamp(1e-6, 1 - 1e-6))
        
        student_soft = torch.sigmoid(student_logits / self.temperature)
        teacher_soft = torch.sigmoid(teacher_logits / self.temperature)
        
        kd_loss = F.mse_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        return self.alpha * task + (1.0 - self.alpha) * kd_loss


# =============================================================================
# METRICS
# =============================================================================

def compute_iou(pred, target, threshold=0.5):
    """Compute Intersection-over-Union for binary segmentation.
    
    Args:
        pred: Predicted probabilities, shape (N, 1, H, W) or (H, W).
        target: Ground truth binary mask, same shape.
        threshold: Threshold to binarize predictions.
        
    Returns:
        IoU score (float).
    """
    if isinstance(pred, torch.Tensor):
        pred = (pred > threshold).float()
        intersection = (pred * target).sum().item()
        union = ((pred + target) > 0).float().sum().item()
    else:
        pred = (pred > threshold).astype(np.float32)
        intersection = (pred * target).sum()
        union = ((pred + target) > 0).astype(np.float32).sum()
    
    if union == 0:
        return 1.0  # Both empty
    return intersection / union


def compute_miou(pred, target, threshold=0.5):
    """Compute mean IoU over both classes (drivable and non-drivable).
    
    Args:
        pred: Predicted probabilities, shape (N, 1, H, W).
        target: Ground truth binary mask, same shape.
        threshold: Threshold to binarize predictions.
        
    Returns:
        Tuple of (mIoU, iou_drivable, iou_non_drivable).
    """
    if isinstance(pred, torch.Tensor):
        pred_bin = (pred > threshold).float()
    else:
        pred_bin = (pred > threshold).astype(np.float32)
    
    # IoU for drivable class (label = 1)
    iou_drivable = compute_iou(pred, target, threshold)
    
    # IoU for non-drivable class (label = 0)
    iou_non_drivable = compute_iou(1 - pred, 1 - target, threshold)
    
    miou = (iou_drivable + iou_non_drivable) / 2.0
    return miou, iou_drivable, iou_non_drivable


def compute_confusion_matrix(pred, target, threshold=0.5):
    """Compute confusion matrix for binary segmentation.
    
    Args:
        pred: Predicted probabilities (numpy array or tensor).
        target: Ground truth binary mask.
        threshold: Threshold to binarize predictions.
        
    Returns:
        2×2 confusion matrix as numpy array.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    pred_flat = (pred.flatten() > threshold).astype(int)
    target_flat = target.flatten().astype(int)
    
    return sklearn_confusion_matrix(target_flat, pred_flat, labels=[0, 1])


# =============================================================================
# POST-PROCESSING
# =============================================================================

def boundary_refinement(mask, kernel_size=5):
    """Apply morphological boundary refinement to clean jagged mask edges.
    
    Applies closing (dilation→erosion) to fill small holes,
    then opening (erosion→dilation) to remove small noise.
    
    Args:
        mask: Binary mask as numpy array (H, W), values 0 or 1.
        kernel_size: Size of morphological kernel.
        
    Returns:
        Refined binary mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    # Closing: fill small holes in drivable area
    refined = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    # Opening: remove small noise blobs
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
    return refined.astype(np.float32)


def test_time_augmentation(model, image, device='cpu'):
    """Apply Test-Time Augmentation (horizontal flip averaging).
    
    Averages predictions from the original image and its horizontal flip
    for a free mIoU improvement.
    
    Args:
        model: Trained segmentation model.
        image: Input tensor, shape (1, 3, H, W).
        device: Device to run inference on.
        
    Returns:
        Averaged prediction tensor, shape (1, 1, H, W).
    """
    model.eval()
    with torch.no_grad():
        # Original prediction
        pred_orig = model(image.to(device))
        
        # Flipped prediction
        image_flip = torch.flip(image, dims=[3])  # Horizontal flip
        pred_flip = model(image_flip.to(device))
        pred_flip = torch.flip(pred_flip, dims=[3])  # Flip back
        
        # Average
        pred_avg = (pred_orig + pred_flip) / 2.0
    
    return pred_avg


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_overlay(image, mask, alpha=0.5):
    """Create visualization overlay with green=drivable, red=non-drivable.
    
    Args:
        image: RGB image as numpy array (H, W, 3), values 0-255.
        mask: Binary mask (H, W), values 0 or 1.
        alpha: Transparency of overlay.
        
    Returns:
        Overlay image as numpy array (H, W, 3).
    """
    overlay = image.copy()
    
    # Green for drivable
    green_mask = np.zeros_like(image)
    green_mask[:, :, 1] = 255  # Green channel
    
    # Red for non-drivable  
    red_mask = np.zeros_like(image)
    red_mask[:, :, 2] = 255  # Red channel (BGR) or index 0 for RGB
    
    # Apply masks
    drivable = mask > 0.5
    non_drivable = ~drivable
    
    overlay[drivable] = cv2.addWeighted(
        image[drivable], 1 - alpha, green_mask[drivable], alpha, 0
    )
    overlay[non_drivable] = cv2.addWeighted(
        image[non_drivable], 1 - alpha / 2, red_mask[non_drivable], alpha / 2, 0
    )
    
    return overlay


def save_comparison(image, gt_mask, pred_mask, save_path):
    """Save side-by-side comparison of ground truth and prediction.
    
    Args:
        image: RGB image (H, W, 3).
        gt_mask: Ground truth mask (H, W).
        pred_mask: Predicted mask (H, W).
        save_path: Path to save the visualization.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    overlay = create_overlay(image, pred_mask)
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# DATASET STATISTICS
# =============================================================================

def compute_dataset_stats(dataloader):
    """Compute per-channel mean and std of images in the dataset.
    
    Args:
        dataloader: PyTorch DataLoader providing (image, mask) tuples.
            Images should be normalized to [0, 1].
            
    Returns:
        Tuple of (mean, std) — each a list of 3 floats (per channel).
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_pixels = 0
    
    for images, _ in dataloader:
        batch_size = images.size(0)
        images = images.view(batch_size, 3, -1)
        n_pixels += images.size(2) * batch_size
        mean += images.sum(dim=[0, 2])
        std += (images ** 2).sum(dim=[0, 2])
    
    mean /= n_pixels
    std = torch.sqrt(std / n_pixels - mean ** 2)
    
    return mean.tolist(), std.tolist()


def count_parameters(model):
    """Count total trainable parameters in a model.
    
    Args:
        model: PyTorch nn.Module.
        
    Returns:
        Total parameter count (int).
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
