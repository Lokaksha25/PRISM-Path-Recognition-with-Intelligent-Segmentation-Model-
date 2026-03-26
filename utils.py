"""
utils.py — Utility functions for LiteSeg drivable space segmentation.

Contains:
- Loss functions (FocalLoss, TverskyLoss, PRISMLossV2, legacy ComboLoss)
- Boundary-aware loss components
- Knowledge distillation loss
- Metrics (IoU, mIoU, Precision, Recall, FPR, confusion matrix)
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
# LOSS FUNCTIONS — NEW (Logit-based, false-positive-aware)
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for binary segmentation — focuses on hard pixels.
    
    Lin et al., 2017 — "Focal Loss for Dense Object Detection"
    
    FL(p) = -alpha * (1 - p)^gamma * log(p)   for positive class
    FL(p) = -(1-alpha) * p^gamma * log(1-p)    for negative class
    
    gamma > 0 reduces loss for well-classified pixels, focusing training
    on hard, misclassified examples (boundary pixels, confusing textures).
    
    alpha < 0.5 down-weights the dominant non-drivable class.
    
    Operates on RAW LOGITS (uses BCEWithLogitsLoss internally for stability).
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        """Initialize FocalLoss.
        
        Args:
            alpha: Weight for the positive (drivable) class. Default 0.25
                   means non-drivable gets weight 0.75 — penalizing FPs.
            gamma: Focusing parameter. Higher = more focus on hard pixels.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, target):
        """Compute Focal Loss on raw logits.
        
        Args:
            logits: Raw model output, shape (N, 1, H, W).
            target: Ground truth binary mask, shape (N, 1, H, W), values 0 or 1.
            
        Returns:
            Scalar focal loss.
        """
        # Sigmoid probability
        p = torch.sigmoid(logits)
        
        # BCE loss per pixel (no reduction)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        
        # p_t = probability of correct class
        p_t = p * target + (1 - p) * (1 - target)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss — Generalizes Dice with separate FP/FN weights.
    
    Salehi et al., 2017 — "Tversky loss function for image segmentation"
    
    Tversky Index = TP / (TP + alpha*FP + beta*FN)
    
    Setting alpha > beta penalizes False Positives more heavily.
    This directly attacks the "building segmented as road" problem.
    
    Setting alpha=0.5, beta=0.5 reduces to standard Dice.
    """
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        """Initialize TverskyLoss.
        
        Args:
            alpha: Weight for False Positives. Higher = more FP penalty.
            beta: Weight for False Negatives. Higher = more FN penalty.
            smooth: Smoothing factor to prevent division by zero.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, logits, target):
        """Compute Tversky Loss.
        
        Args:
            logits: Raw model output, shape (N, 1, H, W).
            target: Ground truth binary mask, shape (N, 1, H, W).
            
        Returns:
            Scalar Tversky loss.
        """
        pred = torch.sigmoid(logits)
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (pred_flat * target_flat).sum()
        FP = (pred_flat * (1 - target_flat)).sum()
        FN = ((1 - pred_flat) * target_flat).sum()
        
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        
        return 1.0 - tversky_index


class BoundaryFocalLoss(nn.Module):
    """Boundary-focused loss that applies extra supervision near edges.
    
    Extracts boundary pixels using morphological dilation/erosion,
    then applies weighted focal BCE only to those boundary pixels.
    This forces crisp edge predictions without affecting global loss.
    """
    
    def __init__(self, kernel_size=5, boundary_weight=3.0):
        """Initialize BoundaryFocalLoss.
        
        Args:
            kernel_size: Kernel for morphological boundary extraction.
            boundary_weight: Extra weight for boundary pixels.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.boundary_weight = boundary_weight
    
    def _get_boundary_mask(self, target):
        """Extract boundary band using morphological gradient.
        
        Args:
            target: Binary mask (N, 1, H, W).
            
        Returns:
            Boundary mask (N, 1, H, W) — 1 at boundaries, 0 elsewhere.
        """
        k = self.kernel_size
        p = k // 2
        dilated = F.max_pool2d(target, k, stride=1, padding=p)
        eroded = -F.max_pool2d(-target, k, stride=1, padding=p)
        boundary = (dilated - eroded).clamp(0, 1)
        return boundary
    
    def forward(self, logits, target):
        """Compute boundary-focused loss.
        
        Args:
            logits: Raw logits (N, 1, H, W).
            target: Ground truth (N, 1, H, W).
            
        Returns:
            Scalar boundary loss.
        """
        boundary = self._get_boundary_mask(target)
        
        # Weight map: higher weight at boundaries
        weight_map = 1.0 + boundary * (self.boundary_weight - 1.0)
        
        # Weighted BCE on logits
        bce = F.binary_cross_entropy_with_logits(
            logits, target, weight=weight_map, reduction='mean'
        )
        return bce


class PRISMLossV2(nn.Module):
    """PRISM Loss V2 — Combined loss for false-positive-aware segmentation.
    
    Combines three complementary objectives:
    1. Focal Loss: Focuses on hard/misclassified pixels, handles class imbalance
    2. Tversky Loss: Penalizes False Positives (buildings/sky → road) heavily
    3. Boundary Loss: Extra supervision at drivable area edges
    
    All operate on raw logits for numerical stability.
    
    PRISMLossV2 = w_focal * FocalLoss + w_tversky * TverskyLoss + w_boundary * BoundaryLoss
    """
    
    def __init__(self, w_focal=0.5, w_tversky=0.3, w_boundary=0.2,
                 focal_alpha=0.25, focal_gamma=2.0,
                 tversky_alpha=0.7, tversky_beta=0.3):
        """Initialize PRISMLossV2.
        
        Args:
            w_focal: Weight for focal loss component.
            w_tversky: Weight for Tversky loss component.
            w_boundary: Weight for boundary loss component.
            focal_alpha: Alpha for focal loss (positive class weight).
            focal_gamma: Gamma for focal loss (focusing parameter).
            tversky_alpha: Alpha for Tversky (FP weight). Higher = more FP penalty.
            tversky_beta: Beta for Tversky (FN weight).
        """
        super().__init__()
        self.w_focal = w_focal
        self.w_tversky = w_tversky
        self.w_boundary = w_boundary
        
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.boundary = BoundaryFocalLoss(kernel_size=5, boundary_weight=3.0)
    
    def forward(self, logits, target):
        """Compute combined PRISM loss.
        
        Args:
            logits: Raw model output (N, 1, H, W).
            target: Ground truth binary mask (N, 1, H, W).
            
        Returns:
            Scalar combined loss.
        """
        loss_focal = self.focal(logits, target)
        loss_tversky = self.tversky(logits, target)
        loss_boundary = self.boundary(logits, target)
        
        return (self.w_focal * loss_focal + 
                self.w_tversky * loss_tversky + 
                self.w_boundary * loss_boundary)


# =============================================================================
# LEGACY LOSS FUNCTIONS (backward compatibility)
# =============================================================================

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation — handles class imbalance.
    
    Operates on probabilities (apply sigmoid before calling if using logits).
    """
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """Compute Dice loss on probabilities.
        
        Args:
            pred: Predicted probabilities (N, 1, H, W).
            target: Ground truth (N, 1, H, W).
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice


class ComboLoss(nn.Module):
    """Legacy Combination of BCE and Dice Loss (for backward compatibility).
    
    Now operates on logits: applies sigmoid internally.
    """
    
    def __init__(self, alpha=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, logits, target):
        """Compute combo loss on logits.
        
        Args:
            logits: Raw logits (N, 1, H, W).
            target: Ground truth (N, 1, H, W).
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, target)
        pred = torch.sigmoid(logits)
        dice_loss = self.dice(pred, target)
        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss


class BoundaryAwareLoss(nn.Module):
    """Legacy boundary-aware loss. Now operates on logits."""
    
    def __init__(self, alpha=0.5, boundary_weight=2.0, kernel_size=5):
        super().__init__()
        self.alpha = alpha
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size
        self.dice = DiceLoss()
    
    def _get_boundary_mask(self, target):
        kernel = self.kernel_size
        padding = kernel // 2
        dilated = F.max_pool2d(target, kernel, stride=1, padding=padding)
        eroded = -F.max_pool2d(-target, kernel, stride=1, padding=padding)
        return dilated - eroded
    
    def forward(self, logits, target):
        boundary = self._get_boundary_mask(target)
        weight_map = 1.0 + boundary * (self.boundary_weight - 1.0)
        bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight_map)
        pred = torch.sigmoid(logits)
        dice_loss = self.dice(pred, target)
        return self.alpha * bce + (1.0 - self.alpha) * dice_loss


class DistillationLoss(nn.Module):
    """Knowledge distillation loss using PRISMLossV2 as task loss.
    
    Loss = alpha * task_loss(student, gt) + (1-alpha) * KD_loss(student, teacher)
    
    Both student and teacher outputs are raw logits.
    """
    
    def __init__(self, alpha=0.5, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.task_loss = PRISMLossV2()
    
    def forward(self, student_logits, teacher_logits, target):
        """Compute distillation loss.
        
        Args:
            student_logits: Student raw logits (N, 1, H, W).
            teacher_logits: Teacher raw logits (N, 1, H, W).
            target: Ground truth (N, 1, H, W).
        """
        # Task loss against ground truth
        task = self.task_loss(student_logits, target)
        
        # Soft target loss with temperature scaling
        student_soft = torch.sigmoid(student_logits / self.temperature)
        teacher_soft = torch.sigmoid(teacher_logits / self.temperature)
        
        kd_loss = F.mse_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        return self.alpha * task + (1.0 - self.alpha) * kd_loss


# =============================================================================
# METRICS — Enhanced with false-positive detection
# =============================================================================

def compute_iou(pred, target, threshold=0.5):
    """Compute IoU for binary segmentation.
    
    Args:
        pred: Predicted probabilities or logits, shape (N, 1, H, W) or (H, W).
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
        return 1.0
    return intersection / union


def compute_miou(pred, target, threshold=0.5):
    """Compute mean IoU over both classes.
    
    Args:
        pred: Predicted probabilities (after sigmoid), shape (N, 1, H, W).
        target: Ground truth binary mask, same shape.
        threshold: Threshold to binarize predictions.
        
    Returns:
        Tuple of (mIoU, iou_drivable, iou_non_drivable).
    """
    iou_drivable = compute_iou(pred, target, threshold)
    iou_non_drivable = compute_iou(1 - pred, 1 - target, threshold)
    miou = (iou_drivable + iou_non_drivable) / 2.0
    return miou, iou_drivable, iou_non_drivable


def compute_detailed_metrics(pred, target, threshold=0.5):
    """Compute comprehensive metrics including false-positive detection.
    
    THIS IS THE KEY METRIC FUNCTION that exposes the false-positive bleeding
    that mIoU alone hides.
    
    Args:
        pred: Predicted probabilities (after sigmoid), shape (N, 1, H, W).
        target: Ground truth binary mask, shape (N, 1, H, W).
        threshold: Binarization threshold.
        
    Returns:
        Dict with: mIoU, iou_drivable, iou_non_drivable,
                   precision, recall, f1, fpr (false positive rate)
    """
    if isinstance(pred, torch.Tensor):
        pred_bin = (pred > threshold).float()
        target_f = target.float()
        
        TP = (pred_bin * target_f).sum().item()
        FP = (pred_bin * (1 - target_f)).sum().item()
        FN = ((1 - pred_bin) * target_f).sum().item()
        TN = ((1 - pred_bin) * (1 - target_f)).sum().item()
    else:
        pred_bin = (pred > threshold).astype(np.float32)
        target_f = target.astype(np.float32)
        
        TP = (pred_bin * target_f).sum()
        FP = (pred_bin * (1 - target_f)).sum()
        FN = ((1 - pred_bin) * target_f).sum()
        TN = ((1 - pred_bin) * (1 - target_f)).sum()
    
    # Precision: of all predicted-drivable, how many are actually drivable?
    precision = TP / (TP + FP + 1e-8)
    
    # Recall: of all actually-drivable, how many did we find?
    recall = TP / (TP + FN + 1e-8)
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # FALSE POSITIVE RATE: of all non-drivable pixels, how many did we
    # incorrectly predict as drivable? This is the KEY metric for detecting
    # "building/sky/pole segmented as road"
    fpr = FP / (FP + TN + 1e-8)
    
    # Standard mIoU
    miou, iou_drv, iou_non = compute_miou(pred, target, threshold)
    
    return {
        'miou': miou,
        'iou_drivable': iou_drv,
        'iou_non_drivable': iou_non,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'tp': TP,
        'fp': FP,
        'fn': FN,
        'tn': TN,
    }


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
    
    Args:
        mask: Binary mask as numpy array (H, W), values 0 or 1.
        kernel_size: Size of morphological kernel.
        
    Returns:
        Refined binary mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    refined = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
    return refined.astype(np.float32)


def test_time_augmentation(model, image, device='cpu'):
    """Apply Test-Time Augmentation (horizontal flip averaging).
    
    Uses model.predict() to get probabilities.
    
    Args:
        model: Trained segmentation model (returns logits from forward).
        image: Input tensor, shape (1, 3, H, W).
        device: Device to run inference on.
        
    Returns:
        Averaged prediction tensor, shape (1, 1, H, W).
    """
    model.eval()
    with torch.no_grad():
        pred_orig = model.predict(image.to(device))
        image_flip = torch.flip(image, dims=[3])
        pred_flip = model.predict(image_flip.to(device))
        pred_flip = torch.flip(pred_flip, dims=[3])
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
    
    green_mask = np.zeros_like(image)
    green_mask[:, :, 1] = 255
    
    red_mask = np.zeros_like(image)
    red_mask[:, :, 2] = 255
    
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
            
    Returns:
        Tuple of (mean, std) — each a list of 3 floats.
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