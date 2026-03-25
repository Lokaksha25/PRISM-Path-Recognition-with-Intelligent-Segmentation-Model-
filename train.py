"""
train.py — Training pipeline for LiteSeg drivable space segmentation.

Features:
- AdamW optimizer with CosineAnnealingLR schedule
- ComboLoss (BCE + Dice) with optional boundary-aware weighting
- TensorBoard logging of loss, mIoU, learning rate
- Best checkpoint saving based on validation mIoU
- Knowledge distillation support (teacher → student)
- tqdm progress bars
- Full argparse configurability

Usage:
    python train.py --dataroot ./ --mask_dir masks --epochs 50 --batch_size 16
    python train.py --distill --teacher_weights teacher_best.pth --epochs 50
"""

import os
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from model import LiteSegNet, LiteSegTeacher, get_model_info
from dataset import get_dataloaders
from utils import (
    ComboLoss, BoundaryAwareLoss, DistillationLoss,
    FocalTverskyLoss, BFABoundaryLoss, PRISMLoss,
    compute_miou, count_parameters
)


def parse_args():
    """Parse training arguments.
    
    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description='Train LiteSeg for drivable space segmentation')
    
    # Data
    parser.add_argument('--dataroot', type=str, default='./',
                        help='Root directory of nuScenes dataset')
    parser.add_argument('--mask_dir', type=str, default='masks',
                        help='Directory containing generated masks')
    parser.add_argument('--img_height', type=int, default=256,
                        help='Input image height')
    parser.add_argument('--img_width', type=int, default=448,
                        help='Input image width')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for AdamW')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (0 for Windows)')
    
    # Loss
    parser.add_argument('--loss', type=str, default='prism',
                        choices=['combo', 'boundary', 'prism'],
                        help='Loss function: combo, boundary-aware, or prism (recommended)')
    parser.add_argument('--loss_alpha', type=float, default=0.5,
                        help='BCE weight in combo loss (1-alpha for Dice)')
    
    # Knowledge distillation
    parser.add_argument('--distill', action='store_true',
                        help='Enable knowledge distillation')
    parser.add_argument('--teacher_weights', type=str, default='teacher_best.pth',
                        help='Path to trained teacher model weights')
    parser.add_argument('--distill_alpha', type=float, default=0.5,
                        help='Weight for task loss vs distillation loss')
    parser.add_argument('--distill_temp', type=float, default=3.0,
                        help='Temperature for distillation')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory for checkpoints and logs')
    parser.add_argument('--log_dir', type=str, default='runs',
                        help='TensorBoard log directory')
    parser.add_argument('--save_name', type=str, default='best_model.pth',
                        help='Filename for best checkpoint')
    
    # Model
    parser.add_argument('--train_teacher', action='store_true',
                        help='Train the teacher model instead of student')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch,
                    teacher_model=None, distill_loss_fn=None):
    """Train the model for one epoch.
    
    Args:
        model: The segmentation model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Current epoch number.
        teacher_model: Optional teacher model for distillation.
        distill_loss_fn: Optional distillation loss function.
        
    Returns:
        Dict with average metrics for the epoch.
    """
    model.train()
    
    running_loss = 0.0
    running_iou = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]', leave=False)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        predictions = model(images)
        
        # Compute loss
        if teacher_model is not None and distill_loss_fn is not None:
            teacher_model.eval()
            with torch.no_grad():
                teacher_pred = teacher_model(images)
            loss = distill_loss_fn(predictions, teacher_pred, masks)
        else:
            loss = criterion(predictions, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            miou, iou_drv, iou_non = compute_miou(predictions, masks)
        
        running_loss += loss.item()
        running_iou += miou
        n_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mIoU': f'{miou:.4f}'
        })
    
    avg_loss = running_loss / max(n_batches, 1)
    avg_iou = running_iou / max(n_batches, 1)
    
    return {'loss': avg_loss, 'miou': avg_iou}


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    """Validate the model.
    
    Args:
        model: The segmentation model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Device to run on.
        epoch: Current epoch number.
        
    Returns:
        Dict with validation metrics.
    """
    model.eval()
    
    running_loss = 0.0
    running_miou = 0.0
    running_iou_drv = 0.0
    running_iou_non = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]  ', leave=False)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        predictions = model(images)
        loss = criterion(predictions, masks)
        
        miou, iou_drv, iou_non = compute_miou(predictions, masks)
        
        running_loss += loss.item()
        running_miou += miou
        running_iou_drv += iou_drv
        running_iou_non += iou_non
        n_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mIoU': f'{miou:.4f}'
        })
    
    n = max(n_batches, 1)
    return {
        'loss': running_loss / n,
        'miou': running_miou / n,
        'iou_drivable': running_iou_drv / n,
        'iou_non_drivable': running_iou_non / n,
    }


def measure_fps(model, device, input_size=(1, 3, 256, 448), n_runs=100):
    """Measure inference FPS.
    
    Args:
        model: The segmentation model.
        device: Device to benchmark on.
        input_size: Input tensor shape.
        n_runs: Number of forward passes to average.
        
    Returns:
        FPS as float.
    """
    model.eval()
    dummy = torch.randn(*input_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)
    
    # Benchmark
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(dummy)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    fps = n_runs / elapsed
    
    return fps


def main():
    """Main training loop."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"LiteSeg Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Auto-adjust batch size for CPU
    if device.type == 'cpu' and args.batch_size > 8:
        print(f"CPU detected: reducing batch_size from {args.batch_size} to 4")
        args.batch_size = 4
    
    # Create model
    if args.train_teacher:
        model = LiteSegTeacher().to(device)
        model_name = "LiteSegTeacher"
    else:
        model = LiteSegNet().to(device)
        model_name = "LiteSegNet"
    
    n_params = count_parameters(model)
    print(f"Model: {model_name}")
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Create data loaders
    train_loader, val_loader, data_stats = get_dataloaders(
        dataroot=args.dataroot,
        mask_dir=args.mask_dir,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Loss function
    if args.loss == 'prism':
        criterion = PRISMLoss(
            alpha=0.3, beta=0.7, gamma=0.75,
            boundary_weight=0.4, dilation_kernel=7
        ).to(device)
        print("Loss: PRISMLoss (FocalTversky + BFA boundary) — recommended")
    elif args.loss == 'boundary':
        criterion = BoundaryAwareLoss(alpha=args.loss_alpha).to(device)
        print("Loss: BoundaryAwareLoss")
    else:
        criterion = ComboLoss(alpha=args.loss_alpha).to(device)
        print("Loss: ComboLoss (BCE + Dice)")
    
    # Knowledge distillation setup
    teacher_model = None
    distill_loss_fn = None
    if args.distill:
        print(f"Knowledge distillation enabled (teacher: {args.teacher_weights})")
        teacher_model = LiteSegTeacher().to(device)
        teacher_ckpt = torch.load(args.teacher_weights, map_location=device)
        teacher_model.load_state_dict(teacher_ckpt['model_state_dict'])
        teacher_model.eval()
        distill_loss_fn = DistillationLoss(
            alpha=args.distill_alpha,
            temperature=args.distill_temp
        ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_miou = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_miou = ckpt.get('best_miou', 0.0)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Save training config
    config = vars(args)
    config['device'] = str(device)
    config['model_params'] = n_params
    config['data_stats'] = data_stats
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_miou': [], 'val_miou': [], 'lr': []}
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"{'='*60}\n")
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            teacher_model=teacher_model, distill_loss_fn=distill_loss_fn
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # Step scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_miou'].append(train_metrics['miou'])
        history['val_miou'].append(val_metrics['miou'])
        history['lr'].append(current_lr)
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('mIoU/train', train_metrics['miou'], epoch)
        writer.add_scalar('mIoU/val', val_metrics['miou'], epoch)
        writer.add_scalar('IoU/drivable', val_metrics['iou_drivable'], epoch)
        writer.add_scalar('IoU/non_drivable', val_metrics['iou_non_drivable'], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Save best checkpoint
        is_best = val_metrics['miou'] > best_miou
        if is_best:
            best_miou = val_metrics['miou']
            save_path = os.path.join(args.output_dir, args.save_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'config': config,
                'data_stats': data_stats,
            }, save_path)
        
        # Save latest checkpoint
        latest_path = os.path.join(args.output_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou,
            'config': config,
            'data_stats': data_stats,
        }, latest_path)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        best_marker = ' ★ BEST' if is_best else ''
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Train mIoU: {train_metrics['miou']:.4f} | "
              f"Val mIoU: {val_metrics['miou']:.4f} | "
              f"IoU(drv): {val_metrics['iou_drivable']:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s{best_marker}")
        
        # Detailed report every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n{'─'*60}")
            print(f"  Checkpoint Report @ Epoch {epoch+1}")
            print(f"  Best Val mIoU: {best_miou:.4f}")
            print(f"  Val IoU (drivable): {val_metrics['iou_drivable']:.4f}")
            print(f"  Val IoU (non-drivable): {val_metrics['iou_non_drivable']:.4f}")
            
            # Measure FPS
            fps = measure_fps(model, device)
            print(f"  Inference FPS ({device}): {fps:.1f}")
            writer.add_scalar('FPS', fps, epoch)
            print(f"{'─'*60}\n")
    
    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    writer.close()
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate training curves
    generate_training_curves(history, args.output_dir)
    
    # Final FPS measurement
    fps = measure_fps(model, device)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Val mIoU: {best_miou:.4f}")
    print(f"Inference FPS ({device}): {fps:.1f}")
    print(f"Model saved to: {os.path.join(args.output_dir, args.save_name)}")
    print(f"TensorBoard logs: {args.log_dir}")
    print(f"Training history: {history_path}")
    print(f"{'='*60}")


def generate_training_curves(history, output_dir):
    """Generate and save training curve plots.
    
    Args:
        history: Dict with training history lists.
        output_dir: Directory to save plots.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # mIoU curves
    axes[1].plot(epochs, history['train_miou'], 'b-', label='Train mIoU', linewidth=2)
    axes[1].plot(epochs, history['val_miou'], 'r-', label='Val mIoU', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('Training & Validation mIoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {os.path.join(output_dir, 'training_curves.png')}")


if __name__ == '__main__':
    main()