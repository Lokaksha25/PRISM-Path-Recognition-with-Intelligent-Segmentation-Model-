"""
evaluate.py — Evaluation pipeline for LiteSeg drivable space segmentation.

Computes:
- mIoU (drivable class and overall binary)
- Confusion matrix
- Per-scene mIoU breakdown (best 3 and worst 3)
- FPS benchmarks (CPU and GPU)
- 10 overlay visualizations
- Edge case analysis

Usage:
    python evaluate.py --weights output/best_model.pth --dataroot ./ --mask_dir masks
"""

import os
import json
import time
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

from model import LiteSegNet, LiteSegTeacher
from dataset import get_dataloaders, get_val_augmentations, NuScenesDrivableDataset
from utils import (
    compute_miou, compute_confusion_matrix, boundary_refinement,
    test_time_augmentation, create_overlay, save_comparison, count_parameters
)


def parse_args():
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser(description='Evaluate LiteSeg model')
    parser.add_argument('--weights', type=str, default='output/best_model.pth',
                        help='Path to model weights')
    parser.add_argument('--dataroot', type=str, default='./',
                        help='Root directory of nuScenes dataset')
    parser.add_argument('--mask_dir', type=str, default='masks',
                        help='Directory containing ground truth masks')
    parser.add_argument('--output_dir', type=str, default='eval_output',
                        help='Output directory for evaluation results')
    parser.add_argument('--img_height', type=int, default=256)
    parser.add_argument('--img_width', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--use_tta', action='store_true',
                        help='Use Test-Time Augmentation')
    parser.add_argument('--use_boundary_refinement', action='store_true',
                        help='Apply morphological boundary refinement')
    parser.add_argument('--teacher', action='store_true',
                        help='Evaluate teacher model instead of student')
    parser.add_argument('--num_visualizations', type=int, default=10,
                        help='Number of overlay visualizations to generate')
    return parser.parse_args()


def load_model(weights_path, device, is_teacher=False):
    """Load a trained model from checkpoint.
    
    Args:
        weights_path: Path to the checkpoint file.
        device: Device to load model on.
        is_teacher: If True, load teacher model architecture.
        
    Returns:
        Tuple of (model, checkpoint_dict).
    """
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    if is_teacher:
        model = LiteSegTeacher().to(device)
    else:
        model = LiteSegNet().to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


@torch.no_grad()
def evaluate_full(model, dataloader, device, use_tta=False, 
                  use_boundary_ref=False):
    """Run full evaluation on the validation set.
    
    Args:
        model: Trained model.
        dataloader: Validation DataLoader.
        device: Device to run on.
        use_tta: Whether to apply Test-Time Augmentation.
        use_boundary_ref: Whether to apply boundary refinement.
        
    Returns:
        Dict with all evaluation metrics and per-batch predictions.
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_mious = []
    all_iou_drv = []
    all_iou_non = []
    
    for images, masks in tqdm(dataloader, desc='Evaluating'):
        images = images.to(device)
        masks = masks.to(device)
        
        if use_tta:
            predictions = test_time_augmentation(model, images, device)
        else:
            predictions = model(images)
        
        # Apply boundary refinement if requested
        if use_boundary_ref:
            pred_np = predictions.cpu().numpy()
            for i in range(pred_np.shape[0]):
                pred_np[i, 0] = boundary_refinement(
                    (pred_np[i, 0] > 0.5).astype(np.float32)
                )
            predictions = torch.from_numpy(pred_np).to(device)
        
        # Compute per-batch metrics
        miou, iou_drv, iou_non = compute_miou(predictions, masks)
        all_mious.append(miou)
        all_iou_drv.append(iou_drv)
        all_iou_non.append(iou_non)
        
        all_preds.append(predictions.cpu())
        all_targets.append(masks.cpu())
    
    # Aggregate
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Overall metrics
    overall_miou, overall_iou_drv, overall_iou_non = compute_miou(all_preds, all_targets)
    
    # Confusion matrix
    cm = compute_confusion_matrix(all_preds, all_targets)
    
    return {
        'miou': overall_miou,
        'iou_drivable': overall_iou_drv,
        'iou_non_drivable': overall_iou_non,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'targets': all_targets,
    }


def measure_fps_detailed(model, device, input_size=(1, 3, 256, 448), n_runs=200):
    """Measure detailed FPS statistics.
    
    Args:
        model: Model to benchmark.
        device: Device to run on.
        input_size: Input tensor shape.
        n_runs: Number of runs.
        
    Returns:
        Dict with fps, latency stats.
    """
    model.eval()
    dummy = torch.randn(*input_size).to(device)
    
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy)
    
    latencies = []
    for _ in range(n_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        latencies.append(time.perf_counter() - start)
    
    latencies = np.array(latencies) * 1000  # Convert to ms
    
    return {
        'fps': 1000.0 / np.mean(latencies),
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
    }


def per_scene_evaluation(model, dataroot, mask_dir, device, 
                         img_height=256, img_width=448, mean=None, std=None):
    """Compute mIoU breakdown per scene.
    
    Args:
        model: Trained model.
        dataroot: Dataset root.
        mask_dir: Mask directory.
        device: Device.
        img_height: Image height.
        img_width: Image width.
        mean: Normalization mean.
        std: Normalization std.
        
    Returns:
        Dict mapping scene_name to mIoU.
    """
    from dataset import get_scene_split
    
    # Load scene info
    with open(os.path.join(dataroot, 'v1.0-mini', 'scene.json')) as f:
        scenes = json.load(f)
    with open(os.path.join(dataroot, 'v1.0-mini', 'sample.json')) as f:
        samples = json.load(f)
    with open(os.path.join(dataroot, 'v1.0-mini', 'sample_data.json')) as f:
        sample_data_list = json.load(f)
    
    # Build mappings
    scene_map = {s['token']: s['name'] for s in scenes}
    sample_to_scene = {s['token']: s['scene_token'] for s in samples}
    
    # Map filenames to scene tokens
    filename_to_scene = {}
    for sd in sample_data_list:
        if sd.get('is_key_frame') and 'sample_token' in sd:
            sample_token = sd['sample_token']
            scene_token = sample_to_scene.get(sample_token)
            if scene_token:
                filename_to_scene[os.path.basename(sd['filename']).replace('.jpg', '_mask.png')] = scene_map.get(scene_token, 'unknown')
    
    transform = get_val_augmentations(img_height, img_width, mean, std)
    
    # Load all mask files and group by scene
    scene_results = {}
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_mask.png')]
    
    dataset = NuScenesDrivableDataset(dataroot, mask_dir, transform=transform)
    
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, mask = dataset[idx]
            image = image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            
            pred = model(image)
            miou, iou_drv, _ = compute_miou(pred, mask)
            
            # Try to map to scene
            img_path = dataset.samples[idx][0]
            mask_fn = os.path.basename(dataset.samples[idx][1])
            scene_name = filename_to_scene.get(mask_fn, 'unknown')
            
            if scene_name not in scene_results:
                scene_results[scene_name] = []
            scene_results[scene_name].append(miou)
    
    # Average per scene
    scene_avg = {}
    for name, ious in scene_results.items():
        scene_avg[name] = np.mean(ious)
    
    return scene_avg


def generate_visualizations(model, dataroot, mask_dir, device, output_dir,
                            n_vis=10, img_height=256, img_width=448,
                            mean=None, std=None):
    """Generate overlay visualizations of model predictions.
    
    Args:
        model: Trained model.
        dataroot: Dataset root.
        mask_dir: Mask directory.
        device: Device.
        output_dir: Output directory for visualizations.
        n_vis: Number of visualizations to generate.
        img_height: Image height.
        img_width: Image width.
        mean: Normalization mean.
        std: Normalization std.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    transform = get_val_augmentations(img_height, img_width, mean, std)
    dataset = NuScenesDrivableDataset(dataroot, mask_dir, transform=transform)
    
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    model.eval()
    
    # Select evenly spaced samples
    indices = np.linspace(0, len(dataset) - 1, min(n_vis, len(dataset)), dtype=int)
    
    for i, idx in enumerate(indices):
        image_tensor, mask_tensor = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(device))
        
        pred_np = pred.cpu().numpy()[0, 0]
        mask_np = mask_tensor.numpy()[0]
        
        # Denormalize image for visualization
        image_np = image_tensor.numpy().transpose(1, 2, 0)
        image_np = image_np * np.array(std) + np.array(mean)
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        
        # Create visualization
        pred_binary = (pred_np > 0.5).astype(np.float32)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_np, cmap='gray')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(pred_binary, cmap='gray')
        axes[1, 0].set_title(f'Prediction (mIoU: {compute_miou(pred.cpu(), mask_tensor.unsqueeze(0))[0]:.3f})')
        axes[1, 0].axis('off')
        
        overlay = create_overlay(image_np, pred_binary)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (green=drivable)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'eval_sample_{i:02d}.png'), dpi=150)
        plt.close()
    
    print(f"Saved {len(indices)} visualizations to {vis_dir}")


def plot_confusion_matrix(cm, output_dir):
    """Plot and save confusion matrix.
    
    Args:
        cm: 2×2 confusion matrix as list of lists.
        output_dir: Directory to save plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    cm = np.array(cm)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Non-Drivable', 'Drivable']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='Ground Truth',
           title='Confusion Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], ',d'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()


def main():
    """Main evaluation pipeline."""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"LiteSeg Evaluation")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Weights: {args.weights}")
    
    # Load model
    model, checkpoint = load_model(args.weights, device, is_teacher=args.teacher)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Get normalization stats from checkpoint
    data_stats = checkpoint.get('data_stats', {})
    mean = data_stats.get('mean', [0.485, 0.456, 0.406])
    std = data_stats.get('std', [0.229, 0.224, 0.225])
    
    # Create validation dataloader
    _, val_loader, _ = get_dataloaders(
        dataroot=args.dataroot,
        mask_dir=args.mask_dir,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
        num_workers=0,
        mean=mean, std=std
    )
    
    # Full evaluation
    print("\nRunning evaluation...")
    results = evaluate_full(
        model, val_loader, device,
        use_tta=args.use_tta,
        use_boundary_ref=args.use_boundary_refinement
    )
    
    # FPS benchmark
    print("\nBenchmarking FPS...")
    fps_stats = measure_fps_detailed(model, device)
    
    # Per-scene evaluation
    print("\nPer-scene evaluation...")
    scene_ious = per_scene_evaluation(
        model, args.dataroot, args.mask_dir, device,
        args.img_height, args.img_width, mean, std
    )
    
    # Sort scenes by mIoU
    sorted_scenes = sorted(scene_ious.items(), key=lambda x: x[1], reverse=True)
    best_3 = sorted_scenes[:3]
    worst_3 = sorted_scenes[-3:]
    
    # Generate visualizations
    print(f"\nGenerating {args.num_visualizations} visualizations...")
    generate_visualizations(
        model, args.dataroot, args.mask_dir, device, args.output_dir,
        n_vis=args.num_visualizations,
        img_height=args.img_height, img_width=args.img_width,
        mean=mean, std=std
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], args.output_dir)
    
    # =========================================================================
    # FINAL METRICS TABLE
    # =========================================================================
    metrics = {
        'miou_overall': results['miou'],
        'miou_drivable': results['iou_drivable'],
        'miou_non_drivable': results['iou_non_drivable'],
        'fps': fps_stats['fps'],
        'mean_latency_ms': fps_stats['mean_latency_ms'],
        'parameters': n_params,
        'confusion_matrix': results['confusion_matrix'],
        'per_scene_miou': scene_ious,
        'best_3_scenes': best_3,
        'worst_3_scenes': worst_3,
        'tta_enabled': args.use_tta,
        'boundary_refinement': args.use_boundary_refinement,
    }
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print final report
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Metric':<30} {'Value':>15} {'Target':>15}")
    print(f"{'─'*60}")
    print(f"{'mIoU (overall binary)':<30} {results['miou']:>15.4f} {'> 0.75':>15}")
    print(f"{'mIoU (drivable class)':<30} {results['iou_drivable']:>15.4f} {'> 0.72':>15}")
    print(f"{'mIoU (non-drivable)':<30} {results['iou_non_drivable']:>15.4f} {'—':>15}")
    print(f"{'Inference FPS (' + str(device) + ')':<30} {fps_stats['fps']:>15.1f} {'> 30 (CPU)':>15}")
    print(f"{'Mean Latency (ms)':<30} {fps_stats['mean_latency_ms']:>15.2f} {'—':>15}")
    print(f"{'Model Parameters':<30} {n_params:>15,} {'< 3M':>15}")
    print(f"{'TTA Enabled':<30} {str(args.use_tta):>15}")
    print(f"{'Boundary Refinement':<30} {str(args.use_boundary_refinement):>15}")
    
    print(f"\n{'─'*60}")
    print(f"Per-Scene mIoU Breakdown:")
    print(f"{'─'*60}")
    print(f"\nBest 3 scenes:")
    for name, iou in best_3:
        print(f"  {name}: {iou:.4f}")
    print(f"\nWorst 3 scenes:")
    for name, iou in worst_3:
        print(f"  {name}: {iou:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    print(f"  {'':>20} {'Pred Non-Drv':>15} {'Pred Drv':>15}")
    print(f"  {'GT Non-Drivable':>20} {cm[0,0]:>15,} {cm[0,1]:>15,}")
    print(f"  {'GT Drivable':>20} {cm[1,0]:>15,} {cm[1,1]:>15,}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
