"""
dataset.py — PyTorch Dataset for nuScenes drivable space segmentation.

Loads CAM_FRONT images and their corresponding drivable area masks.
Applies albumentations augmentations for training robustness.

Usage:
    from dataset import get_dataloaders
    train_loader, val_loader = get_dataloaders(dataroot='./', mask_dir='masks')
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =============================================================================
# AUGMENTATION PIPELINES
# =============================================================================

def get_train_augmentations(img_height=256, img_width=448, mean=None, std=None):
    """Get HARDENED training augmentation pipeline.
    
    Designed for from-scratch training without pretrained weights.
    Forces the model to learn textures and geometry, not colors.
    
    Groups:
    1. Spatial: flip, crop, perspective, grid distortion
    2. Lighting edge cases: night sim, overexposure, fog, sun flare
    3. Color breaking: aggressive jitter, CLAHE, channel shuffle
    4. Occlusion/noise: dropout, blur, gaussian noise
    
    Uses A.OneOf blocks to prevent over-stacking (max ~3 augmentations per sample).
    
    Args:
        img_height: Target image height.
        img_width: Target image width.
        mean: Per-channel mean for normalization (list of 3 floats).
        std: Per-channel std for normalization (list of 3 floats).
        
    Returns:
        albumentations.Compose pipeline.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    return A.Compose([
        # Resize to target resolution
        A.Resize(height=img_height, width=img_width),
        
        # ── GROUP 1: SPATIAL (geometry learning) ──
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(
            size=(img_height, img_width),
            scale=(0.75, 1.0),
            ratio=(1.5, 2.0),
            p=0.3
        ),
        A.OneOf([
            A.Perspective(scale=(0.02, 0.06), p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=1.0),
            A.ElasticTransform(alpha=30, sigma=5, p=1.0),
        ], p=0.25),
        
        # ── GROUP 2: LIGHTING EDGE CASES (night/glare simulation) ──
        A.OneOf([
            # Night simulation: aggressive darkening + gamma
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.2), contrast_limit=0.3, p=1.0),
                A.RandomGamma(gamma_limit=(40, 80), p=1.0),
            ]),
            # Overexposure / glare simulation
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(0.2, 0.5), contrast_limit=(-0.2, 0.1), p=1.0),
            ]),
            # Fog simulation
            A.RandomFog(fog_coef_range=(0.1, 0.35), alpha_coef=0.1, p=1.0),
            # Sun flare (glare from headlights / sun)
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                src_radius=120,
                num_flare_circles_range=(3, 6),
                p=1.0
            ),
            # Shadow simulation (cast shadows on road surface)
            A.RandomShadow(
                shadow_roi=(0, 0.3, 1, 1),
                num_shadows_limit=(1, 3),
                shadow_dimension=5,
                p=1.0
            ),
        ], p=0.4),
        
        # ── GROUP 3: COLOR BREAKING (force texture/geometry learning) ──
        A.OneOf([
            # CLAHE — contrast-limited adaptive histogram equalization
            # Simulates different camera ISP pipelines
            A.CLAHE(clip_limit=(2, 6), tile_grid_size=(8, 8), p=1.0),
            # Aggressive color jitter — breaks color memorization
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.15, p=1.0),
            # Hue/Saturation shift
            A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=40, val_shift_limit=30, p=1.0),
            # Random tone curve (simulates different exposures)
            A.RandomToneCurve(scale=0.2, p=1.0),
            # Channel shuffle — completely breaks color dependence
            A.ChannelShuffle(p=1.0),
        ], p=0.5),
        
        # Mild brightness/contrast on top (always useful)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        
        # ── GROUP 4: OCCLUSION & NOISE ──
        # Simulate sensor artifacts / partial occlusion
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(10, 40),
            hole_width_range=(10, 40),
            fill_value=0,
            p=0.3
        ),
        
        # Blur (motion / defocus / rain on lens)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 9)),
            A.MedianBlur(blur_limit=5),
        ], p=0.25),
        
        # Sensor noise
        A.GaussNoise(std_range=(0.02, 0.08), p=0.25),
        
        # ── NORMALIZE & CONVERT ──
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_augmentations(img_height=256, img_width=448, mean=None, std=None):
    """Get validation/test augmentation pipeline (minimal transforms).
    
    Only resize and normalize — no random augmentations.
    
    Args:
        img_height: Target image height.
        img_width: Target image width.
        mean: Per-channel mean for normalization.
        std: Per-channel std for normalization.
        
    Returns:
        albumentations.Compose pipeline.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# =============================================================================
# DATASET CLASS
# =============================================================================

class NuScenesDrivableDataset(Dataset):
    """PyTorch Dataset for nuScenes drivable space segmentation.
    
    Loads CAM_FRONT images and corresponding pre-generated drivable masks.
    Masks are expected to be single-channel PNGs with 255 for drivable pixels.
    
    Attributes:
        dataroot: Root directory of nuScenes dataset.
        mask_dir: Directory containing mask PNGs.
        samples: List of (image_path, mask_path) tuples.
        transform: Albumentations augmentation pipeline.
    """
    
    def __init__(self, dataroot, mask_dir, sample_list=None, transform=None):
        """Initialize the dataset.
        
        Args:
            dataroot: Root directory of nuScenes dataset.
            mask_dir: Directory containing generated mask PNGs.
            sample_list: Optional list of (img_rel_path, mask_filename) tuples.
                If None, auto-discovers from mask_dir.
            transform: Albumentations transform pipeline.
        """
        self.dataroot = dataroot
        self.mask_dir = mask_dir
        self.transform = transform
        
        if sample_list is not None:
            self.samples = []
            for img_rel, mask_fn in sample_list:
                img_path = os.path.join(dataroot, img_rel)
                mask_path = os.path.join(mask_dir, mask_fn)
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))
        else:
            self.samples = self._discover_samples()
        
        print(f"Dataset initialized with {len(self.samples)} samples")
    
    def _discover_samples(self):
        """Auto-discover image-mask pairs from the mask directory.
        
        Looks for mask files ending in '_mask.png' and finds corresponding
        image files in the samples/CAM_FRONT/ directory.
        
        Returns:
            List of (image_path, mask_path) tuples.
        """
        samples = []
        mask_files = [f for f in os.listdir(self.mask_dir) 
                      if f.endswith('_mask.png')]
        
        for mask_fn in mask_files:
            # Derive image filename from mask filename
            img_fn = mask_fn.replace('_mask.png', '.jpg')
            img_path = os.path.join(self.dataroot, 'samples', 'CAM_FRONT', img_fn)
            mask_path = os.path.join(self.mask_dir, mask_fn)
            
            if os.path.exists(img_path):
                samples.append((img_path, mask_path))
        
        return samples
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single (image, mask) pair.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (image_tensor, mask_tensor):
            - image_tensor: (3, H, W) float tensor, normalized.
            - mask_tensor: (1, H, W) float tensor, values 0 or 1.
        """
        img_path, mask_path = self.samples[idx]
        
        # Load image (BGR → RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (single channel, 0 or 255)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # (3, H, W) tensor
            mask = augmented['mask']    # (H, W) tensor
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()
        
        # Add channel dimension to mask: (H, W) → (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask


# =============================================================================
# DATA LOADING
# =============================================================================

def compute_dataset_mean_std(dataroot, mask_dir, img_height=256, img_width=448):
    """Compute per-channel mean and std from the actual dataset images.
    
    This is called once before training to get the real normalization values
    instead of using ImageNet defaults.
    
    Args:
        dataroot: Root directory of nuScenes dataset.
        mask_dir: Directory containing masks.
        img_height: Target height for resize.
        img_width: Target width for resize.
        
    Returns:
        Tuple of (mean_list, std_list) — each a list of 3 floats.
    """
    # Create dataset with no normalization
    simple_transform = A.Compose([
        A.Resize(height=img_height, width=img_width),
        ToTensorV2(),
    ])
    
    dataset = NuScenesDrivableDataset(
        dataroot=dataroot,
        mask_dir=mask_dir,
        transform=simple_transform
    )
    
    if len(dataset) == 0:
        print("WARNING: Empty dataset, using default ImageNet stats")
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_pixels = 0
    
    print("Computing dataset mean and std...")
    for images, _ in loader:
        # images: (B, 3, H, W), values 0-255 from ToTensorV2
        images = images.float() / 255.0
        batch_size = images.size(0)
        images_flat = images.view(batch_size, 3, -1)
        n_pixels += images_flat.size(2) * batch_size
        mean += images_flat.sum(dim=[0, 2])
        std += (images_flat ** 2).sum(dim=[0, 2])
    
    mean /= n_pixels
    std = torch.sqrt(std / n_pixels - mean ** 2)
    
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    print(f"Dataset mean: {mean_list}")
    print(f"Dataset std:  {std_list}")
    
    return mean_list, std_list


def get_scene_split(dataroot, val_scenes=None):
    """Split samples by scene for train/val.
    
    Uses scene-level splits to prevent data leakage: consecutive frames
    from the same scene should not appear in both train and val.
    
    Args:
        dataroot: Root directory of nuScenes dataset.
        val_scenes: List of scene names for validation. Default: last 2 scenes.
        
    Returns:
        Tuple of (train_sample_tokens, val_sample_tokens).
    """
    scenes_path = os.path.join(dataroot, "v1.0-mini", "scene.json")
    samples_path = os.path.join(dataroot, "v1.0-mini", "sample.json")
    
    with open(scenes_path) as f:
        scenes = json.load(f)
    with open(samples_path) as f:
        samples = json.load(f)
    
    if val_scenes is None:
        # Default: use last 2 scenes as validation
        val_scenes = [s['name'] for s in scenes[-2:]]
    
    val_scene_tokens = {s['token'] for s in scenes if s['name'] in val_scenes}
    
    train_tokens = []
    val_tokens = []
    for sample in samples:
        if sample['scene_token'] in val_scene_tokens:
            val_tokens.append(sample['token'])
        else:
            train_tokens.append(sample['token'])
    
    print(f"Scene split — Val scenes: {val_scenes}")
    print(f"  Train samples: {len(train_tokens)}")
    print(f"  Val samples:   {len(val_tokens)}")
    
    return train_tokens, val_tokens


def get_dataloaders(dataroot='./', mask_dir='masks', img_height=256, img_width=448,
                    batch_size=16, num_workers=0, val_scenes=None,
                    mean=None, std=None):
    """Create train and validation DataLoaders.
    
    Args:
        dataroot: Root directory of nuScenes dataset.
        mask_dir: Directory containing generated masks.
        img_height: Target image height.
        img_width: Target image width.
        batch_size: Training batch size.
        num_workers: Number of data loading workers.
        val_scenes: Optional list of scene names for validation.
        mean: Dataset mean for normalization (computed if None).
        std: Dataset std for normalization (computed if None).
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, dataset_stats) where
        dataset_stats is a dict with mean, std, and dataset sizes.
    """
    # Load file mapping
    mapping_path = os.path.join(mask_dir, "file_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path) as f:
            file_mapping = json.load(f)
    else:
        file_mapping = None
    
    # Compute or use provided normalization stats
    if mean is None or std is None:
        mean, std = compute_dataset_mean_std(dataroot, mask_dir, img_height, img_width)
    
    # Get scene-based train/val split
    train_tokens, val_tokens = get_scene_split(dataroot, val_scenes)
    
    # Split file mapping by scene
    # Load sample data to map filenames to scene tokens
    sd_path = os.path.join(dataroot, "v1.0-mini", "sample_data.json")
    sample_path = os.path.join(dataroot, "v1.0-mini", "sample.json")
    
    with open(sd_path) as f:
        sample_data_list = json.load(f)
    with open(sample_path) as f:
        samples = json.load(f)
    
    sample_token_to_scene = {s['token']: s['scene_token'] for s in samples}
    
    # Build mapping from filename → sample_token
    filename_to_sample_token = {}
    for sd in sample_data_list:
        if sd.get('is_key_frame') and 'sample_token' in sd:
            filename_to_sample_token[sd['filename']] = sd['sample_token']
    
    val_token_set = set(val_tokens)
    
    if file_mapping:
        train_list = []
        val_list = []
        for img_rel, mask_fn in file_mapping:
            sample_token = filename_to_sample_token.get(img_rel)
            if sample_token and sample_token in val_token_set:
                val_list.append((img_rel, mask_fn))
            else:
                train_list.append((img_rel, mask_fn))
    else:
        # Fall back to auto-discovery
        train_list = None
        val_list = None
    
    # Create augmentation pipelines
    train_transform = get_train_augmentations(img_height, img_width, mean, std)
    val_transform = get_val_augmentations(img_height, img_width, mean, std)
    
    # Create datasets
    train_dataset = NuScenesDrivableDataset(
        dataroot=dataroot, mask_dir=mask_dir,
        sample_list=train_list, transform=train_transform
    )
    
    val_dataset = NuScenesDrivableDataset(
        dataroot=dataroot, mask_dir=mask_dir,
        sample_list=val_list, transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    dataset_stats = {
        'mean': mean,
        'std': std,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
    }
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Batch size: {batch_size}")
    print(f"  Mean: {mean}")
    print(f"  Std: {std}")
    
    return train_loader, val_loader, dataset_stats


if __name__ == '__main__':
    # Test the dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./')
    parser.add_argument('--mask_dir', default='masks')
    args = parser.parse_args()
    
    train_loader, val_loader, stats = get_dataloaders(
        dataroot=args.dataroot, mask_dir=args.mask_dir,
        batch_size=4, num_workers=0
    )
    
    # Test a batch
    for images, masks in train_loader:
        print(f"Image batch: {images.shape}, dtype={images.dtype}")
        print(f"Mask batch:  {masks.shape}, dtype={masks.dtype}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Mask unique values: {masks.unique().tolist()}")
        break
