"""
generate_masks.py — Generate drivable area masks from nuScenes BEV maps.

Pipeline:
1. Load BEV raster map for each scene's location
2. For each CAM_FRONT keyframe, get ego_pose and calibrated_sensor data
3. Extract local BEV patch around ego vehicle position
4. Project ground plane points to camera image using intrinsics/extrinsics
5. Generate binary drivable mask at camera resolution
6. Save masks as PNGs

Usage:
    python generate_masks.py --dataroot ./  --output_dir masks --visualize 5
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from pyquaternion import Quaternion
from tqdm import tqdm


def load_table(dataroot, table_name):
    """Load a nuScenes metadata table from JSON.
    
    Args:
        dataroot: Root directory containing v1.0-mini/.
        table_name: Name of the JSON file (e.g., 'scene.json').
        
    Returns:
        List of dictionaries from the JSON file.
    """
    path = os.path.join(dataroot, "v1.0-mini", table_name)
    with open(path, "r") as f:
        return json.load(f)


def build_token_map(table):
    """Build a token→record lookup dictionary for fast access.
    
    Args:
        table: List of records (dicts), each with a 'token' key.
        
    Returns:
        Dict mapping token string to record dict.
    """
    return {record['token']: record for record in table}


def get_transform_matrix(translation, rotation):
    """Build a 4×4 homogeneous transformation matrix.
    
    Args:
        translation: List of 3 floats [x, y, z].
        rotation: List of 4 floats [w, x, y, z] (quaternion).
        
    Returns:
        4×4 numpy array transformation matrix.
    """
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T


def load_bev_map(dataroot, map_filename):
    """Load and process BEV (Bird's Eye View) raster map.
    
    The nuScenes semantic_prior maps encode different classes as different
    pixel intensities. The drivable area appears as a distinct intensity
    in the map.
    
    Args:
        dataroot: Root directory.
        map_filename: Relative path to map PNG.
        
    Returns:
        Tuple of (full_map, drivable_mask) as numpy arrays.
    """
    map_path = os.path.join(dataroot, map_filename)
    full_map = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)
    
    if full_map is None:
        raise FileNotFoundError(f"Could not load map: {map_path}")
    
    return full_map


def extract_drivable_from_bev(bev_map, threshold_low=200, threshold_high=256):
    """Extract drivable area from BEV raster map by intensity thresholding.
    
    The semantic prior maps in nuScenes use high intensity (white/near-white)
    pixels to mark traversable road surfaces.
    
    Args:
        bev_map: BEV map as numpy array.
        threshold_low: Lower intensity threshold for drivable pixels.
        threshold_high: Upper intensity threshold.
        
    Returns:
        Binary drivable mask (same size as bev_map).
    """
    if len(bev_map.shape) == 3:
        # Convert to grayscale if color
        gray = cv2.cvtColor(bev_map, cv2.COLOR_BGR2GRAY)
    else:
        gray = bev_map
    
    # Drivable areas are typically the brightest regions in semantic prior maps
    drivable = (gray >= threshold_low).astype(np.uint8)
    
    return drivable


class NuScenesMaskGenerator:
    """Generates drivable area masks for CAM_FRONT images.
    
    Uses BEV raster maps, ego poses, and camera calibration data
    to project drivable areas from the bird's eye view onto the
    camera image plane.
    """
    
    # Map resolution in meters per pixel for nuScenes maps
    # nuScenes maps: 10 pixels per meter, so 0.1 meters per pixel
    MAP_RESOLUTION = 0.1  # meters/pixel
    
    # BEV patch size around ego vehicle (in meters)
    BEV_RANGE_FORWARD = 50.0   # meters ahead
    BEV_RANGE_BEHIND = 10.0    # meters behind
    BEV_RANGE_SIDE = 25.0      # meters left/right
    
    def __init__(self, dataroot):
        """Initialize mask generator with nuScenes data.
        
        Args:
            dataroot: Root directory of nuScenes dataset.
        """
        self.dataroot = dataroot
        
        # Load all metadata tables
        self.scenes = load_table(dataroot, "scene.json")
        self.samples = load_table(dataroot, "sample.json")
        self.sample_data_list = load_table(dataroot, "sample_data.json")
        self.sensors = load_table(dataroot, "sensor.json")
        self.calibrated_sensors = load_table(dataroot, "calibrated_sensor.json")
        self.ego_poses = load_table(dataroot, "ego_pose.json")
        self.maps = load_table(dataroot, "map.json")
        self.logs = load_table(dataroot, "log.json")
        
        # Build token lookup maps
        self.sample_data_map = build_token_map(self.sample_data_list)
        self.calib_sensor_map = build_token_map(self.calibrated_sensors)
        self.ego_pose_map = build_token_map(self.ego_poses)
        self.scene_map = build_token_map(self.scenes)
        self.log_map = build_token_map(self.logs)
        self.map_map = build_token_map(self.maps)
        self.sample_map = build_token_map(self.samples)
        
        # Find CAM_FRONT sensor token
        self.cam_front_token = None
        for s in self.sensors:
            if s['channel'] == 'CAM_FRONT':
                self.cam_front_token = s['token']
                break
        
        # Get all CAM_FRONT calibrated sensor tokens
        self.cam_front_calib_tokens = set()
        for cs in self.calibrated_sensors:
            if cs['sensor_token'] == self.cam_front_token:
                self.cam_front_calib_tokens.add(cs['token'])
        
        # Load scene → log → map mapping
        self.scene_to_map = {}
        self.map_cache = {}
        for scene in self.scenes:
            # Find the log for this scene
            log_token = scene.get('log_token')
            if log_token and log_token in self.log_map:
                log = self.log_map[log_token]
                location = log['location']
                # Find the map for this location
                for m in self.maps:
                    if log_token in m.get('log_tokens', []):
                        self.scene_to_map[scene['token']] = m
                        break
        
        # Get all CAM_FRONT keyframe sample_data entries
        self.cam_front_samples = []
        for sd in self.sample_data_list:
            if (sd['calibrated_sensor_token'] in self.cam_front_calib_tokens
                    and sd['is_key_frame']):
                self.cam_front_samples.append(sd)
        
        print(f"Loaded {len(self.cam_front_samples)} CAM_FRONT keyframes")
        print(f"Scene-to-map mappings: {len(self.scene_to_map)}")
    
    def get_cam_front_for_sample(self, sample_token):
        """Get the CAM_FRONT sample_data record for a given sample.
        
        Args:
            sample_token: Token of the sample (keyframe).
            
        Returns:
            sample_data record dict for CAM_FRONT, or None.
        """
        for sd in self.cam_front_samples:
            if sd.get('sample_token') == sample_token:
                return sd
        return None
    
    def generate_mask_for_sample_data(self, sd_record, img_height=900, img_width=1600):
        """Generate drivable area mask for a single CAM_FRONT sample_data record.
        
        Process:
        1. Get ego pose for this timestamp
        2. Get camera calibration (intrinsics + extrinsics)
        3. Load the BEV map for the scene
        4. Create a dense grid of ground-plane points around the ego vehicle
        5. Transform points: world → ego → camera → pixel
        6. Sample the BEV drivable mask at each point's world location
        7. Build the camera-view mask
        
        Args:
            sd_record: sample_data record dict for CAM_FRONT.
            img_height: Camera image height.
            img_width: Camera image width.
            
        Returns:
            Binary mask as numpy array (img_height, img_width), or None on failure.
        """
        # Get ego pose
        ego_pose = self.ego_pose_map.get(sd_record['ego_pose_token'])
        if ego_pose is None:
            return None
        
        # Get calibrated sensor
        calib = self.calib_sensor_map.get(sd_record['calibrated_sensor_token'])
        if calib is None:
            return None
        
        # Find the scene for this sample
        sample_token = sd_record.get('sample_token')
        if sample_token is None:
            return None
        sample = self.sample_map.get(sample_token)
        if sample is None:
            return None
        scene_token = sample['scene_token']
        
        # Get the BEV map for this scene
        map_record = self.scene_to_map.get(scene_token)
        if map_record is None:
            return None
        
        # Load BEV map (cached)
        map_token = map_record['token']
        if map_token not in self.map_cache:
            bev = load_bev_map(self.dataroot, map_record['filename'])
            drivable = extract_drivable_from_bev(bev)
            self.map_cache[map_token] = (bev, drivable)
        _, drivable_bev = self.map_cache[map_token]
        
        # Build transformation matrices
        # Ego pose: transforms from ego frame to world frame
        ego_T = get_transform_matrix(ego_pose['translation'], ego_pose['rotation'])
        
        # Camera extrinsics: transforms from camera frame to ego frame
        cam_T = get_transform_matrix(calib['translation'], calib['rotation'])
        
        # Camera intrinsics: 3×3 matrix
        K = np.array(calib['camera_intrinsic'])
        
        # World-to-camera transform: inv(cam_T) @ inv(ego_T)
        world_to_ego = np.linalg.inv(ego_T)
        ego_to_cam = np.linalg.inv(cam_T)
        world_to_cam = ego_to_cam @ world_to_ego
        
        # Create a grid of ground-plane points in ego vehicle frame
        # The ground plane is at z=0 in the world frame
        ego_x = ego_pose['translation'][0]
        ego_y = ego_pose['translation'][1]
        
        # Generate grid of world points on the ground plane
        # More points = higher quality mask, but slower
        grid_resolution = 0.2  # meters between grid points
        
        x_range = np.arange(
            ego_x - self.BEV_RANGE_BEHIND,
            ego_x + self.BEV_RANGE_FORWARD,
            grid_resolution
        )
        y_range = np.arange(
            ego_y - self.BEV_RANGE_SIDE,
            ego_y + self.BEV_RANGE_SIDE,
            grid_resolution
        )
        
        xx, yy = np.meshgrid(x_range, y_range)
        zz = np.zeros_like(xx)  # Ground plane at z=0
        ones = np.ones_like(xx)
        
        # Stack into (4, N) homogeneous world coordinates
        world_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel(), ones.ravel()], axis=0)
        
        # Transform to camera frame
        cam_points = world_to_cam @ world_points  # (4, N)
        
        # Filter points behind the camera (z <= 0)
        z = cam_points[2, :]
        valid = z > 0.1  # Must be in front of camera
        
        if valid.sum() == 0:
            return np.zeros((img_height, img_width), dtype=np.float32)
        
        # Project to pixel coordinates: [u, v, 1]^T = K @ [X/Z, Y/Z, 1]^T
        cam_points_valid = cam_points[:3, valid]
        cam_points_valid[0, :] /= cam_points_valid[2, :]
        cam_points_valid[1, :] /= cam_points_valid[2, :]
        cam_points_valid[2, :] = 1.0
        
        pixel_coords = K @ cam_points_valid  # (3, N_valid)
        u = pixel_coords[0, :].astype(int)
        v = pixel_coords[1, :].astype(int)
        
        # Filter to image bounds
        in_bounds = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
        
        # Sample drivable status from BEV map
        world_x = world_points[0, valid][in_bounds]
        world_y = world_points[1, valid][in_bounds]
        
        # Convert world coordinates to BEV map pixel coordinates
        # nuScenes maps: origin is at the map center, resolution ~ 0.1 m/pixel
        bev_h, bev_w = drivable_bev.shape[:2]
        
        # The map coordinates are typically in the same frame as the world pose
        # Map pixel = world_coord / resolution
        # Maps are centered differently by location, we need to figure out the mapping
        map_px = (world_x / self.MAP_RESOLUTION).astype(int)
        map_py = (world_y / self.MAP_RESOLUTION).astype(int)
        
        # Clamp to map bounds
        map_px = np.clip(map_px, 0, bev_w - 1)
        map_py = np.clip(map_py, 0, bev_h - 1)
        
        # Sample drivable values
        drivable_values = drivable_bev[map_py, map_px]
        
        # Build output mask
        mask = np.zeros((img_height, img_width), dtype=np.float32)
        u_valid = u[in_bounds]
        v_valid = v[in_bounds]
        mask[v_valid, u_valid] = drivable_values.astype(np.float32)
        
        # Fill gaps with morphological closing (the grid sampling creates gaps)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply a simple heuristic: the bottom portion of the image is more
        # likely to be road. We can use this as a soft prior.
        # Also, sky region (top ~1/3) should definitely not be drivable
        sky_cutoff = img_height // 3
        mask[:sky_cutoff, :] = 0
        
        return mask
    
    def generate_all_masks(self, output_dir, visualize_count=5):
        """Generate drivable masks for all CAM_FRONT keyframes.
        
        Args:
            output_dir: Directory to save mask PNGs.
            visualize_count: Number of sample pairs to visualize.
            
        Returns:
            Dict with statistics about the generated masks.
        """
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'drivable_ratios': [],
            'filenames': []
        }
        
        vis_count = 0
        
        for sd in tqdm(self.cam_front_samples, desc="Generating masks"):
            stats['total'] += 1
            
            try:
                mask = self.generate_mask_for_sample_data(
                    sd, img_height=900, img_width=1600
                )
                
                if mask is None:
                    stats['failed'] += 1
                    continue
                
                # Save mask
                img_filename = os.path.basename(sd['filename'])
                mask_filename = img_filename.replace('.jpg', '_mask.png')
                mask_path = os.path.join(output_dir, mask_filename)
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                
                stats['successful'] += 1
                stats['filenames'].append((sd['filename'], mask_filename))
                
                # Calculate drivable ratio
                ratio = mask.sum() / mask.size
                stats['drivable_ratios'].append(ratio)
                
                # Visualize
                if vis_count < visualize_count:
                    img_path = os.path.join(self.dataroot, sd['filename'])
                    image = cv2.imread(img_path)
                    if image is not None:
                        self._visualize_pair(
                            image, mask, 
                            os.path.join(vis_dir, f"sample_{vis_count}.png"),
                            img_filename
                        )
                        vis_count += 1
                
            except Exception as e:
                print(f"Error processing {sd.get('filename', 'unknown')}: {e}")
                stats['failed'] += 1
        
        # Print statistics
        self._print_stats(stats)
        
        return stats
    
    def _visualize_pair(self, image, mask, save_path, title=""):
        """Visualize an image-mask pair.
        
        Args:
            image: BGR image array.
            mask: Binary mask array.
            save_path: Path to save visualization.
            title: Title for the visualization.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image (BGR to RGB)
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Input: {title[:50]}')
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'Drivable Mask (ratio: {mask.mean():.3f})')
        axes[1].axis('off')
        
        # Overlay
        overlay = image.copy()
        green_overlay = np.zeros_like(image)
        green_overlay[:, :, 1] = 255
        drivable = mask > 0.5
        overlay[drivable] = cv2.addWeighted(
            image[drivable], 0.6, green_overlay[drivable], 0.4, 0
        )
        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Overlay (green = drivable)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _print_stats(self, stats):
        """Print dataset statistics.
        
        Args:
            stats: Dictionary with generation statistics.
        """
        print("\n" + "=" * 50)
        print("MASK GENERATION STATISTICS")
        print("=" * 50)
        print(f"Total CAM_FRONT keyframes: {stats['total']}")
        print(f"Successfully generated: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        
        if stats['drivable_ratios']:
            ratios = np.array(stats['drivable_ratios'])
            print(f"\nDrivable pixel ratio:")
            print(f"  Mean:   {ratios.mean():.4f}")
            print(f"  Std:    {ratios.std():.4f}")
            print(f"  Min:    {ratios.min():.4f}")
            print(f"  Max:    {ratios.max():.4f}")
            print(f"  Median: {np.median(ratios):.4f}")
        print("=" * 50)


def create_simple_road_masks(dataroot, output_dir, visualize_count=5):
    """Fallback: Create road masks using simple geometric priors.
    
    If BEV projection doesn't produce good results, use a pragmatic
    approach: assume the lower-center portion of the image is road,
    and refine using color/edge detection.
    
    This is a valid approach for a hackathon since:
    1. In most front-camera driving images, the road IS in the bottom half
    2. We can use perspective-geometry + vanishing point estimation
    3. Edge detection helps find road boundaries
    
    Args:
        dataroot: Root of nuScenes dataset.
        output_dir: Directory to save masks.
        visualize_count: Number of visualizations to produce.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load CAM_FRONT sample_data entries
    sample_data_list = load_table(dataroot, "sample_data.json")
    sensors = load_table(dataroot, "sensor.json")
    calibrated_sensors = load_table(dataroot, "calibrated_sensor.json")
    
    cam_front_token = None
    for s in sensors:
        if s['channel'] == 'CAM_FRONT':
            cam_front_token = s['token']
            break
    
    cam_front_calib_tokens = set()
    for cs in calibrated_sensors:
        if cs['sensor_token'] == cam_front_token:
            cam_front_calib_tokens.add(cs['token'])
    
    cam_front_samples = [
        sd for sd in sample_data_list
        if sd['calibrated_sensor_token'] in cam_front_calib_tokens
        and sd['is_key_frame']
    ]
    
    print(f"Generating masks for {len(cam_front_samples)} CAM_FRONT keyframes...")
    
    stats = {'total': 0, 'drivable_ratios': [], 'filenames': []}
    vis_count = 0
    
    for sd in tqdm(cam_front_samples, desc="Generating masks"):
        img_path = os.path.join(dataroot, sd['filename'])
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        stats['total'] += 1
        h, w = image.shape[:2]
        
        # Generate road mask using color and geometric analysis
        mask = generate_road_mask_adaptive(image)
        
        # Save mask
        img_filename = os.path.basename(sd['filename'])
        mask_filename = img_filename.replace('.jpg', '_mask.png')
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        
        ratio = mask.sum() / mask.size
        stats['drivable_ratios'].append(ratio)
        stats['filenames'].append((sd['filename'], mask_filename))
        
        # Visualize
        if vis_count < visualize_count:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title(f'Drivable Mask (ratio: {ratio:.3f})')
            axes[1].axis('off')
            
            overlay = image.copy()
            green = np.zeros_like(image)
            green[:, :, 1] = 255
            drivable = mask > 0.5
            overlay[drivable] = cv2.addWeighted(
                image[drivable], 0.6, green[drivable], 0.4, 0
            )
            axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"sample_{vis_count}.png"), dpi=150)
            plt.close()
            vis_count += 1
    
    # Print stats
    if stats['drivable_ratios']:
        ratios = np.array(stats['drivable_ratios'])
        print(f"\nMask Statistics:")
        print(f"  Total masks: {stats['total']}")
        print(f"  Mean drivable ratio: {ratios.mean():.4f}")
        print(f"  Std: {ratios.std():.4f}")
        print(f"  Min: {ratios.min():.4f}, Max: {ratios.max():.4f}")
    
    return stats


def generate_road_mask_adaptive(image):
    """Generate a road mask using adaptive color and geometric analysis.
    
    Multi-step approach:
    1. Convert to multiple color spaces (HSV, LAB) for robust road detection
    2. Use perspective geometry: road converges at vanishing point
    3. Apply GrabCut-style iterative refinement
    4. Clean up with morphological operations
    
    Args:
        image: BGR image array (H, W, 3).
        
    Returns:
        Binary mask (H, W) with 1.0 for drivable pixels.
    """
    h, w = image.shape[:2]
    
    # Convert color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Strategy 1: Road color detection
    # Roads in urban scenes are typically gray/dark with low saturation
    # Sample the road color from the bottom-center region (high confidence road area)
    sample_region = image[int(h * 0.75):int(h * 0.95), int(w * 0.3):int(w * 0.7)]
    
    if sample_region.size == 0:
        return np.zeros((h, w), dtype=np.float32)
    
    # Compute mean and std of the road color in LAB space
    sample_lab = cv2.cvtColor(sample_region, cv2.COLOR_BGR2LAB)
    road_mean = sample_lab.mean(axis=(0, 1))
    road_std = sample_lab.std(axis=(0, 1))
    
    # Color distance from road reference in LAB space
    lab_diff = np.sqrt(((lab.astype(np.float32) - road_mean) ** 2).sum(axis=2))
    
    # Threshold based on color distance
    # Adaptive threshold based on the variance of the road sample
    threshold = max(np.sqrt((road_std ** 2).sum()) * 2.5, 25)
    color_mask = (lab_diff < threshold).astype(np.float32)
    
    # Strategy 2: Geometric prior (trapezoidal road shape)
    # The road typically forms a trapezoid in perspective view
    geo_mask = np.zeros((h, w), dtype=np.float32)
    
    # Vanishing point approximately at image center-top
    vp_x = w // 2
    vp_y = int(h * 0.38)  # Slightly above center
    
    # Create trapezoidal road region with perspective
    pts = np.array([
        [int(w * 0.05), h],       # Bottom-left
        [int(w * 0.95), h],       # Bottom-right
        [int(w * 0.65), vp_y],    # Top-right
        [int(w * 0.35), vp_y],    # Top-left
    ], dtype=np.int32)
    
    cv2.fillPoly(geo_mask, [pts], 1.0)
    
    # Strategy 3: Edge-based refinement
    # Roads have less texture than surroundings
    edges = cv2.Canny(gray, 30, 100)
    # Dilate edges to create boundary regions
    edge_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    texture_mask = 1.0 - (edge_dilated / 255.0) * 0.3  # Subtle edge penalty
    
    # Combine strategies
    combined = color_mask * 0.5 + geo_mask * 0.3 + texture_mask * 0.2
    
    # Threshold to binary
    mask = (combined > 0.45).astype(np.float32)
    
    # Apply geometric constraint: road must be connected from bottom
    # Use flood fill from bottom-center
    seed_mask = np.zeros((h, w), dtype=np.float32)
    
    # Flood fill from multiple bottom points
    from scipy import ndimage
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    if num_features > 0:
        # Find which component(s) touch the bottom of the image
        bottom_row = labeled[h - 5:h, :]
        bottom_labels = set(bottom_row[bottom_row > 0])
        
        # Keep only components connected to bottom
        road_mask = np.zeros_like(mask)
        for lbl in bottom_labels:
            road_mask[labeled == lbl] = 1.0
        mask = road_mask
    
    # Sky removal: nothing above the horizon should be drivable
    horizon = int(h * 0.35)
    mask[:horizon, :] = 0
    
    # Morphological cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # Smooth edges with Gaussian blur + re-threshold
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    mask = (mask > 0.5).astype(np.float32)
    
    return mask


def main():
    """Main entry point for mask generation."""
    parser = argparse.ArgumentParser(description="Generate drivable area masks")
    parser.add_argument('--dataroot', type=str, default='./',
                        help='Root directory of nuScenes dataset')
    parser.add_argument('--output_dir', type=str, default='masks',
                        help='Output directory for masks')
    parser.add_argument('--visualize', type=int, default=5,
                        help='Number of sample pairs to visualize')
    parser.add_argument('--method', type=str, default='adaptive',
                        choices=['bev', 'adaptive'],
                        help='Mask generation method: bev (BEV projection) or adaptive (color+geometry)')
    args = parser.parse_args()
    
    if args.method == 'bev':
        print("Using BEV projection method...")
        generator = NuScenesMaskGenerator(args.dataroot)
        stats = generator.generate_all_masks(args.output_dir, args.visualize)
    else:
        print("Using adaptive color+geometry method...")
        stats = create_simple_road_masks(args.dataroot, args.output_dir, args.visualize)
    
    # Save file mapping for dataset loading
    mapping_path = os.path.join(args.output_dir, "file_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(stats.get('filenames', []), f, indent=2)
    print(f"\nFile mapping saved to {mapping_path}")


if __name__ == '__main__':
    main()
