"""
generate_masks.py — Generate drivable area masks from nuScenes bitmap maps.

Uses the raw semantic_prior bitmap PNGs (binary: 0=non-drivable, 255=drivable)
with auto-calibrated coordinate mapping. The calibration works by finding the
pixel offset that places ego vehicle positions on drivable (white) pixels —
since the vehicle drives on roads, this is guaranteed to be correct.

Pipeline:
1. Load raw bitmap maps (one per location)
2. Auto-calibrate pixel↔world mapping using ego poses as ground truth
3. For each CAM_FRONT keyframe:
   a. Create dense ground-plane grid around the ego vehicle
   b. Look up each grid point in the calibrated bitmap
   c. Project drivable points to camera image
   d. Fill gaps with morphological closing
4. Save masks as PNGs

Usage:
    python generate_masks.py --dataroot ./ --output_dir masks --visualize 10
"""

import os

# Allow loading very large bitmaps — must be set BEFORE importing cv2
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)

import json
import argparse
import numpy as np
import cv2
from pyquaternion import Quaternion
from tqdm import tqdm


# =============================================================================
# UTILITIES
# =============================================================================

def load_table(dataroot, table_name):
    """Load a nuScenes metadata table from JSON."""
    with open(os.path.join(dataroot, "v1.0-mini", table_name), "r") as f:
        return json.load(f)


def build_token_map(table):
    """Build token→record lookup dict."""
    return {r['token']: r for r in table}


def get_transform_matrix(translation, rotation):
    """Build 4×4 homogeneous transform from translation + quaternion."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T


# =============================================================================
# BITMAP MAP HANDLER
# =============================================================================

class BitmapMap:
    """Handles a single semantic_prior bitmap map with auto-calibrated coords.
    
    The bitmap is binary (0 or 255). This class finds the correct
    pixel↔world coordinate mapping by testing ego positions against the bitmap.
    """
    
    RESOLUTION = 0.1  # meters per pixel (standard nuScenes)
    
    def __init__(self, bitmap_path, ego_positions):
        """Load bitmap and calibrate coordinate mapping.
        
        Args:
            bitmap_path: Path to the semantic_prior PNG file.
            ego_positions: List of (x, y) tuples of ego vehicle positions
                          that MUST be on drivable (white) pixels.
        """
        self.bitmap = cv2.imread(bitmap_path, cv2.IMREAD_GRAYSCALE)
        if self.bitmap is None:
            raise RuntimeError(f"Failed to load bitmap: {bitmap_path}")
        
        self.height, self.width = self.bitmap.shape
        self.drivable = (self.bitmap >= 200).astype(np.uint8)
        
        print(f"  Bitmap shape: {self.bitmap.shape}")
        print(f"  Drivable pixels: {self.drivable.sum()} ({self.drivable.sum()/self.drivable.size*100:.1f}%)")
        
        # Auto-calibrate
        self.origin_x = 0.0  # world_x that maps to pixel col 0
        self.origin_y = 0.0  # world_y that maps to pixel row 0
        self._calibrate(ego_positions)
    
    def _calibrate(self, ego_positions):
        """Find the origin offset that maximizes ego positions on drivable pixels.
        
        Since the ego vehicle drives on roads, the correct mapping will place
        most/all ego positions on drivable (white) pixels in the bitmap.
        
        Strategy: Use resolution=0.1, search for (origin_x, origin_y) offsets.
        """
        if not ego_positions:
            print("  ⚠ No ego positions for calibration!")
            return
        
        ego_xy = np.array(ego_positions)  # (N, 2)
        ego_x_mean = ego_xy[:, 0].mean()
        ego_y_mean = ego_xy[:, 1].mean()
        
        # The ego centroid should map roughly to the center of the bitmap
        # Initial estimate: origin_x = ego_x_mean - (width/2) * resolution
        est_origin_x = ego_x_mean - (self.width / 2) * self.RESOLUTION
        est_origin_y = ego_y_mean - (self.height / 2) * self.RESOLUTION
        
        best_score = -1
        best_ox = est_origin_x
        best_oy = est_origin_y
        best_flip = False
        
        # Search around the initial estimate
        search_range = 500  # meters
        search_step = 5     # meters
        
        offsets_x = np.arange(est_origin_x - search_range, 
                              est_origin_x + search_range, search_step)
        offsets_y = np.arange(est_origin_y - search_range, 
                              est_origin_y + search_range, search_step)
        
        for flip_y in [False, True]:
            for ox in offsets_x:
                for oy in offsets_y:
                    # Map ego positions to pixels
                    cols = ((ego_xy[:, 0] - ox) / self.RESOLUTION).astype(int)
                    
                    if flip_y:
                        # Y increases upward in world, but row 0 is top of image
                        rows = ((oy + self.height * self.RESOLUTION - ego_xy[:, 1]) 
                                / self.RESOLUTION).astype(int)
                    else:
                        rows = ((ego_xy[:, 1] - oy) / self.RESOLUTION).astype(int)
                    
                    # Check bounds
                    valid = ((cols >= 0) & (cols < self.width) & 
                             (rows >= 0) & (rows < self.height))
                    
                    if valid.sum() < len(ego_xy) * 0.8:
                        continue  # Too many out of bounds
                    
                    # Score: how many ego positions are on drivable pixels
                    score = 0
                    for i in range(len(ego_xy)):
                        if valid[i]:
                            if self.drivable[rows[i], cols[i]] > 0:
                                score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_ox = ox
                        best_oy = oy
                        best_flip = flip_y
        
        self.origin_x = best_ox
        self.origin_y = best_oy
        self.flip_y = best_flip
        
        hit_rate = best_score / len(ego_xy)
        print(f"  Calibration: origin=({best_ox:.1f}, {best_oy:.1f}), "
              f"flip_y={best_flip}, hit_rate={hit_rate:.2%} ({best_score}/{len(ego_xy)})")
        
        if hit_rate < 0.5:
            print(f"  ⚠ WARNING: Low hit rate — calibration may be wrong!")
    
    def is_drivable(self, world_x, world_y):
        """Check if world coordinates are on drivable area.
        
        Args:
            world_x: Array of world X coordinates.
            world_y: Array of world Y coordinates.
            
        Returns:
            Boolean array: True where drivable.
        """
        cols = ((world_x - self.origin_x) / self.RESOLUTION).astype(int)
        
        if self.flip_y:
            rows = ((self.origin_y + self.height * self.RESOLUTION - world_y) 
                    / self.RESOLUTION).astype(int)
        else:
            rows = ((world_y - self.origin_y) / self.RESOLUTION).astype(int)
        
        # Bounds check
        valid = ((cols >= 0) & (cols < self.width) & 
                 (rows >= 0) & (rows < self.height))
        
        result = np.zeros(len(world_x), dtype=bool)
        valid_idx = np.where(valid)[0]
        
        if len(valid_idx) > 0:
            result[valid_idx] = self.drivable[rows[valid_idx], cols[valid_idx]] > 0
        
        return result


# =============================================================================
# MASK GENERATOR
# =============================================================================

class NuScenesMaskGenerator:
    """Generates drivable area masks using raw bitmap maps.
    
    Auto-calibrates coordinate mapping per location using ego poses,
    then projects drivable ground-plane points to camera view.
    """
    
    # Grid parameters
    GRID_FORWARD = 50.0    # meters ahead of ego
    GRID_BEHIND = 15.0     # meters behind ego
    GRID_SIDE = 25.0       # meters left/right
    GRID_RES = 0.05         # meters between grid points
    
    def __init__(self, dataroot):
        """Initialize with nuScenes data."""
        self.dataroot = dataroot
        
        # Load metadata
        self.scenes = load_table(dataroot, "scene.json")
        self.samples = load_table(dataroot, "sample.json")
        self.sample_data_list = load_table(dataroot, "sample_data.json")
        self.sensors = load_table(dataroot, "sensor.json")
        self.calibrated_sensors = load_table(dataroot, "calibrated_sensor.json")
        self.ego_poses = load_table(dataroot, "ego_pose.json")
        self.maps_table = load_table(dataroot, "map.json")
        self.logs = load_table(dataroot, "log.json")
        
        # Build lookups
        self.calib_sensor_map = build_token_map(self.calibrated_sensors)
        self.ego_pose_map = build_token_map(self.ego_poses)
        self.scene_token_map = build_token_map(self.scenes)
        self.log_map = build_token_map(self.logs)
        self.sample_map = build_token_map(self.samples)
        self.sd_map = build_token_map(self.sample_data_list)
        
        # Find CAM_FRONT sensor
        self.cam_front_token = None
        for s in self.sensors:
            if s['channel'] == 'CAM_FRONT':
                self.cam_front_token = s['token']
                break
        
        self.cam_front_calib_tokens = set()
        for cs in self.calibrated_sensors:
            if cs['sensor_token'] == self.cam_front_token:
                self.cam_front_calib_tokens.add(cs['token'])
        
        # Scene → location → map
        self.scene_to_location = {}
        self.location_to_map_record = {}
        
        for scene in self.scenes:
            log = self.log_map.get(scene.get('log_token', ''))
            if log:
                self.scene_to_location[scene['token']] = log['location']
        
        for m in self.maps_table:
            for lt in m.get('log_tokens', []):
                if lt in self.log_map:
                    loc = self.log_map[lt]['location']
                    self.location_to_map_record[loc] = m
        
        # CAM_FRONT keyframes
        self.cam_front_samples = [
            sd for sd in self.sample_data_list
            if sd['calibrated_sensor_token'] in self.cam_front_calib_tokens
            and sd['is_key_frame']
        ]
        
        # Collect ego positions per location for calibration
        location_ego_positions = {}
        for sd in self.cam_front_samples:
            sample = self.sample_map.get(sd.get('sample_token', ''))
            if sample is None:
                continue
            scene = self.scene_token_map.get(sample['scene_token'])
            if scene is None:
                continue
            loc = self.scene_to_location.get(scene['token'])
            if loc is None:
                continue
            ego = self.ego_pose_map.get(sd['ego_pose_token'])
            if ego:
                location_ego_positions.setdefault(loc, []).append(
                    (ego['translation'][0], ego['translation'][1])
                )
        
        # Load and calibrate bitmap maps
        self.bitmap_maps = {}
        print(f"\nLoading bitmap maps...")
        for loc, map_record in self.location_to_map_record.items():
            map_path = os.path.join(dataroot, map_record['filename'])
            if not os.path.exists(map_path):
                print(f"  ✗ Map file not found: {map_path}")
                continue
            
            ego_pos = location_ego_positions.get(loc, [])
            print(f"\n  Location: {loc} ({len(ego_pos)} ego positions)")
            
            try:
                bm = BitmapMap(map_path, ego_pos)
                self.bitmap_maps[loc] = bm
                print(f"  ✓ Loaded and calibrated")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        print(f"\nLoaded {len(self.cam_front_samples)} CAM_FRONT keyframes")
        print(f"Calibrated {len(self.bitmap_maps)} bitmap maps")
    
    def generate_mask_for_sample_data(self, sd_record, img_h=900, img_w=1600):
        """Generate drivable mask for one CAM_FRONT keyframe.
        
        1. Get ego pose and camera calibration
        2. Create ground-plane grid around ego
        3. Check bitmap for drivable status
        4. Project drivable points to camera
        5. Fill and clean mask
        """
        ego_pose = self.ego_pose_map.get(sd_record['ego_pose_token'])
        calib = self.calib_sensor_map.get(sd_record['calibrated_sensor_token'])
        if ego_pose is None or calib is None:
            return None
        
        sample = self.sample_map.get(sd_record.get('sample_token', ''))
        if sample is None:
            return None
        scene_token = sample['scene_token']
        location = self.scene_to_location.get(scene_token)
        
        bm = self.bitmap_maps.get(location)
        if bm is None:
            return None
        
        # Camera transform
        ego_T = get_transform_matrix(ego_pose['translation'], ego_pose['rotation'])
        cam_T = get_transform_matrix(calib['translation'], calib['rotation'])
        K = np.array(calib['camera_intrinsic'])
        
        world_to_cam = np.linalg.inv(cam_T) @ np.linalg.inv(ego_T)
        
        # Create ground-plane grid around ego
        ego_x, ego_y = ego_pose['translation'][0], ego_pose['translation'][1]
        ego_rot = Quaternion(ego_pose['rotation'])
        yaw = ego_rot.yaw_pitch_roll[0]
        
        fwd = np.arange(-self.GRID_BEHIND, self.GRID_FORWARD, self.GRID_RES)
        lat = np.arange(-self.GRID_SIDE, self.GRID_SIDE, self.GRID_RES)
        ff, ll = np.meshgrid(fwd, lat)
        ff, ll = ff.ravel(), ll.ravel()
        
        # Rotate to world frame
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        wx = ego_x + ff * cos_y - ll * sin_y
        wy = ego_y + ff * sin_y + ll * cos_y
        
        # Check drivable status in bitmap
        drivable = bm.is_drivable(wx, wy)
        
        if drivable.sum() == 0:
            return np.zeros((img_h, img_w), dtype=np.float32)
        
        # Build 3D world points on ground (z=0)
        n = len(wx)
        world_pts = np.stack([wx, wy, np.zeros(n), np.ones(n)], axis=0)  # (4, N)
        
        # Project to camera
        cam_pts = world_to_cam @ world_pts  # (4, N)
        z = cam_pts[2, :]
        
        # Keep points in front of camera AND drivable
        valid = (z > 0.5) & drivable
        
        if valid.sum() == 0:
            return np.zeros((img_h, img_w), dtype=np.float32)
        
        # Perspective projection
        cam_3 = cam_pts[:3, valid].copy()
        cam_3[0] /= cam_3[2]
        cam_3[1] /= cam_3[2]
        cam_3[2] = 1.0
        
        px = K @ cam_3
        u = px[0].astype(int)
        v = px[1].astype(int)
        
        in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        
        # Build mask
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask[v[in_img], u[in_img]] = 255
        
        # Near-field (bottom 40%): aggressive closing for dense projected points
        kernel_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        near_field = mask[int(img_h * 0.6):, :]
        near_field = cv2.morphologyEx(near_field, cv2.MORPH_CLOSE, kernel_near)
        mask[int(img_h * 0.6):, :] = near_field

        # Far-field: standard closing
        kernel_far = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_far)

        # Light opening to remove salt noise only
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # Sky cutoff: top 25% can't be road
        mask[:int(img_h * 0.25), :] = 0

        seed_x = img_w // 2
        seed_y = img_h - 5
        if mask[seed_y, seed_x] == 255:
            flood = mask.copy()
            cv2.floodFill(flood, None, (seed_x, seed_y), 128)
            mask = ((flood == 255) | (flood == 128)).astype(np.uint8) * 255
        
        return (mask > 127).astype(np.float32)
    
    def generate_all_masks(self, output_dir, vis_count=10):
        """Generate masks for all CAM_FRONT keyframes."""
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        stats = {'total': 0, 'ok': 0, 'fail': 0, 
                 'no_img': 0, 'ratios': [], 'files': []}
        vis_done = 0
        
        for sd in tqdm(self.cam_front_samples, desc="Generating masks"):
            stats['total'] += 1
            
            try:
                mask = self.generate_mask_for_sample_data(sd)
                if mask is None:
                    stats['fail'] += 1
                    continue
                
                img_path = os.path.join(self.dataroot, sd['filename'])
                if not os.path.exists(img_path):
                    stats['no_img'] += 1
                    continue
                
                fn = os.path.basename(sd['filename'])
                mask_fn = fn.replace('.jpg', '_mask.png')
                cv2.imwrite(os.path.join(output_dir, mask_fn), 
                           (mask * 255).astype(np.uint8))
                
                stats['ok'] += 1
                stats['files'].append((sd['filename'], mask_fn))
                
                ratio = mask.sum() / mask.size
                stats['ratios'].append(ratio)
                
                if vis_done < vis_count:
                    img = cv2.imread(img_path)
                    if img is not None:
                        self._vis(img, mask,
                                  os.path.join(vis_dir, f"sample_{vis_done:02d}.png"),
                                  fn, ratio)
                        vis_done += 1
                        
            except Exception as e:
                print(f"\nError on {sd.get('filename','?')}: {e}")
                import traceback
                traceback.print_exc()
                stats['fail'] += 1
        
        self._stats(stats)
        return stats
    
    def _vis(self, image, mask, path, title, ratio):
        """Save visualization."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Input: {title[:50]}')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'Mask (ratio: {ratio:.3f})')
        axes[1].axis('off')
        
        overlay = image.copy()
        green = np.zeros_like(image); green[:,:,1] = 255
        d = mask > 0.5
        overlay[d] = cv2.addWeighted(image[d], 0.5, green[d], 0.5, 0)
        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle('Bitmap-Projected Drivable Mask', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _stats(self, s):
        """Print stats."""
        print("\n" + "=" * 60)
        print("MASK GENERATION STATISTICS")
        print("=" * 60)
        print(f"Total:       {s['total']}")
        print(f"Successful:  {s['ok']}")
        print(f"Failed:      {s['fail']}")
        print(f"No image:    {s['no_img']}")
        
        if s['ratios']:
            r = np.array(s['ratios'])
            print(f"\nDrivable ratio:")
            print(f"  Mean:   {r.mean():.4f}")
            print(f"  Std:    {r.std():.4f}")
            print(f"  Min:    {r.min():.4f}")
            print(f"  Max:    {r.max():.4f}")
            print(f"  Median: {np.median(r):.4f}")
            
            if r.mean() > 0.40:
                print("  ⚠ May over-predict")
            elif r.mean() < 0.08:
                print("  ⚠ May under-predict")
            else:
                print("  ✓ Looks reasonable")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./')
    parser.add_argument('--output_dir', default='masks')
    parser.add_argument('--visualize', type=int, default=10)
    args = parser.parse_args()
    
    print("=" * 60)
    print("PRISM — Drivable Mask Generator (Bitmap + Auto-Calibration)")
    print("=" * 60)
    
    if not os.path.exists(os.path.join(args.dataroot, "v1.0-mini")):
        print("ERROR: v1.0-mini/ not found!")
        return
    
    gen = NuScenesMaskGenerator(args.dataroot)
    stats = gen.generate_all_masks(args.output_dir, args.visualize)
    
    with open(os.path.join(args.output_dir, "file_mapping.json"), 'w') as f:
        json.dump(stats.get('files', []), f, indent=2)
    
    print(f"\nDone! {stats['ok']} masks saved to '{args.output_dir}'")


if __name__ == '__main__':
    main()
