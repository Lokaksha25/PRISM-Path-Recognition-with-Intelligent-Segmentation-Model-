"""
inference.py — Standalone inference pipeline for LiteSeg.

Features:
- Single image inference with mask + overlay output
- ONNX export and ONNX Runtime inference
- Dynamic quantization for faster CPU inference
- FPS benchmarking (PyTorch, ONNX, quantized)
- Demo video generation from scene image sequences

Usage:
    python inference.py --image path/to/image.jpg --weights output/best_model.pth
    python inference.py --export_onnx --weights output/best_model.pth
    python inference.py --benchmark --weights output/best_model.pth
    python inference.py --demo_video --weights output/best_model.pth --dataroot ./
"""

import os
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn

from model import LiteSegNet, LiteSegTeacher
from utils import boundary_refinement, test_time_augmentation, create_overlay


def parse_args():
    """Parse inference arguments."""
    parser = argparse.ArgumentParser(description='LiteSeg Inference Pipeline')
    
    # Core
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image')
    parser.add_argument('--weights', type=str, default='output/best_model.pth',
                        help='Path to model weights')
    parser.add_argument('--output_dir', type=str, default='inference_output',
                        help='Output directory')
    
    # Model
    parser.add_argument('--img_height', type=int, default=256)
    parser.add_argument('--img_width', type=int, default=448)
    parser.add_argument('--teacher', action='store_true',
                        help='Use teacher model')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Segmentation threshold')
    
    # Post-processing
    parser.add_argument('--refine', action='store_true',
                        help='Apply boundary refinement')
    parser.add_argument('--tta', action='store_true',
                        help='Apply Test-Time Augmentation')
    
    # Export & benchmark
    parser.add_argument('--export_onnx', action='store_true',
                        help='Export model to ONNX format')
    parser.add_argument('--onnx_path', type=str, default='model.onnx',
                        help='Path for ONNX export')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run FPS benchmarks')
    parser.add_argument('--quantize', action='store_true',
                        help='Apply dynamic quantization and benchmark')
    
    # Demo video
    parser.add_argument('--demo_video', action='store_true',
                        help='Generate demo video from scene sequence')
    parser.add_argument('--dataroot', type=str, default='./',
                        help='nuScenes data root (for demo video)')
    parser.add_argument('--scene_name', type=str, default=None,
                        help='Scene name for demo video (default: first scene)')
    
    return parser.parse_args()


class LiteSegInference:
    """Standalone inference class for LiteSeg model.
    
    Handles preprocessing, model inference, post-processing, and visualization
    in a single clean pipeline.
    """
    
    def __init__(self, weights_path, device=None, img_height=256, img_width=448,
                 is_teacher=False, threshold=0.5):
        """Initialize inference pipeline.
        
        Args:
            weights_path: Path to trained model checkpoint.
            device: Device to run on (auto-detected if None).
            img_height: Input height for the model.
            img_width: Input width for the model.
            is_teacher: Whether to use teacher model.
            threshold: Binary threshold for mask.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_height = img_height
        self.img_width = img_width
        self.threshold = threshold
        
        # Load model
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        if is_teacher:
            self.model = LiteSegTeacher().to(self.device)
        else:
            self.model = LiteSegNet().to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Get normalization stats
        data_stats = checkpoint.get('data_stats', {})
        self.mean = np.array(data_stats.get('mean', [0.485, 0.456, 0.406]))
        self.std = np.array(data_stats.get('std', [0.229, 0.224, 0.225]))
        
        print(f"Model loaded on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def preprocess(self, image):
        """Preprocess a single image for inference.
        
        Args:
            image: BGR image as numpy array (H, W, 3).
            
        Returns:
            Preprocessed tensor (1, 3, H, W).
        """
        # Store original size for later
        self.original_size = (image.shape[0], image.shape[1])
        
        # Resize
        resized = cv2.resize(image, (self.img_width, self.img_height))
        
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] then apply dataset stats
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std
        
        # HWC to CHW, add batch dimension
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, image, refine=False, tta=False):
        """Run full inference pipeline on a single image.
        
        Args:
            image: BGR image as numpy array.
            refine: Apply boundary refinement.
            tta: Apply Test-Time Augmentation.
            
        Returns:
            Tuple of (binary_mask, probability_map) at original resolution.
        """
        # Preprocess
        tensor = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            if tta:
                probs = test_time_augmentation(self.model, tensor, self.device)
            else:
                probs = self.model.predict(tensor)
        
        # Post-process
        prob_map = probs.cpu().numpy()[0, 0]
        
        if refine:
            binary = boundary_refinement((prob_map > self.threshold).astype(np.float32))
        else:
            binary = (prob_map > self.threshold).astype(np.float32)
        
        # Resize back to original resolution
        binary = cv2.resize(binary, (self.original_size[1], self.original_size[0]),
                           interpolation=cv2.INTER_NEAREST)
        prob_map = cv2.resize(prob_map, (self.original_size[1], self.original_size[0]),
                             interpolation=cv2.INTER_LINEAR)
        
        return binary, prob_map
    
    def predict_and_visualize(self, image, refine=False, tta=False):
        """Run inference and create visualization.
        
        Args:
            image: BGR image.
            refine: Apply boundary refinement.
            tta: Apply TTA.
            
        Returns:
            Tuple of (binary_mask, overlay_image, fps).
        """
        # Time the prediction
        start = time.perf_counter()
        binary, prob_map = self.predict(image, refine=refine, tta=tta)
        elapsed = time.perf_counter() - start
        fps = 1.0 / elapsed
        
        # Create overlay
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = create_overlay(rgb_image, binary)
        
        return binary, overlay, fps


def export_to_onnx(weights_path, onnx_path, img_height=256, img_width=448,
                   is_teacher=False):
    """Export PyTorch model to ONNX format.
    
    Args:
        weights_path: Path to PyTorch checkpoint.
        onnx_path: Output path for ONNX model.
        img_height: Input height.
        img_width: Input width.
        is_teacher: Whether to export teacher model.
    """
    device = torch.device('cpu')
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    if is_teacher:
        model = LiteSegTeacher()
    else:
        model = LiteSegNet()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dummy_input = torch.randn(1, 3, img_height, img_width)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"ONNX model exported to {onnx_path}")
    print(f"ONNX model size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")


def benchmark_onnx(onnx_path, img_height=256, img_width=448, n_runs=200):
    """Benchmark ONNX Runtime inference.
    
    Args:
        onnx_path: Path to ONNX model.
        img_height: Input height.
        img_width: Input width.
        n_runs: Number of benchmark runs.
        
    Returns:
        FPS as float.
    """
    import onnxruntime as ort
    
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    dummy = np.random.randn(1, 3, img_height, img_width).astype(np.float32)
    
    # Warmup
    for _ in range(20):
        session.run(None, {input_name: dummy})
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {input_name: dummy})
    elapsed = time.perf_counter() - start
    
    fps = n_runs / elapsed
    latency = elapsed / n_runs * 1000
    
    print(f"ONNX Runtime FPS: {fps:.1f}")
    print(f"ONNX Runtime Latency: {latency:.2f} ms")
    
    return fps


def apply_quantization(weights_path, output_path='quantized_model.pth',
                       is_teacher=False):
    """Apply dynamic quantization to the model.
    
    Args:
        weights_path: Path to original model checkpoint.
        output_path: Path to save quantized model.
        is_teacher: Whether to quantize teacher model.
        
    Returns:
        Tuple of (quantized_model, original_size_mb, quantized_size_mb).
    """
    device = torch.device('cpu')
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    if is_teacher:
        model = LiteSegTeacher()
    else:
        model = LiteSegNet()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dynamic quantization on linear and conv layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    
    # Save quantized model
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'data_stats': checkpoint.get('data_stats', {}),
    }, output_path)
    
    original_size = os.path.getsize(weights_path) / 1024 / 1024
    quantized_size = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")
    
    return quantized_model, original_size, quantized_size


def benchmark_pytorch(model, device, img_height=256, img_width=448, n_runs=200):
    """Benchmark PyTorch inference.
    
    Args:
        model: PyTorch model.
        device: Device.
        img_height: Input height.
        img_width: Input width.
        n_runs: Number of runs.
        
    Returns:
        FPS as float.
    """
    model.eval()
    dummy = torch.randn(1, 3, img_height, img_width).to(device)
    
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy)
    
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
    latency = elapsed / n_runs * 1000
    
    print(f"PyTorch FPS ({device}): {fps:.1f}")
    print(f"PyTorch Latency: {latency:.2f} ms")
    
    return fps


def generate_demo_video(weights_path, dataroot, output_path='demo_video.mp4',
                        scene_name=None, img_height=256, img_width=448,
                        is_teacher=False, fps=5):
    """Generate a demo video with segmentation overlay on a scene sequence.
    
    Args:
        weights_path: Path to model weights.
        dataroot: nuScenes data root.
        output_path: Output video path.
        scene_name: Scene to use (default: first scene).
        img_height: Model input height.
        img_width: Model input width.
        is_teacher: Use teacher model.
        fps: Video frame rate.
    """
    import json
    
    print(f"Generating demo video...")
    
    # Initialize inference
    inferencer = LiteSegInference(
        weights_path, img_height=img_height, img_width=img_width,
        is_teacher=is_teacher
    )
    
    # Load scene data
    with open(os.path.join(dataroot, 'v1.0-mini', 'scene.json')) as f:
        scenes = json.load(f)
    with open(os.path.join(dataroot, 'v1.0-mini', 'sample.json')) as f:
        samples = json.load(f)
    with open(os.path.join(dataroot, 'v1.0-mini', 'sample_data.json')) as f:
        sample_data_list = json.load(f)
    with open(os.path.join(dataroot, 'v1.0-mini', 'sensor.json')) as f:
        sensors = json.load(f)
    with open(os.path.join(dataroot, 'v1.0-mini', 'calibrated_sensor.json')) as f:
        calib_sensors = json.load(f)
    
    # Find scene
    if scene_name:
        scene = next((s for s in scenes if s['name'] == scene_name), scenes[0])
    else:
        scene = scenes[0]
    
    print(f"Using scene: {scene['name']}: {scene['description']}")
    
    # Get CAM_FRONT sensor
    cam_front_token = next(s['token'] for s in sensors if s['channel'] == 'CAM_FRONT')
    cam_front_calib = {cs['token'] for cs in calib_sensors if cs['sensor_token'] == cam_front_token}
    
    # Get ordered samples for this scene
    sample_map = {s['token']: s for s in samples}
    sd_map = {}
    for sd in sample_data_list:
        if sd['calibrated_sensor_token'] in cam_front_calib and sd.get('is_key_frame'):
            if sd.get('sample_token'):
                sd_map[sd['sample_token']] = sd
    
    # Get ordered sample tokens for the scene
    ordered_samples = []
    current_token = scene['first_sample_token']
    while current_token:
        sample = sample_map.get(current_token)
        if sample is None:
            break
        ordered_samples.append(current_token)
        current_token = sample.get('next', '')
        if not current_token:
            break
    
    # Get image paths for scene
    image_paths = []
    for st in ordered_samples:
        sd = sd_map.get(st)
        if sd:
            image_paths.append(os.path.join(dataroot, sd['filename']))
    
    if not image_paths:
        print("No images found for scene")
        return
    
    print(f"Processing {len(image_paths)} frames...")
    
    # Process frames
    frames = []
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        binary, overlay, frame_fps = inferencer.predict_and_visualize(image, refine=True)
        
        # Convert overlay RGB to BGR for video
        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Add FPS text
        cv2.putText(overlay_bgr, f'FPS: {frame_fps:.1f}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(overlay_bgr, f'Scene: {scene["name"]}', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Resize for video
        frame = cv2.resize(overlay_bgr, (1280, 720))
        frames.append(frame)
    
    # Write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Demo video saved to {output_path} ({len(frames)} frames)")


def main():
    """Main inference entry point."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Single image inference
    if args.image:
        print(f"\n{'='*60}")
        print(f"LiteSeg Inference")
        print(f"{'='*60}")
        
        image = cv2.imread(args.image)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {args.image}")
        
        print(f"Input: {args.image} ({image.shape[1]}×{image.shape[0]})")
        
        inferencer = LiteSegInference(
            args.weights, img_height=args.img_height, img_width=args.img_width,
            is_teacher=args.teacher, threshold=args.threshold
        )
        
        binary, overlay, fps = inferencer.predict_and_visualize(
            image, refine=args.refine, tta=args.tta
        )
        
        # Save outputs
        mask_path = os.path.join(args.output_dir, 'mask.png')
        overlay_path = os.path.join(args.output_dir, 'overlay.png')
        
        cv2.imwrite(mask_path, (binary * 255).astype(np.uint8))
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        drivable_ratio = binary.sum() / binary.size
        
        print(f"\nResults:")
        print(f"  FPS: {fps:.1f}")
        print(f"  Drivable ratio: {drivable_ratio:.3f}")
        print(f"  Mask saved to: {mask_path}")
        print(f"  Overlay saved to: {overlay_path}")
    
    # ONNX export
    if args.export_onnx:
        print(f"\n{'='*60}")
        print(f"ONNX Export")
        print(f"{'='*60}")
        export_to_onnx(
            args.weights, args.onnx_path,
            args.img_height, args.img_width, args.teacher
        )
    
    # Benchmarks
    if args.benchmark:
        print(f"\n{'='*60}")
        print(f"FPS Benchmarks")
        print(f"{'='*60}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
        if args.teacher:
            model = LiteSegTeacher()
        else:
            model = LiteSegNet()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # PyTorch benchmark
        print("\n1. PyTorch Inference:")
        pytorch_fps = benchmark_pytorch(model, device, args.img_height, args.img_width)
        
        # ONNX benchmark
        if os.path.exists(args.onnx_path):
            print("\n2. ONNX Runtime Inference:")
            onnx_fps = benchmark_onnx(args.onnx_path, args.img_height, args.img_width)
        else:
            print("\n2. ONNX model not found — exporting first...")
            export_to_onnx(args.weights, args.onnx_path, args.img_height, args.img_width, args.teacher)
            onnx_fps = benchmark_onnx(args.onnx_path, args.img_height, args.img_width)
        
        # Quantized benchmark
        if args.quantize:
            print("\n3. Quantized PyTorch:")
            quantized_model, orig_size, quant_size = apply_quantization(args.weights, is_teacher=args.teacher)
            quant_fps = benchmark_pytorch(quantized_model, torch.device('cpu'), args.img_height, args.img_width)
        
        print(f"\n{'─'*40}")
        print(f"Summary:")
        print(f"  PyTorch ({device}): {pytorch_fps:.1f} FPS")
        print(f"  ONNX Runtime:  {onnx_fps:.1f} FPS")
        if args.quantize:
            print(f"  Quantized CPU:  {quant_fps:.1f} FPS")
    
    # Demo video
    if args.demo_video:
        print(f"\n{'='*60}")
        print(f"Demo Video Generation")
        print(f"{'='*60}")
        
        video_path = os.path.join(args.output_dir, 'demo_video.mp4')
        generate_demo_video(
            args.weights, args.dataroot, video_path,
            scene_name=args.scene_name,
            img_height=args.img_height, img_width=args.img_width,
            is_teacher=args.teacher
        )


if __name__ == '__main__':
    main()
