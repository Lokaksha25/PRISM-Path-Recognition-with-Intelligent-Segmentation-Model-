"""
model.py — LiteSeg: Lightweight Segmentation Network for Drivable Space.

Custom architecture built entirely from scratch:
- Encoder: MobileNetV2-style inverted residual blocks (depthwise separable convolutions)
- Decoder: Lightweight ASPP (dilation rates 6, 12, 18) + U-Net skip connections
- Output: Binary segmentation head (sigmoid → threshold 0.5)

Target: <3M parameters, >30 FPS on CPU, >100 FPS on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class ConvBNReLU(nn.Module):
    """Standard Convolution → BatchNorm → ReLU6 block.
    
    ReLU6 clips activations at 6, which is more robust for
    quantization and mobile deployment.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1):
        """Initialize ConvBNReLU block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Convolution padding.
            dilation: Convolution dilation rate.
            groups: Number of groups for grouped convolution.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        """Forward pass."""
        return self.relu(self.bn(self.conv(x)))


class InvertedResidualBlock(nn.Module):
    """MobileNetV2 Inverted Residual Block.
    
    Architecture:
        Input (narrow) → 1×1 Expand → 3×3 Depthwise → 1×1 Project (narrow)
        
    The 'inverted' aspect: expands channels first, then compresses.
    Uses residual connection only when input/output channels match and stride=1.
    
    Depthwise separable convolution = depthwise conv + pointwise conv
    This reduces computation by ~k² factor compared to standard convolution
    (where k is kernel size).
    """
    
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        """Initialize InvertedResidualBlock.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for the depthwise convolution (1 or 2).
            expand_ratio: Expansion factor for the hidden dimension.
        """
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # 1×1 pointwise expansion (skip if expand_ratio == 1)
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1, padding=0))
        
        # 3×3 depthwise convolution (groups=hidden_dim makes it depthwise)
        layers.append(ConvBNReLU(
            hidden_dim, hidden_dim, kernel_size=3, stride=stride,
            padding=1, groups=hidden_dim
        ))
        
        # 1×1 pointwise projection (linear — no ReLU to preserve info)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass with optional residual connection."""
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


# =============================================================================
# ENCODER
# =============================================================================

class MobileNetV2Encoder(nn.Module):
    """Custom MobileNetV2-style encoder built from scratch.
    
    Progressively downsamples the input through inverted residual blocks,
    producing feature maps at 1/2, 1/4, 1/8, and 1/16 resolution.
    
    Channel widths are reduced compared to standard MobileNetV2
    to keep total model under 3M parameters.
    
    Skip connections are extracted at 1/4, 1/8, and 1/16 for the decoder.
    """
    
    def __init__(self, in_channels=3):
        """Initialize encoder.
        
        Args:
            in_channels: Number of input image channels (3 for RGB).
        """
        super().__init__()
        
        # Initial convolution: 3 → 32 channels, stride 2 (1/2 resolution)
        self.stem = ConvBNReLU(in_channels, 32, kernel_size=3, stride=2, padding=1)
        
        # Stage 1: 32 → 16, stride 1 (still 1/2 resolution)
        self.stage1 = InvertedResidualBlock(32, 16, stride=1, expand_ratio=1)
        
        # Stage 2: 16 → 24, stride 2 (1/4 resolution) — SKIP CONNECTION
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(16, 24, stride=2, expand_ratio=6),
            InvertedResidualBlock(24, 24, stride=1, expand_ratio=6),
        )
        
        # Stage 3: 24 → 32, stride 2 (1/8 resolution) — SKIP CONNECTION
        self.stage3 = nn.Sequential(
            InvertedResidualBlock(24, 32, stride=2, expand_ratio=6),
            InvertedResidualBlock(32, 32, stride=1, expand_ratio=6),
            InvertedResidualBlock(32, 32, stride=1, expand_ratio=6),
        )
        
        # Stage 4: 32 → 64, stride 2 (1/16 resolution)
        self.stage4 = nn.Sequential(
            InvertedResidualBlock(32, 64, stride=2, expand_ratio=6),
            InvertedResidualBlock(64, 64, stride=1, expand_ratio=6),
            InvertedResidualBlock(64, 64, stride=1, expand_ratio=6),
            InvertedResidualBlock(64, 64, stride=1, expand_ratio=6),
        )
        
        # Stage 5: 64 → 96, stride 1 (still 1/16 resolution) — feeds into ASPP
        self.stage5 = nn.Sequential(
            InvertedResidualBlock(64, 96, stride=1, expand_ratio=6),
            InvertedResidualBlock(96, 96, stride=1, expand_ratio=6),
            InvertedResidualBlock(96, 96, stride=1, expand_ratio=6),
        )
        
        # Stage 6: 96 → 160, stride 1 (still 1/16 resolution) — final features
        self.stage6 = nn.Sequential(
            InvertedResidualBlock(96, 160, stride=1, expand_ratio=6),
            InvertedResidualBlock(160, 160, stride=1, expand_ratio=6),
        )
    
    def forward(self, x):
        """Forward pass returning multi-scale features for skip connections.
        
        Args:
            x: Input tensor (N, 3, H, W).
            
        Returns:
            Tuple of (skip_4x, skip_8x, features_16x) tensors:
            - skip_4x: Features at 1/4 resolution (24 channels)
            - skip_8x: Features at 1/8 resolution (32 channels)
            - features_16x: Features at 1/16 resolution (160 channels)
        """
        x = self.stem(x)      # 1/2
        x = self.stage1(x)    # 1/2
        
        x = self.stage2(x)    # 1/4
        skip_4x = x           # Save for decoder
        
        x = self.stage3(x)    # 1/8
        skip_8x = x           # Save for decoder
        
        x = self.stage4(x)    # 1/16
        x = self.stage5(x)    # 1/16
        x = self.stage6(x)    # 1/16
        
        return skip_4x, skip_8x, x


# =============================================================================
# ASPP (Atrous Spatial Pyramid Pooling)
# =============================================================================

class LightweightASPP(nn.Module):
    """Lightweight Atrous Spatial Pyramid Pooling module.
    
    Captures multi-scale context using parallel dilated convolutions
    with different dilation rates: 6, 12, 18.
    
    Also includes:
    - 1×1 convolution for local features
    - Global average pooling for image-level features
    
    All branches are concatenated and fused through a 1×1 convolution.
    """
    
    def __init__(self, in_channels=160, out_channels=128):
        """Initialize ASPP module.
        
        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        
        # Branch 1: 1×1 convolution (local features)
        self.conv1x1 = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)
        
        # Branch 2: 3×3 convolution with dilation=6 (medium context)
        self.conv_d6 = ConvBNReLU(
            in_channels, out_channels, kernel_size=3,
            padding=6, dilation=6
        )
        
        # Branch 3: 3×3 convolution with dilation=12 (large context)
        self.conv_d12 = ConvBNReLU(
            in_channels, out_channels, kernel_size=3,
            padding=12, dilation=12
        )
        
        # Branch 4: 3×3 convolution with dilation=18 (very large context)
        self.conv_d18 = ConvBNReLU(
            in_channels, out_channels, kernel_size=3,
            padding=18, dilation=18
        )
        
        # Branch 5: Global average pooling + 1×1 conv (image-level features)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True),
        )
        
        # Fusion: concatenate all 5 branches → 1×1 conv to reduce channels
        self.fuse = ConvBNReLU(out_channels * 5, out_channels, kernel_size=1, padding=0)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        """Forward pass through parallel ASPP branches.
        
        Args:
            x: Input features (N, C, H, W).
            
        Returns:
            Multi-scale fused features (N, out_channels, H, W).
        """
        size = x.shape[2:]
        
        b1 = self.conv1x1(x)
        b2 = self.conv_d6(x)
        b3 = self.conv_d12(x)
        b4 = self.conv_d18(x)
        b5 = self.global_pool(x)
        b5 = self.global_conv(b5)
        b5 = F.interpolate(b5, size=size, mode='bilinear', align_corners=True)
        
        # Concatenate all branches
        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        out = self.fuse(out)
        out = self.dropout(out)
        
        return out


# =============================================================================
# DECODER (U-Net style)
# =============================================================================

class DecoderBlock(nn.Module):
    """Single decoder block: upsample → concatenate skip → refine.
    
    Uses bilinear upsampling (not transposed convolution) for efficiency
    and to avoid checkerboard artifacts.
    """
    
    def __init__(self, in_channels, skip_channels, out_channels):
        """Initialize decoder block.
        
        Args:
            in_channels: Channels from the previous decoder level (or ASPP).
            skip_channels: Channels from the encoder skip connection.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.refine = nn.Sequential(
            ConvBNReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x, skip):
        """Forward pass with skip connection.
        
        Args:
            x: Input features from previous level.
            skip: Skip connection features from encoder.
            
        Returns:
            Refined features at 2× resolution.
        """
        # Upsample to match skip connection size
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        # Concatenate with skip
        x = torch.cat([x, skip], dim=1)
        # Refine
        return self.refine(x)


class UNetDecoder(nn.Module):
    """U-Net style decoder with bilinear upsampling and skip connections.
    
    Takes ASPP output (1/16 resolution) and progressively upsamples
    using skip connections from the encoder at 1/8 and 1/4 resolution.
    """
    
    def __init__(self, aspp_channels=128, skip_8x_channels=32, skip_4x_channels=24):
        """Initialize U-Net decoder.
        
        Args:
            aspp_channels: Channels from ASPP output.
            skip_8x_channels: Channels from encoder 1/8 skip.
            skip_4x_channels: Channels from encoder 1/4 skip.
        """
        super().__init__()
        
        # 1/16 → 1/8 (with 32-channel skip from encoder stage3)
        self.decode_8x = DecoderBlock(aspp_channels, skip_8x_channels, 64)
        
        # 1/8 → 1/4 (with 24-channel skip from encoder stage2)
        self.decode_4x = DecoderBlock(64, skip_4x_channels, 32)
    
    def forward(self, aspp_out, skip_8x, skip_4x):
        """Forward pass through decoder.
        
        Args:
            aspp_out: ASPP output at 1/16 resolution.
            skip_8x: Encoder features at 1/8 resolution.
            skip_4x: Encoder features at 1/4 resolution.
            
        Returns:
            Decoded features at 1/4 resolution (32 channels).
        """
        x = self.decode_8x(aspp_out, skip_8x)  # 1/16 → 1/8
        x = self.decode_4x(x, skip_4x)          # 1/8 → 1/4
        return x


# =============================================================================
# FULL MODEL
# =============================================================================

class LiteSegNet(nn.Module):
    """LiteSeg: Complete lightweight segmentation network.
    
    Architecture:
        Input → MobileNetV2 Encoder → ASPP → U-Net Decoder → Segmentation Head
        
    Produces binary drivable/non-drivable segmentation mask.
    Target: <3M parameters, real-time inference.
    """
    
    def __init__(self, in_channels=3, num_classes=1):
        """Initialize LiteSeg model.
        
        Args:
            in_channels: Input image channels (3 for RGB).
            num_classes: Number of output classes (1 for binary segmentation).
        """
        super().__init__()
        
        self.encoder = MobileNetV2Encoder(in_channels)
        self.aspp = LightweightASPP(in_channels=160, out_channels=128)
        self.decoder = UNetDecoder(
            aspp_channels=128,
            skip_8x_channels=32,
            skip_4x_channels=24
        )
        
        # Final segmentation head: 1×1 conv to produce logits
        self.seg_head = nn.Sequential(
            ConvBNReLU(32, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Kaiming initialization.
        
        Kaiming init is optimal for networks with ReLU activations,
        as it accounts for the variance reduction from zeroing negative values.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through complete LiteSeg network.
        
        Args:
            x: Input tensor (N, 3, H, W).
            
        Returns:
            Segmentation probabilities (N, 1, H, W) via sigmoid.
        """
        input_size = x.shape[2:]
        
        # Encoder with skip connections
        skip_4x, skip_8x, features = self.encoder(x)
        
        # ASPP for multi-scale context
        aspp_out = self.aspp(features)
        
        # Decoder with skip connections
        decoded = self.decoder(aspp_out, skip_8x, skip_4x)
        
        # Upsample from 1/4 to full resolution
        decoded = F.interpolate(decoded, size=input_size, mode='bilinear', align_corners=True)
        
        # Segmentation head
        logits = self.seg_head(decoded)
        
        return torch.sigmoid(logits)


# =============================================================================
# TEACHER MODEL (for knowledge distillation)
# =============================================================================

class LiteSegTeacher(nn.Module):
    """Larger teacher model (~5M params) for knowledge distillation.
    
    Same architecture as LiteSegNet but with wider channels
    throughout the encoder and decoder to increase capacity.
    """
    
    def __init__(self, in_channels=3, num_classes=1):
        """Initialize teacher model with wider channels.
        
        Args:
            in_channels: Input image channels.
            num_classes: Number of output classes.
        """
        super().__init__()
        
        # Wider encoder
        self.stem = ConvBNReLU(in_channels, 48, kernel_size=3, stride=2, padding=1)
        self.stage1 = InvertedResidualBlock(48, 24, stride=1, expand_ratio=1)
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(24, 32, stride=2, expand_ratio=6),
            InvertedResidualBlock(32, 32, stride=1, expand_ratio=6),
            InvertedResidualBlock(32, 32, stride=1, expand_ratio=6),
        )
        self.stage3 = nn.Sequential(
            InvertedResidualBlock(32, 48, stride=2, expand_ratio=6),
            InvertedResidualBlock(48, 48, stride=1, expand_ratio=6),
            InvertedResidualBlock(48, 48, stride=1, expand_ratio=6),
            InvertedResidualBlock(48, 48, stride=1, expand_ratio=6),
        )
        self.stage4 = nn.Sequential(
            InvertedResidualBlock(48, 96, stride=2, expand_ratio=6),
            InvertedResidualBlock(96, 96, stride=1, expand_ratio=6),
            InvertedResidualBlock(96, 96, stride=1, expand_ratio=6),
            InvertedResidualBlock(96, 96, stride=1, expand_ratio=6),
            InvertedResidualBlock(96, 96, stride=1, expand_ratio=6),
        )
        self.stage5 = nn.Sequential(
            InvertedResidualBlock(96, 128, stride=1, expand_ratio=6),
            InvertedResidualBlock(128, 128, stride=1, expand_ratio=6),
            InvertedResidualBlock(128, 128, stride=1, expand_ratio=6),
        )
        self.stage6 = nn.Sequential(
            InvertedResidualBlock(128, 224, stride=1, expand_ratio=6),
            InvertedResidualBlock(224, 224, stride=1, expand_ratio=6),
            InvertedResidualBlock(224, 224, stride=1, expand_ratio=6),
        )
        
        # Wider ASPP
        self.aspp = LightweightASPP(in_channels=224, out_channels=192)
        
        # Wider decoder
        self.decode_8x = DecoderBlock(192, 48, 96)
        self.decode_4x = DecoderBlock(96, 32, 48)
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            ConvBNReLU(48, 24, kernel_size=3, padding=1),
            nn.Conv2d(24, num_classes, kernel_size=1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through teacher model.
        
        Args:
            x: Input tensor (N, 3, H, W).
            
        Returns:
            Segmentation probabilities (N, 1, H, W).
        """
        input_size = x.shape[2:]
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        skip_4x = x
        x = self.stage3(x)
        skip_8x = x
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        
        aspp_out = self.aspp(x)
        x = self.decode_8x(aspp_out, skip_8x)
        x = self.decode_4x(x, skip_4x)
        
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        logits = self.seg_head(x)
        
        return torch.sigmoid(logits)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_info(model, input_size=(1, 3, 256, 448)):
    """Print model information including parameter count and output shape.
    
    Args:
        model: PyTorch model.
        input_size: Input tensor size for forward pass test.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{'='*50}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Parameters (M): {total_params / 1e6:.2f}M")
    print(f"Under 3M: {'✓ YES' if total_params < 3_000_000 else '✗ NO'}")
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy = torch.randn(*input_size).to(device)
    with torch.no_grad():
        output = model(dummy)
    print(f"Input shape:  {list(dummy.shape)}")
    print(f"Output shape: {list(output.shape)}")
    print(f"{'='*50}")
    
    return total_params


if __name__ == '__main__':
    # Test both models
    print("\n--- Student Model (LiteSegNet) ---")
    student = LiteSegNet()
    get_model_info(student)
    
    print("\n--- Teacher Model (LiteSegTeacher) ---")
    teacher = LiteSegTeacher()
    get_model_info(teacher)
