"""
model.py — LiteSeg: Lightweight Segmentation Network for Drivable Space.

Custom architecture built entirely from scratch:
- Encoder: MobileNetV2-style inverted residual blocks (depthwise separable convolutions)
- CoordConv: Injects (x, y) positional channels for spatial awareness
- Squeeze-Excite attention in decoder for channel recalibration
- Decoder: Lightweight ASPP (dilation rates 6, 12, 18) + U-Net skip connections
- Output: Raw logits (sigmoid applied externally for numerical stability)

Target: <3M parameters, >30 FPS on CPU, >100 FPS on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SPATIAL AWARENESS: CoordConv
# =============================================================================

class AddCoords(nn.Module):
    """Concatenate normalized (x, y) coordinate channels to input tensor.
    
    Critical for training from scratch: gives the network absolute position
    awareness so it can learn spatial priors like 'sky is at top, road at bottom'
    without pretrained ImageNet features.
    
    Adds 2 channels: normalized x ∈ [-1, 1] and y ∈ [-1, 1].
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Add coordinate channels to input.
        
        Args:
            x: Input tensor (N, C, H, W).
            
        Returns:
            Tensor (N, C+2, H, W) with x and y coordinate channels appended.
        """
        B, _, H, W = x.shape
        
        # Create normalized coordinate grids [-1, 1]
        y_coords = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        x_coords = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        
        # Expand to (1, 1, H, W) and (1, 1, H, W)
        y_grid = y_coords.view(1, 1, H, 1).expand(B, 1, H, W)
        x_grid = x_coords.view(1, 1, 1, W).expand(B, 1, H, W)
        
        return torch.cat([x, x_grid, y_grid], dim=1)


class CoordConv(nn.Module):
    """CoordConv = AddCoords + Standard Conv2d.
    
    Liu et al., 2018 — "An Intriguing Failing of Convolutional Neural Networks
    and the CoordConv Solution". Particularly effective for:
    - Coordinate transforms (which we need: pixel position → class label)
    - Training from scratch (no pretrained spatial priors)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=False):
        super().__init__()
        self.add_coords = AddCoords()
        self.conv = nn.Conv2d(
            in_channels + 2, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
    
    def forward(self, x):
        x = self.add_coords(x)
        return self.conv(x)


# =============================================================================
# CHANNEL ATTENTION: Squeeze-and-Excite
# =============================================================================

class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation channel attention block.
    
    Learns to recalibrate channel-wise feature responses.
    In decoder: suppresses irrelevant skip-connection channels
    (e.g., building texture features when predicting road).
    
    Only adds ~2*C parameters per block — negligible cost.
    """
    
    def __init__(self, channels, reduction=4):
        """Initialize SE block.
        
        Args:
            channels: Number of input/output channels.
            reduction: Channel reduction ratio for bottleneck.
        """
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        """Apply channel attention.
        
        Args:
            x: Input tensor (N, C, H, W).
            
        Returns:
            Channel-recalibrated tensor (N, C, H, W).
        """
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


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
    
    Uses CoordConv at the stem to inject spatial position awareness.
    
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
        
        # CoordConv stem: RGB + (x, y) → 32 channels, stride 2 (1/2 resolution)
        # CoordConv adds 2 coordinate channels, so input is in_channels + 2
        self.coord_conv = CoordConv(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_relu = nn.ReLU6(inplace=True)
        
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
        # CoordConv stem (position-aware)
        x = self.stem_relu(self.stem_bn(self.coord_conv(x)))  # 1/2
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
# DECODER (U-Net style with SE attention)
# =============================================================================

class DecoderBlock(nn.Module):
    """Single decoder block: upsample → concatenate skip → refine → SE attend.
    
    Uses bilinear upsampling (not transposed convolution) for efficiency
    and to avoid checkerboard artifacts. SE attention recalibrates channels
    after skip concatenation to suppress irrelevant encoder features.
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
        # Channel attention after refinement
        self.se = SqueezeExcite(out_channels, reduction=4)
    
    def forward(self, x, skip):
        """Forward pass with skip connection and SE attention.
        
        Args:
            x: Input features from previous level.
            skip: Skip connection features from encoder.
            
        Returns:
            Refined, attention-recalibrated features at 2× resolution.
        """
        # Upsample to match skip connection size
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        # Concatenate with skip
        x = torch.cat([x, skip], dim=1)
        # Refine
        x = self.refine(x)
        # Channel attention
        x = self.se(x)
        return x


class UNetDecoder(nn.Module):
    """U-Net style decoder with bilinear upsampling, skip connections, and SE attention.
    
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
        Input → CoordConv Stem → MobileNetV2 Encoder → ASPP → 
        U-Net Decoder (with SE attention) → Segmentation Head
        
    Produces binary drivable/non-drivable segmentation.
    
    IMPORTANT: forward() returns RAW LOGITS (not sigmoid).
    Use predict() for inference with sigmoid applied.
    
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
        
        - fan_in mode: correct for layers followed by ReLU/ReLU6
        - Final conv bias initialized to prior that most pixels are non-drivable
          (prevents the model from starting by predicting everything as road)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize final conv bias to prior: ~30% of pixels are drivable
        # bias = -log((1 - prior) / prior) = -log(0.7 / 0.3) ≈ -0.847
        # This ensures the model starts by predicting "mostly non-drivable"
        final_conv = self.seg_head[-1]
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, -0.847)
    
    def forward(self, x):
        """Forward pass through complete LiteSeg network.
        
        RETURNS RAW LOGITS — no sigmoid applied.
        Use predict() for inference.
        
        Args:
            x: Input tensor (N, 3, H, W).
            
        Returns:
            Raw logits (N, 1, H, W). Apply sigmoid externally for probabilities.
        """
        input_size = x.shape[2:]
        
        # Encoder with skip connections (CoordConv at stem)
        skip_4x, skip_8x, features = self.encoder(x)
        
        # ASPP for multi-scale context
        aspp_out = self.aspp(features)
        
        # Decoder with skip connections and SE attention
        decoded = self.decoder(aspp_out, skip_8x, skip_4x)
        
        # Upsample from 1/4 to full resolution
        decoded = F.interpolate(decoded, size=input_size, mode='bilinear', align_corners=True)
        
        # Segmentation head → raw logits
        logits = self.seg_head(decoded)
        
        return logits
    
    def predict(self, x):
        """Inference-time forward pass with sigmoid.
        
        Args:
            x: Input tensor (N, 3, H, W).
            
        Returns:
            Segmentation probabilities (N, 1, H, W) via sigmoid.
        """
        return torch.sigmoid(self.forward(x))


# =============================================================================
# TEACHER MODEL (for knowledge distillation)
# =============================================================================

class LiteSegTeacher(nn.Module):
    """Larger teacher model (~5M params) for knowledge distillation.
    
    Same architecture as LiteSegNet but with wider channels
    throughout the encoder and decoder to increase capacity.
    
    Also uses CoordConv stem and SE attention in decoder.
    Returns raw logits; use predict() for inference.
    """
    
    def __init__(self, in_channels=3, num_classes=1):
        """Initialize teacher model with wider channels.
        
        Args:
            in_channels: Input image channels.
            num_classes: Number of output classes.
        """
        super().__init__()
        
        # CoordConv stem (wider)
        self.coord_conv = CoordConv(in_channels, 48, kernel_size=3, stride=2, padding=1)
        self.stem_bn = nn.BatchNorm2d(48)
        self.stem_relu = nn.ReLU6(inplace=True)
        
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
        
        # Wider decoder with SE attention
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
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Prior: most pixels are non-drivable
        final_conv = self.seg_head[-1]
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, -0.847)
    
    def forward(self, x):
        """Forward pass through teacher model.
        
        Returns raw logits.
        
        Args:
            x: Input tensor (N, 3, H, W).
            
        Returns:
            Raw logits (N, 1, H, W).
        """
        input_size = x.shape[2:]
        
        x = self.stem_relu(self.stem_bn(self.coord_conv(x)))
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
        
        return logits
    
    def predict(self, x):
        """Inference-time forward pass with sigmoid.
        
        Args:
            x: Input tensor (N, 3, H, W).
            
        Returns:
            Segmentation probabilities (N, 1, H, W).
        """
        return torch.sigmoid(self.forward(x))


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
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}] (logits)")
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
