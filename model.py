"""
model.py — LiteSeg: Lightweight Segmentation Network for Drivable Space.

Custom architecture built entirely from scratch:
- Encoder: MobileNetV2-style inverted residual blocks (depthwise separable convolutions)
- CoordConv: Injects (x, y) positional channels for spatial awareness
- RAU: Reflection Attention Unit — detects water puddles via sky-ground vertical correlation
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
        B, _, H, W = x.shape
        y_coords = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        x_coords = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        y_grid = y_coords.view(1, 1, H, 1).expand(B, 1, H, W)
        x_grid = x_coords.view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, x_grid, y_grid], dim=1)


class CoordConv(nn.Module):
    """CoordConv = AddCoords + Standard Conv2d."""
    
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
    """Squeeze-and-Excitation channel attention block."""
    
    def __init__(self, channels, reduction=4):
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
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


# =============================================================================
# REFLECTION ATTENTION UNIT (RAU) — water puddle detection
# =============================================================================

class ReflectionAttentionUnit(nn.Module):
    """Reflection Attention Unit for detecting water puddles and reflective surfaces.

    From: Han et al., "Single Image Water Hazard Detection using FCN with
    Reflection Attention Units", ECCV 2018.

    Core insight: a water puddle on the road reflects the sky above it.
    This creates a specific vertical relationship in the feature space —
    a ground-level pixel that looks like sky is almost certainly a reflection.

    Mechanism:
        1. Vertically flip the feature map. This aligns each ground pixel
           with the sky pixel it would reflect if the surface were a mirror.
        2. Concatenate original + flipped features channel-wise.
        3. A small conv network learns to detect where these two streams
           match (i.e., where ground looks like sky = reflection = puddle).
        4. The resulting spatial attention map suppresses reflective pixels
           so the decoder classifies them correctly as road surface, not sky.

    Gated residual design:
        output = x + tanh(gate) * refined_attended_features
        gate initialised to 0 → tanh(0) = 0 → identity at epoch 0.
        The gate gradually opens as training progresses, so the RAU cannot
        destabilise early training. Safe for from-scratch training.

    Where it lives: between encoder stage5 and stage6 (1/16 resolution,
    96 channels in student / 128 in teacher). Deep enough for semantic
    sky/ground understanding, before final feature compression.

    Args:
        channels: Number of input/output feature channels.
        reduction: Channel reduction for the cross-attention bottleneck.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 16)

        # Cross-attention branch: detects vertical sky-ground correspondence.
        # Input: [original | vertically_flipped] → 2*channels
        # Output: spatial attention map in [0, 1]
        self.cross_attn = nn.Sequential(
            nn.Conv2d(channels * 2, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Refinement: processes attended features before gated residual add.
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Learnable gate scalar — initialised to 0 (identity at start of training).
        # tanh keeps it bounded in (-1, 1) so the residual never explodes.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Apply reflection attention.

        Args:
            x: Feature map (N, C, H, W) from encoder deep context path.

        Returns:
            Attended feature map (N, C, H, W), same shape as input.
        """
        # Step 1: vertically flip — aligns ground pixels with the sky they reflect
        x_flip = torch.flip(x, dims=[2])

        # Step 2: cross-attention from vertical correlation
        concat = torch.cat([x, x_flip], dim=1)   # (N, 2C, H, W)
        attn = self.cross_attn(concat)             # (N, 1, H, W) in [0, 1]

        # Step 3: suppress reflective pixels, refine
        attended = x * attn
        refined   = self.refine(attended)

        # Step 4: gated residual — gate=0 → identity at init, opens during training
        return x + torch.tanh(self.gate) * refined


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class ConvBNReLU(nn.Module):
    """Standard Convolution → BatchNorm → ReLU6 block."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InvertedResidualBlock(nn.Module):
    """MobileNetV2 Inverted Residual Block."""
    
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1, padding=0))
        layers.append(ConvBNReLU(
            hidden_dim, hidden_dim, kernel_size=3, stride=stride,
            padding=1, groups=hidden_dim
        ))
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


# =============================================================================
# ENCODER
# =============================================================================

class MobileNetV2Encoder(nn.Module):
    """Custom MobileNetV2-style encoder with CoordConv stem and RAU.

    RAU is inserted between stage5 and stage6 — the deep context path
    at 1/16 resolution where semantic sky/ground features are richest.
    """
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # CoordConv stem: RGB + (x, y) → 32 channels, stride 2 (1/2 resolution)
        self.coord_conv = CoordConv(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.stem_bn    = nn.BatchNorm2d(32)
        self.stem_relu  = nn.ReLU6(inplace=True)
        
        self.stage1 = InvertedResidualBlock(32, 16, stride=1, expand_ratio=1)
        
        # 1/4 resolution — SKIP CONNECTION
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(16, 24, stride=2, expand_ratio=6),
            InvertedResidualBlock(24, 24, stride=1, expand_ratio=6),
        )
        
        # 1/8 resolution — SKIP CONNECTION
        self.stage3 = nn.Sequential(
            InvertedResidualBlock(24, 32, stride=2, expand_ratio=6),
            InvertedResidualBlock(32, 32, stride=1, expand_ratio=6),
            InvertedResidualBlock(32, 32, stride=1, expand_ratio=6),
        )
        
        # 1/16 resolution
        self.stage4 = nn.Sequential(
            InvertedResidualBlock(32, 64, stride=2, expand_ratio=6),
            InvertedResidualBlock(64, 64, stride=1, expand_ratio=6),
            InvertedResidualBlock(64, 64, stride=1, expand_ratio=6),
            InvertedResidualBlock(64, 64, stride=1, expand_ratio=6),
        )
        self.stage5 = nn.Sequential(
            InvertedResidualBlock(64, 96, stride=1, expand_ratio=6),
            InvertedResidualBlock(96, 96, stride=1, expand_ratio=6),
            InvertedResidualBlock(96, 96, stride=1, expand_ratio=6),
        )

        # RAU inserted here: 96-channel features at 1/16 resolution.
        # Deep enough for sky/road semantic understanding.
        # Safe: gated residual starts as identity (gate=0).
        self.rau = ReflectionAttentionUnit(channels=96, reduction=8)

        self.stage6 = nn.Sequential(
            InvertedResidualBlock(96, 160, stride=1, expand_ratio=6),
            InvertedResidualBlock(160, 160, stride=1, expand_ratio=6),
        )
    
    def forward(self, x):
        """Forward pass returning multi-scale features for skip connections.

        Returns:
            (skip_4x, skip_8x, features_16x):
            - skip_4x:       (N, 24,  H/4,  W/4)
            - skip_8x:       (N, 32,  H/8,  W/8)
            - features_16x:  (N, 160, H/16, W/16)
        """
        x = self.stem_relu(self.stem_bn(self.coord_conv(x)))  # 1/2
        x = self.stage1(x)    # 1/2

        x = self.stage2(x)    # 1/4
        skip_4x = x

        x = self.stage3(x)    # 1/8
        skip_8x = x

        x = self.stage4(x)    # 1/16
        x = self.stage5(x)    # 1/16
        x = self.rau(x)       # 1/16 — reflection attention (water puddles)
        x = self.stage6(x)    # 1/16

        return skip_4x, skip_8x, x


# =============================================================================
# ASPP (Atrous Spatial Pyramid Pooling)
# =============================================================================

class LightweightASPP(nn.Module):
    """Lightweight Atrous Spatial Pyramid Pooling module."""
    
    def __init__(self, in_channels=160, out_channels=128):
        super().__init__()
        self.conv1x1  = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv_d6  = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=6,  dilation=6)
        self.conv_d12 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv_d18 = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True),
        )
        self.fuse    = ConvBNReLU(out_channels * 5, out_channels, kernel_size=1, padding=0)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        size = x.shape[2:]
        b1 = self.conv1x1(x)
        b2 = self.conv_d6(x)
        b3 = self.conv_d12(x)
        b4 = self.conv_d18(x)
        b5 = self.global_conv(self.global_pool(x))
        b5 = F.interpolate(b5, size=size, mode='bilinear', align_corners=True)
        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return self.dropout(self.fuse(out))


# =============================================================================
# DECODER (U-Net style with SE attention)
# =============================================================================

class DecoderBlock(nn.Module):
    """Single decoder block: upsample → concatenate skip → refine → SE attend."""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.refine = nn.Sequential(
            ConvBNReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.se      = SqueezeExcite(out_channels, reduction=4)
        self.dropout = nn.Dropout2d(0.15)
    
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.refine(x)
        x = self.se(x)
        return self.dropout(x)


class UNetDecoder(nn.Module):
    """U-Net style decoder with bilinear upsampling, skip connections, and SE attention."""
    
    def __init__(self, aspp_channels=128, skip_8x_channels=32, skip_4x_channels=24):
        super().__init__()
        self.decode_8x = DecoderBlock(aspp_channels, skip_8x_channels, 64)
        self.decode_4x = DecoderBlock(64, skip_4x_channels, 32)
    
    def forward(self, aspp_out, skip_8x, skip_4x):
        x = self.decode_8x(aspp_out, skip_8x)
        x = self.decode_4x(x, skip_4x)
        return x


# =============================================================================
# FULL MODEL
# =============================================================================

class LiteSegNet(nn.Module):
    """LiteSeg: Complete lightweight segmentation network.

    Architecture:
        Input
        → CoordConv Stem      (position-aware, no pretrained weights needed)
        → MobileNetV2 Stages 1-5
        → RAU                 (reflection attention — water puddle handling)
        → MobileNetV2 Stage 6
        → ASPP                (multi-scale context)
        → U-Net Decoder       (skip connections + SE channel attention)
        → Segmentation Head
        → Raw logits

    IMPORTANT: forward() returns RAW LOGITS — no sigmoid.
    Apply sigmoid externally (in loss/metric functions) for numerical stability.
    Use predict() for inference.
    """
    
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.encoder  = MobileNetV2Encoder(in_channels)
        self.aspp     = LightweightASPP(in_channels=160, out_channels=128)
        self.decoder  = UNetDecoder(aspp_channels=128, skip_8x_channels=32, skip_4x_channels=24)
        self.seg_head = nn.Sequential(
            ConvBNReLU(32, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )
        # Auxiliary boundary head — taps from same 32-ch decoder features.
        # Predicts drivable area edges for explicit boundary supervision.
        self.boundary_head = nn.Conv2d(32, 1, kernel_size=1)
        self._init_weights()
    
    def _init_weights(self):
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
        # Prior: ~30% of pixels are drivable → bias = -log(0.7/0.3) ≈ -0.847
        final_conv = self.seg_head[-1]
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, -0.847)
        # Boundary head: ~10% of pixels are boundary → bias = -log(0.9/0.1) ≈ -2.2
        nn.init.constant_(self.boundary_head.bias, -2.2)
        # RAU gate explicitly zero (already default, but be explicit)
        nn.init.zeros_(self.encoder.rau.gate)
    
    def forward(self, x):
        """Forward pass.
        
        Training:  returns (seg_logits, boundary_logits) — both (N, 1, H, W).
        Eval:      returns seg_logits only (N, 1, H, W) — backward compatible.
        """
        input_size          = x.shape[2:]
        skip_4x, skip_8x, features = self.encoder(x)
        aspp_out            = self.aspp(features)
        decoded             = self.decoder(aspp_out, skip_8x, skip_4x)
        decoded             = F.interpolate(decoded, size=input_size, mode='bilinear', align_corners=True)
        seg_logits = self.seg_head(decoded)
        
        if self.training:
            boundary_logits = self.boundary_head(decoded)
            return seg_logits, boundary_logits
        return seg_logits
    
    def predict(self, x):
        """Inference: returns sigmoid probabilities (N, 1, H, W)."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# =============================================================================
# TEACHER MODEL (for knowledge distillation)
# =============================================================================

class LiteSegTeacher(nn.Module):
    """Larger teacher model (~5M params) for knowledge distillation.

    Identical structure to LiteSegNet but wider channels.
    RAU inserted at equivalent position: between stage5 and stage6
    at 128-channel, 1/16 resolution features.
    """
    
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        
        self.coord_conv = CoordConv(in_channels, 48, kernel_size=3, stride=2, padding=1)
        self.stem_bn    = nn.BatchNorm2d(48)
        self.stem_relu  = nn.ReLU6(inplace=True)
        
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

        # RAU at 128 channels — same position as student
        self.rau = ReflectionAttentionUnit(channels=128, reduction=8)

        self.stage6 = nn.Sequential(
            InvertedResidualBlock(128, 224, stride=1, expand_ratio=6),
            InvertedResidualBlock(224, 224, stride=1, expand_ratio=6),
            InvertedResidualBlock(224, 224, stride=1, expand_ratio=6),
        )
        
        self.aspp     = LightweightASPP(in_channels=224, out_channels=192)
        self.decode_8x = DecoderBlock(192, 48, 96)
        self.decode_4x = DecoderBlock(96, 32, 48)
        self.seg_head = nn.Sequential(
            ConvBNReLU(48, 24, kernel_size=3, padding=1),
            nn.Conv2d(24, num_classes, kernel_size=1),
        )
        # Auxiliary boundary head — same design as student
        self.boundary_head = nn.Conv2d(48, 1, kernel_size=1)
        self._init_weights()
    
    def _init_weights(self):
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
        final_conv = self.seg_head[-1]
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, -0.847)
        nn.init.constant_(self.boundary_head.bias, -2.2)
        nn.init.zeros_(self.rau.gate)
    
    def forward(self, x):
        """Training: returns (seg_logits, boundary_logits). Eval: seg_logits only."""
        input_size = x.shape[2:]
        x = self.stem_relu(self.stem_bn(self.coord_conv(x)))
        x = self.stage1(x)
        x = self.stage2(x);  skip_4x = x
        x = self.stage3(x);  skip_8x = x
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.rau(x)       # reflection attention
        x = self.stage6(x)
        aspp_out = self.aspp(x)
        x = self.decode_8x(aspp_out, skip_8x)
        x = self.decode_4x(x, skip_4x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        seg_logits = self.seg_head(x)
        
        if self.training:
            boundary_logits = self.boundary_head(x)
            return seg_logits, boundary_logits
        return seg_logits
    
    def predict(self, x):
        """Inference: returns sigmoid probabilities (N, 1, H, W)."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_info(model, input_size=(1, 3, 256, 448)):
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'='*50}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Parameters (M): {total_params / 1e6:.2f}M")
    print(f"Under 3M: {'✓ YES' if total_params < 3_000_000 else '✗ NO'}")
    device = next(model.parameters()).device
    dummy  = torch.randn(*input_size).to(device)
    model.eval()
    with torch.no_grad():
        output = model(dummy)
    if isinstance(output, tuple):
        seg, boundary = output
        print(f"Input shape:     {list(dummy.shape)}")
        print(f"Seg output:      {list(seg.shape)}")
        print(f"Boundary output: {list(boundary.shape)}")
    else:
        print(f"Input shape:  {list(dummy.shape)}")
        print(f"Output shape: {list(output.shape)}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}] (logits)")
    print(f"{'='*50}")
    return total_params


if __name__ == '__main__':
    print("\n--- Student Model (LiteSegNet) ---")
    student = LiteSegNet()
    get_model_info(student)
    
    print("\n--- Teacher Model (LiteSegTeacher) ---")
    teacher = LiteSegTeacher()
    get_model_info(teacher)