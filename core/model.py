import torch
import torch.nn as nn
import torch.nn.functional as F


class GradScale(torch.autograd.Function):
    """
    Identity in the forward pass; scales the gradient by `scale` in the backward pass.
    Used to reduce the height decoder's gradient contribution to the shared encoder
    without fully detaching it (Option A gradient hook).
    """
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def grad_scale(x, alpha: float):
    return GradScale.apply(x, alpha)


# ==========================================
# 1. LIGHT UNET COMPONENTS
# ==========================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpsampleBlock(nn.Module):
    """
    Bilinear Upsampling + Convolution.
    Smoother than PixelShuffle/TransposeConv, avoids checkerboard artifacts.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LightUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(LightUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Architecture: Light version (32->64->128->256)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        self.up1 = UpsampleBlock(256, 128)
        self.conv1 = DoubleConv(256, 128)

        self.up2 = UpsampleBlock(128, 64)
        self.conv2 = DoubleConv(128, 64)

        self.up3 = UpsampleBlock(64, 32)
        self.conv3 = DoubleConv(64, 32)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)

        logits = self.outc(x)
        return logits


# ==========================================
# 2. M2-OPTIMIZED DECODER COMPONENTS
# ==========================================
#
# class DecoderEasyM2(nn.Module):
#     """Fast, lightweight decoder avoiding ConvTranspose2d."""
#
#     def __init__(self, in_channels=768, out_channels=4):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True)
#         )
#
#         self.up1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#
#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#
#         self.up3 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
#         )
#
#     def forward(self, x):
#         x = self.proj(x)
#         x = self.up1(x)
#         x = self.up2(x)
#         return self.up3(x)
#
#
# class DepthwiseSeparableConv(nn.Module):
#     """M2-Optimized Convolution: Computes spatial and channel features separately."""
#
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return self.bn(x)
#
#
# class ResidualBlockM2(nn.Module):
#     """Lightweight residual block using Depthwise-Separable Convolutions."""
#
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
#         self.act = nn.GELU()
#         self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
#
#         self.shortcut = (
#             nn.Identity()
#             if in_channels == out_channels
#             else nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_channels),
#             )
#         )
#
#     def forward(self, x):
#         residual = self.shortcut(x)
#         x = self.conv1(x)
#         x = self.act(x)
#         x = self.conv2(x)
#         x = x + residual
#         return self.act(x)
#
#
# class UpsampleFusionBlockM2(nn.Module):
#     """Hardware-accelerated upsampling with M2-friendly projections."""
#
#     def __init__(self, in_channels, out_channels, skip_channels):
#         super().__init__()
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.up_proj = nn.Sequential(
#             DepthwiseSeparableConv(in_channels, out_channels),
#             nn.GELU(),
#         )
#         self.skip_proj = nn.Sequential(
#             nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#         )
#         self.fuse = ResidualBlockM2(out_channels * 2, out_channels)
#
#     def forward(self, x, skip):
#         x = self.upsample(x)
#         x = self.up_proj(x)
#
#         if skip.shape[-2:] != x.shape[-2:]:
#             skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
#         skip = self.skip_proj(skip)
#
#         x = torch.cat([x, skip], dim=1)
#         return self.fuse(x)
#
#
# class DecoderResidualM2(nn.Module):
#     """Fully M2-Optimized deeper embedding decoder."""
#
#     def __init__(self, in_channels=768, out_channels=4, widths=(320, 256, 192, 128, 96), dropout=0.1):
#         super().__init__()
#         if len(widths) != 5:
#             raise ValueError("widths must contain exactly 5 values for 4 upsampling stages")
#
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(in_channels, widths[0], kernel_size=1, bias=False),
#             nn.BatchNorm2d(widths[0]),
#             nn.GELU(),
#             ResidualBlockM2(widths[0], widths[0]),
#         )
#
#         self.global_skip = nn.Sequential(
#             nn.Conv2d(in_channels, widths[-1], kernel_size=1, bias=False),
#             nn.BatchNorm2d(widths[-1]),
#             nn.GELU(),
#         )
#
#         self.up1 = UpsampleFusionBlockM2(widths[0], widths[1], widths[-1])
#         self.up2 = UpsampleFusionBlockM2(widths[1], widths[2], widths[-1])
#         self.up3 = UpsampleFusionBlockM2(widths[2], widths[3], widths[-1])
#         self.up4 = UpsampleFusionBlockM2(widths[3], widths[4], widths[-1])
#
#         self.head = nn.Sequential(
#             ResidualBlockM2(widths[4], widths[4]),
#             nn.Dropout2d(p=dropout),
#             DepthwiseSeparableConv(widths[4], 64),
#             nn.GELU(),
#             nn.Conv2d(64, out_channels, kernel_size=1),
#         )
#
#     def forward(self, x):
#         skip = self.global_skip(x)
#         x = self.bottleneck(x)
#         x = self.up1(x, skip)
#         x = self.up2(x, skip)
#         x = self.up3(x, skip)
#         x = self.up4(x, skip)
#         return self.head(x)
#
#
# # ==========================================
# # 3. MODEL BUILDER
# # ==========================================
#
# def infer_model_type(n_channels):
#     if n_channels == 768:
#         return "only_decoder"
#     return "lightunet"
#
#
#
#
# class DepthwiseSeparableConv(nn.Module):
#     """M2-Optimized Convolution: Computes spatial and channel features separately."""
#
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = nn.GELU()
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return self.act(x)
#
#

class StandardUpsampleBlock(nn.Module):
    """
    Uses standard dense convolutions.
    Blazingly fast on Apple Silicon MPS, unlike grouped/depthwise convs.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Standard 3x3 convolution (groups=1) which the M2 GPU loves
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class EfficientDecoder256Fast(nn.Module):
    """
    High-speed, memory-safe decoder for 16x16 -> 256x256 upsampling on M2 Max.
    """

    def __init__(self, in_channels=768, out_channels=4):
        super().__init__()

        # THE SQUEEZE: 768 -> 256 at 16x16 resolution. (Prevents memory blowup)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )

        # PROGRESSIVE UPSAMPLING: Halving channels as resolution doubles.
        self.up1 = StandardUpsampleBlock(256, 128)  # 16x16   -> 32x32
        self.up2 = StandardUpsampleBlock(128, 64)  # 32x32   -> 64x64
        self.up3 = StandardUpsampleBlock(64, 32)  # 64x64   -> 128x128
        self.up4 = StandardUpsampleBlock(32, 16)  # 128x128 -> 256x256

        # PREDICTION HEAD
        self.head = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.head(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        """
        F_g: gating signal channels (from decoder)
        F_x: skip connection channels (from encoder)
        F_int: intermediate representation channels
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1)
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Align spatial dimensions of gating signal to skip connection if they differ
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        alpha = self.sigmoid(psi)
        return x * alpha


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.act(out)


class AttentionFusedDecoder(nn.Module):
    def __init__(self, pixel_channels=128, patch_channels=768, out_channels=4):
        super().__init__()

        # --- ENCODER (128 -> 64 -> 128 -> 256 -> 512) ---
        self.inc = DoubleConv(pixel_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # MaxPool to obtain 16x16 resolution bottleneck features from the encoder
        self.down4_pool = nn.MaxPool2d(2)

        # --- BOTTLENECK FUSION ---
        # Concatenate 512 (encoder bottleneck) + 768 (patch embedding) = 1280 channels
        # Pass this combined tensor through a convolution to reduce channel dimensions to 512.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(512 + patch_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # --- DECODER with Attention Gated Skip Connections ---
        # Step 1: 16x16 -> 32x32.
        self.up1 = StandardUpsampleBlock(512, 256)
        self.att1 = AttentionGate(F_g=256, F_x=512, F_int=256)
        self.dec_conv1 = ResidualConvBlock(768, 256)

        # Step 2: 32x32 -> 64x64.
        self.up2 = StandardUpsampleBlock(256, 128)
        self.att2 = AttentionGate(F_g=128, F_x=256, F_int=128)
        self.dec_conv2 = ResidualConvBlock(384, 128)

        # Step 3: 64x64 -> 128x128.
        self.up3 = StandardUpsampleBlock(128, 64)
        self.att3 = AttentionGate(F_g=64, F_x=128, F_int=64)
        self.dec_conv3 = ResidualConvBlock(192, 64)

        # Step 4: 128x128 -> 256x256.
        self.up4 = StandardUpsampleBlock(64, 32)
        self.att4 = AttentionGate(F_g=32, F_x=64, F_int=32)
        self.dec_conv4 = ResidualConvBlock(96, 32)

        # --- OUTPUT HEAD ---
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, pixel_emb, patch_emb):
        # 1. Encoder forward pass
        x1 = self.inc(pixel_emb)                # [B, 64, 256, 256]
        x2 = self.down1(x1)                     # [B, 128, 128, 128]
        x3 = self.down2(x2)                     # [B, 256, 64, 64]
        x4 = self.down3(x3)                     # [B, 512, 32, 32]
        
        # Encoder bottleneck representation at 16x16
        enc_bottleneck = self.down4_pool(x4)    # [B, 512, 16, 16]

        # 2. Bottleneck Fusion
        fused = torch.cat([enc_bottleneck, patch_emb], dim=1)  # [B, 512+768, 16, 16]
        fused = self.fusion_conv(fused)                        # [B, 512, 16, 16]

        # 3. Decoder with Attention-Gated skip connections
        # Stage 1: 16x16 -> 32x32
        d1 = self.up1(fused)                    # [B, 256, 32, 32]
        gated_x4 = self.att1(x4, d1)            # [B, 512, 32, 32]
        d1 = torch.cat([d1, gated_x4], dim=1)   # [B, 768, 32, 32]
        d1 = self.dec_conv1(d1)                 # [B, 256, 32, 32]

        # Stage 2: 32x32 -> 64x64
        d2 = self.up2(d1)                       # [B, 128, 64, 64]
        gated_x3 = self.att2(x3, d2)            # [B, 256, 64, 64]
        d2 = torch.cat([d2, gated_x3], dim=1)   # [B, 384, 64, 64]
        d2 = self.dec_conv2(d2)                 # [B, 128, 64, 64]

        # Stage 3: 64x64 -> 128x128
        d3 = self.up3(d2)                       # [B, 64, 128, 128]
        gated_x2 = self.att3(x2, d3)            # [B, 128, 128, 128]
        d3 = torch.cat([d3, gated_x2], dim=1)   # [B, 192, 128, 128]
        d3 = self.dec_conv3(d3)                 # [B, 64, 128, 128]

        # Stage 4: 128x128 -> 256x256
        d4 = self.up4(d3)                       # [B, 32, 256, 256]
        gated_x1 = self.att4(x1, d4)            # [B, 64, 256, 256]
        d4 = torch.cat([d4, gated_x1], dim=1)   # [B, 96, 256, 256]
        d4 = self.dec_conv4(d4)                 # [B, 32, 256, 256]

        # Output head
        logits = self.head(d4)                  # [B, 4, 256, 256]
        return logits


class YNetDecoder(nn.Module):
    """
    A single decoder branch of the Y-Net. Receives the bottleneck
    and all 4 encoder skip connections, and produces `out_channels` output maps.
    This is instantiated TWICE (once for classification, once for height).
    """

    def __init__(self, out_channels: int):
        super().__init__()
        # Stage 1: 16x16 -> 32x32
        self.up1 = StandardUpsampleBlock(512, 256)
        self.att1 = AttentionGate(F_g=256, F_x=512, F_int=256)
        self.dec_conv1 = ResidualConvBlock(768, 256)

        # Stage 2: 32x32 -> 64x64
        self.up2 = StandardUpsampleBlock(256, 128)
        self.att2 = AttentionGate(F_g=128, F_x=256, F_int=128)
        self.dec_conv2 = ResidualConvBlock(384, 128)

        # Stage 3: 64x64 -> 128x128
        self.up3 = StandardUpsampleBlock(128, 64)
        self.att3 = AttentionGate(F_g=64, F_x=128, F_int=64)
        self.dec_conv3 = ResidualConvBlock(192, 64)

        # Stage 4: 128x128 -> 256x256
        self.up4 = StandardUpsampleBlock(64, 32)
        self.att4 = AttentionGate(F_g=32, F_x=64, F_int=32)
        self.dec_conv4 = ResidualConvBlock(96, 32)

        # Output head
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, fused, x1, x2, x3, x4):
        """fused: [B, 512, 16, 16] | x1-x4: encoder skip connections."""
        d1 = self.up1(fused)                    # [B, 256, 32, 32]
        gated_x4 = self.att1(x4, d1)            # [B, 512, 32, 32]
        d1 = torch.cat([d1, gated_x4], dim=1)   # [B, 768, 32, 32]
        d1 = self.dec_conv1(d1)                 # [B, 256, 32, 32]

        d2 = self.up2(d1)                       # [B, 128, 64, 64]
        gated_x3 = self.att2(x3, d2)            # [B, 256, 64, 64]
        d2 = torch.cat([d2, gated_x3], dim=1)   # [B, 384, 64, 64]
        d2 = self.dec_conv2(d2)                 # [B, 128, 64, 64]

        d3 = self.up3(d2)                       # [B, 64, 128, 128]
        gated_x2 = self.att3(x2, d3)            # [B, 128, 128, 128]
        d3 = torch.cat([d3, gated_x2], dim=1)   # [B, 192, 128, 128]
        d3 = self.dec_conv3(d3)                 # [B, 64, 128, 128]

        d4 = self.up4(d3)                       # [B, 32, 256, 256]
        gated_x1 = self.att4(x1, d4)            # [B, 64, 256, 256]
        d4 = torch.cat([d4, gated_x1], dim=1)   # [B, 96, 256, 256]
        d4 = self.dec_conv4(d4)                 # [B, 32, 256, 256]

        return self.head(d4)


class YNetAttentionFusedDecoder(nn.Module):
    """
    Y-Net: Shared U-Net Encoder + Patch Fusion Bottleneck with TWO
    completely independent decoders to prevent gradient conflicts:
      - decoder_class  : 3 channels (Building, Vegetation, Water)
      - decoder_height : 1 channel  (nDSM Height)

    Outputs are concatenated at the end -> [B, 4, 256, 256].
    BatchNorm is applied after every convolution (already done inside
    DoubleConv, ResidualConvBlock, StandardUpsampleBlock, AttentionGate).
    """

    def __init__(self, pixel_channels=128, patch_channels=768, height_grad_scale=0.1):
        super().__init__()

        # Fraction of height-decoder gradient that reaches the shared encoder.
        # 1.0 = full coupling (original Y-Net), 0.0 = full detach, 0.1 = gentle whisper.
        self.height_grad_scale = height_grad_scale

        # --- SHARED ENCODER (identical to AttentionFusedDecoder) ---
        self.inc   = DoubleConv(pixel_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4_pool = nn.MaxPool2d(2)

        # --- SHARED BOTTLENECK FUSION ---
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(512 + patch_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # --- TWO INDEPENDENT DECODER BRANCHES ---
        self.decoder_class  = YNetDecoder(out_channels=3)  # Building / Veg / Water
        self.decoder_height = YNetDecoder(out_channels=1)  # nDSM Height

    def forward(self, pixel_emb, patch_emb):
        # 1. Shared encoder
        x1 = self.inc(pixel_emb)              # [B,  64, 256, 256]
        x2 = self.down1(x1)                   # [B, 128, 128, 128]
        x3 = self.down2(x2)                   # [B, 256,  64,  64]
        x4 = self.down3(x3)                   # [B, 512,  32,  32]
        enc_bottleneck = self.down4_pool(x4)  # [B, 512,  16,  16]

        # 2. Shared bottleneck fusion with patch embedding
        fused = torch.cat([enc_bottleneck, patch_emb], dim=1)  # [B, 1280, 16, 16]
        fused = self.fusion_conv(fused)                        # [B,  512, 16, 16]

        # 3. Gradient hook: classification decoder owns the encoder at full strength.
        #    Height decoder contributes only height_grad_scale (default 0.1) of its
        #    gradient back through the bottleneck and all skip connections.
        #    GradScale is a no-op in the forward pass — zero extra computation.
        if self.training and self.height_grad_scale < 1.0:
            alpha = self.height_grad_scale
            fused_h = grad_scale(fused, alpha)
            x1_h    = grad_scale(x1,    alpha)
            x2_h    = grad_scale(x2,    alpha)
            x3_h    = grad_scale(x3,    alpha)
            x4_h    = grad_scale(x4,    alpha)
        else:
            fused_h, x1_h, x2_h, x3_h, x4_h = fused, x1, x2, x3, x4

        class_out  = self.decoder_class( fused,   x1,   x2,   x3,   x4)   # [B, 3, 256, 256]
        height_out = self.decoder_height(fused_h, x1_h, x2_h, x3_h, x4_h) # [B, 1, 256, 256]

        # 4. Concatenate -> [B, 4, 256, 256]
        return torch.cat([class_out, height_out], dim=1)


def infer_model_type(n_channels):
    if n_channels == 768:
        return "decoder_residual"
    return "lightunet"


def build_model(model_type, n_channels, n_classes, pixel_channels=128, patch_channels=768):
    selected = model_type.lower()

    if selected == "auto":
        selected = infer_model_type(n_channels)
    if selected == "lightunet":
        return LightUNet(n_channels, n_classes), selected
    if selected == "decoder_residual":
        return EfficientDecoder256Fast(in_channels=n_channels, out_channels=n_classes), selected
    if selected == "attention_fusion":
        return AttentionFusedDecoder(pixel_channels=pixel_channels, patch_channels=patch_channels, out_channels=n_classes), selected
    if selected == "ynet_attention_fusion":
        return YNetAttentionFusedDecoder(pixel_channels=pixel_channels, patch_channels=patch_channels), selected

    raise ValueError(
        f"Unknown model_type '{model_type}'. Use one of: auto, lightunet, decoder_residual, attention_fusion, ynet_attention_fusion"
    )