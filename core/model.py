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


# ==========================================
# 6A: TESSERA Stream + Cross-Attention Fusion
# (see prompts/exp-6-tessera-xattn.md)
# ==========================================

class CrossAttentionBlock(nn.Module):
    """
    One pre-LN transformer block where queries come from the pixel bottleneck
    (16x16 = 256 tokens at dim D) and keys/values come from the patch tokens
    (16x16 = 256 tokens at dim D_kv -> projected to D). Learned positional
    encodings on both sides. Residual + FFN as in a standard transformer.
    """

    def __init__(self, dim=512, kv_dim=1536, num_heads=8, ffn_mult=2,
                 num_q_tokens=256, num_kv_tokens=256, dropout=0.0):
        super().__init__()
        self.q_pos = nn.Parameter(torch.zeros(1, num_q_tokens, dim))
        self.kv_pos = nn.Parameter(torch.zeros(1, num_kv_tokens, dim))
        nn.init.trunc_normal_(self.q_pos, std=0.02)
        nn.init.trunc_normal_(self.kv_pos, std=0.02)

        self.kv_proj = nn.Linear(kv_dim, dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, q_feat, kv_feat):
        # q_feat: [B, D, Hq, Wq]   (pixel bottleneck)
        # kv_feat: [B, Dkv, Hk, Wk] (patch tokens)
        B, D, Hq, Wq = q_feat.shape
        q_tokens = q_feat.flatten(2).transpose(1, 2)              # [B, Hq*Wq, D]
        kv_tokens = kv_feat.flatten(2).transpose(1, 2)            # [B, Hk*Wk, Dkv]
        kv_tokens = self.kv_proj(kv_tokens)                       # [B, Hk*Wk, D]

        q_norm = self.norm_q(q_tokens + self.q_pos)
        kv_norm = self.norm_kv(kv_tokens + self.kv_pos)
        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        q_tokens = q_tokens + attn_out

        q_tokens = q_tokens + self.ffn(self.norm_ffn(q_tokens))

        return q_tokens.transpose(1, 2).reshape(B, D, Hq, Wq)


class YNetTesseraXAttn(nn.Module):
    """
    Experiment 6A: AlphaEarth + TESSERA dual-stream pixel input,
    cross-attention bottleneck fusion with patch tokens (terramind_s1+s2),
    shared U-Net decoder body, split task heads with GradScale on the height
    head's input. The dataset concatenates pixel streams along the channel
    axis (in the order given by --pixel-inputs), and this model splits the
    concatenation back into per-stream stems for instrumentation.
    """

    def __init__(
        self,
        alpha_channels=64,
        tessera_channels=128,
        patch_channels=1536,
        n_classes=4,
        num_heads=8,
        height_grad_scale=0.1,
        fusion_type="xattn",
    ):
        super().__init__()
        assert n_classes == 4, "YNetTesseraXAttn outputs class(3) + height(1) = 4 channels."
        assert fusion_type in ("xattn", "broadcast"), f"fusion_type must be 'xattn' or 'broadcast', got {fusion_type!r}"
        self.alpha_channels = alpha_channels
        self.tessera_channels = tessera_channels  # 0 = alpha-only ablation (n2)
        self.patch_channels = patch_channels
        self.height_grad_scale = height_grad_scale
        self.fusion_type = fusion_type

        if tessera_channels > 0:
            # Dual-stream: each stem -> 32ch, concat to 64ch
            self.alpha_stem = nn.Sequential(
                nn.Conv2d(alpha_channels, 32, kernel_size=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.tessera_stem = nn.Sequential(
                nn.Conv2d(tessera_channels, 32, kernel_size=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
        else:
            # Single-stream alpha-only ablation: stem -> 64ch directly
            self.alpha_stem = nn.Sequential(
                nn.Conv2d(alpha_channels, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.tessera_stem = None

        # Shared U-Net encoder (matches AttentionFusedDecoder / YNetAttentionFusedDecoder)
        self.inc = DoubleConv(64, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4_pool = nn.MaxPool2d(2)

        # Bottleneck patch fusion: either cross-attention (default, n1/n2/n4)
        # or broadcast/concat-conv (n3 ablation that isolates the xattn change).
        if fusion_type == "xattn":
            self.xattn = CrossAttentionBlock(
                dim=512, kv_dim=patch_channels, num_heads=num_heads,
                num_q_tokens=16 * 16, num_kv_tokens=16 * 16,
            )
            self.broadcast_fusion = None
        else:  # "broadcast"
            self.xattn = None
            self.broadcast_fusion = nn.Sequential(
                nn.Conv2d(512 + patch_channels, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

        # Shared decoder body (same attention-gated upsample stack as YNetDecoder)
        self.up1 = StandardUpsampleBlock(512, 256)
        self.att1 = AttentionGate(F_g=256, F_x=512, F_int=256)
        self.dec_conv1 = ResidualConvBlock(768, 256)

        self.up2 = StandardUpsampleBlock(256, 128)
        self.att2 = AttentionGate(F_g=128, F_x=256, F_int=128)
        self.dec_conv2 = ResidualConvBlock(384, 128)

        self.up3 = StandardUpsampleBlock(128, 64)
        self.att3 = AttentionGate(F_g=64, F_x=128, F_int=64)
        self.dec_conv3 = ResidualConvBlock(192, 64)

        self.up4 = StandardUpsampleBlock(64, 32)
        self.att4 = AttentionGate(F_g=32, F_x=64, F_int=32)
        self.dec_conv4 = ResidualConvBlock(96, 32)

        # Split task heads: 2 conv layers each (3-channel class + 1-channel height)
        self.class_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )
        self.height_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, pixel_emb, patch_emb):
        # Dataset concatenates pixel streams in the order given by --pixel-inputs.
        # Full model (n1): --pixel-inputs alpha_earth,tessera   -> 64+128 = 192 channels.
        # Alpha-only (n2): --pixel-inputs alpha_earth           -> 64 channels.
        if self.tessera_stem is not None:
            assert pixel_emb.shape[1] == self.alpha_channels + self.tessera_channels, (
                f"pixel_emb has {pixel_emb.shape[1]} channels, expected "
                f"{self.alpha_channels} (alpha_earth) + {self.tessera_channels} (tessera). "
                f"Run with --pixel-inputs alpha_earth,tessera in that order."
            )
            alpha = pixel_emb[:, :self.alpha_channels]
            tessera = pixel_emb[:, self.alpha_channels:]
            x = torch.cat([self.alpha_stem(alpha), self.tessera_stem(tessera)], dim=1)
        else:
            assert pixel_emb.shape[1] == self.alpha_channels, (
                f"pixel_emb has {pixel_emb.shape[1]} channels, expected "
                f"{self.alpha_channels} (alpha-only ablation). "
                f"Run with --pixel-inputs alpha_earth alone."
            )
            x = self.alpha_stem(pixel_emb)            # [B, 64, 256, 256]

        x1 = self.inc(x)                              # [B,  64, 256, 256]
        x2 = self.down1(x1)                           # [B, 128, 128, 128]
        x3 = self.down2(x2)                           # [B, 256,  64,  64]
        x4 = self.down3(x3)                           # [B, 512,  32,  32]
        bn = self.down4_pool(x4)                      # [B, 512,  16,  16]

        if self.fusion_type == "xattn":
            fused = self.xattn(bn, patch_emb)         # [B, 512, 16, 16]
        else:  # broadcast
            fused = self.broadcast_fusion(torch.cat([bn, patch_emb], dim=1))

        d1 = self.up1(fused)
        d1 = torch.cat([d1, self.att1(x4, d1)], dim=1)
        d1 = self.dec_conv1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, self.att2(x3, d2)], dim=1)
        d2 = self.dec_conv2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, self.att3(x2, d3)], dim=1)
        d3 = self.dec_conv3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, self.att4(x1, d4)], dim=1)
        feat = self.dec_conv4(d4)                     # [B, 32, 256, 256]

        # GradScale only on the input to the height head: encoder + shared decoder
        # see the height task only at α (0.1) strength.
        if self.training and self.height_grad_scale < 1.0:
            feat_h = grad_scale(feat, self.height_grad_scale)
        else:
            feat_h = feat

        class_out = self.class_head(feat)             # [B, 3, 256, 256]
        height_out = self.height_head(feat_h)         # [B, 1, 256, 256]
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
    if selected in ("ynet_tessera_xattn", "ynet_tessera_broadcast"):
        # pixel_channels is the total concat channel count from the dataset.
        #   192 = 64 (alpha_earth) + 128 (tessera)  ->  full model (n1, n3, n4)
        #    64 = alpha_earth alone                 ->  alpha-only ablation (n2)
        #   128 = tessera alone                     ->  tessera-only (kept for completeness)
        if pixel_channels == 64 + 128:
            alpha_ch, tessera_ch = 64, 128
        elif pixel_channels == 64:
            alpha_ch, tessera_ch = 64, 0
        elif pixel_channels == 128:
            alpha_ch, tessera_ch = 128, 0
        else:
            raise ValueError(
                f"{selected} expects pixel_channels in {{64, 128, 192}} "
                f"(alpha-only / tessera-only / both). Got {pixel_channels}. "
                f"Use --pixel-inputs alpha_earth[,tessera]."
            )
        fusion = "broadcast" if selected == "ynet_tessera_broadcast" else "xattn"
        return YNetTesseraXAttn(
            alpha_channels=alpha_ch,
            tessera_channels=tessera_ch,
            patch_channels=patch_channels,
            n_classes=n_classes,
            fusion_type=fusion,
        ), selected

    raise ValueError(
        f"Unknown model_type '{model_type}'. Use one of: auto, lightunet, decoder_residual, attention_fusion, ynet_attention_fusion, ynet_tessera_xattn, ynet_tessera_broadcast"
    )