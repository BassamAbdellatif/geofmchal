import torch
import torch.nn as nn
import torch.nn.functional as F


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


def infer_model_type(n_channels):
    if n_channels == 768:
        return "decoder_residual"
    return "lightunet"


def build_model(model_type, n_channels, n_classes, use_height_bridge=True,
                patch_inputs=("terramind_s1", "terramind_s2"),
                patch_stem_version="v2", xattn_heads=4,
                patch_routing="sensor",
                use_fraction_bridge=False, fraction_bridge_alpha=0.2,
                fraction_head="sigmoid3"):
    selected = model_type.lower()

    if selected == "auto":
        selected = infer_model_type(n_channels)
    if selected == "lightunet":
        return LightUNet(n_channels, n_classes), selected
    if selected == "decoder_residual":
        return EfficientDecoder256Fast(in_channels=n_channels, out_channels=n_classes), selected
    if selected == "dual_enc_dec_fusion":
        # 7A: channels are fixed per modality (alpha=64, tessera=128, patches=768,
        # out=4). n_channels / n_classes are ignored for this architecture.
        return DualEncDualDecFusion(
            use_height_bridge=use_height_bridge,
            patch_inputs=patch_inputs,
            patch_stem_version=patch_stem_version,
            xattn_heads=xattn_heads,
            patch_routing=patch_routing,
            use_fraction_bridge=use_fraction_bridge,
            fraction_bridge_alpha=fraction_bridge_alpha,
            fraction_head=fraction_head,
        ), selected

    raise ValueError(
        f"Unknown model_type '{model_type}'. Use one of: auto, lightunet, "
        f"decoder_residual, dual_enc_dec_fusion"
    )


# =============================================================================
# 7A — Decoupled Dual-Encoder / Dual-Decoder with Sensor-Routed Patch Skips
#
#   alpha_earth (64ch)            tessera (128ch)
#        │ AlphaStem(BN)               │ TesseraStem(BN)
#        │ UNetEncoderHalf             │ UNetEncoderHalf
#   α-skips + α-bottleneck (16×16)  τ-skips + τ-bottleneck (16×16)
#        │                             │
#   FractionDecoder ◄─ S2 patches   HeightDecoder ◄─ S1 patches
#   (α-skips, inject @16,@32)       (τ-skips, inject @16,@32,
#        │                            + α-bottleneck side-input via GradScale)
#   [3 frac + 1 binary-B heads]     [1 height head]
#
# Resolution flow (resolved from the spec's concrete decoder/patch constraints):
#   encoder levels [96,128,192,256,384] at resolutions [256,128,64,32,16];
#   bottleneck 16×16@384; decoder up-blocks 16→32→64→128→256 with patch
#   cross-attention injected at the 16×16 and 32×32 decoder levels.
# Deviation from the spec diagram: the stem is a full-resolution per-modality
# projection (not pre-downsampled to 32×32). Rationale: a 256-res stem skip
# preserves the high-frequency detail that building IoU depends on (6A lesson),
# and it makes the 5-level encoder + 16×16 bottleneck internally consistent.
# =============================================================================

def _gn(num_channels, max_groups=32):
    """GroupNorm with a group count that divides the channel count."""
    g = max_groups
    while num_channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, num_channels)


class _GradScale(torch.autograd.Function):
    """Identity forward; scales gradient by alpha on backward."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None


def grad_scale(x, alpha):
    return _GradScale.apply(x, alpha)


class _DoubleConvGN(nn.Module):
    """(conv 3×3 → GroupNorm → GELU) × 2, optional stride-2 on the first conv."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            _gn(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class _ModalityStem(nn.Module):
    """
    Per-modality input projection at full 256×256 resolution. Uses BatchNorm
    (per-modality statistic calibration, as the spec asks) before any mixing.
    Output: 96 channels at 256×256.
    """

    def __init__(self, in_ch, out_ch=96):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


def AlphaStem():
    return _ModalityStem(in_ch=64, out_ch=96)


def TesseraStem():
    return _ModalityStem(in_ch=128, out_ch=96)


class UNetEncoderHalf(nn.Module):
    """
    5-level U-Net encoder. Takes the 96ch@256 stem output as level 0 and
    downsamples to a 16×16 bottleneck.
        L0: 96  @ 256   (skip, = stem output, passed through)
        L1: 128 @ 128   (skip)
        L2: 192 @ 64    (skip)
        L3: 256 @ 32    (skip)
        L4: 384 @ 16    (bottleneck)
    forward(stem_out) -> (skips=[L0,L1,L2,L3], bottleneck=L4)
    """

    CHANNELS = [96, 128, 192, 256, 384]

    def __init__(self):
        super().__init__()
        c = self.CHANNELS
        self.down1 = _DoubleConvGN(c[0], c[1], stride=2)  # 256 -> 128
        self.down2 = _DoubleConvGN(c[1], c[2], stride=2)  # 128 -> 64
        self.down3 = _DoubleConvGN(c[2], c[3], stride=2)  # 64  -> 32
        self.down4 = _DoubleConvGN(c[3], c[4], stride=2)  # 32  -> 16

    def forward(self, stem_out):
        l0 = stem_out          # 96  @ 256
        l1 = self.down1(l0)    # 128 @ 128
        l2 = self.down2(l1)    # 192 @ 64
        l3 = self.down3(l2)    # 256 @ 32
        l4 = self.down4(l3)    # 384 @ 16  (bottleneck)
        return [l0, l1, l2, l3], l4


class PatchTokenStemV1(nn.Module):
    """
    Original (Phase 5C / 7A_simple) patch token stem.
    Projects raw patch tokens (B, 256, 768) -> (B, 256, out_dim) with a small
    MLP + LayerNorm, and adds a learned 16×16 positional encoding.

    Kept byte-identical to the pre-Phase-5D `PatchTokenStem` (only the name
    changed) so `--patch-stem-version v1` reproduces `7A_simple` exactly.
    """

    def __init__(self, in_dim=768, out_dim=384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )
        self.pos = nn.Parameter(torch.zeros(1, 256, out_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, tokens):
        return self.mlp(tokens) + self.pos


class PatchTokenStemV2(nn.Module):
    """
    Phase 5D enhanced stem — per-modality calibration before patch token fusion.
    LayerNorm normalises embedding statistics independently per modality.
    2-layer MLP with GELU projects to common cross-attention dimension.
    Learned positional encoding added after projection.

    The whole point: foundation-model embeddings (TerraMind vs THOR) have
    mismatched value ranges / covariance; an independent LayerNorm + MLP per
    modality calibrates each space before they are concatenated as cross-attn
    keys/values, so the larger-norm modality cannot hijack a shared projection.
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 384):
        super().__init__()
        self.norm   = nn.LayerNorm(in_dim)
        self.proj1  = nn.Linear(in_dim, out_dim)
        self.act    = nn.GELU()
        self.proj2  = nn.Linear(out_dim, out_dim)
        self.pos_embed = nn.Parameter(torch.randn(256, out_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 256, in_dim]
        x = self.norm(x)
        x = self.act(self.proj1(x))
        x = self.proj2(x)
        return x + self.pos_embed


class PatchTokenStemV2b(nn.Module):
    """
    Phase 5E water-safe variant of the enhanced stem.

    Keeps V2's depth/calibration intent (2-layer GELU MLP + learned positional
    encoding) but DROPS the input LayerNorm. V2's LayerNorm over the raw 768-d
    tokens normalises away per-token magnitude, which empirically suppressed
    rare-class (water) ignition: v1/v2b keep water (IoU_W ~0.68), v2 collapses it
    to 0. V2b normalises the OUTPUT instead (v1-style), so the input token scale
    the water signal lives in is preserved. Cross-modal scale (THOR ≫ TerraMind)
    is already handled upstream by the per-channel cache norm-stats, so the input
    LayerNorm was both redundant and harmful.
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 384):
        super().__init__()
        self.proj1  = nn.Linear(in_dim, out_dim)
        self.act    = nn.GELU()
        self.proj2  = nn.Linear(out_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)          # output norm (v1-style)
        self.pos_embed = nn.Parameter(torch.randn(256, out_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.proj1(x))
        x = self.proj2(x)
        x = self.norm(x)
        return x + self.pos_embed


def _make_patch_stem(version, in_dim=768, out_dim=384):
    if version == "v1":
        return PatchTokenStemV1(in_dim, out_dim)
    if version == "v2":
        return PatchTokenStemV2(in_dim, out_dim)
    if version == "v2b":
        return PatchTokenStemV2b(in_dim, out_dim)
    raise ValueError(f"Unknown patch_stem_version {version!r}; use 'v1', 'v2', or 'v2b'.")


class PatchCrossAttnBlock(nn.Module):
    """
    Cross-attention from decoder features (queries) to patch tokens (keys/values).
    Decoder features (B,C,H,W) attend over the patch token grid. When the decoder
    resolution differs from the 16×16 patch grid, the tokens are bilinearly
    resampled to H×W first. Residual + LayerNorm on the query stream.
    """

    def __init__(self, dim_q, dim_kv=384, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_q, num_heads=heads, kdim=dim_kv, vdim=dim_kv,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim_q)

    def forward(self, x, tokens):
        B, C, H, W = x.shape
        # `tokens` may be a concatenation of G 16×16 patch grids (one per patch
        # modality), shape (B, G*256, Dkv). At the 16×16 decoder level the grids
        # are already at decoder resolution, so they are used directly as K/V
        # (the K dimension simply grows with G). At higher decoder resolutions
        # each 16×16 grid is bilinearly resampled to H×W *independently* (mixing
        # modalities through a single fake grid would scramble them) and then
        # re-concatenated along the token dimension.
        if H * W != 256:
            N = tokens.shape[1]
            G = N // 256
            Dkv = tokens.shape[2]
            t = tokens.reshape(B, G, 256, Dkv).permute(0, 1, 3, 2).reshape(B * G, Dkv, 16, 16)
            t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
            t = t.reshape(B, G, Dkv, H * W).permute(0, 1, 3, 2).reshape(B, G * H * W, Dkv)
        else:
            t = tokens                                              # B, G*256, Dkv
        q = x.flatten(2).transpose(1, 2)                            # B,H*W,C
        attended, _ = self.attn(q, t, t)
        q = self.norm(q + attended)
        return q.transpose(1, 2).reshape(B, C, H, W)


class _UpBlock(nn.Module):
    """
    Decoder up-block: bilinear upsample ×2, concat skip, fuse with a DoubleConv.
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.fuse = _DoubleConvGN(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.reduce(self.up(x))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


# Encoder feature channels, shared by both decoders.
_ENC = UNetEncoderHalf.CHANNELS          # [96,128,192,256,384]
_DEC = [384, 256, 192, 128, 96]          # bottleneck -> up1 -> up2 -> up3 -> up4


class FractionDecoder(nn.Module):
    """
    Fraction decoder: 4 up-blocks from the α-bottleneck (384@16) to 256×256,
    using α-encoder skips. S2 patch tokens are injected by cross-attention at
    the 16×16 (bottleneck) and 32×32 (after up1) decoder levels.
    Outputs: 3 fraction channels + 1 auxiliary binary-building channel.
    """

    def __init__(self, xattn_heads=4, use_bridge=False, bridge_alpha=0.2, num_classes=3):
        super().__init__()
        # [Phase 5E #1] Optional τ-bottleneck → fraction bridge, mirror of the
        # height bridge. Constructed only when enabled, so the default path is
        # byte-identical (no extra params / no init-RNG shift) to 7A_simple.
        self.use_fbridge = use_bridge
        self.fbridge_alpha = bridge_alpha
        if use_bridge:
            self.fbridge = nn.Conv2d(_DEC[0] * 2, _DEC[0], 1, bias=False)
        self.inject16 = PatchCrossAttnBlock(dim_q=_DEC[0], heads=xattn_heads)   # 384 @ 16
        self.up1 = _UpBlock(_DEC[0], _ENC[3], _DEC[1])       # 16->32,  skip 256@32
        self.inject32 = PatchCrossAttnBlock(dim_q=_DEC[1], heads=xattn_heads)   # 256 @ 32
        self.up2 = _UpBlock(_DEC[1], _ENC[2], _DEC[2])       # 32->64,  skip 192@64
        self.up3 = _UpBlock(_DEC[2], _ENC[1], _DEC[3])       # 64->128, skip 128@128
        self.up4 = _UpBlock(_DEC[3], _ENC[0], _DEC[4])       # 128->256,skip 96@256
        # [Phase 9 P2] num_classes=3 (default) -> byte-identical sigmoid3 head;
        # =4 -> softmax-4 simplex [building, veg, water, other]. binary_head is
        # built after frac_head so its init RNG is unchanged when num_classes=3.
        self.frac_head = nn.Conv2d(_DEC[4], num_classes, 1)
        self.binary_head = nn.Conv2d(_DEC[4], 1, 1)

    def forward(self, bottleneck, skips, s2_tokens, tessera_bottleneck=None):
        # [Phase 5E #1] Optional cross-encoder bridge: fuse the τ-bottleneck into
        # the fraction path's first block (GradScale protects the τ/height encoder).
        if self.use_fbridge and tessera_bottleneck is not None:
            t = grad_scale(tessera_bottleneck, self.fbridge_alpha)
            bottleneck = self.fbridge(torch.cat([bottleneck, t], dim=1))
        # s2_tokens is None when no *_s2 patch streams are active -> the fraction
        # decoder runs on the α-encoder features alone (no patch injection).
        x = self.inject16(bottleneck, s2_tokens) if s2_tokens is not None else bottleneck
        x = self.up1(x, skips[3])
        if s2_tokens is not None:
            x = self.inject32(x, s2_tokens)
        x = self.up2(x, skips[2])
        x = self.up3(x, skips[1])
        x = self.up4(x, skips[0])
        return self.frac_head(x), self.binary_head(x)


class HeightDecoder(nn.Module):
    """
    Height decoder: 4 up-blocks from the τ-bottleneck (384@16) to 256×256, using
    τ-encoder skips. The α-bottleneck is concatenated as a side input at the
    first block (the only cross-encoder meeting point); a GradScale(α=0.2) hook
    on that side input keeps height loss from dominating the α-encoder. S1 patch
    tokens are injected at the 16×16 and 32×32 levels. Outputs: 1 height channel.
    """

    def __init__(self, bridge_alpha=0.2, use_bridge=True, xattn_heads=4):
        super().__init__()
        self.bridge_alpha = bridge_alpha
        self.use_bridge = use_bridge
        # Fuse τ-bottleneck (384) + α-bottleneck side input (384) -> 384.
        # Always constructed (so param init/order is identical with or without the
        # bridge); the forward only uses it when use_bridge is True.
        self.bridge = nn.Conv2d(_DEC[0] * 2, _DEC[0], 1, bias=False)
        self.inject16 = PatchCrossAttnBlock(dim_q=_DEC[0], heads=xattn_heads)   # 384 @ 16
        self.up1 = _UpBlock(_DEC[0], _ENC[3], _DEC[1])       # 16->32
        self.inject32 = PatchCrossAttnBlock(dim_q=_DEC[1], heads=xattn_heads)   # 256 @ 32
        self.up2 = _UpBlock(_DEC[1], _ENC[2], _DEC[2])       # 32->64
        self.up3 = _UpBlock(_DEC[2], _ENC[1], _DEC[3])       # 64->128
        self.up4 = _UpBlock(_DEC[3], _ENC[0], _DEC[4])       # 128->256
        self.height_head = nn.Conv2d(_DEC[4], 1, 1)

    def forward(self, tau_bottleneck, tau_skips, alpha_bottleneck, s1_tokens):
        if self.use_bridge:
            a = grad_scale(alpha_bottleneck, self.bridge_alpha)
            x = self.bridge(torch.cat([tau_bottleneck, a], dim=1))
        else:
            # No cross-encoder bridge: height decoder runs on the τ-bottleneck
            # alone (skip the projection entirely; do not pass zeros through it).
            x = tau_bottleneck
        # s1_tokens is None when no *_s1 patch streams are active.
        if s1_tokens is not None:
            x = self.inject16(x, s1_tokens)
        x = self.up1(x, tau_skips[3])
        if s1_tokens is not None:
            x = self.inject32(x, s1_tokens)
        x = self.up2(x, tau_skips[2])
        x = self.up3(x, tau_skips[1])
        x = self.up4(x, tau_skips[0])
        return self.height_head(x)


class DualEncDualDecFusion(nn.Module):
    """
    7A top-level model. See module banner for the full diagram.

    forward(batch) expects a dict with keys:
        alpha_earth  (B,64,256,256)
        tessera      (B,128,256,256)
        and one (B,256,768) entry per active patch stream in `patch_inputs`,
        e.g. terramind_s1, terramind_s2, thor_s1, thor_s2.
    and returns a dict:
        fraction (B,3,256,256) | height (B,1,256,256) | binary (B,1,256,256)

    Patch routing (data-driven by name suffix, Phase 5D):
        any *_s1 stream -> height decoder injection (SAR/structure path)
        any *_s2 stream -> fraction decoder injection (optical/fraction path)
        multiple same-sensor streams (e.g. terramind_s1 + thor_s1) are projected
        by their own per-modality stem and concatenated along the token dim
        before cross-attention (K/V grows; Q unchanged).
        no *_s1 streams -> height decoder gets no patch injection (and likewise
        for *_s2 / fraction decoder).

    With patch_inputs=(terramind_s1, terramind_s2), patch_stem_version='v1',
    xattn_heads=4 the model is byte-identical to the pre-Phase-5D `7A_simple`.

    predict() assembles the 4-channel metric output [B,V,W,height], applying
    sigmoid to the fraction channels (height is left as the raw regression).
    """

    def __init__(self, bridge_alpha=0.2, use_height_bridge=True,
                 patch_inputs=("terramind_s1", "terramind_s2"),
                 patch_stem_version="v2", xattn_heads=4,
                 patch_routing="sensor",
                 use_fraction_bridge=False, fraction_bridge_alpha=0.2,
                 fraction_head="sigmoid3"):
        super().__init__()
        self.patch_inputs = list(patch_inputs)
        self.patch_stem_version = patch_stem_version
        self.xattn_heads = xattn_heads
        # [Phase 9 P2] 'sigmoid3' (default, byte-identical) or 'softmax4' simplex.
        if fraction_head not in ("sigmoid3", "softmax4"):
            raise ValueError(f"Unknown fraction_head {fraction_head!r}; "
                             f"use 'sigmoid3' or 'softmax4'.")
        self.fraction_head = fraction_head
        # [Phase 5E #4] sensor->branch routing. 'sensor' (default) = byte-identical
        # 7A: s1->height, s2->fraction. 's1-both' also feeds s1 into the fraction
        # branch (SAR for water/building). 'all-both' feeds everything to both.
        if patch_routing not in ("sensor", "s1-both", "all-both"):
            raise ValueError(f"Unknown patch_routing {patch_routing!r}; "
                             f"use 'sensor', 's1-both', or 'all-both'.")
        self.patch_routing = patch_routing
        self.alpha_stem = AlphaStem()
        self.tessera_stem = TesseraStem()
        self.alpha_encoder = UNetEncoderHalf()
        self.tessera_encoder = UNetEncoderHalf()
        # One calibration stem per active patch modality, built in patch_inputs
        # order (preserves init RNG order vs the old s1/s2 attribute pair).
        self.token_stems = nn.ModuleDict()
        for name in self.patch_inputs:
            self.token_stems[name] = _make_patch_stem(patch_stem_version, 768, 384)
        self._s1_names = [n for n in self.patch_inputs if n.endswith("_s1")]
        self._s2_names = [n for n in self.patch_inputs if n.endswith("_s2")]
        self.fraction_decoder = FractionDecoder(xattn_heads=xattn_heads,
                                                use_bridge=use_fraction_bridge,
                                                bridge_alpha=fraction_bridge_alpha,
                                                num_classes=4 if fraction_head == "softmax4" else 3)
        self.height_decoder = HeightDecoder(bridge_alpha=bridge_alpha,
                                            use_bridge=use_height_bridge,
                                            xattn_heads=xattn_heads)

    def _route_tokens(self, batch, names):
        if not names:
            return None
        proj = [self.token_stems[n](batch[n]) for n in names]
        return proj[0] if len(proj) == 1 else torch.cat(proj, dim=1)

    @staticmethod
    def _cat_tokens(*toks):
        toks = [t for t in toks if t is not None]
        if not toks:
            return None
        return toks[0] if len(toks) == 1 else torch.cat(toks, dim=1)

    def forward(self, batch):
        alpha = batch["alpha_earth"]
        tessera = batch["tessera"]
        s1 = self._route_tokens(batch, self._s1_names)
        s2 = self._route_tokens(batch, self._s2_names)

        # [Phase 5E #4] decide which token set each decoder attends over.
        if self.patch_routing == "sensor":
            frac_tok, height_tok = s2, s1
        elif self.patch_routing == "s1-both":
            frac_tok, height_tok = self._cat_tokens(s2, s1), s1
        else:  # all-both
            both = self._cat_tokens(s1, s2)
            frac_tok, height_tok = both, both

        a_skips, a_bottleneck = self.alpha_encoder(self.alpha_stem(alpha))
        t_skips, t_bottleneck = self.tessera_encoder(self.tessera_stem(tessera))

        frac, binary = self.fraction_decoder(a_bottleneck, a_skips, frac_tok, t_bottleneck)
        height = self.height_decoder(t_bottleneck, t_skips, a_bottleneck, height_tok)

        return {"fraction": frac, "height": height, "binary": binary}

    @torch.no_grad()
    def predict(self, batch):
        out = self.forward(batch)
        if self.fraction_head == "softmax4":
            # 4-way softmax, then drop the derived 'other' channel for the metric.
            frac = torch.softmax(out["fraction"], dim=1)[:, :3]
        else:
            frac = torch.sigmoid(out["fraction"])
        return torch.cat([frac, out["height"]], dim=1)   # (B,4,256,256)