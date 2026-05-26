import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced segmentation.
    alpha: penalizes False Positives.
    beta: penalizes False Negatives.
    Setting beta > alpha forces the model to capture minority classes (like sparse buildings).
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds, targets):
        batch_size = preds.size(0)
        p = preds.view(batch_size, -1)
        t = targets.view(batch_size, -1)

        TP = torch.sum(p * t, dim=1)
        FP = torch.sum(p * (1 - t), dim=1)
        FN = torch.sum((1 - p) * t, dim=1)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return torch.mean(1.0 - tversky)


class GradientDifferenceLoss(nn.Module):
    """Penalizes differences in image gradients (edges/sharpness)."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])

        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

        loss_x = torch.mean(torch.abs(pred_dx - target_dx))
        loss_y = torch.mean(torch.abs(pred_dy - target_dy))

        return loss_x + loss_y


class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) Loss using a Gaussian window."""

    def __init__(self, window_size=5, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            elif img1.device.type == 'mps':
                window = window.to('mps')

            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


class CompositeLossWithVegBoost(nn.Module):
    """
    ImprovedCompositeLoss + vegetation height boost.

    Recipe:
    1. Split Foreground/Background weighted MAE (buildings & water weighted 3×).
    2. SSIM and Gradient Loss on abundance channels.
    3. Multi-class Tversky Loss (buildings & water weighted higher).
    4. Building-masked height MAE boost.
    5. Vegetation-masked height MAE boost.

    Returns: (total_loss, loss_mae, loss_ssim, loss_grad, loss_tversky)
    """

    def __init__(self, lambdas=None, bg_weight=0.05):
        super().__init__()
        self.ssim = SSIMLoss(window_size=11)
        self.gdl = GradientDifferenceLoss()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.bg_weight = bg_weight
        # lambdas kept for API compat but actual weights are hardcoded below

    def forward(self, preds, targets, **kwargs):
        device = preds.device

        preds_abund = torch.sigmoid(preds[:, :3, :, :])
        preds_height = preds[:, 3:4, :, :]
        target_abund = targets[:, :3, :, :]
        target_height = targets[:, 3:4, :, :]

        preds_norm = torch.cat([preds_abund, preds_height], dim=1)

        # --- Foreground/Background weighted MAE ---
        abs_err = torch.abs(preds_norm - targets)
        fg_mask = (targets > 0).float()
        bg_mask = 1.0 - fg_mask

        fg_sum = torch.sum(fg_mask, dim=(0, 2, 3)) + 1e-6
        bg_sum = torch.sum(bg_mask, dim=(0, 2, 3)) + 1e-6

        mae_fg = torch.sum(abs_err * fg_mask, dim=(0, 2, 3)) / fg_sum
        mae_bg = torch.sum(abs_err * bg_mask, dim=(0, 2, 3)) / bg_sum
        mae_per_channel = mae_fg + (self.bg_weight * mae_bg)

        # Heavily penalize building (ch 0) and water (ch 2) errors
        mae_weights = torch.tensor([3.0, 1.0, 3.0, 1.0]).to(device)
        loss_mae = torch.sum(mae_per_channel * mae_weights)

        # --- Structural & Gradient Loss ---
        loss_ssim = self.ssim(preds_abund, target_abund)
        loss_grad = self.gdl(preds_abund, target_abund)

        # --- Multi-Class Tversky ---
        t_build = self.tversky(preds_abund[:, 0, :, :], target_abund[:, 0, :, :])
        t_veg   = self.tversky(preds_abund[:, 1, :, :], target_abund[:, 1, :, :])
        t_water = self.tversky(preds_abund[:, 2, :, :], target_abund[:, 2, :, :])
        loss_tversky = (2.0 * t_build + 0.5 * t_veg + 2.0 * t_water) / 4.5

        # --- Height Boost: buildings ---
        height_err = torch.abs(preds_height - target_height)
        build_mask = (target_abund[:, 0, :, :].detach() > 0.1).float()
        loss_height_boost = (
            torch.sum(height_err * build_mask) / (build_mask.sum() + 1e-6)
        )

        # --- Height Boost: vegetation ---
        veg_mask = (target_abund[:, 1, :, :].detach() > 0.1).float()
        loss_veg_boost = (
            torch.sum(height_err * veg_mask) / (veg_mask.sum() + 1e-6)
        )

        total_loss = (
            1.0 * loss_mae +
            0.5 * loss_ssim +
            0.5 * loss_grad +
            2.0 * loss_tversky +
            1.0 * loss_height_boost +
            1.0 * loss_veg_boost
        )

        return total_loss, loss_mae, loss_ssim, loss_grad, loss_tversky


class MSEVegBoostLoss(nn.Module):
    """
    Pure MSE for abundance channels + masked MAE for height.

    Rationale: abundance targets are continuous fractions (0–1), not binary.
    MSE is better calibrated for fractional outputs evaluated via hard-IoU-at-0.5.
    Height is trained with a base MAE plus two region-specific boosts:
      - building-masked boost (recover tall structure height)
      - vegetation-masked boost (recover tree canopy / vegetation height)

    Returns: (total_loss, loss_mse, zeros×3)
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, **kwargs):
        preds_abund = torch.sigmoid(preds[:, :3, :, :])
        preds_height = preds[:, 3:4, :, :]
        target_abund = targets[:, :3, :, :]
        target_height = targets[:, 3:4, :, :]

        # Task 1: abundance MSE
        loss_mse = F.mse_loss(preds_abund, target_abund)

        # Task 2: height base MAE
        height_err = torch.abs(preds_height - target_height)
        loss_height_base = height_err.mean()

        # Task 3: building-masked height boost
        build_mask = (target_abund[:, 0, :, :].detach() > 0.1).float()
        loss_build_boost = (
            torch.sum(height_err.squeeze(1) * build_mask) /
            (build_mask.sum() + 1e-6)
        )

        # Task 4: vegetation-masked height boost
        veg_mask = (target_abund[:, 1, :, :].detach() > 0.1).float()
        loss_veg_boost = (
            torch.sum(height_err.squeeze(1) * veg_mask) /
            (veg_mask.sum() + 1e-6)
        )

        total = loss_mse + loss_height_base + loss_build_boost + loss_veg_boost

        _zero = torch.zeros(1, device=preds.device)
        return total, loss_mse, _zero, _zero, _zero


class MSESigmaLoss(nn.Module):
    """
    MSEVegBoostLoss + Kendall uncertainty weighting (Kendall et al., 2018).

    Four learnable log-variance parameters (one per sub-task) are trained jointly
    with the model. The total loss is:
        L = Σ_i [ exp(-log_var_i) * L_i + log_var_i ]
    This automatically balances the four sub-tasks without manual lambda tuning.

    NOTE: the optimizer must include list(criterion.parameters()) so the
    log_vars are updated alongside the model weights.

    Returns: (total_loss, loss_mse, zeros×3)
    """

    def __init__(self):
        super().__init__()
        # log_vars for: [abundance_mse, height_base, build_boost, veg_boost]
        self.log_vars = nn.Parameter(torch.zeros(4))

    def forward(self, preds, targets, **kwargs):
        preds_abund = torch.sigmoid(preds[:, :3, :, :])
        preds_height = preds[:, 3:4, :, :]
        target_abund = targets[:, :3, :, :]
        target_height = targets[:, 3:4, :, :]

        # Task 0: abundance MSE
        loss_mse = F.mse_loss(preds_abund, target_abund)

        # Task 1: height base MAE
        height_err = torch.abs(preds_height - target_height)
        loss_height_base = height_err.mean()

        # Task 2: building-masked height boost
        build_mask = (target_abund[:, 0, :, :].detach() > 0.1).float()
        loss_build_boost = (
            torch.sum(height_err.squeeze(1) * build_mask) /
            (build_mask.sum() + 1e-6)
        )

        # Task 3: vegetation-masked height boost
        veg_mask = (target_abund[:, 1, :, :].detach() > 0.1).float()
        loss_veg_boost = (
            torch.sum(height_err.squeeze(1) * veg_mask) /
            (veg_mask.sum() + 1e-6)
        )

        raw_losses = [loss_mse, loss_height_base, loss_build_boost, loss_veg_boost]
        total = sum(
            torch.exp(-self.log_vars[i]) * l + self.log_vars[i]
            for i, l in enumerate(raw_losses)
        )

        _zero = torch.zeros(1, device=preds.device)
        return total, loss_mse, _zero, _zero, _zero


# Alias for backward compatibility
ImprovedCompositeLoss = CompositeLossWithVegBoost
