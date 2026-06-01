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
        # Flatten: [Batch, H, W] -> [Batch, N]
        batch_size = preds.size(0)
        p = preds.view(batch_size, -1)
        t = targets.view(batch_size, -1)

        # True Positives, False Positives, False Negatives
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


class ImprovedCompositeLoss(nn.Module):
    """
    Combined Recipe:
    1. Split Foreground/Background MAE (L1).
    2. SSIM and Gradient Loss on RGB channels.
    3. Tversky Loss specifically to boost Building recall.
    4. Structure-Boosted Height MAE.
    """

    def __init__(self, lambdas=[1.0, 0.5, 0.5, 2.0], bg_weight=0.05):
        super().__init__()
        self.ssim = SSIMLoss(window_size=11)
        self.gdl = GradientDifferenceLoss()

        # Using Tversky instead of Jaccard to heavily penalize missing buildings
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

        self.w_mae = lambdas[0]
        self.w_ssim = lambdas[1]
        self.w_grad = lambdas[2]
        self.w_structure = lambdas[3]  # Represents both Tversky and Building-Height weights

        self.bg_weight = bg_weight

        # Height is now normalized, so we weight all 4 channels equally in base MAE
        self.mae_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).float()

    def forward(self, preds, targets):
        device = preds.device
        mae_weights = self.mae_weights.to(device)

        # --- 1. Base MAE (Foreground / Background Split) ---
        abs_err = torch.abs(preds - targets)
        fg_mask = (targets > 0).float()
        bg_mask = 1.0 - fg_mask

        fg_sum = torch.sum(fg_mask, dim=(0, 2, 3)) + 1e-6
        bg_sum = torch.sum(bg_mask, dim=(0, 2, 3)) + 1e-6

        mae_fg = torch.sum(abs_err * fg_mask, dim=(0, 2, 3)) / fg_sum
        mae_bg = torch.sum(abs_err * bg_mask, dim=(0, 2, 3)) / bg_sum

        mae_per_channel = mae_fg + (self.bg_weight * mae_bg)
        loss_mae = torch.sum(mae_per_channel * mae_weights)

        # --- 2. Structural & Gradient Loss on Land Cover Only ---
        lc_pred = preds[:, :3, :, :]
        lc_target = targets[:, :3, :, :]

        loss_ssim = self.ssim(lc_pred, lc_target)
        loss_grad = self.gdl(lc_pred, lc_target)

        # --- 3. Multi-Class Tversky Loss ---
        # Apply Tversky to all three land cover channels to balance recall
        t_build = self.tversky(torch.relu(preds[:, 0, :, :]), targets[:, 0, :, :])
        t_veg = self.tversky(torch.relu(preds[:, 1, :, :]), targets[:, 1, :, :])
        t_water = self.tversky(torch.relu(preds[:, 2, :, :]), targets[:, 2, :, :])

        # Average the Tversky losses so no single class dominates the optimization
        loss_tversky = (t_build + t_veg + t_water) / 3.0

        # --- 4. NEW: Height-Aware Building Masking ---
        # Apply a 5x penalty to height errors specifically where building ground truth > 10%
        build_presence_mask = (targets[:, 0, :, :] > 0.1).float()
        height_err = torch.abs(preds[:, 3, :, :] - targets[:, 3, :, :])

        # Base height error + boosted error on buildings
        loss_height_boost = torch.mean(height_err * (1.0 + 5.0 * build_presence_mask))

        # --- Combine Total Loss ---
        total_loss = (self.w_mae * loss_mae) + \
                     (self.w_ssim * loss_ssim) + \
                     (self.w_grad * loss_grad) + \
                     (self.w_structure * loss_tversky) + \
                     (self.w_structure * loss_height_boost)

        # Return tuple matching your training loop signature
        return total_loss, loss_mae, loss_ssim, loss_grad, loss_tversky


# =============================================================================
# 7A — DualPathLoss + GradNormBalancer for DualEncDualDecFusion
# =============================================================================

def _soft_dice(pred, target, smooth=1.0):
    """Soft Dice over (B,C,H,W) or (B,H,W) probability map vs binary target."""
    dims = tuple(range(1, pred.dim()))
    inter = torch.sum(pred * target, dim=dims)
    denom = torch.sum(pred, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return torch.mean(1.0 - dice)


class DualPathLoss(nn.Module):
    """
    Loss for the 7A dual-decoder model.

    Inputs: the model forward() dict (raw logits):
        fraction (B,3,H,W) -> sigmoid for fractions
        height   (B,1,H,W) -> normalised-height regression [0,1.5]
        binary   (B,1,H,W) -> aux building presence
    Target (B,4,H,W): ch0-2 fractions in [0,1], ch3 normalised height.

    forward(outputs, target) -> dict {fraction, height, binary} of scalar losses
    (unweighted; the training loop combines them, statically or via GradNorm).

    No SSIM/GDL: they regularise toward smoothness, hurting the sharp building
    boundaries that drive hard-IoU_B. Soft Dice on the shifted sigmoid is a
    continuous relaxation of hard IoU at the 0.5 threshold.
    """

    def __init__(self, dice_lambda=1.0, dice_k=5.0, huber_delta=0.5,
                 height_bg_weight=0.1, veg_height_boost=0.0, veg_boost_thresh=0.1):
        """
        veg_height_boost: extra weight on the masked Huber for vegetation pixels
            (veg_frac > veg_boost_thresh). Default 0.0 reproduces the Phase-4
            baseline exactly; set >0 (e.g. 2-3) to pull RMSE_V back under the
            3.9 m scoring floor — mirrors the veg_height_boost that helped 2A.
        """
        super().__init__()
        self.dice_lambda = dice_lambda
        self.dice_k = dice_k
        self.huber_delta = huber_delta
        self.height_bg_weight = height_bg_weight
        self.veg_height_boost = veg_height_boost
        self.veg_boost_thresh = veg_boost_thresh

    def forward(self, outputs, target):
        frac_logits = outputs["fraction"]          # (B,3,H,W)
        height_pred = outputs["height"][:, 0]      # (B,H,W)
        bin_logits = outputs["binary"][:, 0]       # (B,H,W)

        frac_target = target[:, :3]                # (B,3,H,W) in [0,1]
        height_target = target[:, 3]               # (B,H,W) normalised
        build_frac = target[:, 0]
        veg_frac = target[:, 1]

        # Fractions: MAE (calibration) + soft Dice at the 0.5 boundary.
        frac_prob = torch.sigmoid(frac_logits)
        loss_mae = torch.mean(torch.abs(frac_prob - frac_target))
        dice_pred = torch.sigmoid(self.dice_k * (frac_logits - 0.5))
        dice_tgt = (frac_target > 0.5).float()
        loss_dice = _soft_dice(dice_pred, dice_tgt)
        loss_fraction = loss_mae + self.dice_lambda * loss_dice

        # Height: Huber, masked (building OR veg present) + small bg tail.
        mask = ((build_frac > 0) | (veg_frac > 0)).float()
        huber = F.huber_loss(height_pred, height_target, delta=self.huber_delta,
                             reduction="none")
        m_sum = mask.sum().clamp_min(1.0)
        bg = 1.0 - mask
        bg_sum = bg.sum().clamp_min(1.0)
        loss_height = (huber * mask).sum() / m_sum \
            + self.height_bg_weight * (huber * bg).sum() / bg_sum

        # Vegetation-height boost: extra weighted Huber where vegetation is
        # present, to drive RMSE_V down (the 3.9 m scoring-floor term). Mean over
        # the veg-pixel count so the term is scale-comparable to the masked Huber.
        if self.veg_height_boost > 0.0:
            veg_mask = (veg_frac > self.veg_boost_thresh).float()
            veg_sum = veg_mask.sum().clamp_min(1.0)
            loss_height = loss_height \
                + self.veg_height_boost * (huber * veg_mask).sum() / veg_sum

        # Aux binary building head: BCE + soft Dice on (build_frac > 0.5).
        bin_target = (build_frac > 0.5).float()
        loss_bce = F.binary_cross_entropy_with_logits(bin_logits, bin_target)
        loss_bin_dice = _soft_dice(torch.sigmoid(bin_logits), bin_target)
        loss_binary = loss_bce + loss_bin_dice

        return {"fraction": loss_fraction, "height": loss_height, "binary": loss_binary}


class GradNormBalancer:
    """
    GradNorm task balancer (Chen et al. 2018). Learns per-task weights so each
    task's gradient magnitude (through a reference parameter in that task's
    relevant encoder) tracks a target set by its relative inverse training rate.

    Per step (call BEFORE the model loss.backward(); uses retain_graph=True):
        balancer.step(per_task_loss, ref_params)   # updates task weights
        w = balancer.weights()                     # detached, for the model loss
        model_loss = sum(w[i] * per_task_loss[name_i])
        model_loss.backward(); optimizer.step()

    The per-task gradient norms are measured as constants (detached); we balance
    the weights, not differentiate through the norm, so no second-order graph.
    """

    def __init__(self, task_names, device, alpha=1.5, lr=0.025):
        self.task_names = list(task_names)
        self.n = len(self.task_names)
        self.alpha = alpha
        self.log_w = torch.zeros(self.n, device=device, requires_grad=True)  # weights init 1
        self.opt = torch.optim.AdamW([self.log_w], lr=lr)
        self.L0 = None

    def weights(self):
        return torch.exp(self.log_w.detach())

    def weight_dict(self):
        w = self.weights()
        return {n: float(w[i]) for i, n in enumerate(self.task_names)}

    def step(self, per_task_loss, ref_params):
        losses = [per_task_loss[n] for n in self.task_names]
        if self.L0 is None:
            self.L0 = [float(l.detach()) + 1e-8 for l in losses]

        # Observed per-task grad norm through the reference param (detached).
        G_norm = []
        for n in self.task_names:
            g = torch.autograd.grad(per_task_loss[n], ref_params[n],
                                    retain_graph=True)[0]
            G_norm.append(g.norm().detach())
        G_norm = torch.stack(G_norm)

        w = torch.exp(self.log_w)               # differentiable
        G = w * G_norm
        G_avg = G.mean().detach()

        loss_ratio = torch.tensor(
            [float(losses[i].detach()) / self.L0[i] for i in range(self.n)],
            device=G.device,
        )
        r = loss_ratio / loss_ratio.mean()
        target = (G_avg * r.pow(self.alpha)).detach()
        grad_loss = torch.abs(G - target).sum()

        self.opt.zero_grad()
        grad_loss.backward()
        self.opt.step()

        # Renormalise so sum(weights) == n (keeps overall loss scale stable).
        with torch.no_grad():
            w_new = torch.exp(self.log_w)
            self.log_w.data += torch.log(torch.tensor(float(self.n), device=G.device)) \
                - torch.log(w_new.sum())

        return float(grad_loss.detach())