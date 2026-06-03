"""Phase 5D byte-identical verification.

Rebuilds DualEncDualDecFusion with --patch-stem-version v1 + terramind-only
(+ xattn_heads=4) using the same seed and the same fixed input captured by
ref_capture.py, and asserts max_abs_diff == 0 against the pre-edit snapshot.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import build_model

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref.pt")


def main():
    ref = torch.load(REF, weights_only=False)
    batch = ref["inputs"]
    ref_out = ref["output"]

    torch.manual_seed(1234)
    model, _ = build_model(
        "dual_enc_dec_fusion", n_channels=64, n_classes=4,
        patch_inputs=("terramind_s1", "terramind_s2"),
        patch_stem_version="v1", xattn_heads=4,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        out = model(batch)

    print(f"[verify] ref n_params={ref_out['_n_params']}  new n_params={n_params}  "
          f"match={ref_out['_n_params'] == n_params}")
    worst = 0.0
    for k in ("fraction", "height", "binary"):
        d = (out[k] - ref_out[k]).abs().max().item()
        worst = max(worst, d)
        print(f"[verify] {k}: max_abs_diff = {d:.3e}")
    print(f"[verify] OVERALL max_abs_diff = {worst:.3e}  "
          f"{'PASS (byte-identical)' if worst == 0.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
