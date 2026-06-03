"""Phase 5D byte-identical reference.

Run BEFORE editing model.py to snapshot the current DualEncDualDecFusion
forward output for a fixed seed + fixed input. After the edits, verify.py
rebuilds the model with --patch-stem-version v1 + terramind-only and confirms
max_abs_diff == 0 against this snapshot.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import build_model

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref.pt")


def make_inputs():
    g = torch.Generator().manual_seed(999)
    return {
        "alpha_earth":  torch.randn(1, 64, 256, 256, generator=g),
        "tessera":      torch.randn(1, 128, 256, 256, generator=g),
        "terramind_s1": torch.randn(1, 256, 768, generator=g),
        "terramind_s2": torch.randn(1, 256, 768, generator=g),
    }


def build():
    torch.manual_seed(1234)
    model, _ = build_model("dual_enc_dec_fusion", n_channels=64, n_classes=4)
    model.eval()
    return model


def main():
    model = build()
    batch = make_inputs()
    with torch.no_grad():
        out = model(batch)
    snap = {k: v.clone() for k, v in out.items()}
    snap["_n_params"] = sum(p.numel() for p in model.parameters())
    torch.save({"inputs": batch, "output": snap}, OUT)
    print(f"[ref] saved {OUT}")
    print(f"[ref] n_params = {snap['_n_params']}")
    for k, v in out.items():
        print(f"[ref] {k}: shape={tuple(v.shape)} mean={v.mean().item():.6f}")


if __name__ == "__main__":
    main()
