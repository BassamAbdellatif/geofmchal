"""Model-level smoke for Phase 5D routing variants (synthetic input, no data)."""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import build_model

CONFIGS = [
    ("baseline_v1", ("terramind_s1", "terramind_s2"), "v1"),
    ("v2stem",      ("terramind_s1", "terramind_s2"), "v2"),
    ("thor_full",   ("terramind_s1", "terramind_s2", "thor_s1", "thor_s2"), "v2"),
    ("thor_s1",     ("terramind_s1", "terramind_s2", "thor_s1"), "v2"),
    ("thor_s2",     ("terramind_s1", "terramind_s2", "thor_s2"), "v2"),
    ("nostem_v1",   ("terramind_s1", "terramind_s2", "thor_s1", "thor_s2"), "v1"),
]


def make_batch(names, B=2):
    g = torch.Generator().manual_seed(7)
    b = {
        "alpha_earth": torch.randn(B, 64, 256, 256, generator=g),
        "tessera":     torch.randn(B, 128, 256, 256, generator=g),
    }
    for n in names:
        b[n] = torch.randn(B, 256, 768, generator=g)
    return b


def main():
    for tag, names, ver in CONFIGS:
        torch.manual_seed(0)
        model, _ = build_model("dual_enc_dec_fusion", 64, 4,
                               patch_inputs=names, patch_stem_version=ver, xattn_heads=4)
        model.train()
        n = sum(p.numel() for p in model.parameters())
        batch = make_batch(names)
        out = model(batch)
        loss = sum(v.float().pow(2).mean() for v in out.values())
        loss.backward()
        finite = all(torch.isfinite(v).all().item() for v in out.values())
        print(f"[{tag:12s}] params={n/1e6:7.3f}M  "
              f"frac={tuple(out['fraction'].shape)} height={tuple(out['height'].shape)} "
              f"binary={tuple(out['binary'].shape)}  finite={finite}  loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
