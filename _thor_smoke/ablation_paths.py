"""Phase 5E #4/#1 model-level smoke: forward+backward + param counts for the
routing and fraction-bridge ablations (synthetic input, no data/cache)."""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import build_model

THOR = ("terramind_s1", "terramind_s2", "thor_s1", "thor_s2")
CONFIGS = [
    ("base(sensor)",   dict(patch_inputs=THOR, patch_stem_version="v2")),
    ("s1-both",        dict(patch_inputs=THOR, patch_stem_version="v2", patch_routing="s1-both")),
    ("all-both",       dict(patch_inputs=THOR, patch_stem_version="v2", patch_routing="all-both")),
    ("frac-bridge",    dict(patch_inputs=THOR, patch_stem_version="v2", use_fraction_bridge=True)),
    ("s1both+fbridge", dict(patch_inputs=THOR, patch_stem_version="v2", patch_routing="s1-both", use_fraction_bridge=True)),
]


def batch(names, B=2):
    g = torch.Generator().manual_seed(7)
    b = {"alpha_earth": torch.randn(B, 64, 256, 256, generator=g),
         "tessera": torch.randn(B, 128, 256, 256, generator=g)}
    for n in names:
        b[n] = torch.randn(B, 256, 768, generator=g)
    return b


def main():
    for tag, kw in CONFIGS:
        torch.manual_seed(0)
        m, _ = build_model("dual_enc_dec_fusion", 64, 4, **kw)
        m.train()
        out = m(batch(kw["patch_inputs"]))
        loss = sum(v.float().pow(2).mean() for v in out.values())
        loss.backward()
        fin = all(torch.isfinite(out[k]).all().item() for k in out)
        print(f"[{tag:16s}] params={sum(p.numel() for p in m.parameters())/1e6:7.3f}M "
              f"finite={fin} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
