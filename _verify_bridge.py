"""Reproducibility check for the --no-height-bridge guard (Phase 5C, Change 2).

Run BEFORE editing model.py with `--save` to capture the pre-edit reference;
run AFTER editing (default, use_height_bridge=True) to confirm byte-identical
output. Same seed -> identical weights -> identical forward output.
"""
import sys, torch
from core.model import build_model

REF = "/tmp/ref_dualenc.pt"
SEED = 1234


def make_batch():
    g = torch.Generator().manual_seed(99)
    return {
        "alpha_earth":  torch.randn(2, 64, 256, 256, generator=g),
        "tessera":      torch.randn(2, 128, 256, 256, generator=g),
        "terramind_s1": torch.randn(2, 256, 768, generator=g),
        "terramind_s2": torch.randn(2, 256, 768, generator=g),
    }


def run():
    torch.manual_seed(SEED)
    model, _ = build_model("dual_enc_dec_fusion", n_channels=64, n_classes=4)
    model.eval()
    batch = make_batch()
    with torch.no_grad():
        out = model(batch)
    return {k: v for k, v in out.items() if v is not None}


if __name__ == "__main__":
    out = run()
    if "--save" in sys.argv:
        torch.save(out, REF)
        print("saved reference:", {k: tuple(v.shape) for k, v in out.items()})
    else:
        ref = torch.load(REF)
        ok = True
        for k in ref:
            same = torch.equal(out[k], ref[k])
            md = (out[k] - ref[k]).abs().max().item()
            print(f"  {k}: identical={same}  max_abs_diff={md:.3e}")
            ok = ok and same
        print("RESULT:", "IDENTICAL ✓" if ok else "DIFFERENT ✗")
        sys.exit(0 if ok else 1)
