"""Confirm training_params.txt -> predict.py architecture round-trip + legacy remap."""
import os
import sys
import tempfile
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import build_model
import predict as P


def write_params(exp, patch_inputs, stem, heads, no_bridge=False):
    with open(os.path.join(exp, "training_params.txt"), "w") as f:
        f.write("--- EXPERIMENT: rt ---\n")
        f.write("MODEL_TYPE: dual_enc_dec_fusion\n")
        f.write(f"PATCH_INPUTS: {patch_inputs}\n")
        f.write(f"PATCH_STEM_VERSION: {stem}\n")
        f.write(f"XATTN_HEADS: {heads}\n")
        f.write(f"NO_HEIGHT_BRIDGE: {no_bridge}\n")


def reconstruct(params):
    """Mirror the reconstruction logic in predict.predict_7a."""
    patch_names = [p.strip() for p in params.get("PATCH_INPUTS", "terramind_s1,terramind_s2").split(",") if p.strip()]
    stem = params.get("PATCH_STEM_VERSION", "v1").strip()
    heads = int(params.get("XATTN_HEADS", "4"))
    bridge = params.get("NO_HEIGHT_BRIDGE", "False").strip().lower() != "true"
    model, _ = build_model("dual_enc_dec_fusion", 64, 4, use_height_bridge=bridge,
                           patch_inputs=patch_names, patch_stem_version=stem, xattn_heads=heads)
    return model, patch_names, stem, heads, bridge


def test_roundtrip(patch_inputs, stem, heads):
    with tempfile.TemporaryDirectory() as exp:
        m1, _ = build_model("dual_enc_dec_fusion", 64, 4,
                            patch_inputs=[p.strip() for p in patch_inputs.split(",")],
                            patch_stem_version=stem, xattn_heads=heads)
        torch.save(m1.state_dict(), os.path.join(exp, "model_best.pth"))
        write_params(exp, patch_inputs, stem, heads)
        params = P.load_experiment_params(exp)
        m2, names, s, h, bridge = reconstruct(params)
        state = torch.load(os.path.join(exp, "model_best.pth"), map_location="cpu")
        state = P._remap_legacy_7a_state_dict(state, m2)
        m2.load_state_dict(state)  # raises on mismatch
        print(f"[roundtrip] '{patch_inputs}' stem={s} heads={h} -> OK "
              f"(names={names}, n_params={sum(p.numel() for p in m2.parameters())/1e6:.3f}M)")


def test_legacy_remap():
    # Old checkpoint: terramind-only v1 with s1_token_stem / s2_token_stem keys.
    m_old, _ = build_model("dual_enc_dec_fusion", 64, 4,
                          patch_inputs=["terramind_s1", "terramind_s2"], patch_stem_version="v1")
    sd = m_old.state_dict()
    legacy = {}
    for k, v in sd.items():
        if k.startswith("token_stems.terramind_s1."):
            legacy["s1_token_stem." + k[len("token_stems.terramind_s1."):]] = v
        elif k.startswith("token_stems.terramind_s2."):
            legacy["s2_token_stem." + k[len("token_stems.terramind_s2."):]] = v
        else:
            legacy[k] = v
    assert any(k.startswith("s1_token_stem.") for k in legacy)
    m_new, _ = build_model("dual_enc_dec_fusion", 64, 4,
                          patch_inputs=["terramind_s1", "terramind_s2"], patch_stem_version="v1")
    remapped = P._remap_legacy_7a_state_dict(legacy, m_new)
    m_new.load_state_dict(remapped)  # raises on mismatch
    print("[legacy]    pre-5D s1/s2_token_stem checkpoint -> loads via remap: OK")


def main():
    test_roundtrip("terramind_s1,terramind_s2", "v1", 4)
    test_roundtrip("terramind_s1,terramind_s2,thor_s1,thor_s2", "v2", 2)
    test_roundtrip("terramind_s1,terramind_s2,thor_s2", "v2", 4)
    test_legacy_remap()
    print("ALL ROUND-TRIP CHECKS PASSED")


if __name__ == "__main__":
    main()
