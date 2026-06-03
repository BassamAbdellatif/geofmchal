"""Phase 5D smoke runner: runs ONE config through the real train_7a (batch 4,
1 epoch, 2 train batches) and reports exact peak GPU memory. One config per
process so the CUDA peak is clean. Usage: run_smoke.py <tag>"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE = "/home/bassam/nvme_cache/cache7a"
COMMON = [
    "--model-type", "dual_enc_dec_fusion",
    "--pixel-inputs", "alpha_earth,tessera",
    "--no-use-gradnorm",                       # spec wrote '--use-gradnorm False' (rejected by argparse)
    "--static-weights", "0.65,0.64,1.70",
    "--veg-height-boost", "0.0",
    "--cache-dir", CACHE,
    "--num-workers", "4",
    "--batch-size", "4", "--epochs", "1", "--max-batches", "2",
    "--cv-fold", "0", "--seed", "0",
]
CONFIGS = {
    "smoke_baseline_v1": ["--patch-inputs", "terramind_s1,terramind_s2", "--patch-stem-version", "v1"],
    "smoke_thor_full":   ["--patch-inputs", "terramind_s1,terramind_s2,thor_s1,thor_s2", "--patch-stem-version", "v2"],
    "smoke_thor_s2":     ["--patch-inputs", "terramind_s1,terramind_s2,thor_s2", "--patch-stem-version", "v2"],
}


def main():
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    tag = sys.argv[1]
    from train import parse_args, train_7a

    sys.argv = ["train.py"] + COMMON + CONFIGS[tag] + ["--experiment-name", tag]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    args = parse_args()
    train_7a(args)
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[SMOKE {tag}] PEAK_GPU_ALLOC = {peak:.3f} GB")


if __name__ == "__main__":
    main()
