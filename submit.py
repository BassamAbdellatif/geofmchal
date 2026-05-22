import os
import argparse
import glob
import zipfile
import numpy as np
import config

def parse_args():
    parser = argparse.ArgumentParser(description="Package predictions into a competition-ready zip file.")
    parser.add_argument("--experiment-name", type=str, required=True, help="Name of the experiment to package.")
    return parser.parse_args()

def main():
    args = parse_args()
    exp_dir = os.path.join(config.SHARED_RUNS_DIR, args.experiment_name)
    predictions_dir = os.path.join(exp_dir, "predictions")

    if not os.path.exists(predictions_dir):
        raise RuntimeError(f"Predictions directory not found: {predictions_dir}. Please run predict.py first.")

    npy_files = glob.glob(os.path.join(predictions_dir, "*.npy"))
    if not npy_files:
        raise RuntimeError(f"No .npy files found in {predictions_dir}")

    print(f"📦 Found {len(npy_files)} prediction files.")

    # Safety Check: Load the first file and assert its shape
    print(f"Checking safety validation for {npy_files[0]}...")
    sample_npy = np.load(npy_files[0])
    if sample_npy.shape != (4, 256, 256):
        raise ValueError(
            f"\n🚨 [SAFETY CHECK FAILED] 🚨\n"
            f"Expected tensor shape to be strictly (4, 256, 256), but got {sample_npy.shape}!\n"
            f"Please check your predict.py output formatting before submitting."
        )
    print(f"✅ Shape validation passed: {sample_npy.shape}")

    zip_filename = os.path.join(exp_dir, f"submission_{args.experiment_name}.zip")
    print(f"🤐 Zipping results to {zip_filename} ...")

    # Platform-Compliant Zipping with internal 'predictions/' folder
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        for npy_path in npy_files:
            npy_filename = os.path.basename(npy_path)
            # The competition platform STRICTLY requires the internal folder to be named 'predictions/'
            arcname = f"predictions/{npy_filename}"
            zf.write(npy_path, arcname=arcname)

    print("🎉 Success! The submission is safe to upload to the competition site.")

if __name__ == "__main__":
    main()
