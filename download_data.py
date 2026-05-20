import os
from eotdl.datasets import download_dataset

def fetch_data():
    # We will store the data inside a 'data' folder within the repo
    base_data_dir = "/mnt/head/users/bassam/data/geofmdata"
    os.makedirs(base_data_dir, exist_ok=True)
    
    dataset_name = "embed2heights -v 1" # Verify this name on the portal if it fails
    
    print(f"Downloading {dataset_name} to {base_data_dir}...")
    try:
        # eotdl will automatically download and extract the dataset here
        download_dataset(dataset_name, path=base_data_dir)
        print("\nDownload complete! Check the ./data folder for train/test splits.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_data()
