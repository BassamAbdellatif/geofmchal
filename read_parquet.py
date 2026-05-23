import pandas as pd

# Load the parquet file
df = pd.read_parquet("/mnt/head/users/bassam/data/geofmdata/embed2heights/catalog.v1.parquet")

# 1. Print all column names to find the exact geometry keys
print("--- Column Names ---")
print(df.columns.tolist())

# 2. Look at the first 3 rows to see the data format
print("\n--- First 3 Rows ---")
print(df.head(3))

# 3. Check the identifier column (usually 'id' or 'patch_id')
# This is crucial for matching the metadata to your AlphaEarth .pt/.tif files