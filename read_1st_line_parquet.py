import pandas as pd
import pprint

# Load the parquet file
df = pd.read_parquet("/mnt/head/users/bassam/data/geofmdata/embed2heights/catalog.v1.parquet")

# Extract the very first row as a standard Python dictionary
first_row = df.iloc[0].to_dict()

# Pretty-print the dictionary so we can read the nested JSON
print("--- FULL METADATA STRUCTURE FOR PATCH 1 ---")
pprint.pprint(first_row, depth=4)

print('--------------------------------')
# Filter out the README and grab the first real patch
real_patches = df[df['id'] != 'README.md']
first_real_patch = real_patches.iloc[0].to_dict()

print("--- FULL METADATA STRUCTURE FOR A REAL PATCH ---")
pprint.pprint(first_real_patch, depth=4)