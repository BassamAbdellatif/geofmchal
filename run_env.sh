#!/bin/bash

# Source conda initialization script
CONDA_SH="/home/bassam/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
else
    # Fallback to initialize conda in the shell
    eval "$(conda shell.bash hook)"
fi

# Activate the target environment
conda activate /scratch/head/geofm_env

# Run python with any arguments passed to this script
exec python "$@"
