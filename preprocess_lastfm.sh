#!/bin/bash

#SBATCH --job-name=lastfm_prep      # Name of the job
#SBATCH --partition=cpu            # Use the CPU partition (faster queue for this)
#SBATCH --nodes=1                  # Run on a single node
#SBATCH --ntasks=1                 # Run a single task
#SBATCH --cpus-per-task=4          # Request 4 CPU cores (good for pandas)
#SBATCH --mem=32G                  # Request 32GB of RAM (adjust based on file size)
#SBATCH --time=02:00:00            # Time limit (4 hours)
#SBATCH --output=logs/prep_%j.log  # Standard output log (%j = Job ID)

# 1. Load the Conda module and activate your environment
module load conda/latest
source activate e4srec

# 2. Run your preprocessing script
# Make sure the paths inside your python script point to the /work directory
python preprocess_lastfm.py