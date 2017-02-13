#!/bin/bash
#
#SBATCH -p large # Partition to submit to
#SBATCH -n 40 # Number of cores
#SBATCH --mem=50000 # Memory pool for all cores
#SBATCH -o out # File to which STDOUT will be written
#SBATCH -e err # File to which STDERR will be written

source ~/.bashrc
python data_prep.py
