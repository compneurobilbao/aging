#!/bin/bash
#
#SBATCH -p large # Partition to submit to
#SBATCH -n 1 # Number of cores
#SBATCH --mem=50000 # Memory pool for all cores
#SBATCH -o runscript_tot.out # File to which STDOUT will be written
#SBATCH -e runscript_tot.err # File to which STDERR will be written

module load Octave/4.0.0-foss-2015a
octave --no-gui --eval "total; exit;"
module unload Octave/4.0.0-foss-2015a
