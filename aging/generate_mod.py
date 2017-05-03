#!/usr/bin/python

import sys
import os

try:
	os.mkdir(".job/")
	os.mkdir(".out/")
	os.mkdir(".job/.out/")
except OSError:
	pass


for i in range(20,22):
	filey = ".job/%s.job" %i
	filey = open(filey,"w")
	filey.writelines("#!/bin/bash\n")
	filey.writelines("#SBATCH --job-name=%s\n" %i)
	filey.writelines("#SBATCH --output=.out/%s.out\n" %i)
	filey.writelines("#SBATCH --error=.out/%s.err\n" %i)
	filey.writelines("#SBATCH -p large\n")
	filey.writelines("#SBATCH --mem=1000\n")
	filey.writelines("\n")
	filey.writelines("module load Octave/4.0.0-foss-2015a\n")
	filey.writelines("octave --no-gui --eval 'generate_mod(%s)'; exit;\n" %i)
	filey.writelines("module unload Octave/4.0.0-foss-2015a\n")
	filey.close()
	os.system("sbatch  " + ".job/%s.job" %i)
