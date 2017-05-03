from nilearn import image
from nilearn import plotting as plt
import matplotlib.pyplot
import nibabel as nib
import os
import numpy as np


mni_template = nib.load('atlas_1000.nii')

for root, dirs, files in os.walk("./paolo_fig3/"):
    for file in files:
        if file.endswith(".nii"):
			print(os.path.join(root, file))


			statmap = nib.load(os.path.join(root, file))


			# First plot the map for the PCC: index 4 in the atlas
			display = plt.plot_stat_map(statmap,
						 cmap=plt.cm.cold_white_hot,
						 draw_cross = False, annotate = False)
							 #title="DMN nodes in MSDL atlas") 
						 #cut_coords=(0, -55, 29), annotate = False
			plt.show()
			display.savefig(os.path.join(root, file)+'.png')
			display.close()  


   
   
os.chdir("/home/asier/Desktop/AGING/motion_correction/figures/")
max_val = 0
min_val = 0.22
# get maximum value for vmax
for root, dirs, files in os.walk("./fig_data"):
	for file in files:
		if file.endswith(".nii"):
			max_i = nib.volumeutils.finite_range(nib.load(os.path.join(root, file)).get_data())[1]
			print(max_i)
			max_val = max(max_i, max_val)
# get minimum value for vmin
#for root, dirs, files in os.walk("./paolo_fig3/"):
#	for file in files:
#		if file.endswith(".nii"):
#			min_i = nib.volumeutils.finite_range(nib.load(os.path.join(root, file)).get_data())[1]
#			print min_i
#			if min_i: 
#				min_val = min(min_i, min_val)


for root, dirs, files in os.walk("./fig_data"):
	for file in files:
         if file.endswith(".nii"):
             print(os.path.join(root, file))
             statmap = nib.load(os.path.join(root, file))

             # First plot the map for the PCC: index 4 in the atlas
             plt.plot_glass_brain(statmap, threshold=0, colorbar=True,cmap=matplotlib.pyplot.cm.autumn, display_mode='lyrz', vmax = max_val, vmin = min_val, output_file = os.path.join(root, file)+'glass_white_max.png')
			
 


statmap = nib.load("/home/asier/Desktop/d.nii.gz")
disp = plt.plot_glass_brain(statmap, colorbar=True)
disp._add_lines(vmax = -3, vmin = 5)
disp.save_figure( '/home/asier/Desktop/glass_white_max.png')
			


statmap = nib.load('./paolo_fig3/internal/intConnect_FCvsSC_negNeg_numModMin_20signif_1998_1mm.nii')
plt.plot_glass_brain(statmap, threshold=0, colorbar=True,cmap=matplotlib.pyplot.cm.autumn, display_mode='lyrz', 
			vmax = max_val, vmin = min_val, output_file = os.path.join(root, file)+'glass_white_max.png')



plt.plot_glass_brain(statmap, colorbar=True,cmap=matplotlib.pyplot.cm.autumn, display_mode='lyrz', output_file ='paolo_2nd_degree.png')

