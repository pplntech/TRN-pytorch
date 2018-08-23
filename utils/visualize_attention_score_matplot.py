# Visualization of attention score and corresponding frames
'''
	run on Validation Set

	hops : multiple hops is possible
	which to show : frame images + hop possibilities + GT class + predicted class


	When store the file,
		gather the same class
		result_root
			-class1_name
				-1.png
				-143.png
				-...
			-class2_name

	Function Arguments
		0. root_dir
		1. img
		2. hop_probabilities
		3. # of frames
		4. # of hops
		5. GT_class (idx)
		6. predicted class (idx)
		7. idx2class dictionary
'''
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import os, sys
import numpy as np
import time
import datetime
import math
from scipy.misc import imsave
import scipy.io

def visualize_attention_score(root_dir, filenames, raw_imgs, hop_probabilities, num_of_frames, num_of_hops, GT_class, Pred_class, idx2class):
	'''
	raw_imgs : ndarray, (bs, num_frames * num_channel, h, w)
	hop_probabilities : ndarray, (bs, num_hops, num_frames)
	'''
	bs = raw_imgs.shape[0]
	for batch_idx in range(bs):

		img = raw_imgs[batch_idx] # (num_frames * num_channel, h, w)
		hop_prob = hop_probabilities[batch_idx] # (num_hops, num_frames)

		num_rows = num_frames + 2
		num_columns = num_hops + 1

		plt.rcParams['figure.figsize'] = (num_rows, num_columns) # (18, 6)
		fig = plt.figure(0)
		axes = [[[]for x in range(num_columns)] for y in range(num_rows)]
		for x in range(num_columns):
			for y in range(num_rows-1):
				axes[y][x] = plt.subplot2grid((num_rows, num_columns), (y,x))
		axes[num_rows-1][0] = plt.subplot2grid((num_rows, num_columns), (num_rows-1,0))
		axes[num_rows-1][1] = plt.subplot2grid((num_rows, num_columns), (num_rows-1,1), colspan=num_of_hops)

		for y in range(num_of_frames):
			axes[y+1][0].imshow(img[y,:,:,:])

		for x in range(num_of_hops):
			for y in range(num_frames):
				axes[y+1][x+1] = hop_prob[x][y] # plot text

		# plot text
		axes[num_rows-1][0] = 'GT : %s' % (idx2class[GT_class])
		axes[num_rows-1][1] = 'Predicted : %s' % (idx2class[Pred_class])

		dirs = os.path.join(root_dir,idx2class[GT_class].replace(' ','_'))
		if os.path.exists(dirs)==False:
			os.makedirs(dirs)
		fig.savefig(os.path.join(dirs, filenames[batch_idx]))
		plt.close('all')
