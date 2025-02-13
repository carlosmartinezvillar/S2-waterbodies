'''
split.py
--------
This script runs a training/validation/split


testing, validation split:
Test dataset.
1. Choose 2/6 UTM zones at random, without replacement.
2. Choose a tile with (a number of raster below the median) at random for each of the two UTM zones.

Validation dataset
3. Of the remaining dataset, choose 2 tiles at random and set as validation. Set the rest as training.
x - 3. Off remaining set, choose 2/6 UTM zones at random, without replacement.
x - 4. Choose a tile at random for each of the two UTM zones. 

'''
import argparse
import os
import numpy as np
import random
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir',default=None,required=True,help='Dataset directory.')
parser.add_argument('-z','--zones',action='store_true',help='Choose tiles by drawing from different UTM zones.')

DEFAULT_SEED = 476


if __name__ == '__main__': #THIS IS ALL EASIER WITH A TREE?

	args = parser.parse_args()
	assert os.path.isdir(args.data_dir), "split.py: Incorrect data dir argument."

	np.random.seed(DEFAULT_SEED)
	random.seed(DEFAULT_SEED)

	band_files  = sorted(glob.glob("*_B0X.tif",root_dir=args.data_dir))
	# label_files = sorted(glob.glob("*_LBL.tif",root_dir=chip_dir))	
	chips       = [_[0:-8] for _ in band_files] #19456 X 53
	all_rasters = [i[0:-11] for i in chips]   #19456 x 42
	all_tiles   = [i[-16:-11] for i in chips] #19456 x 6
	all_zones   = [i[-16:-13] for i in chips] #19456 x 3

	c_per_r_str, c_per_r_cnt = np.unique(all_rasters,return_counts=True) #726 x 42
	c_per_t_str, c_per_t_cnt = np.unique(all_tiles,return_counts=True)   #21 x 6
	c_per_z_str, c_per_z_cnt = np.unique(all_zones,return_counts=True)   #6 x 3

	r_per_t_str, r_per_t_cnt = np.unique([_[-5:] for _ in c_per_r_str],return_counts=True) #21 x 6
	t_per_z_str, t_per_z_cnt = np.unique([_[0:3] for _ in r_per_t_str],return_counts=True) #6 x 3

	#CHOOSE TWO ZONES
	test_zones = np.random.choice(c_per_z_str,2,replace=False)

	#CHOOSE A TILE FOR EACH ZONE
	test_mask = []
	for z in test_zones:
		tiles_in_zone_mask = np.isin([_[0:3] for _ in c_per_t_str], z) #21 bool
		tiles_in_zone      = c_per_t_str[tiles_in_zone_mask]
		tile               = np.random.choice(tiles_in_zone,1,replace=False)
		tile_mask          = np.isin(all_tiles,tile)
		test_mask.append(tile_mask)

	# TEST SET -- UNION/OR OP OF TWO TILES SELECTED
	test_mask = test_mask[0] + test_mask[1] #19456
	test_idxs = np.where(test_mask)[0]

	# NEW TRAINING/VALIDATION DATASET
	trva_set_mask = ~test_mask #19456

	#VALIDATION SET -- CHOOSE 2 TILES AT RANDOM
	trva_tiles    = np.array(all_tiles)[trva_set_mask]
	validation_tiles = np.random.choice(np.unique(trva_tiles),2,replace=False)
	validation_mask  = np.isin(all_tiles,validation_tiles)
	validation_idxs  = np.where(validation_mask)[0]

	#TRAINING SET -- EXCLUDE EVERYTHING ELSE
	training_mask = ~(validation_mask+test_mask) #19456
	training_idxs = np.where(training_mask)[0]

  	#SAVE TO FILE
	with open('../cfg/training.txt','w+') as fp_tr:
		for i in training_idxs:
			fp_tr.write(str(i) + '\n')

	with open('../cfg/validation.txt','w+') as fp_va:
		for i in validation_idxs:
			fp_va.write(str(i) + '\n')

	with open('../cfg/testing.txt','w+') as fp_te:
		for i in test_idxs:
			fp_te.write(str(i) + '\n')

