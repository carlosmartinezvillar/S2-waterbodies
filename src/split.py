'''
split.py
--------
This script runs a training/validation/split
'''
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir',default=None,required=True,help='Dataset directory.')
parser.add_argument('-z','--zones',action='store_true',help='Choose tiles by drawing from different UTM zones.')

DEFAULT_SEED = 476

if __name__ == '__main__':

	args = parser.parser_args()
	assert os.path.isdir(args.data_dir), "split.py: Incorrect data dir argument."

	band_files  = sorted(glob.glob("*_B0X.tif",root_dir=chip_dir))
	label_files = sorted(glob.glob("*_LBL.tif",root_dir=chip_dir))	

    np.random.seed(DEFAULT_SEED)
    random.seed(DEFAULT_SEED)



	all_rasters = [i[0:-19] for i in band_files] #19456
	all_tiles = [i[-25:-19] for i in band_files] #19456
	
	unique_r,r_inv,r_cnt = np.unique(all_rasters,return_inverse=True,return_counts=True) #726
	tiles   = [i[-6:] for i in unique_r] #726
	unique_t,t_inv,t_cnt = np.unique(tiles,return_inverse=True,return_counts=True) #21

	# CHOOSE 4 TILES AT RANDOM
	tile_choices     = np.random.choice(unique_t,4,replace=False)
	validation_tiles = tile_choices[0:2]
	test_tiles       = tile_choices[2:]

	validation_mask  = np.isin(all_tiles,validation_tiles)
	validation_idx   = np.where(validation_mask)[0]
	test_mask        = np.isin(all_tiles,test_tiles)
	test_idx         = np.where(test_mask)[0]
	training_mask    = ~(validation_mask+test_mask)
	training_idx     = np.where(training_mask)[0]
	
	return training_idx,validation_idx,test_idx
