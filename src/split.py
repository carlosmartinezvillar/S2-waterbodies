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
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir',default=None,required=True,
	help='Dataset directory.')
parser.add_argument('-z','--zones',action='store_true',
	help='Choose tiles by drawing from different UTM zones.')

DEFAULT_SEED = 476
NR_TILES = 3 #nr for validation and training, respectively

def build_tree(chip_names):
	'''
	{zone}
	|-> {tile}
		|-> {raster}
			|--> [chips]
	'''
	tree = {}

	# Iterate thru chips
	for chip in chip_names:
		#Get strings
		raster = chip[0:-11]
		tile   = chip[-16:-11]
		zone   = chip[-16:-13]

		# Append to tree
		if zone in tree:
			if tile in tree[zone]:
				if raster in tree[zone][tile]:
					tree[zone][tile][raster].append(chip)
				else:
					tree[zone][tile][raster] = []
					tree[zone][tile][raster].append(chip)
			else:
				tree[zone][tile] = {raster:[chip]}
		else:
			tree[zone] = {tile:{raster:[chip]}}

	return tree


if __name__ == '__main__': #THIS IS ALL EASIER WITH A TREE?

	#CHECK DIR
	args = parser.parse_args()
	assert os.path.isdir(args.data_dir), "split.py: Incorrect data dir argument."

	#FIX SEED
	np.random.seed(DEFAULT_SEED)
	random.seed(DEFAULT_SEED)

	#GET CHIP NAMES
	band_files  = sorted(glob.glob("*_B0X.tif",root_dir=args.data_dir))
	# label_files = sorted(glob.glob("*_LBL.tif",root_dir=chip_dir))	
	chips       = [_[0:-8] for _ in band_files] #19456 X 53

	#TREE
	tree = build_tree(chips)

	#GET TEST TILES
	test_zones = np.random.choice(sorted(tree),NR_TILES,replace=False)
	test_tiles = []
	for z in test_zones:
		test_tiles.append(np.random.choice(sorted(tree[z]),1,replace=False)[0])

	#GET VALIDATION+TRAIN TILES
	trainvalidate_tiles = []
	for z in tree:
		for t in tree[z]:
			if t not in test_tiles:
				trainvalidate_tiles.append(t)
	validate_tiles = np.random.choice(sorted(trainvalidate_tiles),NR_TILES,replace=False)
	training_tiles = set(trainvalidate_tiles).difference(validate_tiles)

	#LOG SPLIT
	print(f"TEST TILES:       {test_tiles}")
	print(f"VALIDATION TILES: {validate_tiles}")
	print(f"TRAIN TILES:      {training_tiles}")
	with open('../cfg/split_summary.txt','w+') as fp:
		fp.write(f"TEST TILES:       {test_tiles}\n")
		fp.write(f"VALIDATION TILES: {validate_tiles}\n")
		fp.write(f"TRAIN TILES:      {training_tiles}\n")

	#SAVE IN SEPARATAE FOLDERS
	data_dir = args.data_dir
	new_dir = '/'.join([*data_dir.split('/')[0:-1],'chips_sorted'])
	os.mkdir(new_dir)
	os.mkdir(f'{new_dir}/training')
	os.mkdir(f'{new_dir}/validation')
	os.mkdir(f'{new_dir}/testing')

	print("COPYING VALIDATION FILES")
	for tile in validate_tiles:
		tile_files = glob.glob(f"*_T{tile}_*.tif",root_dir=args.data_dir)
		for file in tile_files:
			shutil.copy(f"{args.data_dir}/{file}",f"{new_dir}/validation/{file}",follow_symlinks=False)

	print("COPYING TESTING FILES")
	for tile in test_tiles:
		tile_files = glob.glob(f"*_T{tile}_*.tif",root_dir=args.data_dir)
		for file in tile_files:
			shutil.copy(f"{args.data_dir}/{file}",f"{new_dir}/testing/{file}",follow_symlinks=False)

	print("COPYING TRAINING FILES")
	for tile in training_tiles:
		tile_files = glob.glob(f"*_T{tile}_*.tif",root_dir=args.data_dir)
		for file in tile_files:
			shutil.copy(f"{args.data_dir}/{file}",f"{new_dir}/training/{file}",follow_symlinks=False)

	################################################################################
	# OLD VERSION
		################################################################################
			################################################################################
	# all_rasters = np.array([i[0:-11] for i in chips])   #19456 x 42
	# all_tiles   = np.array([i[-16:-11] for i in chips]) #19456 x 6
	# all_zones   = np.array([i[-16:-13] for i in chips]) #19456 x 3

	# c_per_r_str, c_per_r_cnt = np.unique(all_rasters,return_counts=True) #726 x 42
	# c_per_t_str, c_per_t_cnt = np.unique(all_tiles,return_counts=True)   #21 x 6
	# c_per_z_str, c_per_z_cnt = np.unique(all_zones,return_counts=True)   #6 x 3

	# r_per_t_str, r_per_t_cnt = np.unique([_[-5:] for _ in c_per_r_str],return_counts=True) #21 x 6
	# t_per_z_str, t_per_z_cnt = np.unique([_[0:3] for _ in r_per_t_str],return_counts=True) #6 x 3

	# #CHOOSE TWO ZONES
	# test_zones = np.random.choice(c_per_z_str,NR_TILES,replace=False)

	# #CHOOSE A TILE FOR EACH ZONE
	# test_mask = []
	# for z in test_zones:
	# 	tiles_in_zone_mask = np.isin([_[0:3] for _ in c_per_t_str], z) #21 bool
	# 	tiles_in_zone      = c_per_t_str[tiles_in_zone_mask]
	# 	tile               = np.random.choice(tiles_in_zone,1,replace=False)
	# 	tile_mask          = np.isin(all_tiles,tile)
	# 	test_mask.append(tile_mask)

	# # TEST SET -- UNION/OR OP OF TWO TILES SELECTED
	# # test_mask = test_mask[0] + test_mask[1] + test_mask[2] #19456
	# # test_mask = np.sum(test_mask,axis=0)
	# test_mask = np.row_stack(test_mask).any(axis=0)
	# test_idxs = np.where(test_mask)[0]

	# # NEW TRAINING/VALIDATION DATASET
	# trva_set_mask = ~test_mask #19456

	# #VALIDATION SET -- CHOOSE 2 TILES AT RANDOM
	# trva_tiles    = np.array(all_tiles)[trva_set_mask]
	# validation_tiles = np.random.choice(np.unique(trva_tiles),NR_TILES,replace=False)
	# validation_mask  = np.isin(all_tiles,validation_tiles)
	# validation_idxs  = np.where(validation_mask)[0]

	# #TRAINING SET -- EXCLUDE EVERYTHING ELSE
	# training_mask = ~(validation_mask+test_mask) #19456
	# training_idxs = np.where(training_mask)[0]

  	# #SAVE TO FILE
	# with open('../cfg/training.txt','w+') as fp_tr:
	# 	for i in training_idxs:
	# 		fp_tr.write(str(i) + '\n')

	# with open('../cfg/validation.txt','w+') as fp_va:
	# 	for i in validation_idxs:
	# 		fp_va.write(str(i) + '\n')

	# with open('../cfg/testing.txt','w+') as fp_te:
	# 	for i in test_idxs:
	# 		fp_te.write(str(i) + '\n')


	# test_tiles = np.unique(all_tiles[test_mask])
	# train_tiles = np.unique(all_tiles[training_mask])
	# with open('../cfg/split_summary.txt','w+') as fp:
	# 	fp.write(f"TEST TILES:       {test_tiles}\n")
	# 	fp.write(f"VALIDATION TILES: {validation_tiles}\n")
	# 	fp.write(f"TRAIN TILES:      {train_tiles}")
	# print(f"TEST TILES:       {test_tiles}")
	# print(f"VALIDATION TILES: {validation_tiles}")
	# print(f"TRAIN TILES:      {train_tiles}")

	# #SAVE IN SEPARATAE FOLDERS
	# data_dir = args.data_dir
	# new_dir = '/'.join([*data_dir.split('/')[0:-1],'chips_sorted'])
	# os.mkdir(new_dir)
	# os.mkdir(f'{new_dir}/training')
	# os.mkdir(f'{new_dir}/validation')
	# os.mkdir(f'{new_dir}/testing')

	# print("COPYING VALIDATION FILES")
	# for tile in validate_tiles:
	# 	tile_files = glob.glob(f"*_T{tile}_*.tif",root_dir=args.data_dir)
	# 	for file in tile_files:
	# 		shutil.copy(f"{args.data_dir}/{file}",f"{new_dir}/validation/{file}",follow_symlinks=False)

	# print("COPYING TESTING FILES")
	# for tile in test_tiles:
	# 	tile_files = glob.glob(f"*_T{tile}_*.tif",root_dir=args.data_dir)
	# 	for file in tile_files:
	# 		shutil.copy(f"{args.data_dir}/{file}",f"{new_dir}/testing/{file}",follow_symlinks=False)

	# print("COPYING TRAINING FILES")
	# for tile in training_tiles:
	# 	tile_files = glob.glob(f"*_T{tile}_*.tif",root_dir=args.data_dir)
	# 	for file in tile_files:
	# 		shutil.copy(f"{args.data_dir}/{file}",f"{new_dir}/training/{file}",follow_symlinks=False)

