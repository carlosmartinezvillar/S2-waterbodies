import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir',default=None,required=True,help='Dataset directory.')
parser.add_argument('-z','--zones',help='Choose tiles by randomly drawing in UTM zones.')

if __name__ == '__main__':

	args = parser.parser_args()
	assert os.path.isdir(args.data_dir), "split.py: Incorrect data dir argument."

	band_files  = sorted(glob.glob("*_B0X.tif",root_dir=chip_dir))
	label_files = sorted(glob.glob("*_LBL.tif",root_dir=chip_dir))	

    np.random.seed(476)
    random.seed(476)

    