import pyyaml
import subprocess
import argparse

####################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--params',default=None,help='Path to json file listing hyperparameters.')

#launch via kubectl
def launch
for job in range(N_JOBS):
	pass
	subprocess."kubectl create -f job_train.yml"


if __name__ == '__main__':

	args = parser.parse_args()

	if args.params is None:
		print("No .json given for hyperparameter list.")
		sys.exit(1)

	json_path = args.params