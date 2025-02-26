import pyyaml
import subprocess
import argparse

####################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--params',default=None,help='Path to json file listing hyperparameters.')
args = parser.parse_args()

#launch via kubectl
def launch_jobs(start,end):
	for job in range(start,end):
		pass

	old_job_str = obj['metadata']['name']
	new_job_str = old_job_str.replace('-0',f'-{ROW}')

	old_cmd_str = obj['spec']['template']['spec']['containers'][0]['args'][0]
	new_cmd_str = old_str.replace('--row 0;',f'--row {ROW};')



if __name__ == '__main__':

	if args.params is None:
		print("No .json given for hyperparameter list.")
		sys.exit(1)

	json_path = args.params
	assert os.path.isfile(args.params), "kjobs.py: INCORRECT JSON FILE PATH"

	with open(args.params,'r') as fp:
		HP_LIST = json.load(fp)

	assert len(HP_LIST) > 0, "kjobs.py: GOT EMPTY JSON FILE."
	
	N = len(HP_LIST)