import yaml
import subprocess
import argparse
import os
import json

####################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--spec',default=None,required=True,help='Path to original YAML to base calls off')
parser.add_argument('--params',default=None,required=True,help='Path to json file listing hyperparameters.')
parser.add_argument('--start',type=int,default=0,help='Starting row')
parser.add_argument('--end',type=int,default=1,help='End row')
args = parser.parse_args()

#launch via kubectl
def launch_jobs(template_path,start,end,hp_list):

	with open(template_path,'r') as fp:
		obj = yaml.safe_load(fp)

	old_job_str = obj['metadata']['name']
	old_cmd_str = obj['spec']['template']['spec']['containers'][0]['args'][0]

	for row in range(start,end):
		model_id = hp_list[row]['ID']
		new_job_str = old_job_str.replace('-0',f'-{model_id}')
		new_cmd_str = old_cmd_str.replace('--row 0;',f'--row {row};')

		obj['metadata']['name'] = new_job_str
		obj['spec']['template']['spec']['containers'][0]['args'][0] = new_cmd_str

		with open('../cfg/temp.yml','w+') as fp_temp:
			yaml.dump(obj,fp_temp)

		out = subprocess.run(f"kubectl create -f ../cfg/temp.yml",capture_output=True,text=True,shell=True)
		if out.returncode != 0:
			print(f"ERROR launching job for model listed in row {row}")
			print(out.stderr)
		print(out.stdout)

	os.remove('../cfg/temp.yml')

if __name__ == '__main__':

	if args.params is None:
		print("No .json given for hyperparameter list.")
		sys.exit(1)

	assert os.path.isfile(args.params), "kjobs.py: INCORRECT JSON FILE PATH"
	with open(args.params,'r') as fp:
		HP_LIST = json.load(fp)

	N = len(HP_LIST)
	assert N > 0, "kjobs.py: GOT EMPTY JSON FILE."
	assert args.end <= N, "kjobs.py: END INDEX IN ARGS OUT OF RANGE"
	assert args.end > args.start, "kjobs.py: INCORRECT INDICES IN ARGS"

	start_index = args.start

	if args.end < N:
		end_index = args.end
	else:
		end_index = N #for range(start,end) which already n-1

	assert os.path.isfile(args.spec), "kjobs.py: INCORRECT YAML TEMPLATE PATH IN SPEC ARG"

	launch_jobs(args.spec,start_index,end_index,HP_LIST)
