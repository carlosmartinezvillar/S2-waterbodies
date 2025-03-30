import yaml
import subprocess as sp
import argparse
import os
import json
import sys

####################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--spec',default=None,required=False,help='Path to original YAML to base calls off')
parser.add_argument('--params',default=None,required=False,help='Path to json file listing hyperparameters.')
parser.add_argument('--start',type=int,default=None,help='Starting row (zero indexed)')
parser.add_argument('--end',type=int,default=None,help='End row zero-indexed using [0,end)')
parser.add_argument('--clear',required=False,action='store_true',help='Delete jobs')
args = parser.parse_args()

#launch via kubectl
def launch_jobs(template_path,start,end,hp_list):
	'''
	Opens a template YAML for the job. For each number in the range of [start,end) launches a
	kubernetes job. Each job is named to match the model id listed in the hyperparameter json
	file.
	'''

	#base template
	with open(template_path,'r') as fp:
		obj = yaml.safe_load(fp)

	#base strings
	old_job_str = obj['metadata']['name']
	old_cmd_str = obj['spec']['template']['spec']['containers'][0]['args'][0]

	for row in range(start,end):

		#Set the new strings for this row
		model_id = hp_list[row]['ID'] #GET ID in json row
		new_job_str = old_job_str.replace('-0',f'-{model_id}')
		new_cmd_str = old_cmd_str.replace('--row 0;',f'--row {row};')

		#assign new strings to object
		obj['metadata']['name'] = new_job_str
		obj['spec']['template']['spec']['containers'][0]['args'][0] = new_cmd_str

		#create a temp file to pass to the kubernetes command
		with open('../cfg/temp.yml','w+') as fp_temp:
			yaml.dump(obj,fp_temp)

		#run
		out = sp.run("kubectl create -f ../cfg/temp.yml",capture_output=True,text=True,shell=True)
		if out.returncode != 0:
			print(f"ERROR launching job for model listed in row {row}")
			print(out.stderr)
		print(f"STDOUT:{out.stdout}")

	#clear temp file
	os.remove('../cfg/temp.yml')


def clear_jobs(start,end):
	'''
	Get the list of jobs from the cluster and remove them.
	'''
	out   = sp.run(f"kubectl get jobs",capture_output=True,text=True,shell=True)
	lines = out.stdout.split('\n')
	jobs  = [line.split()[0] for line in lines[1:-1]]
	jobnr = [int(job.split('-')[-1]) for job in jobs]

	if (start is None) and (end is None): #delete all
		for job in jobs:
			del_out = sp.run(f"kubectl delete job {job}",capture_output=True,text=True,shell=True)
			print(del_out.stdout)
	
	if (start is None) and (end is not None): #delete (0,end]
		for i in jobnr[:jobnr.index(end)+1]:
			del_out = sp.run(f"kubectl delete job train-job-{i}",capture_output=True,text=True,shell=True)
			print(del_out.stdout)

	if (start is not None) and (end is None): #delete [start,N)
		for i in jobnr[jobnr.index(start):]:
			del_out = sp.run(f"kubectl delete job train-job-{i}",capture_output=True,text=True,shell=True)
			print(del_out.stdout)

	if (start is not None) and (end is not None): #delete [start,end]
		for i in jobnr[jobnr.index(start):jobnr.index(end)+1]:
			del_out = sp.run(f"kubectl delete job train-job-{i}",capture_output=True,text=True,shell=True)
			print(del_out.stdout)

if __name__ == '__main__':
	if args.clear is True: #do this and nothing else.
		print("CLEARING JOBS...")
		clear_jobs(args.start,args.end)
		sys.exit(1)

	#CHECK YAML TEMPLATE EXISTS
	assert os.path.isfile(args.spec), "(kjobs.py): INCORRECT YAML TEMPLATE PATH IN SPEC ARG"

	#CHECK+LOAD HYPERPARAMETER FILE
	assert args.params is not None, "(kjobs.py): No JSON given."
	assert os.path.isfile(args.params), "(kjobs.py): INCORRECT JSON FILE PATH"
	with open(args.params,'r') as fp:
		HP_LIST = [json.loads(l) for l in fp.readlines()]
	N = len(HP_LIST)
	assert N > 0, "kjobs.py: GOT EMPTY JSON FILE."

	#CHECK START+END ARGS
	assert args.start is not None, "(kjobs.py): GOT NONE FOR START ROW"
	assert args.end is not None, "(kjobs.py): GOT NONE FOR END ROW"
	assert args.end <= N, "(kjobs.py): END INDEX OUT OF RANGE"
	assert args.end > args.start, "(kjobs.py): END ROW > START ROW"

	#SET
	start_index = args.start
	if args.end < N:
		end_index = args.end
	else:
		end_index = N #for range(start,end) which already n-1


	#RUN
	launch_jobs(args.spec,start_index,end_index,HP_LIST)
