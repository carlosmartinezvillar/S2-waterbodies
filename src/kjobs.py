import pyyaml
import subprocess
####################################################################################################
N_JOBS = 25
####################################################################################################
#load parameters

#launch via kubectl
for job in range(N_JOBS):
	pass
	kubectl create -f job_template.yml
