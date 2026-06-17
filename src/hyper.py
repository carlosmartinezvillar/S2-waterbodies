'''
Set files with rows of hyperparameter dictionaries.
'''
import itertools
import json
import argparse
import os

def randomize_hyperparameters(n=1): # <<<< MISSING?
	HP = {}
	#same as a grid search but randomize the choice of parameters
	return HP


def sequence_hyperparameters(out_file_path,id_start,trial):
	'''
	Create a list of dict elements each containing a model's hyperparameters.
	The list is stored in out_path in .json format and created using a 
	cross-product (all-by-all) of the parameters provided.
	'''
	# Each parameter -- Trial 0 -- debugging
	if trial == 0:
		seeds = [1]	
		epoch = [10]
		lrate = [0.00001,0.0005]	
		sched = ["none"]
		optim = ["adamw"]
		decay = [0.001,0.0001]
		loss  = ["ce"]	
		batch = [16,32]
		inits = ["random"]
		bands = [3]
		label = [2]
		model = ["UNet3_1","UNet6_1"]

	# Each parameter -- Trial 1
	if trial == 1:
		seeds = [1]	
		epoch = [50]
		lrate = [0.0001,0.00025,0.0005,0.00075,0.001]	
		sched = ["none"]
		optim = ["adamw"]
		decay = [0.01,0.001,0.0001,0.00001]
		loss  = ["ce"]	
		batch = [16,32]
		inits = ["random"]
		bands = [3]
		label = [2]
		model = ["UNet2_1","UNet2_2","UNet2_4",
			"UNet3_1",
			"UNet5_1","UNet5_2","UNet5_4",
			"UNet6_1"]

	# Each parameter -- Trial 2 No residuals
	if trial == 2:
		seeds = [1]
		epoch = [50]
		lrate = [0.0001,0.00025,0.0005,0.00075,0.001]	
		sched = ["none"]
		optim = ["adamw"]
		decay = [0.01,0.001,0.0001,0.00001]
		loss  = ["ce"]	
		batch = [16,32]
		inits = ["random"]
		bands = [3]
		label = [2]
		model = ["UNet1_3","UNet4_3","UNet1_2"] #try no residuals for best in trial 1

	# Each parameter -- Trial 3 RGB+NIR
	if trial == 3:
		seeds = [1]	
		epoch = [50]
		lrate = [0.0001,0.00025,0.0005] #?	
		sched = ["none"]
		optim = ["adamw"]
		decay = [0.01,0.001,0.0001,0.00001] #?
		loss  = ["ce"]	
		batch = [16,32] #?
		inits = ["random"]
		bands = [4]
		label = [2]
		model = ["UNet2_1","UNet2_2","UNet2_4"] #best 5? from trial 1

	# Each parameter -- Trial 4 Best unseeded 100 epochs
	if trial == 4:
		seeds = [1]	
		epoch = [50]
		lrate = [0.0001,0.00025,0.0005] #?	
		sched = ["none"]
		optim = ["adamw"]
		decay = [0.01,0.001,0.0001,0.00001] #?
		loss  = ["ce"]	
		batch = [16,32] #?
		inits = ["random"]
		bands = [4]
		label = [2]
		model = ["UNet6_1"] #best 5? from trial 1

	# Cross-product
	hp = list(itertools.product(seeds,epoch,lrate,sched,optim,decay,loss,batch,inits,bands,label,model))
	HP_NEW = []

	for i in range(len(hp)):
		row_dict = {}
		row_dict["ID"]            = i+id_start
		row_dict["SEED"]          = hp[i][0]
		row_dict["EPOCHS"]        = hp[i][1]		
		row_dict["LEARNING_RATE"] = hp[i][2]
		row_dict["SCHEDULER"]     = hp[i][3]
		row_dict["OPTIM"]         = hp[i][4]
		row_dict["DECAY"]         = hp[i][5]
		row_dict["LOSS"]          = hp[i][6]
		row_dict["BATCH"]         = hp[i][7]
		row_dict["INIT"]          = hp[i][8]
		row_dict["BANDS"]         = hp[i][9]
		row_dict["OUTPUTS"]       = hp[i][10]
		row_dict["MODEL"]         = hp[i][11]
		HP_NEW.append(row_dict)

	with open(out_file_path,'w') as fp:
		for line in HP_NEW:
			# fp.write(str(line))
			# fp.write('\n')
			json.dump(line,fp)
			fp.write('\n')
	print(f"Parameter file written to {out_file_path}")


if __name__ == '__main__':
	# as ARGV?
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--hpo",required=False,default=None,
	# 	help="Create a JSON file with a combination of hyperparameters.")
	# args = parser.parse_args()

	out_file_path = '../hpo/trial0.json'
	# out_file_path = '../hpo/trial1.json'
	# out_file_path = '../hpo/trial2.json'
		
	assert not os.path.isfile(out_file_path), f"Overwriting existing file {out_file_path}"

	sequence_hyperparameters(out_file_path,id_start=10,trial=0)
	# sequence_hyperparameters(out_file_path,id_start=101,trial=1)		
	# sequence_hyperparameters(out_file_path,id_start=421,trial=2)
