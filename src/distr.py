import os
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as v2
import random
import time
from tqdm import tqdm
import argparse
import json
from functools import wraps

#DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group

import utils
import model
import dload

####################################################################################################
# SET GLOBAL VARS FROM ENV OR ARGS
####################################################################################################
parser = argparse.ArgumentParser()
required = parser.add_argument_group('Required arguments')
optional = parser.add_argument_group('Optional arguments')

required.add_argument('--data-dir',required=True,
    help='Input dataset directory.')
required.add_argument('--net-dir',required=True,
    help='Directory where the weights of trained models are dumped.')
required.add_argument('--log-dir',required=True,default='../log',
    help='Training logs directory.')
required.add_argument('--row',required=True,type=int,default=0,
    help='Model id number in JSON hyperparameter file (default ../hpo/params.json).')

optional.add_argument('--full',required=False,action='store_true',default=False,
    help='Train on both training and validation sets (training final model).')
optional.add_argument('-p','--params',required=False,default='../hpo/parameters.json',
    help='Path to JSON hyperparameter file (if not default).')
args = parser.parse_args()

DATA_DIR  = args.data_dir
LOG_DIR   = args.log_dir
MODEL_DIR = args.net_dir


####################################################################################################
# SOME HELPER FUNCTIONS
####################################################################################################
def calculate_metrics(confmat):
    '''
    Calculate precision, recall, accuracy, and IoU for a 
    '''
    with torch.no_grad():
        # Add stuff
        TP = confmat.diagonal()
        FP = confmat.sum(dim=0) - TP #this is silly but explicit
        FN = confmat.sum(dim=1) - TP
        TN = confmat.sum() - TP - FP - FN
        # eps = 0.0000000001

        # the metrics
        ppv = TP / (TP + FP).clamp(min=1) #precision
        tpr = TP / (TP + FN).clamp(min=1) #recall
        acc = (TP+TN) / (TP+FN+FP+TN).clamp(min=1) #accuracy
        iou = TP / (TP + FN + FP).clamp(min=1) #Intersection-over-Union

    return ppv,tpr,acc,iou

@torch.no_grad()
def update_confusion_matrix(confmat,T,Y,n_classes):
    '''
    Update a confusion matrix tensor in gpu. Per-pixel classification.
    '''
    # confmat[0,0] += ((T==0) & (Y==0)).sum() #TN  #<--- this in modulo op below
    # confmat[0,1] += ((T==0) & (Y==1)).sum() #FP
    # confmat[1,0] += ((T==1) & (Y==0)).sum() #FN
    # confmat[1,1] += ((T==1) & (Y==1)).sum() #TP

    for k in range(n_classes*n_classes):
        i = k // n_classes #row
        j = k % n_classes #col
        confmat[i,j] += ((T==i) & (Y==j)).sum()


def ddp_reduce_sum(tensor):
    dist.all_reduce(tensor,op=dist.ReduceOp.SUM)
    return tensor


def total_time_decorator(orig_func):
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        total_time_start = time.time()
        orig_func(*args,**kwargs)
        total_time = time.time() - total_time_start
        print(f'TOTAL TRAINING TIME: {total_time:.2f}s')

    return wrapper

####################################################################################################
# TRAININING+VALIDATION
####################################################################################################
def train_and_validate(model,dataloaders,optimizer,loss_fn,scheduler,epochs,n_classes,device):

    # AUTOMATIC MIXED PRECISION
    scaler = torch.amp.GradScaler("cuda",enabled=True)

    # SET LOG FILE
    if dist.get_rank()==0:
        log_file_header = ["tloss","vloss"]
        log_file_header += [f"tacc{c}" for c in range(n_classes)]
        log_file_header += [f"tiou{c}" for c in range(n_classes)]
        log_file_header += [f"vacc{c}" for c in range(n_classes)]
        log_file_header += [f"vtpr{c}" for c in range(n_classes)]
        log_file_header += [f"vppv{c}" for c in range(n_classes)]
        log_file_header += [f"viou{c}" for c in range(n_classes)]   
        log_file_path = f'{LOG_DIR}/epoch_log_{model.model_id:03}.tsv'   
        logger        = utils.Logger(log_file_path,log_file_header)
        total_time_start = time.time()

    best_iou   = 0.0
    best_epoch = 0

    for epoch in range(epochs):

        # Confusion matrices in GPU
        gpu_mat_tr = torch.zeros((n_classes,n_classes),device=device,dtype=torch.int64) 
        gpu_mat_va = torch.zeros((n_classes,n_classes),device=device,dtype=torch.int64)

        # Time
        if dist.get_rank() == 0:
            epoch_start_time = time.time()
            print(f'\nEpoch {epoch}/{epochs-1}')
            print('-'*80,flush=True)

        ############################################################
        # TRAINING
        ############################################################
        # SHUFFLE
        dataloaders['training'].sampler.set_epoch(epoch) 
        loss_sum_tr = torch.zeros(1,device=device)

        model.train()
        for X,T in dataloaders['training']:

            #TO DEV
            X = X.to(device,non_blocking=True)
            T = T.to(device,non_blocking=True)

            # FORWARD
            with torch.autocast(device_type="cuda", dtype=torch.float16,enabled=True):
                output = model(X)
                loss   = loss_fn(output,T)

            # BACKPROP
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # METRICS -- Loss
            loss_sum_tr   += loss.detach() * X.size(0)
            sample_sum_tr += X.size(0)

            # METRICS -- Confusion matrix in gpu
            Y = output.detach().argmax(axis=1) #keep detach if needed to switch to .max()
            T = T.detach()
            update_confusion_matrix(gpu_mat_tr,T,Y,n_classes)                      


        # SCHEDULER UPDATE
        if scheduler is not None:
            scheduler.step()

        # ---- Sync GPUs ----
        ddp_reduce_sum(loss_sum_tr)
        ddp_reduce_sum(sample_sum_tr)
        ddp_reduce_sum(gpu_mat_tr)

        # TRAINING METRICS
        loss_tr    = (loss_sum_tr/sample_sum_tr).item() #-----------CPU sync
        cpu_mat_tr = gpu_mat_tr.cpu() #-----------------------------CPU sync
        tr_ppv,tr_tpr,tr_acc,tr_iou = calculate_metrics(cpu_mat_tr) #tensors,result per class

        if dist.get_rank() == 0:
            print(f'[T] LOSS: {loss_tr:.5f} | ACC: {tr_acc[-1]:.5f}',end='')
            if n_classes > 2:
                va_miou = va_iou.mean().item()
                print(f' | mIoU: {va_miou:.5f}')
            else:            
                print(f' | IoU_0: {tr_iou[0]:.5f} | IoU_1: {tr_iou[1]:.5f}')


        ############################################################
        # VALIDATION
        ############################################################        
        loss_sum_va   = torch.zeros(1,device=CUDA_DEV)
        sample_sum_va = torch.zeros(1,device=CUDA_DEV)

        # LOOP
        model.eval()
        with torch.no_grad():
            for X,T in dataloaders['validation']:

                X = X.to(device,non_blocking=True)
                T = T.to(device,non_blocking=True)

                # FORWARD
                with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=True):
                    output = model(X)
                    loss   = loss_fn(output,T)
                Y_soft,Y   = torch.max(output,1) #soft-prediction, hard-prediction

                # METRICS -- Loss
                loss_sum_va   += loss.detach() * X.size(0)
                sample_sum_va += X.size(0)                               

                # METRICS -- Confusion matrix
                update_confusion_matrix(gpu_mat_va,T,Y,n_classes)

        # SYNC GPUS
        ddp_reduce_sum(loss_sum_va)
        ddp_reduce_sum(sample_sum_va)
        dpp_reduce_sum(gpu_mat_va)

        # VALIDATION METRICS
        loss_va    = (loss_sum_va / sample_sum_va).item() #-------------sync
        cpu_mat_va = gpu_mat_va.cpu() #---------------------------------sync
        va_ppv,va_tpr,va_acc,va_iou = calculate_metrics(cpu_mat_va)

        if dist.get_rank() == 0:
            print(f'[V] LOSS: {loss_va:.5f} | ACC: {va_acc[-1]:.5f}',end='')
            if n_classes > 2:
                va_miou = va_iou.mean().item()
                print(f' | mIoU: {va_miou:.5f}')
            else:
                print(f' | IoU_0: {va_iou[0]:.5f} | IoU_1: {va_iou[1]:.5f}')


        ############################################################
        # LOG AND CHECKPOINTS -- ONLY RANK 0
        ############################################################
        if dist.get_rank() == 0:

            # EPOCH TIME
            epoch_time = time.time() - epoch_start_time
            print(f'\nEpoch time: {epoch_time:.2f}s')

            # LOG RESULTS STRING
            epoch_result = [loss_tr,loss_va]
            epoch_result += [tr_acc[i] for i in range(n_classes)]
            epoch_result += [tr_iou[i] for i in range(n_classes)]
            epoch_result += [va_acc[i] for i in range(n_classes)]
            epoch_result += [va_tpr[i] for i in range(n_classes)]
            epoch_result += [va_ppv[i] for i in range(n_classes)]
            epoch_result += [va_iou[i] for i in range(n_classes)]
            logger.log(epoch_result)

            # SAVE MODEL
            if n_classes > 2:
                epoch_iou = va_miou #mIoU for 3+ classes
            else:
                epoch_iou = va_iou[1] #true label iou for 2 classes

            if best_iou < epoch_iou:
                best_iou = epoch_iou
                best_epoch = epoch
                utils.save_ddp_checkpoint(MODEL_DIR,model,optimizer,scaler,epoch,loss_tr,loss_va,best=True)        

            print(f'Best validation IoU: {best_iou:.5f} -- Epoch {best_epoch}')
            total_time = time.time() - total_time_start
            print(f'TOTAL TRAINING TIME: {total_time:.2f}s')


def ddp_worker(rank,world_size,HP):
    #---------- SET UP DDP -------------------------------------------------------------------------
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl",rank=rank,world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda",rank)

    #---------- INPUT BANDS -----------------------------------------------------------------------
    assert HP['BANDS'] in [3,4],"INCORRECT BAND NR IN JSON HYPERPARAMETER FILE."
    n_bands = HP['BANDS']

    #---------- OUTPUT CHANNELS -------------------------------------------------------------------
    assert HP['OUTPUTS'] in [2,3], "INCORRECT # OF CLASSES SET IN JSON HP FILE."
    n_classes = HP['OUTPUTS'] 

    #---------- SET ALL SEEDS ----------------------------------------------------------------------
    assert HP['SEED'] in (0,1), "INCORRECT SEED VALUE IN JSON PARAMETER DICT."
    if HP['SEED'] == True:
        utils.set_seed(476)

    #---------- MODEL -----------------------------------------------------------------------------
    model_str = HP['MODEL'][0:4]
    assert model_str in ["vits","unet"], "INCORRECT MODEL STRING."
    if model_str == 'unet':
        net = eval(f"model.UNet{HP['MODEL'][4]}_{HP['MODEL'][6]}({HP['ID']},in_channels={n_bands})")
    if model_str == 'vits':
        net = eval(f"model.ViT{HP['MODEL'][4]}_{HP['MODEL'][6]}({HP['ID']},in_channels={n_bands})")

    # ---> TO EACH GPU
    net = net.to(device)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net,device_ids=[rank])
    net = torch.compile(net)

    #---------- LOSS ------------------------------------------------------------------------------
    assert HP['LOSS'] in ["ce","ew","cw"], "INCORRECT STRING FOR LOSS IN DICT."
    if HP['LOSS'] == "ce":
        loss_fn = torch.nn.CrossEntropyLoss()
    if HP['LOSS'] == "ew":
        loss_fn = None
    if HP['LOSS'] == "cw": #<<< --- Needs some work...
        loss_fn = None

    #---------- OPTIMIZER -------------------------------------------------------------------------
    assert HP["LEARNING_RATE"] in [0.0001,0.00025,0.0005,0.00075,0.001],"LR OUT OF RANGE."
    assert HP["OPTIM"] in ["adam","sgd","adamw"], "INCORRECT STRING FOR OPTIMIZER IN DICT."

    if HP['OPTIM'] == "adam":
        optimizer = torch.optim.Adam(net.parameters(),lr=HP['LEARNING_RATE'])
    if HP['OPTIM'] == "sgd":
        optimizer = torch.optim.SGD(net.parameters(),lr=HP['LEARNING_RATE'])
    if HP['OPTIM'] == 'adamw':
        assert "DECAY" in HP, "DECAY UNDEFINED FOR ADAMW RUN."
        assert HP["DECAY"] in [0.01,0.001,0.0001,0.00001,0.000001], "DECAY OUT OF RANGE."
        optimizer = torch.optim.AdamW(net.parameters(),lr=HP['LEARNING_RATE'],
            weight_decay=HP["DECAY"])

    #---------- LEARNING RATE SCHEDULER ------------------------------------------------------------
    assert HP["SCHEDULER"] in ["step","exp","none"] #making it overly explicit for logs/results
    if HP['SCHEDULER'] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.3)
    if HP['SCHEDULER'] == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
    if HP['SCHEDULER'] == "none":
        scheduler = None

    #---------- DATALOADERS ------------------------------------------------------------------------
    transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)
    ])

    tr_dataset = dload.SentinelDataset(f"{DATA_DIR}/training",
        n_bands=n_bands,
        n_labels=n_classes,
        transform=transform)

    va_dataset = dload.SentinelDataset(f"{DATA_DIR}/validation",
        n_bands=n_bands,
        n_labels=n_classes,
        transform=None)

    dataloaders = {
            'training': torch.utils.data.DataLoader(
                tr_dataset,
                batch_size=HP['BATCH'],
                drop_last=False,
                shuffle=False,
                sampler=DistributedSampler(tr_dataset,num_replicas=world_size,rank=rank),
                num_workers=4,
                pin_memory=True,
                prefetch_factor=8),
            'validation': torch.utils.data.DataLoader(
                va_dataset,
                batch_size=HP['BATCH'],
                drop_last=False,
                shuffle=False,
                sampler=DistributedSampler(va_dataset,num_replicas=world_size,rank=rank),
                num_workers=4,
                pin_memory=True,
                prefetch_factor=8)
    }

    #---------- TRAINING --------------------------------------------------------------------------
    train_and_validate(
        net,
        dataloaders,
        optimizer,
        loss_fn,
        scheduler,
        HP['EPOCHS'],
        n_classes,
        device
    )

    #---------- CLEAN UP DDP ----------------------------------------------------------------------
    dist.destroy_process_group()


if __name__ == '__main__':
    #---------- LOAD HYPERPARAMETERS -------------------------------------------
    assert os.path.isfile(args.params), "INCORRECT JSON FILE PATH"
    with open(args.params,'r') as fp:
        HP_LIST = [json.loads(line) for line in fp.readlines() if line != "\n"]
    assert len(HP_LIST) > 0, "GOT EMPTY JSON FILE."

    # SEARCH BY ID
    hp_list_indexed = {row['ID']:row for row in HP_LIST}
    HPS = hp_list_indexed[args.row]

    #---------- SPAWN PROCESSES ------------------------------------------------    
    world_size = 2
    mp.spawn(ddp_worker, args=(world_size,HPS),nprocs=world_size)    
