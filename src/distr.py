import torch
import os
import argparse

#DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_reduce_confmat(confmat):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(confmat, op=dist.ReduceOp.SUM)
    return confmat

def train_and_validate(model,dataloaders,optimizer,loss_fn,scaler,scheduler=None,n_epochs=50,n_class=2,gpu_id):


    N_tr = len(dataloaders['training'].dataset)
    N_va = len(dataloaders['validation'].dataset)

    log_header   = ["tloss","t_acc","vloss","v_acc","v_tpr","v_ppv","v_iou"]
    log_path     = f'{LOG_DIR}/epoch_log_{model.model_id:03}.tsv'   

    # dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    for epoch in range(n_epochs):

         dataloaders['training'].sampler.set_epoch(epoch) 

        ############################################################
        # LOG AND CHECKPOINTS -- ONLY RANK 0
        ############################################################
        if gpu_id == 0 and epoch % save_every == 0:

            if best_iou < epoch_iou:
                best_iou = epoch_iou
                best_epoch = epoch
                utils.save_ddp_checkpoint(MODEL_DIR,model,optimizer,epoch,loss_tr,loss_va,best=True)        

def main_ddp(rank,world_size,HP):
    #---------- SET UP DDP -------------------------------------------------------------------------
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl",rank=rank,world_size=world_size)

    #---------- SET ALL SEEDS ----------------------------------------------------------------------
    assert HP['SEED'] in (0,1), "INCORRECT SEED IN JSON PARAMETER DICT."
    if HP['SEED'] == True:
        utils.set_seed(476) 

    #---------- INPUT BANDS -----------------------------------------------------------------------
    assert HP['BANDS'] in [3,4],"INCORRECT NR. of BANDS IN JSON HYPERPARAMETER FILE."
    input_bands = HP['BANDS']

    #---------- OUTPUT CHANNELS -------------------------------------------------------------------
    assert HP['CLASS'] in [2,3], "INCORRECT # OF CLASSES SET IN JSON HYPERPARAMETER FILE."
    n_classes = HP['CLASS'] 

    #---------- MODEL -----------------------------------------------------------------------------
    model_str = HP['MODEL'][0:4]
    assert model_str in ["vit","unet"], "INCORRECT MODEL STRING."
    if model_str == 'unet':
        net = eval(f"model.UNet{HP['MODEL'][4]}_{HP['MODEL'][6]}({HP['ID']},in_channels={input_bands})")
    if model_str == 'vit':
        pass
    # ---> TO GPU
    net = net.to(rank)
    net = torch.compile(net)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    ddp_net = DDP(net,device_ids=[rank])

    #---------- OPTIMIZER -------------------------------------------------------------------------
    assert HP["OPTIM"] in ["adam","lamb","adamw"], "INCORRECT STRING FOR OPTIMIZER IN DICT."
    if HP['OPTIM'] == "adam":
        optimizer = torch.optim.Adam(ddp_net.parameters(),lr=HP['LEARNING_RATE'])
    if HP['OPTIM'] == "sgd":
        optimizer = torch.optim.SGD(ddp_net.parameters(),lr=HP['LEARNING_RATE'])
    if HP['OPTIM'] == 'adamw':
        optimizer = torch.optim.AdamW(ddp_net.parameters(),lr=HP['LEARNING_RATE'])

    #---------- LEARNING RATE SCHEDULER ------------------------------------------------------------
    if HP['SCHEDULER'] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.3)
    elif HP['SCHEDULER'] == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
    else:
        scheduler = None

    #---------- LOSS ------------------------------------------------------------------------------
    assert HP['LOSS'] in ["ce","ew","cw"], "INCORRECT STRING FOR LOSS IN DICT."
    if HP['LOSS'] == "ce":
        loss_fn = torch.nn.CrossEntropyLoss()
    if HP['LOSS'] == "ew":
        loss_fn = None
    if HP['LOSS'] == "cw": #<<< --- Needs some work...
        loss_fn = None

    #----------- AUTOMATIC MIXED PRECISION ---------------------------------------------------------
    scaler = torch.amp.GradScaler("cuda",enabled=True)


    #---------- DATALOADERS ------------------------------------------------------------------------
    transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)
    ])

    tr_dataset = dload.SentinelDataset(f"{DATA_DIR}/training",
        n_bands=input_bands,
        n_labels=2,
        transform=transform)

    va_dataset = dload.SentinelDataset(f"{DATA_DIR}/validation",
        n_bands=input_bands,
        n_labels=2,
        transform=None)

    dataloaders = {
            'training': torch.utils.data.DataLoader(
                tr_dataset,
                batch_size=HP['BATCH'],
                drop_last=False,
                shuffle=False,
                sampler=DistributedSampler(tr_dataset),
                num_workers=4,
                pin_memory=True,
                prefetch_factor=8),
            'validation': torch.utils.data.DataLoader(
                va_dataset,
                batch_size=HP['BATCH'],
                drop_last=False,
                shuffle=False,
                sampler=DistributedSampler(va_dataset),
                num_workers=4,
                pin_memory=True,
                prefetch_factor=8)
    }

    #---------- TRAINING --------------------------------------------------------------------------
    train_and_validate_ddp(net,dataloaders,optimizer,loss_fn,scaler,scheduler,HP['EPOCHS'],HP['OUTPUTS'],rank)

    #---------- CLEAN UP DDP ----------------------------------------------------------------------
    destroy_process_group()

if __name__ == '__main__':
    world_size = 2
    mp.spawn(main_ddp, args=(world_size,hyperparameters),nprocs=world_size)    
