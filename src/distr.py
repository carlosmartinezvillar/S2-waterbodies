import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse


def local_train_and_validate(device):
	pass


def net_train_and_validate(rank,world):

	for epoch in n_epochs:


	    inputs = inputs.to(rank)
	    labels = labels.to(rank)


        outputs = ddp_model(inputs)

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Rank: {rank}, Epoch: {epoch}, Loss: {loss.item()}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rank',type=int,required=True,help='Rank of the current process')
    parser.add_argument('--world-size',type=int,required=True,help='Total number of processes')
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8080'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    model = 
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = 
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)


    train_and_validate()

    dist.destroy_process_group()