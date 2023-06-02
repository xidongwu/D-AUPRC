import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

import random

import torch.nn as nn
from utils import *

def DSGD(train_set, test_set, model, args, device):
    print("SGD!")

    Iteration = 0
    rank = dist.get_rank()
    tsize = dist.get_world_size()
    left = ((rank - 1) + tsize) % tsize 
    right = (rank + 1) % tsize

    criterion = torch.nn.BCELoss()

    for epoch in range(args.epochs):
        # model.train()

        for siter, (data, target) in enumerate(train_set):
            model.zero_grad()

            data   = data.to(device)
            target = target.to(device)

            output = torch.sigmoid(model(data))

            loss = criterion(output, target.view(-1,1).float())
            loss.backward()

            # Update
            for i, param in enumerate(model.parameters()):
                param.data.add_(param.grad.data, alpha= -args.lr)
                param.data.copy_(ring_reduce(rank, left, right, param.data, device))

            if Iteration % 10 == 0 and rank == 0:   
                preds, targets = test_classification(model, test_set, device)
                AP = ave_prc(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())
                print(Iteration, AP)
                model.train()

            Iteration += 1

            if Iteration > args.iteration:
                for para in model.parameters():
                    dist.all_reduce(para.data, op=dist.ReduceOp.SUM)
                    para.data.div_(tsize)

                if rank == 0:
                    mat = evalutation(model, test_set, device)

                    filename = "PR/" + args.dataset + '_' + args.method + ".txt"
                    with open(filename, "w") as f:
                        for line in mat:
                            np.savetxt(f, line, fmt='%.2f')
                            # return

                return



