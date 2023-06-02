import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

import random

import torch.nn as nn
from utils import *

class APLOSS(nn.Module):
    def __init__(self, threshold, batch_size, data_length, device):
        '''
        :param threshold: margin for squred hinge loss
        '''
        super(APLOSS, self).__init__()
        # print(data_length)
        self.threshold = threshold
        self.device = device

    def forward(self,f_ps, f_ns, args):
        f_ps = f_ps.view(-1)
        f_ns = f_ns.view(-1)

        vec_dat = torch.cat((f_ps, f_ns), 0)
        mat_data = vec_dat.repeat(len(f_ps), 1)

        f_ps = f_ps.view(-1, 1)

        neg_mask = torch.ones_like(mat_data)
        neg_mask[:, 0:f_ps.size(0)] = 0

        pos_mask = torch.zeros_like(mat_data)
        pos_mask[:, 0:f_ps.size(0)] = 1

        neg_loss = torch.max(self.threshold - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2 * neg_mask
        pos_loss = torch.max(self.threshold - (f_ps - mat_data), torch.zeros_like(mat_data)) ** 2 * pos_mask

        loss = pos_loss + neg_loss

        if f_ps.size(0) == 1:
            u_pos = pos_loss.mean()
            u_all = loss.mean()
        else:
            u_all = loss.mean(1, keepdim=True)
            u_pos = pos_loss.mean(1, keepdim=True)

        p = (u_pos - (u_all) * pos_mask) / (u_all ** 2)

        p.detach_()
        loss = torch.mean(p * loss)

        return loss


def SLATEM(train_set, test_set, model, args, device):

    criterion = APLOSS(threshold=args.thrd, batch_size = args.batch_size, data_length = args.ttnum, device= device)
    rank = dist.get_rank()
    tsize = dist.get_world_size()
    left = ((rank - 1) + tsize) % tsize 
    right = (rank + 1) % tsize
    m_t = [torch.zeros_like(param) for param in model.parameters()]

    Iteration = 0
    for epoch in range(args.epochs):
        model.train()

        for i, (inputs, target) in enumerate(train_set):
            model.zero_grad()
            inputs = inputs.to(device)
            target = target.to(device).float()
            out = model(inputs)

            predScore = torch.nn.Sigmoid()(out)

            loss = criterion(f_ps=predScore[0:args.posNum], f_ns=predScore[args.posNum:],  args = args)

            loss.backward()

            # Update
            with torch.no_grad():
                for i, param in enumerate(model.parameters()):
                    m_t[i].copy_(param.grad.data + (1 - args.alpha) * (m_t[i] - param.grad.data))
                    m_t[i].copy_(ring_reduce(rank, left, right, m_t[i], device))

                    param.data.add_(m_t[i], alpha= -args.lr)
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

