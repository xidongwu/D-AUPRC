# from torchvision import datasets, transforms
import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

import random

import torch.nn as nn
from utils import *

class AUCLOSS(nn.Module):
	def __init__(self, a, b, w, model):
		super(AUCLOSS, self).__init__()
		# self.p = 1 / (1 + 0.2) # randomly remove 80% of the negative data points 
		# self.p = 0.2 / (1+ 0.2)  # randomly remove 80% of the positive data
		self.p = 0.03  # w7a  / w8a 
		self.a = a
		self.b = b
		self.w = w
		self.model = model
	def forward(self, y_pred, y_true):
		'''
		AUC Margin Loss
		'''
		auc_loss = (1 - self.p) * torch.mean((y_pred - self.a)**2 * (1 == y_true).float()) + self.p * torch.mean((y_pred - self.b)**2 * (0 == y_true).float()) + \
		2 * (1+ self.w) * ( torch.mean((self.p * y_pred * (0 == y_true).float() - (1 - self.p) * y_pred * (1==y_true).float()))) - self.p * (1 - self.p) * self.w**2
		return auc_loss
	def zero_grad(self):
		self.model.zero_grad()
		self.a.grad = None
		self.b.grad = None
		self.w.grad = None

def CODA(train_set, test_set, model, args, device):
	print("CODA")
	Iteration = 0
	rank = dist.get_rank()
	tsize = dist.get_world_size()
	left = ((rank - 1) + tsize) % tsize 
	right = (rank + 1) % tsize

	a = torch.ones(1, device = device, requires_grad=True)
	b = torch.zeros(1, device = device, requires_grad=True)
	w = torch.zeros(1, device = device, requires_grad=True)

	criterion = AUCLOSS(a, b, w, model)
	model.train()

	for epoch in range(args.epochs):
		# model.train()

		for siter, (data, target) in enumerate(train_set):
			data   = data.to(device)
			target = target.to(device)

			output = model(data)
			
			loss = criterion(output, target.view(-1,1).float())
			loss.backward()

            # Update
			with torch.no_grad():
				for i, param in enumerate(model.parameters()):
					param.data.add_(param.grad.data, alpha= - args.lr)
					param.data.copy_(ring_reduce(rank, left, right, param.data, device))

			model.zero_grad()
			a.data.copy_(a.data - args.lr * a.grad.data)
			b.data.copy_(b.data - args.lr * b.grad.data)
			w.data.copy_(w.data + args.lr2 * w.grad.data)
			w.data  = torch.clamp(w.data, -10, 10)
			criterion.zero_grad()

			with torch.no_grad():
				a.copy_(ring_reduce(rank, left, right, a, device))
				b.copy_(ring_reduce(rank, left, right, b, device))
				w.copy_(ring_reduce(rank, left, right, w, device))
				w.data  = torch.clamp(w.data, -10, 10)


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
				return
