import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torchvision import datasets, transforms
import numpy as np

from utils import *
from SLATE import *
from SLATEM import *
from SGD import *
from CODA import *

from torch.utils.data import DataLoader
from dist_data import *
from torchvision import datasets
import argparse
import sys

def get_default_device(idx):
    # print(idx)
    if torch.cuda.is_available():
        print("GPU available")
        return torch.device('cuda:' + str(int(idx // 10)))

    else:
        print("GPU NOT available")

        return torch.device('cpu')

def run(rank, size, args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rank = dist.get_rank()
    device = get_default_device(rank)

    if args.dataset == 'mnist':
        train_dataset = DISTMNIST(root='../data/', rank=rank, train=True, download=True, transform=transform_m)
        if rank == 0:
            test_dataset = DISTMNIST(root='../data/', rank=rank, train=False, download=True, transform=transform_m)

        model = MnistModel() 
    elif args.dataset == 'fmnist':
        train_dataset = DISTFashionMNIST(root='../data/', rank=rank, train=True, download=True, transform=transform_f)
        if rank == 0:
            test_dataset = DISTFashionMNIST(root='../data/', rank=rank, train=False, download=True, transform=transform_f)

        model = FMnistModel() 

    elif args.dataset == 'cifar10':
        train_dataset = DISTCIFAR10('../data/cifar10/', rank=rank, train=True, download=True, transform=transform_c)

        if rank == 0:
            test_dataset = DISTCIFAR10('../data/cifar10/', rank=rank, train=False, download=True, transform=transform_c)

        model = CIFARModel() 

    elif args.dataset == 'tiny':
        print('Tiny')
        train_dataset = TINYIMAGENET(root='../data/TINYIMAGENET/train/', train=True, transform = transform_train_t)
        if rank == 0:    
            test_dataset = TINYIMAGENET(root='../data/TINYIMAGENET/val/images', train=False,  transform=transform_val_t)

        model = resnet18()

    elif args.dataset in ['w7a', 'w8a']:
        if args.init and rank == 0:
            loadlib(args.dataset)
        dist.barrier()
        train_dataset = Generated(root='../data/' + args.dataset, train=True)
        args.dim = train_dataset.dim
        if rank == 0:
            test_dataset = Generated(root='../data/' + args.dataset, train=False)

        model = MLP(args.dim)
        # print(model)
    else:
        print("ERORR")

    args.totalPNum =  train_dataset.pos_num
    args.ttnum = train_dataset.pos_num + train_dataset.neg_num

    labels = [0] * (len(train_dataset) - args.totalPNum) + [1] * args.totalPNum

    train_set = DataLoader(train_dataset, batch_size=args.batch_size, 
                            sampler=AUPRCSampler(labels, args.batch_size, posNum=args.posNum), 
                            num_workers=2, pin_memory=True)
    if rank == 0:
        test_set = DataLoader(test_dataset, 500, shuffle=False, num_workers=4, pin_memory=True)
    else:
        test_set = None

    if args.init:
        print("Wegith Init")
        fname = 'model/' + args.dataset + '.pth'
    else:
        print("Wegith Loaded")
        fname = 'model/' + args.dataset + '.pth'
        model.load_state_dict(torch.load(fname))

    model = model.to(device)

    if args.method == 'sgd':
        DSGD(train_set, test_set, model, args, device)
    elif args.method == 'coda':
        CODA(train_set, test_set, model, args, device)
    elif args.method == 'slate':
        print("slate")
        SLATE(train_set, test_set, model, args, device)
    elif args.method == 'slatem':
        print("slatem")
        SLATEM(train_set, test_set, model, args, device)
    else:
        print("ERRORRRR")


def init_process(rank, size, args, fn, backend='gloo'):
# def init_process(rank, size, args, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(args.port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, args)


################## MAIN ######################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 5000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--worker-size', type=int, default=3, metavar='N',
                        help='szie of worker (default: 3)')

    parser.add_argument('--posNum', type=int, default=10, metavar='N',
                        help='sample posNum postive data each time (default: 5)')
    parser.add_argument('--ttnum', type=int, default=1, metavar='N',
                        help='total postive data in dataset (default: 1)')
    parser.add_argument('--totalPNum', type=int, default=1, metavar='N',
                        help='total  data in dataset (default: 1)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr2', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=0.1, metavar='alpha',
                        help='momentum rate alpha')

    parser.add_argument('--thrd', type=float, default=0.5, metavar='alpha',
                        help='Loss threathold')

    parser.add_argument('--inLoop', type=int, default=10, metavar='S',
                        help='inter loop number')
    parser.add_argument('--iteration', type=int, default=10, metavar='S',
                        help='stop iteration number')

    parser.add_argument('--init', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset for trainig')
    parser.add_argument('--method', type=str, default='fedavg',
                        help='Dataset for trainig')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--port', type=int, default=29505, metavar='S',
                        help='random seed (default: 29505)')
    args = parser.parse_args()
    print(args)

    size = args.worker_size 
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, args, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

