import torch
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from torchvision import models
from torchvision.datasets.folder import ImageFolder
import numpy as np
from sklearn.metrics import auc, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
import gflags
import sys
import torch.nn as nn
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

# Flags = gflags.FLAGS

def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1)
    return model

# def resnet18():
#     # model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=200)
#     net = models.resnet18(True)
#     #Finetune Final few layers to adjust for tiny imagenet input
#     net.avgpool = nn.AdaptiveAvgPool2d(1)
#     net.fc.out_features = 1
#     return net


def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1)
    return model

def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1)
    return model

class MnistModel(nn.Module):
    
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 1 / m.bias.numel())
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 1 / m.bias.numel())

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(-1, 800)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FMnistModel(nn.Module):
    
    def __init__(self):
        super(FMnistModel, self).__init__()
        print("Fashion Mnist")
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3)
        self.fc1 = nn.Linear(250, 100)
        self.fc2 = nn.Linear(100, 1)
        self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool2d(2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 1 / m.bias.numel())
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 1 / m.bias.numel())

    def forward(self, x):
        x = self.maxpool(self.tanh(self.conv1(x)))
        x = self.maxpool(self.tanh(self.conv2(x)))
        x = x.view(-1, 250)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFARModel(nn.Module):
    def __init__(self):
        super(CIFARModel, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        # fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        # flattening
        x = x.view(-1, 64 * 4 * 4)
        # fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLP(nn.Module):
   
    def __init__(self, input):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input, 28)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(28, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AUPRCSampler(Sampler):

    def __init__(self, labels, batchSize, posNum=1):
        # positive class: minority class
        # negative class: majority class

        self.labels = labels
        self.posNum = posNum
        self.batchSize = batchSize

        # print("AUPRCSampler", self.posNum, self.batchSize)

        self.clsLabelList = np.unique(labels)
        self.dataDict = {}

        for label in self.clsLabelList:
            self.dataDict[str(label)] = []

        for i in range(len(self.labels)):
            self.dataDict[str(self.labels[i])].append(i)

        self.ret = []


    def __iter__(self):
        minority_data_list = self.dataDict[str(1)]
        majority_data_list = self.dataDict[str(0)]

        # print("AUPRCSampler222", len(minority_data_list), len(majority_data_list))

        # print(len(minority_data_list), len(majority_data_list))
        np.random.shuffle(minority_data_list)
        np.random.shuffle(majority_data_list)

        # In every iteration : sample 1(posNum) positive sample(s), and sample batchSize - 1(posNum) negative samples
        if len(minority_data_list) // self.posNum  > len(majority_data_list)//(self.batchSize - self.posNum): # At this case, we go over the all positive samples in every epoch.
            # extend the length of majority_data_list from  len(majority_data_list) to len(minority_data_list)* (batchSize-posNum)
            majority_data_list.extend(np.random.choice(majority_data_list, len(minority_data_list) // self.posNum * (self.batchSize - self.posNum) - len(majority_data_list), replace=True).tolist())

        elif len(minority_data_list) // self.posNum  < len(majority_data_list)//(self.batchSize - self.posNum): # At this case, we go over the all negative samples in every epoch.
            # extend the length of minority_data_list from len(minority_data_list) to len(majority_data_list)//(batchSize-posNum) + 1s
            minority_data_list.extend(np.random.choice(minority_data_list, len(majority_data_list) // (self.batchSize - self.posNum)*self.posNum - len(minority_data_list), replace=True).tolist())

        # print("AUPRCSampler333", len(minority_data_list), len(majority_data_list))


        self.ret = []
        for i in range(len(minority_data_list) // self.posNum):
            self.ret.extend(minority_data_list[i*self.posNum:(i+1)*self.posNum])
            startIndex = i    *(self.batchSize - self.posNum)
            endIndex   = (i+1)*(self.batchSize - self.posNum)
            self.ret.extend(majority_data_list[startIndex:endIndex])

        return iter(self.ret)

    def __len__ (self):
        return len(self.ret)



def ave_prc(targets, preds):
    return average_precision_score(targets, preds)


def test_classification(model, test_loader,  device):
    model.eval()
    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)

    # if Flags.datasets == 'tiny':
    for (inputs, target) in test_loader:

        inputs = inputs.to(device)
        target = target.to(device).float()
        if torch.max(target) > 1:
            print("ERRRORR----- test_classification ----")
        # else:
        #     target[target <= 4] = 0
        #     target[target > 4] = 1

        with torch.no_grad():
            out = model(inputs)

        if out.shape[1] == 1:
            pred = torch.sigmoid(out)  ### prediction real number between (0,1)
        else:
            print("ERROR !!!")
        preds = torch.cat([preds, pred], dim=0)
        targets = torch.cat([targets, target], dim=0)


    return preds, targets
    # return ave_prc(targets.cpu().detach().numpy(), preds.cpu().detach().numpy()),  


def evalutation(model, test_set,  device):
    model.eval()
    #### testing  #######
    test_pred = []
    test_true = [] 
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_set):
            data = data.to(device)
            # target = target.to(device)
            y_pred = torch.sigmoid(model(data))
            test_pred.append(y_pred.cpu().detach().numpy())
            test_true.append(target.numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    print("============")
    print(ave_prc(test_true, test_pred))

    precision, recall, _ = precision_recall_curve(test_true, test_pred)
    # print("######## PRC ##########")
    # for i in range(len(precision)):
    #     print(precision[i], recall[i])
    an_array = np.concatenate((np.expand_dims(precision, axis = 1), np.expand_dims(recall, axis = 1)), axis=1)
    mat = np.matrix(an_array)
    return mat


def save_pr(preds, targets):
    #### save to .txt ###

    precision, recall, _ = precision_recall_curve(targets.cpu().detach().numpy(), preds.cpu().detach().numpy())

    an_array = np.concatenate((np.expand_dims(precision, axis = 1),
    np.expand_dims(recall, axis = 1)), axis=1)

    mat = np.matrix(an_array)

    filename = "PR/PR_" + Flags.datasets + '_' + Flags.method + ".txt"

    with open(filename, "w") as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')

def ring_reduce(rank, left, right, msg, device):

    msg = msg.contiguous().cpu()

    lef_buff = msg.clone()
    rig_buff = msg.clone()

    ####### simple ####

    req0 = dist.isend(msg, dst=right)
    dist.recv(lef_buff, src=left)
    req0.wait()

    req0 = dist.isend(msg, dst=left)
    dist.recv(rig_buff, src=right)
    req0.wait()

    return (lef_buff + msg +  rig_buff).to(device) / 3

