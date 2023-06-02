import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from collections import Counter
import torch.distributed as dist

# __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
#                    'std': [0.229, 0.224, 0.225]}
#
# __tiny_imagenet_stats = {'mean': [0.4802, 0.4481, 0.3975],
#                    'std': [0.2302, 0.2265, 0.2262]}
#
# __imagenet_pca = {
#     'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
#     'eigvec': torch.Tensor([
#         [-0.5675,  0.7192,  0.4009],
#         [-0.5808, -0.0045, -0.8140],
#         [-0.5836, -0.6948,  0.4203],
#     ])
# }
#
# __cifar10_stats = {'mean': [0.4914, 0.4822, 0.4465],
#                       'std': [0.2023, 0.1994, 0.2010]}
#
# __cifar100_stats = {'mean': [0.5071, 0.4867, 0.4408],
#                        'std': [0.2675, 0.2565, 0.2761]}



transform_c = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])


transform_m =transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])


transform_f = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(
        (0.2860,), (0.3529,)) ])



transform_train_t = transforms.Compose([
            transforms.Resize(256), # Resize images to 256 x 256
            transforms.CenterCrop(224), # Center crop image
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Converting cropped images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])


transform_val_t = transforms.Compose([
            transforms.Resize(256), # Resize images to 256 x 256
            transforms.ToTensor(),  # Converting cropped images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])


class DISTMNIST(torchvision.datasets.MNIST):
    cls_num = 10

    def __init__(self, root, rank, imb_factor=0.2, train=True,
                 transform=None, target_transform=None, download=False):
        super(DISTMNIST, self).__init__(root, train, transform, target_transform, download)

        size = dist.get_world_size()
 
        if train:
            imb_factor = 0.2
        else:
            imb_factor = 1

        img_num_per_cls = []
        new_data = []
        new_targets = []

        classes = np.unique(self.targets.numpy())
        targets_np = np.array(self.targets, dtype=np.int64)
        self.neg_num = 0
        self.pos_num = 0

        for class_ in classes:
            idx = np.where(targets_np == class_)[0]
            np.random.shuffle(idx)
            i_num = len(idx) 
            if class_ < self.cls_num//2:
                new_class = 0
                selec_idx = idx[:i_num]
                # selec_idx = selec_idx[rank::size]
                # self.neg_num += len(selec_idx)
            else:
                i_num = int(i_num * imb_factor)
                new_class = 1
                selec_idx = idx[:i_num]
                # selec_idx = selec_idx[rank::size]
                # self.pos_num += len(selec_idx)

            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([new_class, ] * len(selec_idx))

        # self.data = torch.from_numpy(np.vstack(new_data))
        # self.targets = np.array(new_targets).tolist()

        if train:
            self.data = torch.from_numpy(np.vstack(new_data))[rank::size]
            self.targets = np.array(new_targets).tolist()[rank::size]
        else:
            self.data = torch.from_numpy(np.vstack(new_data))
            self.targets = np.array(new_targets).tolist()

        Y = np.array(self.targets)

        self.neg_num = len(Y[np.any([Y == 0], axis=0)])
        self.pos_num = len(Y[np.any([Y == 1], axis=0)])

        print(rank, self.pos_num, self.neg_num)


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        return img, target

class DISTFashionMNIST(torchvision.datasets.FashionMNIST):
    cls_num = 10

    def __init__(self, root, rank, imb_factor=0.2, train=True,
                 transform=None, target_transform=None, download=False):
        super(DISTFashionMNIST, self).__init__(root, train, transform, target_transform, download)

        size = dist.get_world_size()

        if train:
            imb_factor = 0.2
        else:
            imb_factor = 1

        new_data = []
        new_targets = []

        classes = np.unique(self.targets.numpy())
        targets_np = np.array(self.targets, dtype=np.int64)
        self.neg_num = 0
        self.pos_num = 0

        for class_ in classes:
            idx = np.where(targets_np == class_)[0]
            np.random.shuffle(idx)
            i_num = len(idx) 
            if class_ < self.cls_num//2:
                new_class = 0
                selec_idx = idx[:i_num]
            else:
                new_class = 1
                i_num = int(i_num * imb_factor)

                selec_idx = idx[:i_num]

            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([new_class, ] * len(selec_idx))
        # new_data = np.vstack(new_data)

        # self.data = torch.from_numpy(new_data)
        # self.targets = np.array(new_targets).tolist()

        if train:
            self.data = torch.from_numpy(np.vstack(new_data))[rank::size]
            self.targets = np.array(new_targets).tolist()[rank::size]
        else:
            self.data = torch.from_numpy(np.vstack(new_data))
            self.targets = np.array(new_targets).tolist()

        Y = np.array(self.targets)

        self.neg_num = len(Y[np.any([Y == 0], axis=0)])
        self.pos_num = len(Y[np.any([Y == 1], axis=0)])

        print(rank, self.pos_num, self.neg_num)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        return img, target


class DISTCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    def __init__(self, root, rank, imb_factor=0.2, train=True,
                 transform=None, target_transform=None, download=False):
        super(DISTCIFAR10, self).__init__(root, train, transform, target_transform, download)

        size = dist.get_world_size()

        if train:
            imb_factor = 0.2
        else:
            imb_factor = 1

        new_data = []
        new_targets = []

        targets_np = np.array(self.targets, dtype=np.int64)
        self.neg_num = 0
        self.pos_num = 0

        for class_ in range(self.cls_num):
            idx = np.where(targets_np == class_)[0]
            np.random.shuffle(idx)
            i_num = len(idx) 
            if class_ < self.cls_num//2:
                new_class = 0
                selec_idx = idx[:i_num]
            else:
                new_class = 1
                i_num = int(i_num * imb_factor)
                selec_idx = idx[:i_num]

            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([new_class, ] * len(selec_idx))

        if train:
            self.data = np.vstack(new_data)[rank::size]
            self.targets = np.array(new_targets).tolist()[rank::size]
        else:
            self.data = np.vstack(new_data)
            self.targets = np.array(new_targets).tolist()

        Y = np.array(self.targets)

        self.neg_num = len(Y[np.any([Y == 0], axis=0)])
        self.pos_num = len(Y[np.any([Y == 1], axis=0)])

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        return img, target


class TINYIMAGENET(torchvision.datasets.ImageFolder):
    cls_num = 200

    def __init__(self, root, train=False, transform=None):
        super(TINYIMAGENET, self).__init__(root, train, transform)
        np.random.seed(0)
        rank = dist.get_rank()
        size = dist.get_world_size()

        self.transform = transform
        # self.imb_factor = 0.2

        if train:
            imb_factor = 0.2
        else:
            imb_factor = 1

        img_max = len(self.imgs) / self.cls_num
        img_num_per_cls = []

        for cls_idx in range(self.cls_num // 2):
            img_num_per_cls.append(int(img_max))

        for cls_idx in range(self.cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))

        self.pos_num = sum(img_num_per_cls[:100]) 
        self.neg_num = sum(img_num_per_cls[100:])

        new_data = []
        new_targets = []

        self.targets = [x[1] for x in self.imgs]
        self.imgs = np.array(self.imgs)

        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        for the_class, the_img_num in zip(classes, img_num_per_cls):
            # print(the_class)
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.extend(self.imgs[selec_idx, ...])
            label = the_class // 100
            new_targets.extend([label, ] * the_img_num)

        if train:
            self.samples = new_data[rank::size]
            self.targets = new_targets[rank::size]
            Y = np.array(self.targets)

            self.neg_num = len(Y[np.any([Y == 0], axis=0)])
            self.pos_num = len(Y[np.any([Y == 1], axis=0)])

        else:
            self.samples = new_data 
            self.targets = new_targets 
        # print(len(self.targets))

    def __getitem__(self, index):

        path, target = self.samples[index]
        target = int(target)
        assert os.path.isfile(path) == True, "File not exists"
        img = Image.open(path)
        sample = img.convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return sample, target // 100

def loadlib(dataset):
    # dataset = filename
    X_train, y_train = load_svmlight_file("../data/" + dataset)
    X_test, y_test = load_svmlight_file("../data/" + dataset + 't')  

    # X = torch.tensor(X_train.todense())
    X_train = X_train.todense()
    X_test  = X_test.todense()

    if dataset in ['w7a', 'w8a']:
        y_train = (y_train + 1) / 2
        y_test  = (y_test  + 1) / 2


    # print("ratio", len(idx_p), len(idx_n), len(idx_p) / (len(idx_n) + len(idx_p)))
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)

    np.save("data/" + dataset + "_X_train.npy",X_train)
    np.save("data/" + dataset + "_X_test.npy",X_test)
    np.save("data/" + dataset + "_y_train.npy",y_train)
    np.save("data/" + dataset + "_y_test.npy",y_test)

    print("Data Processed")


class Generated(torch.utils.data.Dataset):

    def __init__(self, root, train=True):
        # super(Generated, self).__init__(root, train, transform, target_transform)
        self.train = train
        # self.transform = transform
        rank = dist.get_rank()
        size = dist.get_world_size()
        if self.train:
            self.data = np.load(root + "_X_train.npy")
            self.targets = np.load(root + "_y_train.npy")
            self.dim = np.shape(self.data)[1]

            new_data = []
            new_targets = []
            num = []
            for class_ in [0, 1]:
                idx = np.where(self.targets == class_)[0]
                np.random.shuffle(idx)
                idx = idx[rank::size]

                num.append(len(idx))

                new_data.append(self.data[idx, ...])
                new_targets.extend([class_, ] * len(idx))

            self.neg_num = num[0]
            self.pos_num = num[1]

            self.data = np.vstack(new_data)
            self.targets = np.array(new_targets).tolist()

        else:
            self.data = np.load(root + "_X_test.npy")
            self.targets = np.load(root + "_y_test.npy").tolist()            


    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        # return index, torch.tensor(data), target
        return torch.tensor(data), target

    def __len__(self):
        return len(self.data)
