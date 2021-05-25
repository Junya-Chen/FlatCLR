import os
import numpy as np
import torch
import torchvision.transforms as transforms
import pickle
from collections import defaultdict
from torchvision import datasets
from PIL import Image
from torch.utils.data import DataLoader, Dataset

def prepare_cifar(ratio=100):
    transform =  transforms.Compose([transforms.RandomResizedCrop(size=32),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()])

    train_dataset = datasets.CIFAR10('./datasets/cifar', train=True, download=True,
                                     transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True)
    train_data = []
    train_label = []
    for x,y in train_loader:
        train_data.append(x)
        train_label.append(y)

    train_data = torch.cat(train_data)
    train_label = torch.cat(train_label).tolist()

    label_dict = defaultdict(list)
    for idx, label in enumerate(train_label):
        label_dict[label].append(idx)

    selected_idx =[]
    for label in label_dict.keys():
        selected_idx.extend(np.random.choice(label_dict[label], len(label_dict[label])//ratio)) 

    subsample_data = []
    subsample_label =[]

    for idx in selected_idx:
        subsample_data.append(train_data[idx])
        subsample_label.append(train_label[idx])

    subsample_data = torch.stack(subsample_data)

    with open('datasets/cifar/sub_cifar.pkl','wb') as f:
        pickle.dump(subsample_data, f)
        pickle.dump(subsample_label, f)
        
def prepare_cifar100(ratio=100):
    transform =  transforms.Compose([transforms.RandomResizedCrop(size=32),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()])

    train_dataset = datasets.CIFAR100('./datasets/cifar100', train=True, download=True,
                                     transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True)
    train_data = []
    train_label = []
    for x,y in train_loader:
        train_data.append(x)
        train_label.append(y)

    train_data = torch.cat(train_data)
    train_label = torch.cat(train_label).tolist()

    label_dict = defaultdict(list)
    for idx, label in enumerate(train_label):
        label_dict[label].append(idx)

    selected_idx =[]
    for label in label_dict.keys():
        selected_idx.extend(np.random.choice(label_dict[label], len(label_dict[label])//ratio)) 

    subsample_data = []
    subsample_label =[]

    for idx in selected_idx:
        subsample_data.append(train_data[idx])
        subsample_label.append(train_label[idx])

    subsample_data = torch.stack(subsample_data)

    with open('datasets/cifar100/sub_cifar100.pkl','wb') as f:
        pickle.dump(subsample_data, f)
        pickle.dump(subsample_label, f)        


def prepare_tinyimagenet(root = 'datasets/tiny_imagenet/tiny-imagenet-200'):
    label_dict = dict()
    with open(os.path.join(root, 'wnids.txt'), 'rb') as f:
        for i, line in enumerate(f):
            label_dict[line.split()[0]] = i

    train_img_path = []
    train_label = []

    for dirs in label_dict.keys():
        cur_path = os.path.join(root, 'train', dirs.decode('UTF-8'), 'images', dirs.decode('UTF-8'))
        for num in range(0,200):
            train_img_path.append(cur_path+'_'+str(num)+'.JPEG')
            train_label.append(label_dict[dirs])

    val_img_path = []
    val_label = []
    with open(os.path.join(root, 'val/val_annotations.txt'), 'rb') as f:
        for line in enumerate(f):
            cur_path = os.path.join(root, 'val/images', line[1].split()[0].decode('UTF-8'))
            val_img_path.append(cur_path)
            val_label.append(label_dict[line[1].split()[1]])
    return train_img_path, train_label, val_img_path, val_label

class TinyDataset(Dataset):
    def __init__(self, root = None, train=True, transform=transforms.ToTensor()):
        train_img_path, train_label, val_img_path, val_label = prepare_tinyimagenet()
        if train:
            img_paths = train_img_path
            labels = train_label
        else:
            img_paths = val_img_path
            labels = val_label
            
        self.img_path = img_paths
        self.targets = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB') 
        if self.transform:
            sample = self.transform(sample)
        label = self.targets[index]
        return sample, label
    
train_loader = DataLoader(TinyDataset(train=True), batch_size=2*128,
                                num_workers=5, drop_last=False, shuffle=True)
test_loader = DataLoader(TinyDataset(train=False), batch_size=2*128,
                                num_workers=5, drop_last=False, shuffle=False)

data = []
label = []
for x,y in train_loader:
    data.append(x)
    label.append(y)
    
train_data = torch.cat(data)
train_label = torch.cat(label)

data = []
label = []
for x,y in test_loader:
    data.append(x)
    label.append(y)
    
test_data = torch.cat(data)
test_label = torch.cat(label)
with open('datasets/tiny_imagenet/train_tiny.pkl', 'wb') as f:
    pickle.dump(train_data, f)
    pickle.dump(train_label, f)
    
with open('datasets/tiny_imagenet/test_tiny.pkl', 'wb') as f:
    pickle.dump(test_data, f)
    pickle.dump(test_label, f)
    
def prepare_tinyimagenet_v2(ratio=100):    
    with open('datasets/tiny_imagenet/train_tiny.pkl', 'rb') as f:
        train_data = pickle.load(f)
        train_label = pickle.load(f)
    
    label_dict = defaultdict(list)
    for idx, label in enumerate(train_label):
        label_dict[label].append(idx)

    selected_idx =[]
    for label in label_dict.keys():
        selected_idx.extend(np.random.choice(label_dict[label], len(label_dict[label])//ratio)) 

    subsample_data = []
    subsample_label =[]

    for idx in selected_idx:
        subsample_data.append(train_data[idx])
        subsample_label.append(train_label[idx])

    with open('datasets/tiny_imagenet/sub_tiny.pkl','wb') as f:
        pickle.dump(subsample_data, f)
        pickle.dump(subsample_label, f)