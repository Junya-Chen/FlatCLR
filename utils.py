import torch
import torch.nn as nn
import os
import time
import numpy as np
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from PIL import Image



class SimpleDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.data = x
        self.targets = y
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)


class ImgDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.paths = x
        self.targets = y
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        img = Image.open(self.paths[0])
        target = self.targets[0]

        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[index]
        return img, target

    def __len__(self):
        return len(self.targets)
    


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class logger(object):
    def __init__(self, path, local_rank=0):
        self.path = path
        self.local_rank = local_rank

    def info(self, msg):
        if self.local_rank == 0:
            print(msg)
            with open(os.path.join(self.path, "log.txt"), 'a') as f:
                f.write(msg + "\n")


def change_batchnorm_momentum(module, value):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = value
    for name, child in module.named_children():
        change_batchnorm_momentum(child, value)