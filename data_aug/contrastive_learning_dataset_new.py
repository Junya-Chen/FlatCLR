import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch import nn
from copy import deepcopy
from torchvision.transforms import transforms
from tqdm import tqdm
import pickle



def load_data(args, device=None, download=True):
    if not device:
        device = args.device
    root_folder = args.data
    name = args.dataset_name
    if name=='cifar10':
        train_dataset = datasets.CIFAR10(root_folder, train=True, download=download)
        test_dataset = datasets.CIFAR10(root_folder, train=False, download=download)
        X_train = train_dataset.data
        X_test = test_dataset.data
        X_train = torch.Tensor(np.array(X_train,dtype=np.float32)).permute([0,3,1,2])/255.
        X_test = torch.Tensor(np.array(X_test,dtype=np.float32)).permute([0,3,1,2])/255.
        y_train = torch.Tensor(train_dataset.targets)
        y_test = torch.Tensor(test_dataset.targets)
        return X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)
    
    if name=='cifar100':
        train_dataset = datasets.CIFAR100(root_folder, train=True, download=download)
        test_dataset = datasets.CIFAR100(root_folder, train=False, download=download)
        X_train = train_dataset.data
        X_test = test_dataset.data
        X_train = torch.Tensor(np.array(X_train,dtype=np.float32)).permute([0,3,1,2])/255.
        X_test = torch.Tensor(np.array(X_test,dtype=np.float32)).permute([0,3,1,2])/255.
        y_train = torch.Tensor(train_dataset.targets)
        y_test = torch.Tensor(test_dataset.targets)
        return X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)

    
    elif name=='tiny_imagenet':
        with open('./datasets/tiny_imagenet/train_tiny.pkl', 'rb') as f:
            X_train = pickle.load(f)
            y_train = pickle.load(f).float()
        with open('./datasets/tiny_imagenet/test_tiny.pkl', 'rb') as f:
            X_test = pickle.load(f)
            y_test = pickle.load(f).float()
        y_train = torch.Tensor(y_train)
        y_test = torch.Tensor(y_test)
        return X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)


class NewContrastiveDataSet(Dataset):
    def __init__(self, X, Y, name, n_view = 2, batch_size = 128):
        self.X = X
        self.Y = Y
        self.n_view = n_view
        self.name = name
        self.X1 = None
        self.X2 = None
        self.idx = None
        self.Y1 = None
        self.batch_size = batch_size
        self.refresh()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        x1 = self.X1[idx]
        x2 = self.X2[idx]
        y1 = self.Y1[idx]
        
        return [x1,x2],y1

    def refresh(self):
        if self.name == 'cifar10':
            size = 32
            s = .5
            gaussian = True
        elif self.name == 'cifar100':
            size = 32
            s = .5
            gaussian = False
        elif self.name == 'tiny_imagenet':
            size = 64
            s = 0.
            gaussian = False
        elif self.name =='stl10':
            size = 96
            s = 1
            gaussian = False
        elif self.name =='imagenet':
            size = 224
            s = 1
            gaussian = True
        device = self.X.device
        
        transforms_batch = get_simclr_pipeline_newtransform(size, device, s, gaussian)
        transforms_individual = []

        idx = np.random.permutation(len(self.X))
        
        self.X1 = deepcopy(self.X)
        self.X2 = deepcopy(self.X)
        self.Y1 = deepcopy(self.Y)
        
        n = len(self.X)
        batch_size = self.batch_size
        
#         print('Individual Trans')
        for k,transform in enumerate(transforms_individual):
            idx = np.random.permutation(len(self.X))
            self.X1 = self.X1[idx]
            self.X2 = self.X2[idx]
            self.Y1 = self.Y1[idx]
            for i in range(n):
                self.X1[i] = transform(self.X1[i])
                self.X2[i] = transform(self.X2[i])
            
        for k,transform in enumerate(transforms_batch):
            idx = np.random.permutation(len(self.X))
            self.X1 = self.X1[idx]
            self.X2 = self.X2[idx]
            self.Y1 = self.Y1[idx]
            
#             print('Group trans %d/%d' % (k+1,len(transforms_batch)))
            for j in range(int(np.ceil(float(n)/batch_size))):
                st = j*batch_size
                ed = min((j+1)*batch_size,n)
                
                x = self.X1[st:ed]
                x = transform(x)
                self.X1[st:ed] = x
                x = self.X2[st:ed]
                x = transform(x)
                self.X2[st:ed] = x
                
        return 
                  
class newGaussianBlur(object):
    """blur a single image on CPU"""
    """blur multi images on GPU"""
    def __init__(self, kernel_size,device='cpu'):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3).to(device)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3).to(device)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
        return img
    
def get_simclr_pipeline_newtransform(size, device, s=1, gaussian = False):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    if gaussian:
        transforms_batch = [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([newGaussianBlur(kernel_size=int(0.1 * size), device=device)],p=.5)
            ]
    else:
         transforms_batch = [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ]
    return transforms_batch
