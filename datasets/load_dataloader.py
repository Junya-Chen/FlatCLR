import torch
import torchvision.transforms as transforms
from torchvision import datasets
from utils import accuracy, SimpleDataset, ImgDataset, setup_seed
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from .prepare_caltech import prepare_caltech_data
import torchvision.datasets.voc as voc


def get_cifar10_data_loaders(root, download, num_workers = 5, shuffle=False, batch_size=256, transfer=False):
    if transfer:
        img_transform = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
    else:
        img_transform = transforms.ToTensor()
        
    train_dataset = datasets.CIFAR10(root, train=True, download=download,
                                      transform=img_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.CIFAR10(root, train=False, download=download,
                                      transform=img_transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar100_data_loaders(root, download, num_workers=5, shuffle=False, batch_size=256, transfer=False):
    if transfer:
        img_transform = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
    else:
        img_transform = transforms.ToTensor()
        
    train_dataset = datasets.CIFAR100(root, train=True, download=download,
                                      transform=img_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=shuffle)

    test_dataset = datasets.CIFAR100(root, train=False, download=download,
                                      transform=img_transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_caltech_data_loaders(root, num_workers=5, shuffle=False, batch_size=256, transfer=True):
    if transfer:
        img_transform = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
    else:
        img_transform = transforms.ToTensor()

    with open(os.path.join(root, '101_ObjectCategories', 'train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
        y_train = pickle.load(f)

    with open(os.path.join(root, '101_ObjectCategories', 'test.pkl'), 'rb') as f:
        X_test = pickle.load(f)
        y_test = pickle.load(f)
            
    train_dataset = SimpleDataset(X_train, y_train, transform=img_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=shuffle)

    test_dataset = SimpleDataset(X_test, y_test, transform=img_transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=shuffle)
    
    return train_loader, test_loader

def get_flower_data_loaders(root, num_workers=5, shuffle=False, batch_size=256, transfer=True):
    if transfer:
        img_transform = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
    else:
        img_transform = transforms.ToTensor()

    train_dataset = datasets.ImageFolder(os.path.join(root, 'flower/train'), transform =img_transform)
    test_dataset = datasets.ImageFolder(os.path.join(root, 'flower/test'), transform =img_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=True)
    return train_loader, test_loader

def get_sun_data_loaders(root, num_workers=5, shuffle=False, batch_size=256, transfer=True):
    if transfer:
        img_transform = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
    else:
        img_transform = transforms.ToTensor()
        
    train_dataset = datasets.ImageFolder(os.path.join(root, 'SUN/train'), transform =img_transform)
    test_dataset = datasets.ImageFolder(os.path.join(root, 'SUN/val'), transform =img_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=True)
    return train_loader, test_loader

def get_imagenet_data_loaders(root, num_workers=5, shuffle=False, batch_size=256):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    val_transform = transforms.Compose([
        transforms.Resize(256,interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(os.path.join(root, 'imagenet/train'), transform =train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(root, 'imagenet/val'), transform =val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=True)
    return train_loader, test_loader

class PascalVOC_Dataset(voc.VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
                (default: alphabetic indexing of VOC's 20 classes).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
    """
    def __init__(self, root, year='2012', image_set='train', download=True, transform=None, target_transform=None):
        
        super().__init__(
             root, 
             year=year, 
             image_set=image_set, 
             download=download, 
             transform=transform, 
             target_transform=target_transform)

    def label_index(self, category):
        object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']
        return object_categories.index(category)
        
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
    
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        data, annotation = super().__getitem__(index)
        label = annotation['annotation']['object'][0]['name']
        label=self.label_index(label)
        return data, label
        
    
    def __len__(self):

        return len(self.images)

def get_voc_data_loaders(root, num_workers=5, shuffle=False, batch_size=256, transfer=True):
    if transfer:
        img_transform = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
    else:
        img_transform = transforms.ToTensor()

    train_dataset = PascalVOC_Dataset(os.path.join(root, 'VOC'),year='2007', image_set='trainval', transform =img_transform)
    test_dataset = PascalVOC_Dataset(os.path.join(root, 'VOC'),year='2007', image_set='test', transform =img_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=True)

    return train_loader, test_loader

def get_subimagenet_data_loaders(root, num_workers=5, shuffle=False, batch_size=256, ratio=0.01):
    img_transform = transforms.Compose([
        transforms.Resize(224, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    if ratio == 0.01:
        file_name='1percent_subset.pkl'
    elif ratio == 0.1:
        file_name='10percent_subset.pkl'

    with open(os.path.join('/expanse/lustre/projects/dku142/cytao/datasets/imagenet', file_name), 'rb') as f:
        train_paths = pickle.load(f)
        train_label = pickle.load(f)

    train_dataset = ImgDataset(train_paths, train_label, transform =img_transform)

    test_dataset = datasets.ImageFolder(os.path.join(root, 'imagenet/val'), transform =img_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                num_workers=num_workers, drop_last=False, shuffle=True)
    return train_loader, test_loader


def load_dataloader(args):
    if args.train_mode=='transfer' and args.transfer_mode=='linear_eval':
        transfer = True
        dataset_name = args.transfer_dataset_name
        batch_size = args.batch_size

    elif args.train_mode=='eval':
        transfer = False
        dataset_name = 'imagenet'
        batch_size = args.fc_batch_size

    elif args.train_mode=='transfer' and args.transfer_mode=='finetune':
        transfer = True
        dataset_name = args.transfer_dataset_name
        batch_size = args.fc_batch_size

    elif args.train_mode=='supervised':
        transfer=False
        dataset_name = args.transfer_dataset_name
        batch_size = args.fc_batch_size

    elif args.train_mode=='semi':
        transfer=False
        dataset_name = 'sub_imagenet'
        batch_size = args.fc_batch_size

    if dataset_name == 'caltech':
        train_loader, test_loader = get_caltech_data_loaders(root=args.data,
                                                             num_workers=args.workers, 
                                                             batch_size=batch_size,
                                                             transfer=transfer)

    elif dataset_name == 'SUN':
        train_loader, test_loader = get_sun_data_loaders(root=args.data,
                                                         num_workers=args.workers, 
                                                         batch_size=batch_size,
                                                         transfer=transfer)

    elif dataset_name == 'voc':
        train_loader, test_loader = get_voc_data_loaders(root=args.data,
                                                         num_workers=args.workers, 
                                                         batch_size=batch_size,
                                                         transfer=transfer)

    elif dataset_name == 'cifar10':
        train_loader, test_loader = get_cifar10_data_loaders(root=args.data,
                                                             download=True,
                                                             num_workers=args.workers, 
                                                             batch_size=batch_size,
                                                             transfer=transfer)
    elif dataset_name == 'cifar100':
        train_loader, test_loader = get_cifar100_data_loaders(root=args.data,
                                                              download=True,
                                                              num_workers=args.workers, 
                                                              batch_size=batch_size,
                                                              transfer=transfer)

    elif dataset_name == 'flower':
        train_loader, test_loader = get_flower_data_loaders(root=args.data,
                                                            num_workers=args.workers, 
                                                            batch_size=batch_size,
                                                            transfer=transfer)
    elif dataset_name == 'imagenet':
        train_loader, test_loader = get_imagenet_data_loaders(root=args.data,
                                                              num_workers=args.workers, 
                                                              batch_size=batch_size)

    elif dataset_name == 'sub_imagenet':
        train_loader, test_loader = get_subimagenet_data_loaders(root=args.data,
                                                              num_workers=args.workers, 
                                                              batch_size=batch_size,
                                                              ratio=args.semi_ratio)
    return train_loader, test_loader
