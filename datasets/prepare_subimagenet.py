import os
import numpy as np
from PIL import Image
import pickle
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

def prepare_sub_imagenet(root, ratio):
    root = os.path.join('/expanse/lustre/projects/dku142/cytao/datasets/imagenet')
    if ratio ==0.1:
        file1 = open(os.path.join(root, '10percent.txt'), 'r')
    elif ratio ==0.01:
        file1 = open(os.path.join(root, '1percent.txt'), 'r')
    Lines = file1.readlines()
    subset_labels = []
    subset_img_paths = []
    for line in Lines:
        line = line.rstrip()
        label = line.split("_")[0]
        subset_labels.append(label)
        subset_img_paths.append(os.path.join(root, 'train', label, line))
        
    return subset_img_paths, subset_labels

def prepare_subimagenet_data(root='/expanse/lustre/projects/dku142/cytao/datasets/imagenet'):
    test_dataset = datasets.ImageFolder(os.path.join(root, 'val'))
    subset_img_paths, subset_labels = prepare_sub_imagenet(root, .1)
    train_data = []
    train_label =[]
    for i, ind in enumerate(subset_labels):
        train_label.append(test_dataset.class_to_idx[ind])
        train_data.append(subset_img_paths[i])
    

    with open(os.path.join(root, '10percent_subset.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
        pickle.dump(train_label, f)

    opc_img_paths, opc_labels=prepare_sub_imagenet(root, .01)
    train_data = []
    train_label =[]
    for i, ind in enumerate(opc_labels):
        train_label.append(test_dataset.class_to_idx[ind])
        train_data.append(opc_img_paths[i])

    with open(os.path.join(root, '1percent_subset.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
        pickle.dump(train_label, f)

if __name__ == '__main__':
    prepare_subimagenet_data()