import torch
import numpy as np
import os
import pickle
import torchvision
from torch.utils.data import DataLoader, Dataset
from utils import accuracy, SimpleDataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from models.load_model import load_model
from datasets.load_dataloader import load_dataloader
from models.utils import SupervisedHead, Identity
from optimizer import load_fc_optimizer_scheduler



def train_supervised(train_loader, test_loader, args):
    best_epoch = 0
    best_top1_accuracy = 0.
    best_top5_accuracy = 0.

    model = load_model(args)
    in_channel = model.fc.in_features

    if args.train_mode =='supervised':
        dataset_name = args.transfer_dataset_name

    if dataset_name == 'cifar10':
        model.fc = SupervisedHead(in_channel, num_classes=10)
    elif dataset_name == 'cifar100':
        model.fc = SupervisedHead(in_channel, num_classes=100)
    elif dataset_name == 'imagenet':
        model.fc = SupervisedHead(in_channel, num_classes=1000)
    elif dataset_name == 'caltech':
        model.fc = SupervisedHead(in_channel, num_classes=101)
    elif dataset_name == 'SUN':
        model.fc = SupervisedHead(in_channel, num_classes=397)
    elif dataset_name == 'food':
        model.fc = SupervisedHead(in_channel, num_classes=100)
    elif dataset_name == 'voc':
        model.fc = SupervisedHead(in_channel, num_classes=20)
    elif dataset_name == 'flower':
        model.fc = SupervisedHead(in_channel, num_classes=102)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(args.device)


    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    fc_optimizer, fc_scheduler =load_fc_optimizer_scheduler(fc_model, args, train_loader)


    for epoch in range(args.fc_epochs):
        top1_train_accuracy = 0
        for counter,(images,labels) in enumerate(train_loader):
            images = images.to(args.device)
            labels = labels.to(args.device).long()
            pred = model(images)
            loss = criterion(pred, labels)
            top1 = accuracy(pred, labels, topk=(1,))
            top1_train_accuracy += top1[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0

        for counter, (images, labels) in enumerate(test_loader):
            images = images.to(args.device)
            labels = labels.to(args.device).long()
            pred = model(images)
            top1, top5 = accuracy(pred, labels, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

        if best_top1_accuracy < top1_accuracy:
            best_top1_accuracy = top1_accuracy
            best_epoch = epoch
        if best_top5_accuracy < top5_accuracy:  
            best_top5_accuracy = top5_accuracy 

        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
    
    return top1_train_accuracy.item(), best_top1_accuracy.item(), best_top5_accuracy.item(), best_epoch 