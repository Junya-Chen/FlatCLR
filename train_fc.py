import torch
import numpy as np
import os
import pickle
import torchvision
from tqdm import tqdm_notebook
from torch.utils.data import DataLoader, Dataset
from utils import accuracy, SimpleDataset
import torchvision.transforms as transforms
from models.load_model import load_model
from models.utils import SupervisedHead, Identity

def train_fc(epo, train_loader, test_loader, args):  
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint_{:04d}.pth.tar'.format(epo)),
                            map_location=args.device) 
    state_dict = checkpoint['model_state_dict']
    model = load_model(args)
    del checkpoint
    in_channel = model.fc.in_features
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            if k.startswith('module') and not k.startswith('module.fc'):
              # remove prefix
              state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    
    if args.dataset_name == 'cifar10':
        model.fc = SupervisedHead(in_channel, num_classes=10)
    elif args.dataset_name == 'cifar100':
        model.fc = SupervisedHead(in_channel, num_classes=100)
    elif args.dataset_name == 'tiny_imagenet':
        model.fc = SupervisedHead(in_channel, num_classes=200)
    elif args.dataset_name == 'imagenet':
        model.fc = SupervisedHead(in_channel, num_classes=1000)
    
    for name, param in model.named_parameters():
        if not name.startswith('fc.linear_layer'):
            param.requires_grad = False
    
    if args.fc_optimizer == 'adam':
        fc_optimizer = torch.optim.Adam(model.parameters(), lr=args.fc_lr, weight_decay=args.fc_weight_decay)
    elif args.fc_optimizer == 'lars':
        fc_optimizer = LARS(model.parameters(), lr=args.fc_lr, weight_decay=args.fc_weight_decay,
                        exclude_from_weight_decay=["batch_normalization", "bias"])
    elif args.fc_optimizer == 'sgd':
        fc_optimizer = torch.optim.SGD(model.parameters(), lr=args.fc_lr, 
                                    weight_decay=args.fc_weight_decay, momentum=args.fc_momentum)          
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    best_epoch = 0
    best_top1_accuracy = 0.
    for epoch in tqdm(range(args.fc_epochs)):
        top1_train_accuracy = 0
        model.train()
        for counter, (x_batch, y_batch) in enumerate(tqdm(train_loader)):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            fc_optimizer.zero_grad()
            loss.backward()
            fc_optimizer.step()
            
        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0
        
        model.eval()
        for counter, (x_batch, y_batch) in enumerate(tqdm(test_loader)):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)
            logits = model(x_batch).detach()
            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        
        if best_top1_accuracy < top1_accuracy:
            best_top1_accuracy = top1_accuracy
            best_epoch = epoch
    return top1_train_accuracy.item(), best_top1_accuracy.item(), top5_accuracy.item(), best_epoch   
