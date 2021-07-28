import torch
import numpy as np
import copy
import os
import pickle
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from utils import accuracy, SimpleDataset
import torchvision.transforms as transforms
from optimizer.lars import LARS
from optimizer.load_optimizer_scheduler import load_fc_optimizer_scheduler
from models.load_model import load_model
from models.utils import SupervisedHead, Identity

def train_fc(epo, train_loader, test_loader, args):      
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint_{:04d}.pth.tar'.format(epo)),
                            map_location=args.device)  

    state_dict = checkpoint['model_state_dict']

    model = load_model(args)
    in_channel = model.fc.in_features
    
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            if k.startswith('module') and not k.startswith('module.fc'):
              # remove prefix
              state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]
    
    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    model.fc = Identity()        
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(args.device)
    
    model.eval()
    features = dict()
    for split, loader in zip(['train', 'test'], [train_loader, test_loader]):
        features[split] = dict()
        features[split]['features'] = []
        features[split]['labels'] = []
        for counter, (x_batch, y_batch) in enumerate(tqdm(loader)):
	        x_batch = x_batch.to(args.device)
	        features[split]['features'].append(copy.deepcopy(model(x_batch).detach().cpu()))
	        features[split]['labels'].append(copy.deepcopy(y_batch))
	    features[split]['features'] = torch.cat(features[split]['features'], 0)
	    features[split]['labels'] = torch.cat(features[split]['labels'], 0)
        
        
  	with open('features.pickle', 'wb') as f:
	    pickle.dump(features, f, protocol=4)
    del model, features, checkpoint

    with open('features.pickle', 'rb') as f:
        features = pickle.load(f)

    train_dataset = SimpleDataset(features['train']['features'], features['train']['labels'])
    test_dataset = SimpleDataset(features['test']['features'], features['test']['labels'])
    
    train_loader = DataLoader(train_dataset, batch_size=args.fc_batch_size,
                              num_workers=args.workers, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.fc_batch_size,
                             num_workers=args.workers, drop_last=True, shuffle=False)
    
    X_train = features['train']['features'].to(args.device)
    y_train = features['train']['labels'].to(args.device)
    X_test = features['test']['features'].to(args.device)
    y_test = features['test']['labels'].to(args.device)
    
    if args.train_mode =='eval':
        dataset_name = args.dataset_name
    elif args.train_mode =='transfer':
        dataset_name = args.transfer_dataset_name
        
    if dataset_name == 'cifar10':
        fc_model = SupervisedHead(in_channel, num_classes=10)
    elif dataset_name == 'cifar100':
        fc_model = SupervisedHead(in_channel, num_classes=100)
    elif dataset_name == 'imagenet':
        fc_model = SupervisedHead(in_channel, num_classes=1000)
    elif dataset_name == 'caltech':
        fc_model = SupervisedHead(in_channel, num_classes=101)
    elif dataset_name == 'SUN':
        fc_model = SupervisedHead(in_channel, num_classes=397)
    elif dataset_name == 'food':
        fc_model = SupervisedHead(in_channel, num_classes=100)
    elif dataset_name == 'voc':
        fc_model = SupervisedHead(in_channel, num_classes=20)
    elif dataset_name == 'flower':
        fc_model = SupervisedHead(in_channel, num_classes=102)
        
    if torch.cuda.device_count() > 1:
        fc_model = torch.nn.DataParallel(fc_model)
    fc_model = fc_model.to(args.device)
    
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    fc_optimizer, fc_scheduler =load_fc_optimizer_scheduler(fc_model, args, train_loader)


    best_epoch = 0
    best_top1_accuracy = 0.
    best_top5_accuracy = 0.
    for epoch in tqdm(range(args.fc_epochs)):
        top1_train_accuracy = 0
        for counter in range(len(train_loader)):
            ind = np.random.choice(len(X_train), args.fc_batch_size)
            x_batch = X_train[ind]
            y_batch = y_train[ind]
            logits = fc_model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            fc_optimizer.zero_grad()
            loss.backward()
            fc_optimizer.step()
            if fc_scheduler:
                fc_scheduler.step()
            
        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0
        
        for counter in range(len(test_loader)):
            st = counter*args.fc_batch_size
            ed = (counter+1)*args.fc_batch_size
            x_batch = X_test[st:ed]
            y_batch = y_test[st:ed]
            logits = fc_model(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        
        if best_top1_accuracy < top1_accuracy:
            best_top1_accuracy = top1_accuracy
            best_epoch = epoch
        if best_top5_accuracy < top5_accuracy:  
            best_top5_accuracy = top5_accuracy 

    return top1_train_accuracy.item(), best_top1_accuracy.item(), best_top5_accuracy.item(), best_epoch
