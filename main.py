import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from models.load_model import load_model
from optimizer.load_optimizer_scheduler import load_optimizer_scheduler
from models.utils import ProjectionHead, SupervisedHead
from datasets.load_dataloader import load_dataloader, get_cifar10_data_loaders
from utils import *
from simclr import SimCLR
from flatclr import FlatCLR
from datetime import datetime
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.contrastive_learning_dataset_new import NewContrastiveDataSet, load_data
from train_fc import train_fc
from fine_tune import train_ft
from train_supervised import train_supervised

parser = argparse.ArgumentParser(description='Pytorch Imagenet Training')
parser.add_argument('--data', type=str, default='.datasets',help='location of the data corpus')
parser.add_argument('--dataset_name', type=str, default='imagenet', help='[imagenet, cifar10, cifar100]')
parser.add_argument('--clr', type=str, default='flatclr', help='training loss [simclr, flatclr]')
parser.add_argument('--faster_version', type=bool, default=False, help='faster version with new dataloader (only supported for cifar10 and cifar100)')
parser.add_argument('--workers', type=int, default=40, help='num of workers')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--lr', default=0.6, type=float, help='optimizer lr (0.3*batch_size/128)')
parser.add_argument('--optimizer', default='lars', type=str, help='optimizer type [lars, adam, sgd]')
parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay for optimizers')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd')
parser.add_argument('--use_scheduler', default=True, type=bool, help='use scheduler or not')
parser.add_argument('--scheduler', default='cosine', type=str, help='lr scheduler type [constant, cosine, None]')
parser.add_argument('--warmup_epochs', default=10, type=int, help='lr warmup epochs')
parser.add_argument('--model', default='res50', type=str, help='model type [res10, res18, res50, res101, res152]')
parser.add_argument('--proj_head_mode', default='nonlinear', type=str, help='[nonlinear, linear, none]')
parser.add_argument('--num_proj_layers', default=2, type=int, help='only for nonlinear proj head')
parser.add_argument('--proj_out_dim', default=128, type=int, help='dim of projection head')
parser.add_argument('--n_views', default=2, type=int, help='n_view=2 only')
parser.add_argument('--save_every_n_epochs', default=2, type=int, help='save frequency /epoch')
parser.add_argument('--temperature', default=0.1, type=float, help='temperature for simclr and flatclr')
parser.add_argument('--fc_epochs', default=100, type=int, help='number of total epochs for downstream tasks to run')
parser.add_argument('--fc_batch_size', default=4096, type=int, help='batch size for downstream tasks')
parser.add_argument('--fc_lr', default=1., type=float, help='lr for downstream tasks')
parser.add_argument('--fc_optimizer', default='lars', type=str, help='fc_optimizer type [lars, adam, sgd, LBFGS]')
parser.add_argument('--reg_weight', default=0., type=float, help='grid search')
parser.add_argument('--fc_scheduler', default='cosine', type=str, help='fc_lr scheduler type [constant, cosine, None]')
parser.add_argument('--fc_weight_decay', default=0., type=float, help='weight decay for downstream tasks')
parser.add_argument('--fc_momentum', default=0.9, type=float, help='momentum for downstream tasks')
parser.add_argument('--disable_cuda', action='store_true')
parser.add_argument('--fp16_precision', action='store_true')
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--train_mode', type=str, default='ssl', help='[ssl, eval, transfer, semi, supervised]')
parser.add_argument('--semi_ratio', type=float, default=0.1, help='[0.1, 0.01]')
parser.add_argument('--transfer_mode', type=str, default='linear_eval', help='[finetune, linear_eval]')
parser.add_argument('--transfer_dataset_name', type=str, default='caltech', help='[imagenet, caltech, voc, flower, cifar10, cifar100, SUN]')
parser.add_argument('--transfer_from_epoch', default=100, type=int, help='resume checkpoint')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='model saving dir')
parser.add_argument('--save_dir', type=str, default='results', help='result saving dir')
parser.add_argument('--log_dir', default=None, type=str, help='None for learning from scratch')
parser.add_argument('--train_from', default=False, type=bool, help='resume checkpoint')
parser.add_argument('--from_pretrained', default=False, type=bool, help='learning from scratch or from pretrained')
parser.add_argument('--seed', type=int, default=42, help='random seed')

###### ssl ######
def main_ssl(args):
    model = load_model(args)
    if args.faster_version:
        if args.dataset_name == 'cifar10' or args.dataset_name == 'cifar100': 
            print(args.dataset_name)
            X_train, y_train, X_test, y_test = load_data(args)
            train_dataset = NewContrastiveDataSet(X_train, y_train, args.dataset_name)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
        else:
            assert False
    else:
        dataset = ContrastiveLearningDataset(args.data)
        train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)
    ch = model.fc.in_features
    model.fc = ProjectionHead(args,ch)
    optimizer, scheduler = load_optimizer_scheduler(model, args, train_loader)
    
    if args.clr == 'simclr':
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)
    elif args.clr == 'flatclr':
        flatclr = FlatCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        flatclr.train(train_loader)

###### linear evaluation ######        
def main_eval(args):
    train_loader,test_loader = load_dataloader(args)
    result = []
    for epo in range(args.save_every_n_epochs, args.epochs+args.save_every_n_epochs, args.save_every_n_epochs):
        train_top1, test_top1, test_top5, best_epoch = train_fc(epo=epo, 
                                                                train_loader=train_loader, 
                                                                test_loader=test_loader, 
                                                                args=args)
        print('epoch', epo, 'train_top1', train_top1, 'test_top1', test_top1, 'test_top5', test_top5, 'best_epoch',best_epoch)   
        result.append((train_top1, test_top1, test_top5))
        with open(os.path.join(args.save_dir, 'linear_eval.pkl'), 'wb') as f:
            pickle.dump(result, f)
    print(result)
    return
            
###### transfer learning ######  
def main_transfer(args):
    print('transfer')
    train_loader, test_loader = load_dataloader(args)
    train_top1, test_top1, test_top5, best_epoch = train_fc(epo=args.transfer_from_epoch,
                                                            train_loader=train_loader, 
                                                            test_loader=test_loader, 
                                                            args=args)
    print('train_top1', train_top1, 'test_top1', test_top1, 'test_top5', test_top5, 'best_epoch', best_epoch) 
    result =[train_top1, test_top1, test_top5]
    with open(os.path.join(args.save_dir, args.transfer_dataset_name + 'transfer.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return 


###### transfer learning via fine-tune ######  
def main_transfer_ft(args):
    train_loader, test_loader = load_dataloader(args)
    train_top1, test_top1, test_top5, best_epoch = train_ft(epo=args.transfer_from_epoch, 
                                                            train_loader=train_loader, 
                                                            test_loader=test_loader, 
                                                            args=args)
    print('train_top1', train_top1, 'test_top1', test_top1, 'test_top5', test_top5, 'best_epoch', best_epoch) 
    result =[train_top1, test_top1, test_top5]
    with open(os.path.join(args.save_dir, args.transfer_dataset_name + 'transfer.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return 

def main_supervised(args):
    train_loader, test_loader = load_dataloader(args)
    train_top1, test_top1, test_top5, best_epoch = train_supervised(train_loader=train_loader, 
                                                                    test_loader=test_loader, 
                                                                    args=args)
    print('train_top1', train_top1, 'test_top1', test_top1, 'test_top5', test_top5, 'best_epoch', best_epoch) 
    result =[train_top1, test_top1, test_top5]
    with open(os.path.join(args.save_dir, args.transfer_dataset_name + 'transfer.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return 

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    if args.train_mode =='ssl':
        main_ssl(args)
    elif args.train_mode =='eval':
        main_eval(args)
    elif args.train_mode == 'transfer' and args.transfer_mode=='linear_eval':
        main_transfer(args)
    elif args.train_mode == 'transfer' and args.transfer_mode=='finetune':
        main_transfer_ft(args)
    elif args.train_mode == 'supervised':
        main_supervised(args)
    elif args.train_mode =='semi':
        args.transfer_dataset_name='imagenet'
        main_transfer_ft(args)
