import torch
import logging
import numpy as np
import os
import pickle
import torchvision
from tqdm import tqdm_notebook,tqdm
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from models.load_model import load_model
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from optimizer.load_optimizer_scheduler import load_fc_optimizer_scheduler
from models.utils import SupervisedHead, Identity
from utils import save_config_file, accuracy, save_checkpoint

def train_fc(epo, train_loader, test_loader, args):  
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint_{:04d}.pth.tar'.format(epo)),
                            map_location=args.device) 
    state_dict = checkpoint['model_state_dict']
    model = load_model(args)
    
    in_channel = model.fc.in_features
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            if k.startswith('module') and not k.startswith('module.fc'):
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
    
    fc_optimizer, fc_scheduler = load_fc_optimizer_scheduler(model, args, train_loader)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(args.device)
            
    if args.fc_train_from:
        print('train_from')
        checkpoint = torch.load(os.path.join(args.fc_log_dir, args.fc_train_from),
                                map_location=args.device) 
        fc_log_dir = args.fc_log_dir
        epo_0 = int(args.fc_train_from[-12:-8])+1
        fc_model_state_dict = checkpoint['fc_model_state_dict']
        model.module.fc.load_state_dict(fc_model_state_dict)
        fc_optimizer_state_dict = checkpoint['fc_optimizer']
        fc_optimizer.load_state_dict(fc_optimizer_state_dict) 
        if fc_scheduler:
            fc_scheduler_state_dict = checkpoint['fc_scheduler']
            fc_scheduler.load_state_dict(fc_scheduler_state_dict) 
        
        best_epoch = checkpoint['best_epoch']
        best_top1_accuracy = checkpoint['best_top1_acc']
        best_top5_accuracy = checkpoint['best_top5_acc']
        del checkpoint

    elif args.fc_from_pretrained==False:
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
        fc_log_dir=os.path.join('results',f"{args.dataset_name}", f"{args.clr}", f"{args.batch_size}_{epo}_fc", dt_string)
        epo_0 = 0  
        best_epoch = 0
        best_top1_accuracy = 0.
        best_top5_accuracy = 0.
        
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    
    writer = SummaryWriter(fc_log_dir)        
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)
    
    scaler = GradScaler(enabled=args.fp16_precision)
    # save config file
    save_config_file(writer.log_dir, args)
    logging.info(f"Start FC training for {args.epochs} epochs.")
    logging.info(f"Training with gpu: {args.disable_cuda}.")
    

    for epoch_counter in tqdm(range(epo_0, args.fc_epochs)):
        top1_train_accuracy = 0
        model.train()
        for counter, (x_batch, y_batch) in enumerate(tqdm(train_loader)):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            fc_optimizer.zero_grad()
            loss.backward()
            fc_optimizer.step()
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]
            if fc_scheduler:
                fc_scheduler.step()
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
        
        print(top1_accuracy.item(), top5_accuracy.item())
        
        if best_top1_accuracy < top1_accuracy.item():
            best_top1_accuracy = top1_accuracy.item()
            best_epoch = epoch_counter
            
        if best_top5_accuracy < top5_accuracy.item():
            best_top5_accuracy = top5_accuracy.item()          
            
        if fc_scheduler:
            print('model_learning_rate', fc_scheduler.get_lr()[0])
        writer.add_scalar('loss', loss, global_step=epoch_counter)
        writer.add_scalar('acc/top1', top1[0], global_step=epoch_counter)
        writer.add_scalar('acc/top5', top5[0], global_step=epoch_counter)
        if fc_scheduler:
            writer.add_scalar('learning_rate', fc_scheduler.get_lr()[0], global_step=epoch_counter)

        if epoch_counter%args.fc_save_every_n_epochs ==0:       
            checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
            if fc_scheduler:
                save_checkpoint({
                        'model': args.model,
                        'fc_model_state_dict': model.module.fc.state_dict(),
                        'fc_optimizer': fc_optimizer.state_dict(),
                        'fc_scheduler': fc_scheduler.state_dict(),
                        'best_top1_acc': best_top1_accuracy,
                        'best_top5_acc': best_top5_accuracy,
                        'best_epoch': best_epoch,
                }, is_best=False, filename=os.path.join(writer.log_dir, checkpoint_name))
            else:
                save_checkpoint({
                        'model': args.model,
                        'fc_model_state_dict': model.module.fc.state_dict(),
                        'fc_optimizer': fc_optimizer.state_dict(),
                        'best_top1_acc': best_top1_accuracy,
                        'best_top5_acc': best_top5_accuracy,
                        'best_epoch': best_epoch,
                }, is_best=False, filename=os.path.join(writer.log_dir, checkpoint_name))

        logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

    logging.info("Training has finished.")
    # save model checkpoints
    checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
    if fc_scheduler:
        save_checkpoint({
                'model': model,
                'fc_model_state_dict': model.module.fc.state_dict(),
                'fc_optimizer': fc_optimizer.state_dict(),
                'fc_scheduler': fc_scheduler.state_dict(),
                'best_top1_acc': best_top1_accuracy,
                'best_top5_acc': best_top5_accuracy,
                'best_epoch': best_epoch,
        }, is_best=False, filename=os.path.join(writer.log_dir, checkpoint_name))
    else:
        save_checkpoint({
                'model': args.model,
                'fc_model_state_dict': model.module.fc.state_dict(),
                'fc_optimizer': fc_optimizer.state_dict(),
                'best_top1_acc': best_top1_accuracy,
                'best_top5_acc': best_top5_accuracy,
                'best_epoch': best_epoch,
        }, is_best=False, filename=os.path.join(writer.log_dir, checkpoint_name))

    logging.info(f"Model checkpoint and metadata has been saved at {writer.log_dir}.")
    return top1_train_accuracy, best_top1_accuracy, best_top5_accuracy, best_epoch
