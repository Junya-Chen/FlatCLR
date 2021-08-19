import logging
import os
import sys
import torch
import torch.nn.functional as F
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)

class FlatCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = kwargs['model']
        if torch.cuda.device_count() > 1:
            print('multiple gpu')
            self.model = torch.nn.DataParallel(kwargs['model'])
        self.model = self.model.to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        
        if self.args.train_from:
            checkpoint = torch.load(os.path.join(self.args.log_dir, self.args.train_from),
                                    map_location=self.args.device) 
            log_dir = self.args.log_dir
            self.epo_0 = int(self.args.train_from[-12:-8])+1
            model_state_dict = checkpoint['model_state_dict']
            self.model.load_state_dict(model_state_dict)
            optimizer_state_dict = checkpoint['optimizer']
            self.optimizer.load_state_dict(optimizer_state_dict) 
            scheduler_state_dict = checkpoint['scheduler']
            self.scheduler.load_state_dict(scheduler_state_dict) 
            del checkpoint
            
        elif self.args.from_pretrained==False:
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
            log_dir=os.path.join('results',f"{self.args.dataset_name}", f"{self.args.batch_size}_FlatCLR", dt_string)
            self.epo_0 = 0
            
        else:            
            checkpoint = torch.load('results/imagenet/512_SimCLR/15-05-2021-06-02-45/checkpoint_0009.pth.tar',
                                    map_location=self.args.device) 
            model_state_dict = checkpoint['model_state_dict']
            self.model.load_state_dict(model_state_dict)
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
            log_dir=os.path.join('results',f"{self.args.dataset_name}",f"{self.args.batch_size}_FlatCLR", dt_string)
            self.epo_0 = 0

        self.writer = SummaryWriter(log_dir)        
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def flat_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)

        labels = torch.zeros(positives.shape[0], dtype=torch.long).to(self.args.device) #-

#        logits = logits / self.args.temperature #- 
        logits = (negatives - positives)/self.args.temperature # (512,510) #+
#         labels = None #+
        return logits, labels


    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Start FlatCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        
        for epoch_counter in range(self.epo_0, self.args.epochs):
            for i, (images, _) in enumerate(tqdm(train_loader)):
                
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    _, features = self.model(images)
                    logits, labels = self.flat_loss(features)
                    v = torch.logsumexp(logits, dim=1, keepdim=True) #(512,1)
                    loss_vec = torch.exp(v-v.detach())
                    
                    assert loss_vec.shape == (len(logits),1)
                    dummy_logits = torch.cat([torch.zeros(logits.size(0),1).to(self.args.device), logits],1)
                    loss = loss_vec.mean()-1 + self.criterion(dummy_logits, labels).detach() #+

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                self.scheduler.step()
                scaler.update()

                
                
            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            print('epoch', epoch_counter)
            print('loss', loss.item())
            print('acc/top1', top1[0].item())
            print('acc/top5', top5[0].item())
            print('model_learning_rate', self.scheduler.get_lr()[0])
            self.writer.add_scalar('loss', loss, global_step=epoch_counter)
            self.writer.add_scalar('acc/top1', top1[0], global_step=epoch_counter)
            self.writer.add_scalar('acc/top5', top5[0], global_step=epoch_counter)
            self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=epoch_counter)

            if epoch_counter%self.args.save_every_n_epochs ==0:       
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
                save_checkpoint({
                        'batch_size': self.args.batch_size,
                        'epoch': self.args.epochs,
                        'model': self.args.model,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                
            
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finishfed.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'batch_size': self.args.batch_size,
            'epoch': self.args.epochs,
            'model': self.args.model,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
