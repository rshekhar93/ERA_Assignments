import torch
import torch.nn as nn
import torch.optim as optim
from utils.metrics import AverageMeter, accuracy
import time

class Trainer:
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config['training']['lr_schedule']['milestones'],
            gamma=config['training']['lr_schedule']['gamma']
        )

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for i, (images, target) in enumerate(train_loader):
            images, target = images.to(self.device), target.to(self.device)

            output = self.model(images)
            loss = self.criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.config['logging']['log_interval'] == 0:
                self.logger.log_metrics({
                    'loss': losses.avg,
                    'acc1': top1.avg,
                    'acc5': top5.avg,
                    'lr': self.optimizer.param_groups[0]['lr']
                }, epoch * len(train_loader) + i)

        return losses.avg, top1.avg, top5.avg

    def validate(self, val_loader, epoch):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images, target = images.to(self.device), target.to(self.device)

                output = self.model(images)
                loss = self.criterion(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

        self.logger.log_metrics({
            'val_loss': losses.avg,
            'val_acc1': top1.avg,
            'val_acc5': top5.avg
        }, epoch)

        return losses.avg, top1.avg, top5.avg

    def train(self, train_loader, val_loader):
        best_acc1 = 0
        for epoch in range(self.config['training']['num_epochs']):
            train_loss, train_acc1, train_acc5 = self.train_epoch(train_loader, epoch)
            val_loss, val_acc1, val_acc5 = self.validate(val_loader, epoch)
            
            self.scheduler.step()

            # Save checkpoint if best accuracy
            if val_acc1 > best_acc1:
                best_acc1 = val_acc1
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=True)

    def save_checkpoint(self, state, is_best):
        filename = f'checkpoint_epoch{state["epoch"]}.pth'
        save_path = os.path.join(self.logger.exp_dir, filename)
        torch.save(state, save_path)
        if is_best:
            best_path = os.path.join(self.logger.exp_dir, 'model_best.pth')
            torch.save(state, best_path) 