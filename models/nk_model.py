import torch
import os
import torch.nn as nn
import time
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
from tensorboard_logger import configure, log_value
from utils.compute_average import AverageMeter
from others import settings


class nkModel(object):
    def __init__(self, args, train_loader, test_loader):
        self.data_path = "./save"
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.criterion = nn.BCELoss()
        self.lr = args.lr
        self.model_name = "nk"
        self.epochs = args.epochs
        self.gamma = args.gamma
        self.best_val_acc = 0
        self.num_train = len(self.train_loader.dataset)
        self.use_tensorboard = args.use_tensorboard
        self.batch_size = args.batch_size
        self.logs_dir = args.logs_dir
        now = datetime.now()
        self.time = now.strftime("%H:%M:%S")
        self.save_dir = './' + args.save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.train_steps, self.test_stets = len(self.train_loader), len(self.test_loader)

        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + args.model_name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        if args.model_name == 'vgg16':
            self.model = models.resnet18()
            self.model.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                                    self.model.parameters()),
                                             lr=self.lr,
                                             momentum=args.momentum,
                                             weight_decay=args.weight_decay)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=60,
                                                   gamma=self.gamma, last_epoch=-1)

        if args.save_load:
            location = args.save_location
            print("Location: ", location)
            checkpoint = torch.load(location)
            self.model.load_state_dict(checkpoint['state_dict'])

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            print('\nEpoch: {}/{} - LR: {:.6f}'.format(epoch + 1, self.epochs, self.optimizer.param_groups[0]['lr'], ))
            print("path: ", self.data_path + '/select_model_{}.pth'.format(epoch))
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        }, self.data_path + '/select_model_{}.pth'.format(epoch))
            self.scheduler.step(epoch)

    def train_one_epoch(self, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        batch_time = AverageMeter()
        tic = time.time()

        with tqdm(total=self.num_train) as pbar:
            for i, (inputs, targets, _) in enumerate(self.train_loader):

                if i == self.train_steps - 1:
                    break

                inputs = inputs.float()
                targets = targets.float()

                if settings.flag_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                output = self.model(inputs)
                loss = self.criterion(output, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print("output: ", output, output.shape)
                print("targets: ", targets, targets.shape)
                prec1 = self.accuracy(output.data, targets)[0]
                losses.update(loss.item(), inputs.size()[0])
                top1.update(prec1.item(), inputs.size()[0])

                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - model1_loss: {:.3f} - model1_acc: {:.3f}".format(
                            (toc - tic), losses.avg, top1.avg
                        )
                    )
                )

                pbar.update(self.batch_size)

                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + 1

                    log_value('train_loss_%d' % (i + 1), losses.avg, iteration)
                    log_value('train_acc_%d' % (i + 1), top1.avg, iteration)

            return losses, top1

    def test(self):
        path = "./save/select_model_499.pth"

        self.load_model(path)
        self.model.eval()
        temp = []
        for i, (inputs, targets, _) in enumerate(self.test_loader):
            print(i + 1)

            outputs = self.model(inputs)

            if outputs.detach().numpy()[0] < 0.5:
                temp.append(0)
            else:
                temp.append(1)

            # _, predicted = torch.max(outputs.data, 1)
            # predicted = predicted.cpu()
            # predicted = predicted.numpy()[0]
            # temp.append(predicted)

        return temp

    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t().float()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def save_checkpoint(self, i, state):
        filename = self.model_name + "_" + \
                   self.time + "_" + str(i + 1) + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.save_dir, filename)
        torch.save(state, ckpt_path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])