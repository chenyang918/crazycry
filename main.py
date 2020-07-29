import os
import logging
import argparse
import shutil
import time
import numpy as np
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import torch.nn.functional as F

import utils
from data import WavDataset, WavTestDataset
from models import DFCNN_TCN




logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='PyTorch Mnist Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=False,
                    action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=False,
                    action='store_true', help='use pre-trained model')
parser.add_argument('--gpu', default='', type=str,
                    help='GPU for using.')
parser.add_argument('--lr-steps', type=str, default='40,70', help='steps of lr changing')

parser.add_argument('--arch', default='dfcnn_tcn', type=str,
                    help='arch of model.')

best_prec1 = 0


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def main():
    global args, best_prec1
    args = parser.parse_args()
    utils.init_cuda(args.gpu)
    utils.occumpy_mem(args.gpu, 0.95)
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # create model
    print("=> creating model...")
    if args.arch == 'dfcnn_tcn':
        model = DFCNN_TCN(nclass=6, nHidden=512, mode='small')
    else:
        print('error arch, one of [dfcnn_tcn]')
    if args.resume:
        cudnn.benchmark = True
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.start_epoch == 0:
                args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            if "state_dict" in checkpoint.keys():
                state_dict = remove_prefix(checkpoint['state_dict'], 'module.')
            else:
                state_dict = remove_prefix(checkpoint, 'module.')
            state_dict = remove_prefix(state_dict, 'module.')
            model.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    
    train_dataset = WavDataset(is_train=True, augment=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=utils.WavFeatureCollate()
    )

    
    val_dataset = WavDataset(is_train=False, augment=False)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=utils.WavFeatureCollate()
    )

    test_dataset = WavTestDataset()
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=utils.WavFeatureCollate()
    )

    # define loss function (criterion) and optimizer
    # criterion = nn.CTCLoss(blank=0, reduction='mean').cuda()
    # criterion = nn.CTCLoss(blank=0, reduction='mean').cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                           # betas=(0.5, 0.999))
    
    model = torch.nn.DataParallel(model).cuda()

    if args.evaluate:
        validate(val_loader, model, criterion)
        final_validate(test_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, epoch)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                  'Acc {top1.val:.5f} ({top1.avg:.5f})\t'
                  .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target, img_path) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target.data)
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            #print(output.data, target.data, img_path)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.5f} ({top1.avg:.5f})'
                      .format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))

        print(' * Acc {top1.avg}'.format(top1=top1))

        return top1.avg

def final_validate(test_loader, model, criterion):
    label2name = {0: 'awake', 1: 'diaper', 2: 'hug', 3: 'hungry', 4: 'sleepy', 5: 'uncomfortable'}
    with torch.no_grad():
        # switch to evaluate mode
        model.eval()
        outputs = []
        names = []
        for i, (input, _, paths) in enumerate(test_loader):
            input = input.cuda()
            output = model(input)
            output = F.softmax(output, dim=1).float().cpu().detach().numpy()
            preds = np.argmax(output, axis=1)
            outputs.extend(preds.tolist())
            names.extend([os.path.basename(p) for p in paths])
        with open('test_result.csv', 'w') as f:
            f.write('id,label\n')
            for o, n in zip(outputs, names):
                f.write(f'{n},{label2name[int(o)]}\n')



def save_checkpoint(state, is_best, epoch):
    global args
    utils.maybe_makedir('./weights/')
    filename='./weights/checkpoint_cry_{}-{:0>4}.pth'.format(args.arch, epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './weights/model_best_cry_{}.pth'.format(args.arch))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    lr_steps = list(map(int, lr_steps.split(',')))
    assert len(lr_steps) >= 1
    gaps = {}
    for i, lr_step in enumerate(lr_steps):
        gaps[lr_step] = i+1
    if gaps.get(epoch, -1) != -1:
        lr = args.lr * (0.1 ** gaps[epoch])
        print('lr change to', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
