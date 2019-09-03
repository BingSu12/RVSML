# coding=utf-8
from __future__ import print_function, absolute_import
import time
from utils import AverageMeter, orth_reg
import  torch
from torch.autograd import Variable
from torch.backends import cudnn

#import bad_grad_viz

cudnn.benchmark = True


def train(epoch, model, criterion, optimizer, train_loader, args):

    losses = AverageMeter()
    batch_time = AverageMeter()
    accuracy = AverageMeter()
    pos_sims = AverageMeter()
    neg_sims = AverageMeter()

    end = time.time()

    freq = min(args.print_freq, len(train_loader))

    for i, data_ in enumerate(train_loader, 0):

        inputs, labels = data_
        #print(inputs)

        # wrap them in Variable
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()

        embed_feat = model(inputs)

        loss = criterion(embed_feat, labels)

        #get_dot = bad_grad_viz.register_hooks(loss)

        if args.orth_reg != 0:
            loss = orth_reg(net=model, loss=loss, cof=args.orth_reg)

        loss.backward()
        optimizer.step()

        #for param in model.parameters():
        #    print(param.grad.size())
        #dot = get_dot()
        #dot.save('tmp.dot')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item())
        #accuracy.update(inter_)
        #pos_sims.update(dist_ap)
        #neg_sims.update(dist_an)

        if (i + 1) % freq == 0 or (i+1) == len(train_loader):
            print('Epoch: [{0:03d}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.8f} \t'.format
                  (epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
                   loss=losses))

        if epoch == 0 and i == 0:
            print('-- HA-HA-HA-HA-AH-AH-AH-AH --')
