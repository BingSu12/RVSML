# coding=utf-8
from __future__ import absolute_import, print_function
import time
import argparse
import os
import sys
#del os.environ['MKL_NUM_THREADS','MKL_DOMAIN_NUM_THREADS']
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
import models
import losses
from utils import FastRandomIdentitySampler, mkdir_if_missing, logging, display
from utils.serialization import save_checkpoint, load_checkpoint
from trainerseq import train
from utils import orth_reg

import DataSet
import numpy as np
import os.path as osp
cudnn.benchmark = True

from Model2Feature import Model2Feature
from Sequence2Feature import Sequence2Feature

import ast 
import scipy.io as scio

#try:
#    import ipdb
#except:
#    import pdb as ipdb

use_gpu = True

# Batch Norm Freezer : bring 2% improvement on CUB 
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def main(args):
    # s_ = time.time()

    save_dir = args.save_dir
    mkdir_if_missing(save_dir)

    sys.stdout = logging.Logger(os.path.join(save_dir, 'log.txt'))
    display(args)
    start = 0

    #ipdb.set_trace()
    #model = models.create(args.net, pretrained=True, dim=args.dim)
    model = models.create(args.net, in_dim=args.in_dim, middle_dim=args.middle_dim, out_dim=args.out_dim, pretrained=False)

    # for vgg and densenet
    if args.resume is None:
        model_dict = model.state_dict()

    else:
        # resume model
        print('load model from {}'.format(args.resume))
        chk_pt = load_checkpoint(args.resume)
        weight = chk_pt['state_dict']
        #start = chk_pt['epoch']
        start = 0
        model.load_state_dict(weight)

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # freeze BN
    if args.freeze_BN is True:
        print(40 * '#', '\n BatchNorm frozen')
        model.apply(set_bn_eval)
    else:
        print(40*'#', 'BatchNorm NOT frozen')
        
    # Fine-tune the model: the learning rate for pre-trained parameter is 1/10
    new_param_ids = set(map(id, model.module.classifier.parameters()))

    new_params = [p for p in model.module.parameters() if
                  id(p) in new_param_ids]

    base_params = [p for p in model.module.parameters() if
                   id(p) not in new_param_ids]

    param_groups = [
                {'params': base_params, 'lr_mult': 1.0},
                {'params': new_params, 'lr_mult': 1.0}]

    print('initial model is save at %s' % save_dir)

    optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                 weight_decay=args.weight_decay)

    criterion = losses.create(args.loss, classnum=args.classnum, L=args.L, margin=args.margin, alpha=args.alpha, base=args.loss_base).cuda()

    # Decor_loss = losses.create('decor').cuda()
    #data = DataSet.create(args.data, ratio=args.ratio, width=args.width, origin_width=args.origin_width, root=args.data_root)
    datatrain = DataSet.create_seq(root=args.data_root, train_flag = True)

    train_loader = torch.utils.data.DataLoader(
        datatrain.seqdata, batch_size=args.batch_size,
        sampler=FastRandomIdentitySampler(datatrain.seqdata, num_instances=args.num_instances),
        drop_last=False, pin_memory=True, num_workers=args.nThreads)

    # save the train information

    for epoch in range(start, args.epochs):

        train(epoch=epoch, model=model, criterion=criterion,
              optimizer=optimizer, train_loader=train_loader, args=args)

        if epoch == 1:
            optimizer.param_groups[0]['lr_mul'] = 1 #0.1

        #if (epoch+1) % 200 == 0 and epoch!=0:
        #    args.lr = args.lr*0.1
        
        if (epoch+1) % args.save_step == 0 or epoch==0:
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'epoch': (epoch+1),
            }, is_best=False, fpath=osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Metric Learning')

    # hype-parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate of new parameters") #1e-5
    parser.add_argument('--batch_size', '-b', default=128, type=int, metavar='N',
                        help='mini-batch size (1 = pure stochastic) Default: 256')
    parser.add_argument('--num_instances', default=8, type=int, metavar='n',
                        help=' number of samples from one class in mini-batch')
    parser.add_argument('--in_dim', default=512, type=int, metavar='n',
                        help='dimension of embedding space')
    parser.add_argument('--middle_dim', default=1024, type=int, metavar='n',
                        help='dimension of embedding space')
    parser.add_argument('--out_dim', default=512, type=int, metavar='n',
                        help='dimension of embedding space')

    parser.add_argument('--alpha', default=30, type=int, metavar='n',
                        help='hyper parameter in NCA and its variants')
    parser.add_argument('--beta', default=0.1, type=float, metavar='n',
                        help='hyper parameter in some deep metric loss functions')
    parser.add_argument('--orth_reg', default=0, type=float,
                        help='hyper parameter coefficient for orth-reg loss')
    parser.add_argument('-k', default=16, type=int, metavar='n',
                        help='number of neighbour points in KNN')
    parser.add_argument('--margin', default=0.5, type=float,
                        help='margin in loss function')
    parser.add_argument('--init', default='random',
                        help='the initialization way of FC layer')

    # network
    parser.add_argument('--freeze_BN', default=False, type=bool, required=False, metavar='N',
                        help='Freeze BN if True')
    parser.add_argument('--data', default='seq', required=True,
                        help='name of Data Set')
    parser.add_argument('--data_root', type=str, default=None,
                        help='path to Data Set')

    parser.add_argument('--net', default='Sequence_Inception')
    parser.add_argument('--loss', default='branch', required=True,
                        help='loss for training network')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='epochs for training process')
    parser.add_argument('--save_step', default=100, type=int, metavar='N',
                        help='number of epochs to save model')

    # Resume from checkpoint
    parser.add_argument('--resume', '-r', default=None,
                        help='the path of the pre-trained model')

    # train
    parser.add_argument('--print_freq', default=20, type=int,
                        help='display frequency of training')

    # basic parameter
    # parser.add_argument('--checkpoints', default='/opt/intern/users/xunwang',
    #                     help='where the trained models save')
    parser.add_argument('--save_dir', default=None,
                        help='where the trained models save')
    parser.add_argument('--nThreads', '-j', default=16, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0) #2e-4

    parser.add_argument('--loss_base', type=float, default=0.75)

    parser.add_argument('--train_flag', type=bool, default=True)
    parser.add_argument('--classnum', type=int, default=20)
    parser.add_argument('--L', type=int, default=8)
    parser.add_argument('--pool_feature', type=ast.literal_eval, default=True, required=False,
                    help='if True extract feature from the last pool layer')


    args = parser.parse_args()
    if args.train_flag:
        main(parser.parse_args())

    
    args.resume = osp.join(args.save_dir, 'ckp_ep500.pth.tar')

    checkpoint = load_checkpoint(args.resume)
    #print(args.pool_feature)
    epoch = checkpoint['epoch']
    #if args.train_flag:
    train_feature, train_labels, test_feature, test_labels = \
    Sequence2Feature(data=args.data, root=args.data_root, net=args.net, checkpoint=checkpoint,
                   in_dim=args.in_dim, middle_dim=args.middle_dim,out_dim=args.out_dim,batch_size=args.batch_size, nThreads=args.nThreads, train_flag=args.train_flag)
    
    train_feature = train_feature.numpy()
    
    test_feature = test_feature.numpy()
    
    savedata_mat = "TransFeatures.mat";
    savedatapath = os.path.join(args.data_root,savedata_mat)
    scio.savemat(savedatapath,{'train_feature':train_feature,'test_feature':test_feature})
