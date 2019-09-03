# coding=utf-8
from __future__ import absolute_import, print_function

import torch
from torch.backends import cudnn
from evaluations import extract_features
import models
import DataSet
from utils.serialization import load_checkpoint
cudnn.benchmark = True
import numpy as np


def Sequence2Feature(data, net, checkpoint, in_dim, middle_dim,out_dim=512, root=None, nThreads=16, batch_size=100, train_flag=True,**kargs):
    dataset_name = data
    model = models.create(net, in_dim, middle_dim, out_dim, pretrained=False)
    # resume = load_checkpoint(ckp_path)
    resume = checkpoint
    model.load_state_dict(resume['state_dict'])
    model = torch.nn.DataParallel(model).cuda()
    datatrain = DataSet.create_seq(root=root,train_flag=True)
    
    if dataset_name in ['shop', 'jd_test','seq','seqfull']:
        gallery_loader = torch.utils.data.DataLoader(
            datatrain.seqdata, batch_size=batch_size, shuffle=False,
            drop_last=False, pin_memory=True, num_workers=nThreads)

        pool_feature = False

        train_feature, train_labels = extract_features(model, gallery_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
        if train_flag == True:
            test_feature = torch.zeros(2,1)
            test_labels = torch.zeros(2,1)
            #test_labels.append(1)
        else:
            datatest = DataSet.create_seq(root=root,train_flag=False)
            query_loader = torch.utils.data.DataLoader(
                datatest.seqdata, batch_size=batch_size,
                shuffle=False, drop_last=False,
                pin_memory=True, num_workers=nThreads)
            test_feature, test_labels = extract_features(model, query_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)

    else:
        data_loader = torch.utils.data.DataLoader(
            data.test, batch_size=batch_size,
            shuffle=False, drop_last=False, pin_memory=True,
            num_workers=nThreads)
        features, labels = extract_features(model, data_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
        gallery_feature, gallery_labels = query_feature, query_labels = features, labels
    #if train_flag:
    #    return train_feature, train_labels
    #else:
    return train_feature, train_labels, test_feature, test_labels
    #return gallery_feature, gallery_labels, query_feature, query_labels



def Sequence2Feature_test(data, net, checkpoint, in_dim, middle_dim,out_dim=512, root=None, nThreads=16, batch_size=100, train_flag=True,**kargs):
    dataset_name = data
    model = models.create(net, in_dim, middle_dim, out_dim, pretrained=False)
    # resume = load_checkpoint(ckp_path)
    resume = checkpoint
    model.load_state_dict(resume['state_dict'])
    model = torch.nn.DataParallel(model).cuda()
    datatrain = DataSet.create_seq(root=root,train_flag=True)

    gallery_loader = torch.utils.data.DataLoader(
        datatrain.seqdata, batch_size=batch_size, shuffle=False,
        drop_last=False, pin_memory=True, num_workers=nThreads)

    pool_feature = False

    train_feature, train_labels = extract_features(model, gallery_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
        
    datatest = DataSet.create_seq(root=root,train_flag=False)
    query_loader = torch.utils.data.DataLoader(
        datatest.seqdata, batch_size=batch_size,
        shuffle=False, drop_last=False,
        pin_memory=True, num_workers=nThreads)
    test_feature, test_labels = extract_features(model, query_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)

    
    return train_feature, train_labels, test_feature, test_labels
    #return gallery_feature, gallery_labels, query_feature, query_labels


def Sequence2Feature_testvec(data, net, checkpoint, in_dim, middle_dim,out_dim=512, root=None, nThreads=16, batch_size=100, train_flag=True,**kargs):
    dataset_name = data
    model = models.create(net, in_dim, middle_dim, out_dim, pretrained=False)
    # resume = load_checkpoint(ckp_path)
    resume = checkpoint
    model.load_state_dict(resume['state_dict'])
    model = torch.nn.DataParallel(model).cuda()
    datatrain = DataSet.create_vec(root=root,train_flag=True)

    gallery_loader = torch.utils.data.DataLoader(
        datatrain.seqdata, batch_size=batch_size, shuffle=False,
        drop_last=False, pin_memory=True, num_workers=nThreads)

    pool_feature = False

    train_feature, train_labels = extract_features(model, gallery_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
        
    datatest = DataSet.create_vec(root=root,train_flag=False)
    query_loader = torch.utils.data.DataLoader(
        datatest.seqdata, batch_size=batch_size,
        shuffle=False, drop_last=False,
        pin_memory=True, num_workers=nThreads)
    test_feature, test_labels = extract_features(model, query_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)

    
    return train_feature, train_labels, test_feature, test_labels
    #return gallery_feature, gallery_labels, query_feature, query_labels