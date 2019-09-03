from __future__ import absolute_import, print_function
"""
CUB-200-2011 data-set for Pytorch
"""
import torch
import torch.utils.data as data
import numpy as np
#from PIL import Image

import os
#from DataSet import transforms
from collections import defaultdict

import scipy.io as scio



class TrainData(data.Dataset):
    def __init__(self, root=None, data_mat=None):

        # Initialization data path and train(gallery or query) txt path
        if root is None:
            self.root = "./datamat/MSRAc3D/"
        self.root = root

        datapath = os.path.join(root,data_mat)
        Dismatdata = scio.loadmat(datapath)
        sequence_data = Dismatdata['traindatafull']
        sequence_data = sequence_data.copy(order='C')
        sequence_data = sequence_data.astype(np.float32)
        sequence_label = Dismatdata['labeldatafull']
        sequence_label = sequence_label.copy(order='C')
        sequence_label = sequence_label.astype(np.float32)
        
        sequences = []
        for i in range(sequence_data.shape[0]):
            sequences.append(sequence_data[i,:])
        labels = []
        intlabels = []
        for i in range(sequence_label.shape[0]):
            labels.append(sequence_label[i,:])
            intlabels.append(int(sequence_label[i,sequence_label.shape[1]-1]))

        classes = list(set(intlabels))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(intlabels):
            Index[label].append(i)

        # Initialization Done
        self.root = root
        self.sequences = sequences
        self.labels = labels
        self.classes = classes
        self.intlabels = intlabels
        self.Index = Index

    def __getitem__(self, index):
        sequence, label = self.sequences[index], self.labels[index]
        return sequence, label

    def __len__(self):
        return len(self.sequences)


class TestData(data.Dataset):
    def __init__(self, root=None, data_mat=None):

        # Initialization data path and train(gallery or query) txt path
        if root is None:
            self.root = "./datamat/MSRAc3D/"
        self.root = root

        datapath = os.path.join(root,data_mat)
        Dismatdata = scio.loadmat(datapath)
        sequence_data = Dismatdata['testdatafull']
        sequence_data = sequence_data.copy(order='C')
        sequence_data = sequence_data.astype(np.float32)
        sequences = []
        for i in range(sequence_data.shape[0]):
            sequences.append(sequence_data[i,:])
        labels = []
        for i in range(sequence_data.shape[0]):
            labels.append(int(i))

        classes = list(set(labels))

        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(labels):
            Index[label].append(i)

        # Initialization Done
        self.root = root
        self.sequences = sequences
        self.labels = labels
        self.classes = classes
        self.Index = Index

    def __getitem__(self, index):
        sequence, label = self.sequences[index], self.labels[index]
        return sequence, label

    def __len__(self):
        return len(self.sequences)


class SequenceTrans:
    def __init__(self, train_flag, root=None):
        # Data loading code
        if root is None:
            root = "./datamat/MSRAc3D/"

        train_mat = "traindatafull.mat"
        test_mat = "testdatafull.mat"

        if train_flag == True:
            self.seqdata = TrainData(root, data_mat=train_mat)
        else:
            self.seqdata = TestData(root, data_mat=test_mat)


def testsequence():
    print(SequenceTrans.__name__)
    data = SequenceTrans()
    print(len(data.test))
    print(len(data.train))
    print(data.train[1])


if __name__ == "__main__":
    testsequence()


