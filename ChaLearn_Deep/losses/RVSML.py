from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import random


class RVSMLLoss(nn.Module):
    def __init__(self, classnum, L, **kwargs):
        super(RVSMLLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum') #nn.MSELoss(reduce = True, size_average = False)
        self.classnum = classnum
        self.L = L
        #self.T = T

    def forward(self, inputs, labels):
        targets = torch.eye(self.classnum*self.L,dtype=torch.float).cuda() 
        n = inputs.size(0)
        cur_loss = torch.zeros(1,dtype=torch.float,requires_grad=True).cuda()
        for i in range(n):           
            c = int(labels[i,self.L])
            for j in range(self.L):
                cur_loss = cur_loss + labels[i,j]*self.mse(inputs[i,:],targets[c*self.L+j,:])
        loss = cur_loss/n
        return loss
        



def main():
    data_size = 32
    input_dim = 6
    L = 3
    num_class = 4
    output_dim = 12
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    # print(x)
    targets = Variable(torch.rand(data_size, L), requires_grad=False)
    #inputs = x.mm(w)
    #y_ = Variable(torch.rand(data_size,1), requires_grad=False)
    y_ = torch.rand(data_size,1)
    #print(y_)
    for i in range(data_size):
        y_[i] = random.randint(0,num_class-1)
        #num = random.randint(0,num_class-1)
        #y_.append(num)
    #y_ = list(range(num_class))
    #print(y_)
    #targets = Variable(torch.rand(data_size, L), requires_grad=False)
    targets_y = Variable(torch.FloatTensor(y_))
    targets = torch.cat((targets,targets_y),1)

    myloss = RVSMLLoss(classnum=num_class,L=L)
    a = myloss(inputs, targets)
    print(a)


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


