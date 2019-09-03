import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

class InceptionTransformation(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=None, normalized=True):
        super(InceptionTransformation, self).__init__()
        self.bn = nn.BatchNorm1d(in_dim, eps=1e-5)
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.dropout = dropout
        self.normalized = normalized

    def forward(self, x):
        x = self.bn(x)
        # x = F.relu(x, inplace=True)
        if self.dropout is not None:
            x = nn.Dropout(p=self.dropout)(x, inplace=True)
        x = self.linear(x)
        if self.normalized:
            norm = x.norm(dim=1, p=2, keepdim=True)
            #if norm>0:
            x = x.div(norm.expand_as(x))
        return x


class SequenceInception(nn.Module):
    def __init__(self, in_dim,middle_dim,out_dim=512):
        super(SequenceInception, self).__init__()
        self.in_dim = in_dim
        self.middle_dim = middle_dim
        self.out_dim = out_dim
        inplace = True

        self.layer1 = nn.Sequential(nn.BatchNorm1d(in_dim, eps=1e-5),nn.Linear(in_dim, middle_dim),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.BatchNorm1d(middle_dim, eps=1e-5),nn.Linear(middle_dim, middle_dim),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.BatchNorm1d(middle_dim, eps=1e-5),nn.Linear(middle_dim, middle_dim),nn.ReLU(True))
        #self.layer1 = nn.Sequential(nn.Linear(in_dim, middle_dim),nn.ReLU(True))
        #self.layer2 = nn.Sequential(nn.Linear(middle_dim, middle_dim),nn.ReLU(True))
        #self.layer3 = nn.Sequential(nn.Linear(middle_dim, middle_dim),nn.ReLU(True))

        self.classifier = InceptionTransformation(self.middle_dim, self.out_dim, normalized=True)

    def features(self, input):
        out1 = self.layer1(input)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        output = self.classifier(out3)
        return output
 
    def forward(self, x):        
        y = self.features(x)
        return y

def Sequence_Inception(in_dim, middle_dim, out_dim=512, pretrained=False, model_path=None):
    model = SequenceInception(in_dim, middle_dim, out_dim)
    if model_path is None:
        model_path = './trainedmodels/ChaLearn/seq_inception.pth'
    if pretrained is True:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def main():
    import torch
    model = Sequence_Inception(in_dim = 227, middle_dim=512, out_dim = 227, pretrained=False)
    # print(model)
    images = Variable(torch.ones(8, 227))
    out_ = model(images)
    print(out_.data.shape)

if __name__ == '__main__':
    main()

