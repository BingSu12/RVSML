from .CUB200 import CUB_200_2011
from .Car196 import Cars196
from .Products import Products
from .In_shop_clothes import InShopClothes
from .SequenceData import SequenceData
from .SequenceTrans import SequenceTrans
from .SequenceTransOne import SequenceTransOne
# from .transforms import *
import os 

__factory = {
    'cub': CUB_200_2011,
    'car': Cars196,
    'product': Products,
    'shop': InShopClothes,
    'seq': SequenceData,
    'seqfull': SequenceTrans,
    'seqone': SequenceTransOne,
}


def names():
    return sorted(__factory.keys())

def get_full_name(name):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name].__name__

def create(name, root=None, *args, **kwargs):
    """
    Create a dataset instance.
    """
    #if root is not None:
    #    root = os.path.join(root, get_full_name(name))
    
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root=root, *args, **kwargs)

def create_seq(root=None, train_flag=True, *args, **kwargs):
    return SequenceTrans(root=root, train_flag=train_flag, *args, **kwargs)

def create_seq_one(root=None, train_flag=True, *args, **kwargs):
    return SequenceTransOne(root=root, train_flag=train_flag, *args, **kwargs)

def create_vec(root=None, train_flag=True, *args, **kwargs):
    return SequenceData(root=root, train_flag=train_flag, *args, **kwargs)