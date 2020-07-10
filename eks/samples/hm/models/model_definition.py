from models.cls_model import DenseNet
from models.seg_model import UNet
from models.rnn_model import SequenceModel

def define_model_ano():
    return UNet()

def define_model_cls():
    return DenseNet(n_class=1)

def define_model_rnn():
    return SequenceModel(n_class=1)