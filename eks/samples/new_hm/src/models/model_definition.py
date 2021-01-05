from models.build_model import build_model
from models.seg_model import VAE_GAN, UNet

def define_model_ano():
    return UNet()

# def define_model_cls():
#     return DenseNet(n_class=1)

# def define_model_rnn():
#     return SequenceModel(n_class=1)

def define_model_cls_seq():
    return build_model('cls_seq','mobilenet-v1',(256,256,3))