
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import group_norm


# =-------=-------=-------=------- Layer Functions



# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    z_shape = K.shape(z_mean)
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=z_shape)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def conv2d(x, filters, ksize, strides=(1, 1), d_rate=(1, 1)):
    l = keras.layers.Conv2D(
        filters, ksize, strides, dilation_rate=d_rate, padding="SAME", use_bias=False
    )
    return l(x)


def dwconv2d(x, ksize, strides=(1, 1)):
    l = keras.layers.DepthwiseConv2D(
        ksize, strides, padding="SAME", depth_multiplier=1, use_bias=False
    )
    return l(x)


def bn(x):
    return keras.layers.BatchNormalization()(x)

def gn(x, group=32):
    N,H,W,C = x.shape.as_list()
    if C<group: group = C
    return group_norm.GroupNormalization(group)(x)


def swish(x):
    return x * keras.backend.sigmoid(x)


def sigmoid(x):
    return keras.backend.sigmoid(x)


def max_pool(x):
    return keras.layers.MaxPool2D((3, 3), (2, 2), "same")(x)


def avg_pool(x):
    return keras.layers.AvgPool2D((3, 3), (2, 2), "same")(x)


def upsample(x, mul):
    _, H, W, _ = x.shape.as_list()
    return tf.image.resize(x, (int(H * mul), int(W * mul)))


# def dropout(x, rate=ph_dropout_rate):
#     return tf.nn.dropout(x, rate)


# =-------=-------=-------=------- 블럭 정의


def S_block(x, out_filters, nor=bn):
    P = nor(conv2d(x, out_filters, 1, (2, 2)))
    #     x = bn(conv2d(swish(x), out_filters, 3))
    x = nor(conv2d(swish(x), out_filters, 5, (1, 1)))
    x = avg_pool(x)
    #     x = dropout(x)
    return x + P


def S_block2(x, out_filters, nor=bn):
    P = nor(conv2d(x, out_filters, 1, (2, 2)))
    x = nor(conv2d((x), out_filters, 3))
    x = nor(conv2d(swish(x), out_filters, 3))
    #         x = dropout( x )
    x = max_pool(x)
    return x + P


def R_block(x, Cout=0, nor=bn):
    _, _, _, Cin = x.shape.as_list()
    if Cout <= 0     : Cout = Cin
    if Cout == Cin   : P = x
    else             : P = conv2d(swish(x), Cout, 1)

    x = nor(conv2d(swish(x), Cout, 3))
    x = nor(conv2d(swish(x), Cout, 3))

    return x + P


# def R_block(x, Cout=0, nor=bn):  ### depth wise 버전
#     _, _, _, Cin = x.shape.as_list()
#     if Cout <= 0     : Cout = Cin
#     if Cout == Cin   : P = x
#     else             : P = conv2d(swish(x), Cout, 1)
    
#     x = nor(conv2d(swish(x), Cout, 1))
#     x = nor(dwconv2d(swish(x), 3))
#     x = nor(conv2d(swish(x), Cout, 1))
#     x = nor(dwconv2d(swish(x), 3))
#     x = nor(conv2d(swish(x), Cout, 1))

#     return x + P

def bot_block(x, out_filters, nor=bn, t=6):
    P = nor(conv2d(x, out_filters, 1))
    x = nor(conv2d(swish(x), out_filters * t, 1))
    x = nor(dwconv2d(swish(x), 3))
    x = nor(conv2d(swish(x), out_filters, 1))
    return x + P

