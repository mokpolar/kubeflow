import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.backend as K
from tensorflow.keras.applications import DenseNet121

from models.model_definition_helper import *


def UNet():
    
    ### INput

    input_shape = [512, 512, 1]
    ph_input = keras.layers.Input(shape=input_shape, name="input", dtype="float32",)

    #%%

    latent_c = 2048

    ##### encoder
    zs = []
    z = ph_input * 2.0 - 1.0  # 512

    z = conv2d(z, 16, 5, (4, 4))  # 128
    z = conv2d(swish(gn(z,16)), 32, 3)

    z = S_block(z, 48, lambda x:gn(x,24))  # 64
    z = R_block(z,0, lambda x:gn(x,24))

    z = S_block(z, 64, gn)  # 32
    z = R_block(z,0,gn)

    z = S_block(z, 128, gn)  # 16
    z = R_block(z,0,gn)

    z = S_block(z, 256, gn)  # 8
    z = R_block(z,0,gn)

    z = conv2d(swish(gn(z)), 512, 3, (2,2)) # turn to flat stage #4
    z = conv2d(swish(gn(z)), 1024, 3, (2,2))#2
    z = conv2d(swish(gn(z)), 2048, 3, (2,2))#1

    z_mean    = conv2d(swish(gn(z)), latent_c, 1)
    z_log_var = conv2d(swish(gn(z)), latent_c, 1)

    # sampling
    from tensorflow.keras.layers import Lambda
    _,H,W,C = z_mean.shape.as_list()
    z_latent = Lambda(sampling, output_shape=(H,W,C), name='latent')([z_mean,z_log_var])

    encoder = keras.Model(inputs=ph_input, 
                          outputs=[z_latent, z_mean, z_log_var],
                          name='encoder')



    ##### decoder
    dec_input_shape = z_latent.shape.as_list()[1:]
    dec_input = keras.layers.Input(shape=dec_input_shape, name="dec_input", dtype="float32",)

    # mapping stage
    z = conv2d(dec_input, 2048, 1)
    p = z
    z = conv2d(swish(gn(z)), 2048, 1)
    z = conv2d(swish(gn(z)), 2048, 1)
    z = z+p
    p = z
    z = conv2d(swish(gn(z)), 2048, 1)
    z = conv2d(swish(gn(z)), 2048, 1)
    z = z+p
    p = z
    z = conv2d(swish(gn(z)), 2048, 1)
    z = conv2d(swish(gn(z)), 2048, 1)
    z = z+p

    z = conv2d(swish(gn(z)), 2048, 1) # turn to conv stage
    z = tf.reshape(z,[-1,8,8,32])   # 8
    z = conv2d(swish(gn(z)), 64, 3)
    z = conv2d(swish(gn(z)), 128, 3)
    z = conv2d(swish(gn(z)), 256, 3)

    z = R_block(z, 256, gn)
    z = R_block(z, 256, gn)

    z = upsample(z, 2)  # 16
    z = R_block(z, 128,gn)
    z = R_block(z, 128,gn)

    z = upsample(z, 2)  # 32
    z = R_block(z, 64,gn)
    z = R_block(z, 64,gn)

    z = upsample(z, 2)  # 64
    z = R_block(z, 48,lambda x:gn(x,24))
    z = R_block(z, 48,lambda x:gn(x,24))

    z = upsample(z, 2)  # 128
    z = R_block(z, 32, gn)
    z = R_block(z, 1, gn)


    z_image = upsample(z, 4)  # 256
    # z = tf.sigmoid(z) #해도되고 안해도되고.

    decoder = keras.Model(inputs=dec_input, outputs=z_image, name='decoder')

    z_image = decoder(encoder(ph_input)[0])
    model = keras.Model(inputs=ph_input, outputs=z_image)

#     model.summary()

    return model