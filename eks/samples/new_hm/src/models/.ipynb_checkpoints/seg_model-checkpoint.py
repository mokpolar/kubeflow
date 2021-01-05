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
    z = conv2d(swish(gn(z, 16)), 32, 3)

    z = S_block(z, 48, lambda x: gn(x, 24))  # 64
    z = R_block(z, 0, lambda x: gn(x, 24))

    z = S_block(z, 64, gn)  # 32
    z = R_block(z, 0, gn)

    z = S_block(z, 128, gn)  # 16
    z = R_block(z, 0, gn)

    z = S_block(z, 256, gn)  # 8
    z = R_block(z, 0, gn)

    z = conv2d(swish(gn(z)), 512, 3, (2, 2))  # turn to flat stage #4
    z = conv2d(swish(gn(z)), 1024, 3, (2, 2))  # 2
    z = conv2d(swish(gn(z)), 2048, 3, (2, 2))  # 1

    z_mean = conv2d(swish(gn(z)), latent_c, 1)
    z_log_var = conv2d(swish(gn(z)), latent_c, 1)

    # sampling
    from tensorflow.keras.layers import Lambda

    _, H, W, C = z_mean.shape.as_list()
    z_latent = Lambda(sampling, output_shape=(H, W, C), name="latent")([z_mean, z_log_var])

    encoder = keras.Model(inputs=ph_input, outputs=[z_latent, z_mean, z_log_var], name="encoder")

    ##### decoder
    dec_input_shape = z_latent.shape.as_list()[1:]
    dec_input = keras.layers.Input(shape=dec_input_shape, name="dec_input", dtype="float32",)

    # mapping stage
    z = conv2d(dec_input, 2048, 1)
    p = z
    z = conv2d(swish(gn(z)), 2048, 1)
    z = conv2d(swish(gn(z)), 2048, 1)
    z = z + p
    p = z
    z = conv2d(swish(gn(z)), 2048, 1)
    z = conv2d(swish(gn(z)), 2048, 1)
    z = z + p
    p = z
    z = conv2d(swish(gn(z)), 2048, 1)
    z = conv2d(swish(gn(z)), 2048, 1)
    z = z + p

    z = conv2d(swish(gn(z)), 2048, 1)  # turn to conv stage
    z = tf.reshape(z, [-1, 8, 8, 32])  # 8
    z = conv2d(swish(gn(z)), 64, 3)
    z = conv2d(swish(gn(z)), 128, 3)
    z = conv2d(swish(gn(z)), 256, 3)

    z = R_block(z, 256, gn)
    z = R_block(z, 256, gn)

    z = upsample(z, 2)  # 16
    z = R_block(z, 128, gn)
    z = R_block(z, 128, gn)

    z = upsample(z, 2)  # 32
    z = R_block(z, 64, gn)
    z = R_block(z, 64, gn)

    z = upsample(z, 2)  # 64
    z = R_block(z, 48, lambda x: gn(x, 24))
    z = R_block(z, 48, lambda x: gn(x, 24))

    z = upsample(z, 2)  # 128
    z = R_block(z, 32, gn)
    z = R_block(z, 1, gn)

    z_image = upsample(z, 4)  # 256
    # z = tf.sigmoid(z) #해도되고 안해도되고.

    decoder = keras.Model(inputs=dec_input, outputs=z_image, name="decoder")

    z_image = decoder(encoder(ph_input)[0])
    model = keras.Model(inputs=ph_input, outputs=z_image)

    #     model.summary()

    return model


# =================== hj_ano_200713
from tensorflow.keras import Model, losses, optimizers
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    LayerNormalization,
    ReLU,
    LeakyReLU,
    Dropout,
    Flatten,
    Dense,
    Reshape,
    Lambda,
    Input,
)

class DecoderGdec(Model):
    def __init__(self, latent_c):
        super(DecoderGdec, self).__init__()
        self.latent_c = latent_c
        # self.intput = Reshape((1, 1, latent_c))

        self.block0_conv = Conv2DTranspose(latent_c // 2, 4, strides=2, padding="same")
        self.block0_bn = BatchNormalization()
        self.block0_relu = ReLU()

        self.block1_conv = Conv2DTranspose(1024 // 2, 4, strides=2, padding="same")
        self.block1_bn = BatchNormalization()
        self.block1_relu = ReLU()
        # self.block1_lrelu = layers.LeakyReLU()

        self.block2_conv = Conv2DTranspose(512 // 2, 4, strides=2, padding="same")
        self.block2_bn = BatchNormalization()
        self.block2_relu = ReLU()
        # self.block2_lrelu = layers.LeakyReLU()
        # self.block2_dr = layers.Dropout(0.5)

        self.block3_conv = Conv2DTranspose(256 // 2, 4, strides=2, padding="same")
        self.block3_bn = BatchNormalization()
        self.block3_relu = ReLU()
        # self.block3_lrelu = layers.LeakyReLU()

        self.block4_conv = Conv2DTranspose(128 // 2, 4, strides=2, padding="same")
        self.block4_bn = BatchNormalization()
        self.block4_relu = ReLU()
        # self.block4_lrelu = layers.LeakyReLU()
        # self.block4_dr = layers.Dropout(0.5)

        self.block5_conv = Conv2DTranspose(64 // 2, 4, strides=2, padding="same")
        self.block5_bn = BatchNormalization()
        self.block5_relu = ReLU()

        self.block6_conv = Conv2DTranspose(32 // 2, 4, strides=2, padding="same")
        self.block6_bn = BatchNormalization()
        self.block6_relu = ReLU()

        self.block7_conv = Conv2DTranspose(16 // 2, 4, strides=2, padding="same")
        self.block7_bn = BatchNormalization()
        self.block7_relu = ReLU()

        self.out = Conv2DTranspose(1, 4, strides=2, padding="same", activation="tanh")

    def call(self, x):
        # x = self.intput(x)

        x = self.block0_conv(x)
        x = self.block0_bn(x)
        x = self.block0_relu(x)

        x = self.block1_conv(x)
        x = self.block1_bn(x)
        x = self.block1_relu(x)

        x = self.block2_conv(x)
        x = self.block2_bn(x)
        x = self.block2_relu(x)

        x = self.block3_conv(x)
        x = self.block3_bn(x)
        x = self.block3_relu(x)

        x = self.block4_conv(x)
        x = self.block4_bn(x)
        x = self.block4_relu(x)

        x = self.block5_conv(x)
        x = self.block5_bn(x)
        x = self.block5_relu(x)

        x = self.block6_conv(x)
        x = self.block6_bn(x)
        x = self.block6_relu(x)

        x = self.block7_conv(x)
        x = self.block7_bn(x)
        x = self.block7_relu(x)

        x = self.out(x)
        return x


class EncoderGenc(Model):
    def __init__(self, latent_c):
        super(EncoderGenc, self).__init__()
        self.latent_c = latent_c

        self.block01_conv = Conv2D(32 // 2, 4, strides=2, padding="same")
        self.block01_bn = BatchNormalization()
        self.block01_lrelu = LeakyReLU()

        self.block02_conv = Conv2D(32 // 2, 4, strides=2, padding="same")
        self.block02_bn = BatchNormalization()
        self.block02_lrelu = LeakyReLU()

        self.block03_conv = Conv2D(64 // 2, 4, strides=2, padding="same")
        self.block03_bn = BatchNormalization()
        self.block03_lrelu = LeakyReLU()

        self.block1_conv = Conv2D(64 // 2, 4, strides=2, padding="same")
        self.block1_bn = BatchNormalization()
        self.block1_lrelu = LeakyReLU()

        self.block2_conv = Conv2D(128 // 2, 4, strides=2, padding="same")
        self.block2_bn = BatchNormalization()
        self.block2_lrelu = LeakyReLU()
        # self.block2_dr = layers.Dropout(0.3)

        self.block3_conv = Conv2D(256 // 2, 4, strides=2, padding="same")
        self.block3_bn = BatchNormalization()
        self.block3_lrelu = LeakyReLU()
        # self.block3_dr = layers.Dropout(0.3)

        self.block4_conv = Conv2D(512 // 2, 4, strides=2, padding="same")
        # self.block4_ln = LayerNormalization()
        self.block4_bn = BatchNormalization()
        self.block4_lrelu = LeakyReLU()
        # self.block4_dr = layers.Dropout(0.3)

        self.block5_conv = Conv2D(1024 // 2, 4, strides=2, padding="same")
        # self.block5_ln = LayerNormalization()
        self.block5_bn = BatchNormalization()
        self.block5_lrelu = LeakyReLU()
        # self.block5_dr = layers.Dropout(0.3)

        self.block6_conv = Conv2D(2048 // 2, 4, strides=2, padding="same")
        # self.block6_ln = LayerNormalization()
        self.block6_bn = BatchNormalization()
        self.block6_lrelu = LeakyReLU()

        self.mean = Conv2D(latent_c // 2, 1, strides=1, padding="same")
        self.log_var = Conv2D(latent_c // 2, 1, strides=1, padding="same")

        # self.flatten = Flatten()
        # self.mean = Dense(latent_c)
        # self.log_var = Dense(latent_c)

    def call(self, x):
        x = self.block01_conv(x)
        x = self.block01_bn(x)
        x = self.block01_lrelu(x)

        x = self.block02_conv(x)
        x = self.block02_bn(x)
        x = self.block02_lrelu(x)

        x = self.block03_conv(x)
        x = self.block03_bn(x)
        x = self.block03_lrelu(x)

        x = self.block1_conv(x)
        x = self.block1_bn(x)
        x = self.block1_lrelu(x)

        x = self.block2_conv(x)
        x = self.block2_bn(x)
        x = self.block2_lrelu(x)

        x = self.block3_conv(x)
        x = self.block3_bn(x)
        x = self.block3_lrelu(x)

        x = self.block4_conv(x)
        x = self.block4_bn(x)
        x = self.block4_lrelu(x)

        x = self.block5_conv(x)
        x = self.block5_bn(x)
        x = self.block5_lrelu(x)

        x = self.block6_conv(x)
        x = self.block6_bn(x)
        x = self.block6_lrelu(x)
        # x = self.flatten(x)

        mean = self.mean(x)
        log_var = self.log_var(x)

        return mean, log_var


def VAE_GAN(latent_c=2048):
    genc__ = EncoderGenc(latent_c)
    gdec__ = DecoderGdec(latent_c)

    def reparameterize(mean, logvar):
        eps = tf.random.normal(shape=mean.shape[-3:])
        z = eps * tf.exp(logvar * 0.5) + mean
        return z

    inputs = Input(shape=(512, 512, 1), name="input_ano", dtype="float32",)
    recon = gdec__(reparameterize(*genc__(inputs)))
    VAE_GAN__ = Model(inputs=inputs, outputs=recon)

    return VAE_GAN__

