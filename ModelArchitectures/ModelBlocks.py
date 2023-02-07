######################
###     IMPORTS    ###
######################
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, Dense, LeakyReLU, Conv2DTranspose

def ConvolutionBlock(conv_params, bn_params, act_params, drop_params, inp, use_bn):
    x = Conv2D(**conv_params)(inp)
    if use_bn:
        x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(**act_params)(x)
    x = tf.keras.layers.Dropout(**drop_params)(x)
    return x

def DeConvolutionBlock(deconv_params, bn_params, act_params, drop_params, inp, use_bn):
    x = Conv2DTranspose(**deconv_params)(inp)
    if use_bn:
        x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(**act_params)(x)
    x = tf.keras.layers.Dropout(**drop_params)(x)
    return x

def DenseBlock(dense_params, bn_params, act_params, drop_params, inp, use_bn):
    x = Dense(**dense_params)(inp)
    if use_bn:
        x = BatchNormalization(**bn_params)(x)
    x = LeakyReLU(**act_params)(x)
    x = tf.keras.layers.Dropout(**drop_params)(x)
    return x