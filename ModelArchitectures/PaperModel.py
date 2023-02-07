######################
###     IMPORTS    ###
######################
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, Dense, LeakyReLU, Flatten, Conv2DTranspose
from ModelArchitectures.ModelBlocks import ConvolutionBlock, DeConvolutionBlock, DenseBlock

# class Generator(tf.keras.Model):
#     def __init__(self, shape, data_format, C1, K):
#         super(Generator, self).__init__()
#         # Block parameters
#         self.shape = shape
#         self.conv_params = lambda n, s: {'filters':n*C1, 'kernel_size': (K, K), 'strides':(s,s), 'use_bias':True, 'padding':'same', 'data_format': data_format}
#         self.deconv_params = lambda n, s, pad: {'filters':n*C1, 'kernel_size': (K, K), 'strides':(s,s), 'use_bias':True, 'padding':'same', 'data_format': data_format, 'output_padding':(pad,pad) if pad != 0 else None}

#         self.bn_params = {'momentum':0.9, 'epsilon':1e-4}
#         self.act_params = {'alpha':0.1}
#         self.drop_params = lambda d: {'rate':d}
#     def call(self, inp, training):
#         axis = 1 if self.shape[0] != self.shape[1] else 3
#         # Generator Blocks
#         x = Input(tensor=inp)
#         CB1 = ConvolutionBlock(self.conv_params(4, 2), self.bn_params, self.act_params, self.drop_params(0))(x, training=training, use_bn=False)
#         CB2 = ConvolutionBlock(self.conv_params(8, 2), self.bn_params, self.act_params, self.drop_params(0))(CB1, training=training, use_bn=False)
#         CB3 = ConvolutionBlock(self.conv_params(8, 2), self.bn_params, self.act_params, self.drop_params(0))(CB2, training=training, use_bn=False)
#         CB4 = ConvolutionBlock(self.conv_params(8, 2), self.bn_params, self.act_params, self.drop_params(0))(CB3, training=training, use_bn=False)
#         CB5 = ConvolutionBlock(self.conv_params(16, 2), self.bn_params, self.act_params, self.drop_params(0))(CB4, training=training, use_bn=False)
#         CB6 = ConvolutionBlock(self.conv_params(16, 2), self.bn_params, self.act_params, self.drop_params(0))(CB5, training=training, use_bn=False)

#         DCB1 = DeConvolutionBlock(self.deconv_params(16, 2, 0), self.bn_params, self.act_params, self.drop_params(0))(CB6, training=training, use_bn=False)
#         DCB2 = DeConvolutionBlock(self.deconv_params(16, 2, 1), self.bn_params, self.act_params, self.drop_params(0))(DCB1, training=training, use_bn=False)
#         DCB3 = DeConvolutionBlock(self.deconv_params(8, 2, 0), self.bn_params, self.act_params, self.drop_params(0))(DCB2, training=training, use_bn=False)
#         DCB4 = DeConvolutionBlock(self.deconv_params(8, 2, 1), self.bn_params, self.act_params, self.drop_params(0))(DCB3, training=training, use_bn=False)
#         DCB5 = DeConvolutionBlock(self.deconv_params(4, 2, 1), self.bn_params, self.act_params, self.drop_params(0))(DCB4, training=training, use_bn=False)
#         DCB6 = DeConvolutionBlock(self.deconv_params(2, 2, 0), self.bn_params, self.act_params, self.drop_params(0))(DCB5, training=training, use_bn=False)
#         DCB7 = DeConvolutionBlock(self.self.deconv_params(1, 2, 0), self.bn_params, self.act_params, self.drop_params(0))(DCB6,  use_bn=False)

#         out = Conv2DTranspose(1, (4,4), strides=(2,2), use_bias=True, padding='same', data_format=self.conv_params(4, 2)['data_format'], activation='sigmoid')(DCB7)
#         return out

class ClipConstraint(tf.keras.constraints.Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)


def Generator(shape, data_format, C1, K, multipliers):
    axis = 1 if shape[0] != shape[1] else 3
    # Block parameters
    shape = shape
    conv_params = lambda n, s, regularize_bool: {'filters':n*C1, 'kernel_size': (K, K), 'strides':(s,s), 'use_bias':True, 'padding':'same', 'data_format': data_format, 'kernel_regularizer':'l1_l2' if regularize_bool else None}
    deconv_params = lambda n, s, pad, regularize_bool: {'filters':n*C1, 'kernel_size': (K, K), 'strides':(s,s), 'use_bias':True, 'padding':'same', 'data_format': data_format, 'output_padding':(pad,pad) if pad != 0 else None, 'kernel_regularizer':'l1_l2' if regularize_bool else None}

    bn_params = {'momentum':0.9, 'epsilon':1e-4}
    act_params = {'alpha':0.1}
    drop_params = lambda d: {'rate':d}

    # Model Blocks
    inp = Input(shape=shape)
    #CBSharpen = ConvolutionBlock(conv_params(multipliers[0], 1), bn_params, act_params, drop_params(0), inp, use_bn=True)
    CB1 = ConvolutionBlock(conv_params(multipliers[1], 2, False), bn_params, act_params, drop_params(0), inp, use_bn=True)
    CB2 = ConvolutionBlock(conv_params(multipliers[2], 2, False), bn_params, act_params, drop_params(0), CB1, use_bn=True)
    CB3 = ConvolutionBlock(conv_params(multipliers[3], 2, False), bn_params, act_params, drop_params(0), CB2, use_bn=True)
    CB4 = ConvolutionBlock(conv_params(multipliers[4], 2, False), bn_params, act_params, drop_params(0), CB3, use_bn=True)
    CB5 = ConvolutionBlock(conv_params(multipliers[4], 2, False), bn_params, act_params, drop_params(0.3), CB4, use_bn=True)
    CB6 = ConvolutionBlock(conv_params(multipliers[4], 2, False), bn_params, act_params, drop_params(0.3), CB5, use_bn=True)

    DCB1 = DeConvolutionBlock(deconv_params(multipliers[4], 2, 0, False), bn_params, act_params, drop_params(0.5), CB6, use_bn=True)
    concat1 = tf.concat([CB5, DCB1], axis=axis)
    DCB2 = DeConvolutionBlock(deconv_params(multipliers[4], 2, 1, False), bn_params, act_params, drop_params(0.5), concat1, use_bn=True)
    concat2 = tf.concat([CB4, DCB2], axis=axis)
    DCB3 = DeConvolutionBlock(deconv_params(multipliers[4], 2, 0, False), bn_params, act_params, drop_params(0.5), concat2, use_bn=True)
    concat3 = tf.concat([CB3, DCB3], axis=axis)
    DCB4 = DeConvolutionBlock(deconv_params(multipliers[4], 2, 1, False), bn_params, act_params, drop_params(0.3), concat3, use_bn=True)
    concat4 = tf.concat([CB2, DCB4], axis=axis)
    DCB5 = DeConvolutionBlock(deconv_params(multipliers[3], 2, 1, False), bn_params, act_params, drop_params(0), concat4, use_bn=True) # multipliers[1]
    concat5 = tf.concat([CB1, DCB5], axis=axis)
    DCB6 = DeConvolutionBlock(deconv_params(multipliers[2], 2, 0, False), bn_params, act_params, drop_params(0), concat5, use_bn=True)
    DCB7 = DeConvolutionBlock(deconv_params(multipliers[1], 2, 0, False), bn_params, act_params, drop_params(0), DCB6, use_bn=True)

    #DCB7Sharpen = DeConvolutionBlock(deconv_params(multipliers[0], 2, 0), bn_params, act_params, drop_params(0), DCB7, use_bn=True)
    out = Conv2DTranspose(1, (4,4), strides=(2,2), use_bias=True, padding='same', data_format=conv_params(4, 1, False)['data_format'], activation='sigmoid')(DCB7)
    return tf.keras.Model(inp, out)

def Discriminator(shape, data_format, C1, K, clip):
    axis = 1 if shape[0] != shape[1] else 3

    const = ClipConstraint(0.01)
    # Block parameters
    shape = shape
    conv_params = lambda n, s: {'filters':n*C1, 'kernel_size': (K, K), 'strides':(s,s), 'use_bias':True, 'padding':'same', 'data_format': data_format, 'kernel_constraint':const}
    dense_params = lambda n: {'units':n*C1, 'use_bias':True}
    bn_params = {'momentum':0.9, 'epsilon':1e-4}
    act_params = {'alpha':0.1}
    drop_params = lambda d: {'rate':d}

    # Model Blocks
    inp = Input(shape=shape)
    CB1 = ConvolutionBlock(conv_params(4, 2), bn_params, act_params, drop_params(0), inp, use_bn=False)
    CB2 = ConvolutionBlock(conv_params(8, 2), bn_params, act_params, drop_params(0), CB1, use_bn=True)
    CB3 = ConvolutionBlock(conv_params(8, 2), bn_params, act_params, drop_params(0), CB2, use_bn=True)
    CB4 = ConvolutionBlock(conv_params(8, 2), bn_params, act_params, drop_params(0), CB3, use_bn=True)
    CB5 = ConvolutionBlock(conv_params(16, 2), bn_params, act_params, drop_params(0), CB4, use_bn=True)
    FCB5 = Flatten()(CB5)
    #D1 = DenseBlock(dense_params(2), bn_params, act_params, drop_params(0), FCB4, use_bn=False)
    out = Dense(1, use_bias=True, activation='linear')(FCB5)
    return tf.keras.Model(inp, out)

def CreateGridPaperModel(shape, data_format, C1, K1, d, B1, B2, LR, momentum):
    axis = 1 if shape[0] != shape[1] else 3
    inp = Input(shape=shape)

    # Layer 3
    lay3_1 = Conv2D(4*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(inp)
    lay3_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay3_1)
    act3 = LeakyReLU(alpha=LR)(lay3_2)

    # Layer 4
    lay4_1 = Conv2D(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(act3)
    lay4_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay4_1)
    act4 = LeakyReLU(alpha=LR)(lay4_2)

    # Layer 5
    lay5_1 = Conv2D(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(act4)
    lay5_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay5_1)
    act5 = LeakyReLU(alpha=LR)(lay5_2)
    drop5 = tf.keras.layers.Dropout(d)(act5)

    # Layer 6
    lay6_1 = Conv2D(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(drop5)
    lay6_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay6_1)
    act6 = LeakyReLU(alpha=LR)(lay6_2)
    drop6 = tf.keras.layers.Dropout(d)(act6)

    # Layer 7
    lay7_1 = Conv2D(16*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(drop6)
    lay7_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay7_1)
    act7 = LeakyReLU(alpha=LR)(lay7_2)
    drop7 = tf.keras.layers.Dropout(d)(act7)

    # Layer 8
    lay8_1 = Conv2D(16*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(drop7)
    lay8_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay8_1)
    act8 = LeakyReLU(alpha=LR)(lay8_2)
    drop8 = tf.keras.layers.Dropout(d)(act8)
    
    # Layer 9
    lay9_1 = Conv2DTranspose(16*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(drop8)
    lay9_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay9_1)
    act9 = LeakyReLU(alpha=LR)(lay9_2)
    drop9 = tf.keras.layers.Dropout(d)(act9)
    concat7 = tf.concat([drop9, drop7], axis=axis)
    # Layer 10
    lay10_1 = Conv2DTranspose(16*C1, (4,4), strides=(2,2), use_bias=B1, padding='same', output_padding=(1,1), data_format=data_format)(concat7)
    lay10_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay10_1)
    act10 = LeakyReLU(alpha=LR)(lay10_2)
    drop10 = tf.keras.layers.Dropout(d)(act10)
    concat6 = tf.concat([drop10, drop6], axis=axis)

    # Layer 11
    lay11_1 = Conv2DTranspose(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(concat6)
    lay11_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay11_1)
    act11 = LeakyReLU(alpha=LR)(lay11_2)
    drop11 = tf.keras.layers.Dropout(d)(act11)
    concat5 = tf.concat([drop11, drop5], axis=axis)
    # Layer 12
    lay12_1 = Conv2DTranspose(8*C1, (4,4), strides=(2,2), use_bias=B1, padding='same', output_padding=(1,1), data_format=data_format)(concat5)
    lay12_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay12_1)
    act12 = LeakyReLU(alpha=LR)(lay12_2)
    drop12 = tf.keras.layers.Dropout(d)(act12)
    concat4 = tf.concat([drop12, act4], axis=axis)

    # Layer 13
    lay13_1 = Conv2DTranspose(4*C1, (4,4), strides=(2,2), use_bias=B1, padding='same', output_padding=(1,1), data_format=data_format)(concat4)
    lay13_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay13_1)
    act13 = LeakyReLU(alpha=LR)(lay13_2)
    concat3 = tf.concat([act13, act3], axis=axis)

    # Layer 14
    lay14_1 = Conv2DTranspose(2*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(concat3)
    lay14_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay14_1)
    act14 = LeakyReLU(alpha=LR)(lay14_2)

    # Layer 15
    lay15_1 = Conv2DTranspose(C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(act14)
    lay15_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay15_1)
    act15 = LeakyReLU(alpha=LR)(lay15_2)

    # Layer 16
    lay16_1 = Conv2DTranspose(1, (K1,K1), strides=(2,2), use_bias=B2, padding='same', data_format=data_format)(act15)
    act16 = tf.keras.layers.Activation(activation='sigmoid', dtype="float32")(lay16_1)

    # Model
    build = tf.keras.Model(inp, act16)
    return build

def CreatePaperModel(shape, data_format):
    axis = 1 if shape[0] != shape[1] else 3
    C1, K1, d, B1, B2, LR, momentum = 42, 4, 0.5, True, True, 0.1, 0.9
    inp = Input(shape=shape)

    # # Layer 1
    # lay1_1 = Conv2D(C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(inp)
    # lay1_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay1_1)
    # act1 = LeakyReLU(alpha=LR)(lay1_2)

    # # Layer 2
    # lay2_1 = Conv2D(2*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(act1)
    # lay2_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay2_1)
    # act2 = LeakyReLU(alpha=LR)(lay2_2)

    # Layer 3
    lay3_1 = Conv2D(4*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(inp)
    lay3_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay3_1)
    act3 = LeakyReLU(alpha=LR)(lay3_2)

    # Layer 4
    lay4_1 = Conv2D(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(act3)
    lay4_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay4_1)
    act4 = LeakyReLU(alpha=LR)(lay4_2)

    # Layer 5
    lay5_1 = Conv2D(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(act4)
    lay5_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay5_1)
    act5 = LeakyReLU(alpha=LR)(lay5_2)
    drop5 = tf.keras.layers.Dropout(d)(act5)

    # Layer 6
    lay6_1 = Conv2D(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(drop5)
    lay6_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay6_1)
    act6 = LeakyReLU(alpha=LR)(lay6_2)
    drop6 = tf.keras.layers.Dropout(d)(act6)

    # Layer 7
    lay7_1 = Conv2D(16*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(drop6)
    lay7_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay7_1)
    act7 = LeakyReLU(alpha=LR)(lay7_2)
    drop7 = tf.keras.layers.Dropout(d)(act7)

    # Layer 8
    lay8_1 = Conv2D(16*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(drop7)
    lay8_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay8_1)
    act8 = LeakyReLU(alpha=LR)(lay8_2)
    drop8 = tf.keras.layers.Dropout(d)(act8)
    
    # Layer 9
    lay9_1 = Conv2DTranspose(16*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(drop8)
    lay9_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay9_1)
    act9 = LeakyReLU(alpha=LR)(lay9_2)
    drop9 = tf.keras.layers.Dropout(d)(act9)
    concat7 = tf.concat([drop9, drop7], axis=axis)
    # Layer 10
    lay10_1 = Conv2DTranspose(16*C1, (4,4), strides=(2,2), use_bias=B1, padding='same', output_padding=(1,1), data_format=data_format)(concat7)
    lay10_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay10_1)
    act10 = LeakyReLU(alpha=LR)(lay10_2)
    drop10 = tf.keras.layers.Dropout(d)(act10)
    concat6 = tf.concat([drop10, drop6], axis=axis)

    # Layer 11
    lay11_1 = Conv2DTranspose(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(concat6)
    lay11_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay11_1)
    act11 = LeakyReLU(alpha=LR)(lay11_2)
    drop11 = tf.keras.layers.Dropout(d)(act11)
    concat5 = tf.concat([drop11, drop5], axis=axis)
    # Layer 12
    lay12_1 = Conv2DTranspose(8*C1, (4,4), strides=(2,2), use_bias=B1, padding='same', output_padding=(1,1), data_format=data_format)(concat5)
    lay12_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay12_1)
    act12 = LeakyReLU(alpha=LR)(lay12_2)
    drop12 = tf.keras.layers.Dropout(d)(act12)
    concat4 = tf.concat([drop12, act4], axis=axis)

    # Layer 13
    lay13_1 = Conv2DTranspose(4*C1, (4,4), strides=(2,2), use_bias=B1, padding='same', output_padding=(1,1), data_format=data_format)(concat4)
    lay13_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay13_1)
    act13 = LeakyReLU(alpha=LR)(lay13_2)
    concat3 = tf.concat([act13, act3], axis=axis)

    # Layer 14
    lay14_1 = Conv2DTranspose(2*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(concat3)
    lay14_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay14_1)
    act14 = LeakyReLU(alpha=LR)(lay14_2)

    # Layer 15
    lay15_1 = Conv2DTranspose(C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format=data_format)(act14)
    lay15_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay15_1)
    act15 = LeakyReLU(alpha=LR)(lay15_2)

    # Layer 16
    lay16_1 = Conv2DTranspose(1, (K1,K1), strides=(2,2), use_bias=B2, padding='same', data_format=data_format)(act15)
    act16 = tf.keras.layers.Activation(activation='sigmoid', dtype="float32")(lay16_1)

    # Model
    build = tf.keras.Model(inp, act16)
    return build


# def CreatePaperModel():
#     inp = Input(shape=(424, 424, 3))

#     # Layer 1
#     lay1 = Conv2D(32, (3,3), strides=(2,2),  use_bias=False, padding='same', data_format="channels_last")(inp) # Activate bias maybe
#     act1 = LeakyReLU(alpha=0.2)(lay1)

#     # Layer 2
#     lay2_1 = Conv2D(64, (3,3), strides=(2,2), use_bias=False, padding='same', data_format="channels_last")(act1)
#     lay2_2 = BatchNormalization(momentum=0.9, epsilon=1e-4)(lay2_1)
#     act2 = LeakyReLU(alpha=0.2)(lay2_2)

#     # Layer 3
#     lay3_1 = Conv2D(128, (4,4), strides=(2,2), use_bias=False, padding='same', data_format="channels_last")(act2)
#     lay3_2 = BatchNormalization(momentum=0.9, epsilon=1e-4)(lay3_1)
#     act3 = LeakyReLU(alpha=0.2)(lay3_2)

#     # Layer 4
#     lay4_1 = Conv2D(256, (4,4), strides=(2,2), use_bias=False, padding='same', data_format="channels_last")(act3)
#     lay4_2 = BatchNormalization(momentum=0.9, epsilon=1e-4)(lay4_1)
#     act4 = LeakyReLU(alpha=0.2)(lay4_2)

#     # Layer 5
#     lay5_1 = Conv2D(256, (4,4), strides=(2,2), use_bias=False, padding='same', data_format="channels_last")(act4)
#     lay5_2 = BatchNormalization(momentum=0.9, epsilon=1e-4)(lay5_1)
#     act5 = LeakyReLU(alpha=0.2)(lay5_2)
#     drop5 = tf.keras.layers.Dropout(0.5)(act5)

#     # Layer 6
#     lay6_1 = Conv2D(256, (4,4), strides=(2,2), use_bias=False, padding='same', data_format="channels_last")(drop5)
#     lay6_2 = BatchNormalization(momentum=0.9, epsilon=1e-4)(lay6_1)
#     act6 = LeakyReLU(alpha=0.2)(lay6_2)
#     drop6 = tf.keras.layers.Dropout(0.5)(act6)

#     # Layer 7
#     lay7_1 = Conv2D(512, (4,4), strides=(2,2), use_bias=False, padding='same', data_format="channels_last")(drop6)
#     lay7_2 = BatchNormalization(momentum=0.9, epsilon=1e-4)(lay7_1)
#     act7 = LeakyReLU(alpha=0.2)(lay7_2)
#     drop7 = tf.keras.layers.Dropout(0.5)(act7)

#     # Layer 8
#     lay8_1 = Conv2D(512, (4,4), strides=(2,2), use_bias=False, padding='same', data_format="channels_last")(drop7)
#     lay8_2 = BatchNormalization(momentum=0.9, epsilon=1e-4)(lay8_1)
#     act8 = LeakyReLU(alpha=0.2)(lay8_2)
#     drop8 = tf.keras.layers.Dropout(0.5)(act8)
    
#     # Layer 9
#     lay9_1 = Conv2DTranspose(512, (4,4), strides=(2,2), use_bias=False, padding='same', data_format="channels_last")(drop8)
#     lay9_2 = BatchNormalization(momentum=0.9, epsilon=1e-4)(lay9_1)
#     act9 = LeakyReLU(alpha=0.2)(lay9_2)
#     drop9 = tf.keras.layers.Dropout(0.5)(act9)
#     concat7 = tf.concat([drop9, drop7], axis=3)
#     # Layer 10
#     lay10_1 = Conv2DTranspose(512, (4,4), strides=(2,2), use_bias=False, padding='same', output_padding=(1,1), data_format="channels_last")(concat7)
#     lay10_2 = BatchNormalization(momentum=0.9, epsilon=1e-4)(lay10_1)
#     act10 = LeakyReLU(alpha=0.2)(lay10_2)
#     drop10 = tf.keras.layers.Dropout(0.5)(act10)
#     concat6 = tf.concat([drop10, drop6], axis=3)

#     # Layer 11
#     lay11_1 = Conv2DTranspose(256, (4,4), strides=(2,2), use_bias=False, padding='same', data_format="channels_last")(concat6)
#     lay11_2 = BatchNormalization(momentum=0.9, epsilon=1e-4)(lay11_1)
#     act11 = LeakyReLU(alpha=0.2)(lay11_2)
#     drop11 = tf.keras.layers.Dropout(0.5)(act11)
#     concat5 = tf.concat([drop11, drop5], axis=3)
#     # Layer 12
#     lay12_1 = Conv2DTranspose(256, (4,4), strides=(2,2), use_bias=False, padding='same', output_padding=(1,1), data_format="channels_last")(concat5)
#     lay12_2 = BatchNormalization(momentum=0.9, epsilon=1e-4)(lay12_1)
#     act12 = LeakyReLU(alpha=0.2)(lay12_2)
#     drop12 = tf.keras.layers.Dropout(0.5)(act12)
#     concat4 = tf.concat([drop12, act4], axis=3)

#     # Layer 13
#     lay13_1 = Conv2DTranspose(128, (4,4), strides=(2,2), use_bias=False, padding='same', output_padding=(1,1), data_format="channels_last")(concat4)
#     lay13_2 = BatchNormalization(momentum=0.9, epsilon=1e-4)(lay13_1)
#     act13 = LeakyReLU(alpha=0.2)(lay13_2)
#     concat3 = tf.concat([act13, act3], axis=3)

#     # Layer 14
#     lay14_1 = Conv2DTranspose(64, (4,4), strides=(2,2), use_bias=False, padding='same', data_format="channels_last")(concat3)
#     lay14_2 = BatchNormalization(momentum=0.9, epsilon=1e-4)(lay14_1)
#     act14 = LeakyReLU(alpha=0.2)(lay14_2)
#     concat2 = tf.concat([act14, act2], axis=3)

#     # Layer 15
#     lay15_1 = Conv2DTranspose(32, (3,3), strides=(2,2), use_bias=False, padding='same', data_format="channels_last")(concat2)
#     lay15_2 = BatchNormalization(momentum=0.9, epsilon=1e-4)(lay15_1)
#     act15 = LeakyReLU(alpha=0.2)(lay15_2)
#     concat1 = tf.concat([act15, act1], axis=3)

#     # Layer 16
#     lay16_1 = Conv2DTranspose(1, (3,3), strides=(2,2), use_bias=False, padding='same', data_format="channels_last")(concat1)
#     act16 = tf.keras.layers.Activation(activation='sigmoid', dtype="float32")(lay16_1)

#     # Model
#     build = tf.keras.Model(inp, act16)
#     return build