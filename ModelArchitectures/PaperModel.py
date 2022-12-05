######################
###     IMPORTS    ###
######################
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, Dense, LeakyReLU, Conv2DTranspose

def CreateGridPaperModel(C1, K1, d, B1, B2, LR, momentum):
    inp = Input(shape=(424, 424, 3))

    # Layer 1
    lay1 = Conv2D(C1, (K1,K1), strides=(2,2),  use_bias=B1, padding='same', data_format="channels_last")(inp) # Activate bias maybe
    act1 = LeakyReLU(alpha=LR)(lay1)

    # Layer 2
    lay2_1 = Conv2D(2*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(act1)
    lay2_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay2_1)
    act2 = LeakyReLU(alpha=LR)(lay2_2)

    # Layer 3
    lay3_1 = Conv2D(4*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(act2)
    lay3_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay3_1)
    act3 = LeakyReLU(alpha=LR)(lay3_2)

    # Layer 4
    lay4_1 = Conv2D(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(act3)
    lay4_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay4_1)
    act4 = LeakyReLU(alpha=LR)(lay4_2)

    # Layer 5
    lay5_1 = Conv2D(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(act4)
    lay5_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay5_1)
    act5 = LeakyReLU(alpha=LR)(lay5_2)
    drop5 = tf.keras.layers.Dropout(d)(act5)

    # Layer 6
    lay6_1 = Conv2D(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(drop5)
    lay6_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay6_1)
    act6 = LeakyReLU(alpha=LR)(lay6_2)
    drop6 = tf.keras.layers.Dropout(d)(act6)

    # Layer 7
    lay7_1 = Conv2D(16*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(drop6)
    lay7_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay7_1)
    act7 = LeakyReLU(alpha=LR)(lay7_2)
    drop7 = tf.keras.layers.Dropout(d)(act7)

    # Layer 8
    lay8_1 = Conv2D(16*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(drop7)
    lay8_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay8_1)
    act8 = LeakyReLU(alpha=LR)(lay8_2)
    drop8 = tf.keras.layers.Dropout(d)(act8)
    
    # Layer 9
    lay9_1 = Conv2DTranspose(16*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(drop8)
    lay9_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay9_1)
    act9 = LeakyReLU(alpha=LR)(lay9_2)
    drop9 = tf.keras.layers.Dropout(d)(act9)
    concat7 = tf.concat([drop9, drop7], axis=3)
    # Layer 10
    lay10_1 = Conv2DTranspose(16*C1, (4,4), strides=(2,2), use_bias=B1, padding='same', output_padding=(1,1), data_format="channels_last")(concat7)
    lay10_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay10_1)
    act10 = LeakyReLU(alpha=LR)(lay10_2)
    drop10 = tf.keras.layers.Dropout(d)(act10)
    concat6 = tf.concat([drop10, drop6], axis=3)

    # Layer 11
    lay11_1 = Conv2DTranspose(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(concat6)
    lay11_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay11_1)
    act11 = LeakyReLU(alpha=LR)(lay11_2)
    drop11 = tf.keras.layers.Dropout(d)(act11)
    concat5 = tf.concat([drop11, drop5], axis=3)
    # Layer 12
    lay12_1 = Conv2DTranspose(8*C1, (4,4), strides=(2,2), use_bias=B1, padding='same', output_padding=(1,1), data_format="channels_last")(concat5)
    lay12_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay12_1)
    act12 = LeakyReLU(alpha=LR)(lay12_2)
    drop12 = tf.keras.layers.Dropout(d)(act12)
    concat4 = tf.concat([drop12, act4], axis=3)

    # Layer 13
    lay13_1 = Conv2DTranspose(4*C1, (4,4), strides=(2,2), use_bias=B1, padding='same', output_padding=(1,1), data_format="channels_last")(concat4)
    lay13_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay13_1)
    act13 = LeakyReLU(alpha=LR)(lay13_2)
    concat3 = tf.concat([act13, act3], axis=3)

    # Layer 14
    lay14_1 = Conv2DTranspose(2*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(concat3)
    lay14_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay14_1)
    act14 = LeakyReLU(alpha=LR)(lay14_2)
    concat2 = tf.concat([act14, act2], axis=3)

    # Layer 15
    lay15_1 = Conv2DTranspose(C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(concat2)
    lay15_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay15_1)
    act15 = LeakyReLU(alpha=LR)(lay15_2)
    concat1 = tf.concat([act15, act1], axis=3)

    # Layer 16
    lay16_1 = Conv2DTranspose(1, (K1,K1), strides=(2,2), use_bias=B2, padding='same', data_format="channels_last")(concat1)
    act16 = tf.keras.layers.Activation(activation='sigmoid', dtype="float32")(lay16_1)

    # Model
    build = tf.keras.Model(inp, act16)
    return build

def CreatePaperModel():
    C1, K1, d, B1, B2, LR, momentum = 32, 4, 0.3, False, False, 0.2, 0.9
    inp = Input(shape=(106, 106, 3))

    # Layer 3
    lay3_1 = Conv2D(4*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(inp)
    lay3_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay3_1)
    act3 = LeakyReLU(alpha=LR)(lay3_2)

    # Layer 4
    lay4_1 = Conv2D(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(act3)
    lay4_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay4_1)
    act4 = LeakyReLU(alpha=LR)(lay4_2)

    # Layer 5
    lay5_1 = Conv2D(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(act4)
    lay5_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay5_1)
    act5 = LeakyReLU(alpha=LR)(lay5_2)
    drop5 = tf.keras.layers.Dropout(d)(act5)

    # Layer 6
    lay6_1 = Conv2D(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(drop5)
    lay6_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay6_1)
    act6 = LeakyReLU(alpha=LR)(lay6_2)
    drop6 = tf.keras.layers.Dropout(d)(act6)

    # Layer 7
    lay7_1 = Conv2D(16*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(drop6)
    lay7_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay7_1)
    act7 = LeakyReLU(alpha=LR)(lay7_2)
    drop7 = tf.keras.layers.Dropout(d)(act7)

    # Layer 8
    lay8_1 = Conv2D(16*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(drop7)
    lay8_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay8_1)
    act8 = LeakyReLU(alpha=LR)(lay8_2)
    drop8 = tf.keras.layers.Dropout(d)(act8)
    
    # Layer 9
    lay9_1 = Conv2DTranspose(16*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(drop8)
    lay9_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay9_1)
    act9 = LeakyReLU(alpha=LR)(lay9_2)
    drop9 = tf.keras.layers.Dropout(d)(act9)
    concat7 = tf.concat([drop9, drop7], axis=3)
    # Layer 10
    lay10_1 = Conv2DTranspose(16*C1, (4,4), strides=(2,2), use_bias=B1, padding='same', output_padding=(1,1), data_format="channels_last")(concat7)
    lay10_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay10_1)
    act10 = LeakyReLU(alpha=LR)(lay10_2)
    drop10 = tf.keras.layers.Dropout(d)(act10)
    concat6 = tf.concat([drop10, drop6], axis=3)

    # Layer 11
    lay11_1 = Conv2DTranspose(8*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(concat6)
    lay11_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay11_1)
    act11 = LeakyReLU(alpha=LR)(lay11_2)
    drop11 = tf.keras.layers.Dropout(d)(act11)
    concat5 = tf.concat([drop11, drop5], axis=3)
    # Layer 12
    lay12_1 = Conv2DTranspose(8*C1, (4,4), strides=(2,2), use_bias=B1, padding='same', output_padding=(1,1), data_format="channels_last")(concat5)
    lay12_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay12_1)
    act12 = LeakyReLU(alpha=LR)(lay12_2)
    drop12 = tf.keras.layers.Dropout(d)(act12)
    concat4 = tf.concat([drop12, act4], axis=3)

    # Layer 13
    lay13_1 = Conv2DTranspose(4*C1, (4,4), strides=(2,2), use_bias=B1, padding='same', output_padding=(1,1), data_format="channels_last")(concat4)
    lay13_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay13_1)
    act13 = LeakyReLU(alpha=LR)(lay13_2)
    concat3 = tf.concat([act13, act3], axis=3)

    # Layer 14
    lay14_1 = Conv2DTranspose(2*C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(concat3)
    lay14_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay14_1)
    act14 = LeakyReLU(alpha=LR)(lay14_2)

    # Layer 15
    lay15_1 = Conv2DTranspose(C1, (K1,K1), strides=(2,2), use_bias=B1, padding='same', data_format="channels_last")(act14)
    lay15_2 = BatchNormalization(momentum=momentum, epsilon=1e-4)(lay15_1)
    act15 = LeakyReLU(alpha=LR)(lay15_2)

    # Layer 16
    lay16_1 = Conv2DTranspose(1, (K1,K1), strides=(2,2), use_bias=B2, padding='same', data_format="channels_last")(act15)
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