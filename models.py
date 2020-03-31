import tensorflow
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Conv2D, Input, MaxPool2D, UpSampling2D)
from tensorflow.keras.models import Model, load_model, model_from_json


def ResidualModel1(image, detail, is_training):
        conv_regularizer = tensorflow.keras.regularizers.l2(l=1e-10)
        conv_prev = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                           kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(detail)

        for _ in range(12):
            conv_temp = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                               kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(conv_prev)
            conv_temp = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                               kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(conv_temp)

            conv_prev = Add()([conv_prev, conv_temp])

        neg_residual = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                              kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(conv_prev)
        final_out = Add()([image, neg_residual])

        res_model = Model(inputs=[image, detail], outputs=final_out)
        return res_model


def ResidualModel2(image, detail, is_training):
    conv_regularizer = tensorflow.keras.regularizers.l2(l=1e-10)
    l1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(detail)
    l2 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(l1)
    l3 = MaxPool2D(pool_size=(2, 2))(l2)

    l4 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(l3)
    l5 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(l4)
    l6 = MaxPool2D(pool_size=(2, 2))(l5)

    l7 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(l6)
    l8 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(l7)
    l9 = MaxPool2D(pool_size=(2, 2))(l8)

    l9 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(l9)

    l_8 = UpSampling2D(size=(2, 2), interpolation='bilinear')(l9)
    l_8 = Add()([l_8, l8])
    l_8 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                    kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(l_8)
    l_8 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                    kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(l_8)
    l_5 = UpSampling2D(size=(2, 2), interpolation='bilinear')(l_8)
    l_5 = Add()([l_5, l5])

    l_5 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                    kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(l_5)
    l_5 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                    kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(l_5)
    l_2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(l_5)
    l_2 = Add()([l_2, l2])
    l_2 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                    kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(l_2)
    l_2 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                    kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(l_2)

    neg = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=True,
                    kernel_initializer='glorot_uniform', kernel_regularizer=conv_regularizer)(l_2)

    final_out = Add()([image, neg])

    res_model = Model(inputs=[image, detail], outputs=final_out)
    print(res_model.summary())
    return res_model
