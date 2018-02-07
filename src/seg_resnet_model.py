from keras import models, layers
from keras import utils
import keras.backend as K

def bn_block(input_tensor, filters, kernel_size, strides, dropout=None):
    ''' Standard convolutional block with batch normalization,
    activation using a Rectified Linear Unit and dropout.abs
    Parameters:
    -----------
    input_tensor : 4d tensor of shape (samples, nrows, ncols, channels)
    filters : int scaler of number of filters 
    kernel_size : int tuple of convolution kernel size
    strides : int tuple of stride for convolution
    dropout : float of dropout rate
    '''
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, \
            padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    return(x)

def identity_block(input_tensor, filters, kernel_size):
    '''
    Identity block is a residual block that
    does not convolve input and merges to final layer

    Parameters:
    -----------
    input_tensor : <keras tensor>
    filters : list(<int>, <int>, <int>) 
    kernel_size : tuple(<int>, <int>)
    '''
    x = bn_block(input_tensor, filters[0], kernel_size=(1,1), \
            strides=(1,1))
    # convolution
    x = bn_block(x, filters[1], kernel_size=kernel_size,\
            strides=(1,1))
    # 1x1 conv for filter match
    x = layers.Conv2D(filters[2], kernel_size=(1,1), strides=(1,1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return(x)


def conv_block(input_tensor, filters, kernel_size, stride, \
        dropout=None):
    '''
    Residual block creates a short cut layer that connects
    input tensor to output of 3 convolutions. 

    Parameters:
    -----------
    input_tensor : <keras tensor>
    filters : list(<int>, <int>, <int>) 
    kernel_size : tuple(<int>, <int>)
    stride : tuple(<int>, <int>)
    dropout : <float>
    ''' 
    shortcut = layers.Conv2D(filters[2], kernel_size=kernel_size, \
            strides=stride, padding='same')(input_tensor)
    # 1x1 conv
    x = bn_block(input_tensor, filters[0], kernel_size=(1,1), \
            strides=stride, dropout=dropout)
    # convolution
    x = bn_block(x, filters[1], kernel_size=kernel_size,\
            strides=(1,1), dropout=dropout)
    # 1x1 conv for filter match
    x = layers.Conv2D(filters[2], kernel_size=(1,1), strides=(1,1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return(x)

def main_branch(input_tensor, filters, kernel_size, \
        stride, dropout):
    x = bn_block(input_tensor, filters[0], (7,7), stride, dropout)
    x = bn_block(x, filters[0], (3,3), stride, dropout)
    x = conv_block(x, filters, kernel_size, stride,\
            dropout)
    x = identity_block(x, filters, kernel_size)
    x = identity_block(x, filters, kernel_size)
    x = conv_block(x, filters*2, kernel_size, stride,\
            dropout)
    x = identity_block(x, filters*2, kernel_size)
    x = identity_block(x, filters*2, kernel_size)
    x = identity_block(x, filters*2, kernel_size)
    #x = layers.AveragePooling2D((7,7))(x)
    return(x)

def deconv_net(input_shape, output_shape):
    input_layer = layers.Input(shape=input_shape)
    short_cut = layers.Conv2D(16, (1,1), strides=(1,1), activation='relu')(input_layer)
    x = main_branch(input_layer, filters=[32,64,128], kernel_size=(3,3), stride=(2,2), dropout=0.25)
    x = layers.Conv2DTranspose(128, (2,2), strides=(2,2) ,activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = identity_block(x, filters=[32,64,128], kernel_size=(3,3))
    x = identity_block(x, filters=[32,64,128], kernel_size=(3,3))
    x = layers.Conv2DTranspose(64, (2,2), strides=(2,2), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = identity_block(x, filters=[32,64,64], kernel_size=(3,3))
    x = identity_block(x, filters=[32,64,64], kernel_size=(3,3))
    x = layers.Conv2DTranspose(32, (2,2), strides=(2,2), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = identity_block(x, filters=[32,32,32], kernel_size=(3,3))
    x = identity_block(x, filters=[32,32,32], kernel_size=(3,3))
    x = layers.Conv2DTranspose(16, (2,2), strides=(2,2), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = identity_block(x, filters=[32,32,16], kernel_size=(3,3))
    x = identity_block(x, filters=[32,32,16], kernel_size=(3,3))
    x = layers.Add()([x,short_cut])
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(2, (1,1), strides=(1,1), activation='softmax')(x)
    output_layer = layers.Reshape(output_shape)(x)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return(model)
