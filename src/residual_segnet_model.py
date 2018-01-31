from keras import layers
from keras import models

def bn_block(x, filters, kernel_size, strides, dropout):
    ''' Standard convolutional block with batch normalization,
        activation using a Rectified Linear Unit and dropout.
    
    Parameters:
    -----------
    x : 4d tensor of shape (samples, nrows, ncols, channels)
    filters : int scaler of number of filters 
    kernel_size : int tuple of convolution kernel size
    strides : int tuple of stride for convolution
    dropout : float of dropout rate
    '''
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, \
            padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    return(x)

def residual_block(x, filters, kernel_size, strides, dropout, conv_layers):
    '''
    Residual block constructor 
    
    Parameters:
    -----------
    x : 4d tensor of shape (samples, nrows, ncols, channels)
    filters : int scaler of number of filters 
    kernel_size : int tuple of convolution kernel size
    strides : int tuple of stride for convolution
    dropout : float of dropout rate
    conv_layers : int scalar number of bn_block layers to construct
    '''
    shortcut = layers.Conv2D(filters, kernel_size=(1,1), strides=(1,1), \
            padding='valid')(x)
    for c in range(conv_layers-1):
        x = bn_block(x, filters, kernel_size, strides, dropout)
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,\
            padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout)(x)
    return(x)

def deconv_block(shortcut_x, x, filters, kernel_size, strides, dropout, conv_layers):
    '''
    '''
    x = layers.UpSampling2D(size=(2,2))(x)
    x = layers.concatenate([shortcut_x, x])
    x = residual_block(x, filters, kernel_size, strides, dropout, conv_layers)
    return(x)

def pooling_block(x, filters, kernel_size, strides, dropout, conv_layers):
    '''
    '''
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = residual_block(x, filters, kernel_size, strides, dropout, conv_layers)
    return(x)

def residual_segnet(input_shape, output_shape, n_classes, ifilter, num_blocks):
    '''
    '''
    input_layer = layers.Input(shape=input_shape)
    a = []
    # encoder layers
    for i in range(num_blocks):
        if i == 0: #first block no max pooling
            x = residual_block(input_layer, ifilter*(2**i), kernel_size=(3,3), strides=(1,1), \
                dropout=0.4, conv_layers=2)
        else:
            x = pooling_block(x, ifilter*(2**i), kernel_size=(3,3), strides=(1,1), \
                dropout=0.4, conv_layers=2)   
        a.append(x) #save the output blocks for shortcuts on deconvolution
    # deepest layer
    x = bn_block(x, ifilter*(2**num_blocks), kernel_size=(3,3), strides=(1,1), dropout=0.4) 
    x = residual_block(x, ifilter*(2**num_blocks), kernel_size=(3,3), strides=(1,1), \
            dropout=0.4, conv_layers=3)
    x = bn_block(x, ifilter*(2**(num_blocks-1)), kernel_size=(3,3), strides=(1,1), dropout=0.4) 
    # decoder layers
    for i in range(num_blocks, 1, -1):
        x = deconv_block(a[i-2], x, ifilter*(2**(i-1)), kernel_size=(3,3), strides=(1,1), \
                dropout=0.4, conv_layers=2)
    x = layers.Conv2D(n_classes, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = layers.Reshape(target_shape=output_shape)(x)
    output_layer = layers.Activation('softmax')(x)
    model = models.Model(inputs=[input_layer], outputs=[output_layer])
    return(model)
