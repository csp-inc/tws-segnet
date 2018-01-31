from keras.models import Sequential
from keras.layers import Layer, Convolution2D, Activation, \
        MaxPooling2D, Dropout, ZeroPadding2D, BatchNormalization, \
        UpSampling2D, Reshape
from keras import utils
from keras.models import model_from_json

def import_model(model_json, model_weights):
    '''
    Imports a keras model architecture and 
    associated weights.

    Parameters:
    -----------
    model_json : <str> of keras model in json
    format

    model_weights : <str> of keras model parameters weights
    '''

    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights)
    return(loaded_model)

def create_encoding_layers(filter_size=8,pool_size=2,dropout_rate=0.45):
    return[ 
    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(filter_size*4, kernel_size=(3,3), padding='valid', name='Conv_1a'),
    BatchNormalization(),
    Activation('relu'),
    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(filter_size*4, kernel_size=(3,3), padding='valid', name='Conv_1b'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    Dropout(dropout_rate),

    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(filter_size*8, kernel_size=(3,3), padding='valid', name='Conv_2a'),
    BatchNormalization(),
    Activation('relu'),
    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(filter_size*8, kernel_size=(3,3), padding='valid', name='Conv_2b'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    Dropout(dropout_rate),
           
    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(filter_size*16, kernel_size=(3,3), padding='valid', name='Conv_3a'),
    BatchNormalization(),
    Activation('relu'),
    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(filter_size*16, kernel_size=(3,3), padding='valid', name='Conv_3b'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    Dropout(dropout_rate),

    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(filter_size*32, kernel_size=(3,3), padding='valid', name='Conv_4'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    Dropout(dropout_rate),

    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(filter_size*64, kernel_size=(3,3), padding='valid', name='Conv_5'),
    BatchNormalization(),
    Activation('relu')
    ]
    
def create_decoding_layers(filter_size=8,pool_size=2,dropout_rate=0.45):
    return[
    UpSampling2D(size=(pool_size, pool_size), name='uPool_1'),
    Dropout(dropout_rate),
    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(filter_size*64, kernel_size=(3,3), padding='valid', name='deConv_1a'),
    BatchNormalization(),
    Activation('relu'),
    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(filter_size*32, kernel_size=(3,3), padding='valid', name='deConv_1b'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(size=(pool_size, pool_size), name='uPool_2'),
    Dropout(dropout_rate),
    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(filter_size*16, kernel_size=(3,3), padding='valid', name='deConv_2a'),
    BatchNormalization(),
    Activation('relu'),
    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(filter_size*8, kernel_size=(3,3), padding='valid', name='deConv_2b'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(size=(pool_size, pool_size), name='uPool_3'),
    Dropout(dropout_rate),
    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(filter_size*4, kernel_size=(3,3), padding='valid', name='deConv_3a'),
    BatchNormalization(),
    Activation('relu'),
    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(filter_size*4, kernel_size=(3,3), padding='valid', name='deConv_3b'),
    BatchNormalization(),
    Activation('relu'),
    
    UpSampling2D(size=(pool_size, pool_size), name='uPool_4'),
    ]

def create_classification_layers(num_classes, input_shape, output_shape):
    return[
    ZeroPadding2D(padding=(1, 1)),
    Convolution2D(num_classes, kernel_size=(3,3), padding='valid', name='deConv_4'),
    Reshape((output_shape, num_classes), input_shape=input_shape),
    Activation('softmax')
    ]

def build_segnet_basic(num_classes, input_shape, output_shape):
    segnet = Sequential()
    segnet.add(Layer(input_shape=input_shape))
    
    segnet.encoding_layers = create_encoding_layers(filter_size=8,\
            pool_size=2,dropout_rate=0.45)
    for l in segnet.encoding_layers:
        segnet.add(l)
   
    segnet.decoding_layers = create_decoding_layers(filter_size=8,\
            pool_size=2,dropout_rate=0.45)
    for l in segnet.decoding_layers:
        segnet.add(l)
    
    segnet.classification_layers = create_classification_layers(num_classes, \
            input_shape, output_shape)
    for l in segnet.classification_layers:
        segnet.add(l)
    print(segnet.summary())
    return(segnet)
