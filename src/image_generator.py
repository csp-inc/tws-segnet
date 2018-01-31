# Using the keras ImageDataGenerator class
'''
from keras.preprocessing.image import import ImageDataGenerator
#from src.image2 import ImageDataGenerator #use custom ImageDataGenerator

def pre_process_y(y):
    n_classes = np.shape(y)[-1]
    y_oneband = y[:,:,0]
    y_out = np.zeros((y.shape[0], \
    y.shape[1], n_classes))
    for i in range(n_classes):
    y_out[:,:,i][y_oneband==i] = 1
    out = y_out.reshape(y_out.shape[0], -1, n_classes)
    return(out.astype('uint8'))

data_xgen_args = dict(rescale=1./255, horizontal_flip=True, vertical_flip=True)
data_ygen_args = dict(horizontal_flip=True, vertical_flip=True, \
        preprocessing_function=pre_process_y)

image_datagen = ImageDataGenerator(**data_xgen_args)
mask_datagen= ImageDataGenerator(**data_ygen_args)

train_data_dir ='/contents/images'
x_dir = '%s/xtrain'%train_data_dir
y_dir = '%s/ytrain'%train_data_dir

seed = 90
batch_size = 10
img_width = 256
img_height = 256

image_generator = image_datagen.flow_from_directory(x_dir, \
    target_size=(img_width, img_height),\
    class_mode=None,\
    batch_size=batch_size,\
    shuffle=True,\
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(y_dir, \
    class_mode = None,\
    batch_size=batch_size,\
    shuffle=True,\
    seed=seed)


# combine generators into one which yields image and masks
x_train = next(image_generator)
y_train = next(mask_generator)
#train_generator = zip(image_generator, mask_generator)

#train_image_list = train_generator.filenames

print(np.shape(y_train))
plt.imshow(y_train[0,:,:,0])
'''
