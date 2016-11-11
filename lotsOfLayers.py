from __future__ import print_function
import numpy as np
import pandas as pd
import cv2
np.random.seed(1337)  # for reproducibility
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# function to clean an image
def cleanImage(image):
    mean = cv2.meanStdDev(image)[0]
    std = cv2.meanStdDev(image)[1]
    if (mean > 220):
        ret,image = cv2.threshold(image,min(mean+std+5,250),255,cv2.THRESH_BINARY)
    else:
        ret,image = cv2.threshold(image,min(max(mean+2*std,150),245),255,cv2.THRESH_BINARY)
    return image

# function to clean the data
def cleanData(data):
    clean_data = []
    for i in range(len(data)):
        clean_data.append(cleanImage(data[i]))
    clean_data = np.array(clean_data, dtype='float32')
    clean_data = clean_data.reshape((100000,3600))
    return clean_data

# data with shape (examples, 1d array)
def resizeData(data,fx=(0.5),fy=(0.5)):
    resized = []
    for i in range(len(data)):
        resized.append(cv2.resize(data[i]),None,fx=fx,fy=fy,interpolation=cv2.INTER_AREA)
    resized = np.array(resized, dtype='float32')    
    return resized

# generate More Images with certain amount of noise.

# put all the preprocessing into its own function, makes things a lot easier
def preprocessing():
    # import labels
    df = pd.read_csv('./train_y.csv', dtype='int32')
    y = np.array(df['Prediction'])

    # import images
    X = np.fromfile('./train_x.bin', dtype='uint8')
    X = X.reshape((100000,60,60))

    # clean the data
    print ('Cleaning Data')
    clean_data = cleanData(X)
    print ('Cleaning Done')

    # resize the data so we do calculations faster ==> moar experiments
    print ('Resizing')
    resized = []
    for i in range(len(clean_data)):
        resized.append(cv2.resize(clean_data[i].reshape((60,60)),None,fx=(28./60.),fy=(28./60.),interpolation=cv2.INTER_AREA))
    resized = np.array(resized)
    out_x = resized.reshape((100000,784))
    out_y = y
    print ('Resizing done')
    print ('Preprocessing Done')
    return out_x, out_y

# uncomment the line below to show an example image
# plt.imshow(clean_data[0].reshape((60,60)),cmap=plt.cm.gray_r)
out_x, out_y = preprocessing()
# the data, shuffled and split between train and test sets
max_examples = 70000
X_train = out_x[:max_examples]
y_train = out_y[:max_examples]
X_test = out_x[max_examples:]
y_test = out_y[max_examples:]

batch_size = 32
nb_classes = 19
nb_epoch = 100

#input image dimensions
img_rows, img_cols = 28,28

# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size regular
kernel_size = (5, 5)
# convolution kernel size small
kernel_size_small = (3, 3)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# CNN
model = Sequential()

# layer 1
model.add(Convolution2D(16, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))

model.add(Activation('relu'))

for i in range(2):
	model.add(Convolution2D(32, kernel_size[0], kernel_size[1]))
	model.add(Activation('relu'))


# layer 2
model.add(Convolution2D(32, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

# layer 3 we down size
model.add(Convolution2D(32, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

# max pooling 1
# model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(0.25))

# layer 4
model.add(Convolution2D(256, kernel_size_small[0], kernel_size_small[1]))
model.add(Activation('relu'))

model.add(Convolution2D(256, kernel_size_small[0], kernel_size_small[1]))
model.add(Activation('relu'))

# max pooling 2
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

# flatten pooling
model.add(Flatten())

# 1st dense layer
model.add(Dense(256))
model.add(Activation('relu'))

# 2nd dense layer
model.add(Dense(512))
model.add(Activation('relu'))

# class dense layer
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# compilation
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# logging
filepath="./lotsOfLayers-best.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
csv_logger = callbacks.CSVLogger('./lotsOfLayers.log')

# we stop when val_loss stops decreasing, 5 epochs of patience
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10)

callbacks_list = [checkpoint,csv_logger,early_stopping]

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# flow X_train data into datagenerator
datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        verbose=1,
                        validation_data=(X_test, Y_test), 
                        callbacks=callbacks_list)

score = model.evaluate(X_test, Y_test, verbose=0)
tprint('Test score:', score[0])
print('Test accuracy:', score[1])

