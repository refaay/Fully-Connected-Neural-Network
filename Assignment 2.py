import numpy as np
import time, json

import keras as keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU
from keras.activations import relu, tanh, elu
from keras.utils import np_utils
from keras.optimizers import Adagrad, Adam, Nadam, SGD
from keras.regularizers import l2, l1_l2
from keras.layers.normalization import BatchNormalization
from keras.callbacks import *

import datetime

now = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

np.random.seed(100)

batch_size = 300
nb_classes = 10
nb_epoch = 1000
data_augmentation = True

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train -= np.mean(X_train, axis = 0)
X_train /= np.std(X_train, axis = 0)

X_test -= np.mean(X_test, axis = 0)
X_test /= np.std(X_test, axis = 0)

# to 1-hot encoding
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

datagen = ImageDataGenerator(
featurewise_center=False, # set input mean to 0 over the dataset
samplewise_center=False, # set each sample mean to 0
featurewise_std_normalization=False, # divide inputs by std of the dataset
samplewise_std_normalization=False, # divide each input by its std
zca_whitening=False, # apply ZCA whitening
rotation_range=1, # randomly rotate images in the range (degrees, 0 to 180) ###############5 shedeed
width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
horizontal_flip=True, # randomly flip images
vertical_flip=False) # randomly flip images


datagen.fit(X_train[:50000])

input_shape = X_train.shape[1:40000]
use_bias = True

model = Sequential()


model.add(Flatten(input_shape = input_shape))

#model.add(Activation('relu'))
#model.add(Dropout(0.3))
#model.add(BatchNormalization())

model.add(Dense(2500))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

#model.add(Dense(2300))
#model.add(Activation('relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.5))

#model.add(Dense(1500))
#model.add(Activation('relu'))
#model.add(Dropout(0.1))
#model.add(BatchNormalization())

model.add(Dense(1500))
model.add(Activation('relu'))
model.add(Dropout(0.35))
model.add(BatchNormalization())

model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()


tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + now , histogram_freq=0, write_graph=True, write_images=False)
adam = keras.optimizers.Adam(lr=1e-4, decay=1e-6)

model.compile(
    loss='categorical_crossentropy'
    , optimizer = 'adam'
    , metrics=['accuracy']
)

checkpointer = ModelCheckpoint(filepath="./weights9.hdf5", verbose=1, save_best_only=True, monitor='val_acc', mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
              patience=2, min_lr=1e-7)

model.fit_generator(datagen.flow(X_train[:40000], Y_train[:40000], batch_size=batch_size),
                    steps_per_epoch=40000//batch_size,
                    nb_epoch=nb_epoch,
                    validation_data=(X_train[40000:50000], Y_train[40000:50000])
                    ,callbacks=[tensorboard,checkpointer,reduce_lr])

scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', scores[0])
print('Test accuracy:', scores[1])

# Evaluate CCRn
classes = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
y_test = y_test.flatten()
accuracy_per_class = [0.] * 10

print("classes ")
print(classes)
print(len(classes))

print("ytest")
print(y_test)
print(len(y_test))

yy = y_test.astype(np.int64)
for i in range(classes.shape[0]):
   if classes[i] == y_test[i]:
        accuracy_per_class[yy[i]] += 1
for i in range(10):
    accuracy_per_class[i] /= 1000

c = ['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck']
for i in range(10):
    print("\nCCrn of %s is %f" % (c[i], accuracy_per_class[i]))
