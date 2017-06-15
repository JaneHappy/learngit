# coding: utf-8
# dogs vs cats
# modified




from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img

from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import Activation, Dropout, Flatten, Dense

img_width, img_height = 150, 150
trn_data_dir = '../../dogs_cats/train'
tst_data_dir = '../../dogs_cats/validation'
nb_trn_samples = 1024-64
nb_tst_samples = 64
epochs = 1
batch_size = 64
input_shape = (1,)+(img_width, img_height, 3)  #input_size




model = Sequential()
model.add(Conv3D(32, (3,3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2,2,2)))

model.add(Conv3D(32, (3,3,3)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2,2,2)))

model.add(Conv3D(64, (3,3,3)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2,2,2)))

# this model so far outputs 3D features maps
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])




# this is the augmentation configuration we will use for training
trn_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
tst_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in subfolders of 'data/train', and indefinitely generate batches of augmented image data.
trn_generator = trn_datagen.flow_from_directory(trn_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')
tst_generator = tst_datagen.flow_from_directory(tst_data_dir, target_size=(img_width,img_height), batch_size=batch_size, class_mode='binary')

model.fit_generator(
	trn_generator, steps_per_epoch=nb_trn_samples//batch_size, 
	epochs=epochs, 
	validation_data=tst_generator, validation_steps=nb_tst_samples//batch_size)

model.save_weights('second_try.h5')







