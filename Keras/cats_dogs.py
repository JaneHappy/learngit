# coding: utf-8
# https://keras-cn.readthedocs.io/en/latest/blog/image_classification_using_very_little_data/




from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
'''
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')  #rescale=1./255, 

img = load_img('../../dogs_cats/train/cat.0.jpg')#'data/train/cats/cat.0.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3,150,150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1,3,150,150)

# the .flow() command below generates batches of randomly transformed images and saves the results to the 'preview/' directory
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='../../dogs_cats/preview', save_prefix='cat', save_format='jpeg'):  #save_to_dir='preview'
	i += 1
	if i>20:
		break  # otherwise the generator would loop indefinitely
'''


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D  #Convolution2D,
from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K 
# dimensions of our images.
img_width, img_height = 150,150
trn_data_dir = '../../learngit_data/dogs_cats/train'#'../../dogs_cats/train'
validation_data_dir = '../../learngit_data/dogs_cats/validation'
nb_train_samples = 1024-64
nb_validation_samples = 64
epochs = 1 #50
batch_size = 64 #16
if K.image_data_format() == 'channels_first':
	input_shape = (3,img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)





model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=input_shape))#(150,150,3)))#Convolution2D(32,3,3, input_shape=(3,150,150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))#Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))#Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# the model so far outputs 3D features maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 10 feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1)) #2
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# this is the augmentation configuration we will use for testing: only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in subfolders of 'data/train', and indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(
	trn_data_dir,#'../../dogs_cats/train',#'data/train',  # this is the target directory
	target_size=(img_width, img_height),#150, 150),  # all images will be resized to 150x150
	batch_size=batch_size,#32,
	class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,#'../../dogs_cats/validation',#'data/validation',
	target_size=(img_width, img_height),#150,150),
	batch_size=batch_size,#32,
	class_mode='binary')


#model.fit_generator(train_generator, samples_per_epoch=2000, nb_epoch=50, validation_data=validation_generator, nb_val_samples=800)
model.fit_generator(train_generator,
	steps_per_epoch=nb_train_samples//batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=nb_validation_samples//batch_size)
model.save_weights('first_try.h5')  # always save your weights after training or during training








'''
'''

