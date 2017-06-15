# coding: utf-8
# blog.csdn.net/sinat_26917383/article/details/72861152




import h5py
from keras.models import model_from_json
#import sys
#saveout = sys.stdout
#fsock = open('classes5_out.log', 'w')
#sys.stdout = fsock



from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img

# load and model building
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())  # his converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))



# multi-classification
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])




trn_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
tst_datagen = ImageDataGenerator(rescale=1./255)

trn_generator = trn_datagen.flow_from_directory(
	'../../learngit_data/re/train',#'../../re/train',#'re/train',
	target_size=(150,150),  # all images will be resized to 150x150
	batch_size=64, #128,64,  #batch_size=32
	class_mode='categorical')
tst_generator = tst_datagen.flow_from_directory(
	'../../learngit_data/re/test',#'re/test',
	target_size=(150,150), batch_size=64, class_mode='categorical')



# train region
model.fit_generator(trn_generator, samples_per_epoch=400, nb_epoch=1, validation_data=tst_generator, nb_val_samples=800)  #samples_per_epoch=2000, nb_epoch=50
#model.save_weights('re/first_try_animals.h5')
model.save_weights('first_try_animals.h5')



#sys.stdout = saveout
#fsock.close()





'''
Using TensorFlow backend.
Found 400 images belonging to 5 classes.
Found 100 images belonging to 5 classes.
/home/ubuntu/Program/learngit/Keras/classes5.py:58: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras.pre..., validation_data=<keras.pre..., steps_per_epoch=31, epochs=1, validation_steps=800)`
  model.fit_generator(trn_generator, samples_per_epoch=2000, nb_epoch=1, validation_data=tst_generator, nb_val_samples=800)  #nb_epoch=50
Epoch 1/1
2017-06-15 17:18:23.455761: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-15 17:18:23.455801: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-15 17:18:23.455806: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.

 1/31 [..............................] - ETA: 898s - loss: 1.5952 - acc: 0.2500
 2/31 [>.............................] - ETA: 733s - loss: 6.4975 - acc: 0.2188
 3/31 [=>............................] - ETA: 716s - loss: 8.2502 - acc: 0.2292
 4/31 [==>...........................] - ETA: 657s - loss: 9.3899 - acc: 0.2188
 5/31 [===>..........................] - ETA: 622s - loss: 9.8771 - acc: 0.2219
 6/31 [====>.........................] - ETA: 598s - loss: 10.2913 - acc: 0.2214
 7/31 [=====>........................] - ETA: 512s - loss: 10.2603 - acc: 0.2433
 8/31 [======>.......................] - ETA: 488s - loss: 10.6169 - acc: 0.2344
 9/31 [=======>......................] - ETA: 466s - loss: 10.9404 - acc: 0.2240
10/31 [========>.....................] - ETA: 444s - loss: 11.0300 - acc: 0.2281
11/31 [=========>....................] - ETA: 421s - loss: 11.1263 - acc: 0.2301
12/31 [==========>...................] - ETA: 399s - loss: 11.3117 - acc: 0.2253
13/31 [===========>..................] - ETA: 377s - loss: 11.3336 - acc: 0.2284
14/31 [============>.................] - ETA: 338s - loss: 11.3156 - acc: 0.2344
15/31 [=============>................] - ETA: 319s - loss: 11.4343 - acc: 0.2313
16/31 [==============>...............] - ETA: 300s - loss: 11.4594 - acc: 0.2334
17/31 [===============>..............] - ETA: 283s - loss: 11.4816 - acc: 0.2353
18/31 [================>.............] - ETA: 263s - loss: 11.5293 - acc: 0.2352
19/31 [=================>............] - ETA: 244s - loss: 11.5853 - acc: 0.2344
20/31 [==================>...........] - ETA: 224s - loss: 11.6637 - acc: 0.2313
21/31 [===================>..........] - ETA: 196s - loss: 11.6839 - acc: 0.2321
22/31 [====================>.........] - ETA: 177s - loss: 11.6680 - acc: 0.2351
23/31 [=====================>........] - ETA: 158s - loss: 11.7082 - acc: 0.2344
24/31 [======================>.......] - ETA: 139s - loss: 11.7660 - acc: 0.2324
25/31 [=======================>......] - ETA: 119s - loss: 11.8192 - acc: 0.2306
26/31 [========================>.....] - ETA: 99s - loss: 11.8869 - acc: 0.2278 
27/31 [=========================>....] - ETA: 79s - loss: 11.9059 - acc: 0.2274
28/31 [==========================>...] - ETA: 58s - loss: 11.9786 - acc: 0.2238
29/31 [===========================>..] - ETA: 38s - loss: 11.9998 - acc: 0.2236
30/31 [============================>.] - ETA: 19s - loss: 11.9524 - acc: 0.2276
31/31 [==============================] - 2011s - loss: 11.9736 - acc: 0.2272 - val_loss: 12.0640 - val_acc: 0.2400
Traceback (most recent call last):
  File "/home/ubuntu/Program/learngit/Keras/classes5.py", line 59, in <module>
    model.save_weights('re/first_try_animals.h5')
  File "/usr/local/lib/python2.7/dist-packages/keras/models.py", line 723, in save_weights
    raise ImportError('`save_weights` requires h5py.')
ImportError: `save_weights` requires h5py.
[Finished in 2031.4s with exit code 1]
[shell_cmd: python -u "/home/ubuntu/Program/learngit/Keras/classes5.py"]
[dir: /home/ubuntu/Program/learngit/Keras]
[path: /home/ubuntu/bin:/home/ubuntu/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin]









Using TensorFlow backend.
Found 400 images belonging to 5 classes.
Found 100 images belonging to 5 classes.
/home/ubuntu/Program/learngit/Keras/classes5.py:67: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras.pre..., validation_data=<keras.pre..., steps_per_epoch=6, epochs=1, validation_steps=800)`
  model.fit_generator(trn_generator, samples_per_epoch=400, nb_epoch=1, validation_data=tst_generator, nb_val_samples=800)  #samples_per_epoch=2000, nb_epoch=50
Epoch 1/1
2017-06-15 19:46:18.327729: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-15 19:46:18.327770: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-15 19:46:18.327776: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.

1/6 [====>.........................] - ETA: 159s - loss: 1.6256 - acc: 0.1719
2/6 [=========>....................] - ETA: 99s - loss: 4.1917 - acc: 0.1875 
3/6 [==============>...............] - ETA: 65s - loss: 6.4249 - acc: 0.1979
4/6 [===================>..........] - ETA: 41s - loss: 7.3582 - acc: 0.2383
5/6 [========================>.....] - ETA: 20s - loss: 8.3994 - acc: 0.2188
6/6 [==============================] - 2906s - loss: 9.1553 - acc: 0.2109 - val_loss: 11.6860 - val_acc: 0.2000
[Finished in 2923.7s]

'''




