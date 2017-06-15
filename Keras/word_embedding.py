# coding: utf-8
# https://keras-cn.readthedocs.io/en/latest/blog/word_embedding/




from __future__ import print_function

import os
import sys
import numpy as np 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding 
from keras.models import Model 


BASE_DIR = '../../learngit_data/word_embedding/'
#BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove.6B/'#'/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '20_newsgroup/'#'/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2  



# first, build index mapping words in the embeddings set to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
	path = os.path.join(TEXT_DATA_DIR, name)
	if os.path.isdir(path):
		label_id = len(labels_index)
		labels_index[name] = label_id
		for fname in sorted(os.listdir(path)):
			if fname.isdigit():
				fpath = os.path.join(path, fname)
				if sys.version_info < (3,):
					f = open(fpath)
				else:
					f = open(fpath, encoding='latin-1')
				t = f.read()
				i = t.find('\n\n')  # skip header
				if 0<i:
					t = t[i:]
				texts.append(t)
				f.close()
				labels.append(label_id)

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')



# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word,i in word_index.items():
	if i>MAX_NB_WORDS:
		continue
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.fit(x_train,y_train, batch_size=2048, epochs=1, validation_data=(x_val,y_val))  #batch_size=128, epochs=10





'''
Using TensorFlow backend.
Indexing word vectors.
Found 400000 word vectors.
Processing text dataset
Found 19997 texts.
Found 174105 unique tokens.
Shape of data tensor: (19997, 1000)
Shape of label tensor: (19997, 20)
Preparing embedding matrix.
Training model.
2017-06-15 14:22:34.376018: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-15 14:22:34.381282: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-15 14:22:34.386241: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
Train on 15998 samples, validate on 3999 samples
Epoch 1/1

  128/15998 [..............................] - ETA: 938s - loss: 2.9957 - acc: 0.0312
  256/15998 [..............................] - ETA: 909s - loss: 3.2040 - acc: 0.0430
  384/15998 [..............................] - ETA: 891s - loss: 4.9537 - acc: 0.0417
  512/15998 [..............................] - ETA: 879s - loss: 6.8044 - acc: 0.0410
  640/15998 [>.............................] - ETA: 869s - loss: 8.0543 - acc: 0.0391
  768/15998 [>.............................] - ETA: 860s - loss: 8.9814 - acc: 0.0365
  896/15998 [>.............................] - ETA: 852s - loss: 9.7427 - acc: 0.0391
 1024/15998 [>.............................] - ETA: 842s - loss: 10.3986 - acc: 0.0381
 1152/15998 [=>............................] - ETA: 833s - loss: 10.8626 - acc: 0.0391
 1280/15998 [=>............................] - ETA: 825s - loss: 11.1053 - acc: 0.0406
 1408/15998 [=>............................] - ETA: 817s - loss: 11.3977 - acc: 0.0405
 1536/15998 [=>............................] - ETA: 810s - loss: 11.5630 - acc: 0.0443
 1664/15998 [==>...........................] - ETA: 803s - loss: 11.6880 - acc: 0.0457
 1792/15998 [==>...........................] - ETA: 793s - loss: 11.8254 - acc: 0.0469
 1920/15998 [==>...........................] - ETA: 786s - loss: 11.9858 - acc: 0.0464
 2048/15998 [==>...........................] - ETA: 777s - loss: 12.0981 - acc: 0.0469
 2176/15998 [===>..........................] - ETA: 769s - loss: 12.2255 - acc: 0.0483
 2304/15998 [===>..........................] - ETA: 762s - loss: 12.3364 - acc: 0.0508
 2432/15998 [===>..........................] - ETA: 755s - loss: 12.4349 - acc: 0.0526
 2560/15998 [===>..........................] - ETA: 747s - loss: 12.5415 - acc: 0.0523
 2688/15998 [====>.........................] - ETA: 741s - loss: 12.6240 - acc: 0.0521
 2816/15998 [====>.........................] - ETA: 734s - loss: 12.7315 - acc: 0.0518
 2944/15998 [====>.........................] - ETA: 727s - loss: 12.8360 - acc: 0.0503
 3072/15998 [====>.........................] - ETA: 720s - loss: 12.8857 - acc: 0.0498
 3200/15998 [=====>........................] - ETA: 713s - loss: 12.9424 - acc: 0.0497
 3328/15998 [=====>........................] - ETA: 706s - loss: 13.0049 - acc: 0.0487
 3456/15998 [=====>........................] - ETA: 698s - loss: 13.0215 - acc: 0.0483
 3584/15998 [=====>........................] - ETA: 691s - loss: 13.0891 - acc: 0.0480
 3712/15998 [=====>........................] - ETA: 686s - loss: 13.1330 - acc: 0.0490
 3840/15998 [======>.......................] - ETA: 679s - loss: 13.1821 - acc: 0.0487
 3968/15998 [======>.......................] - ETA: 672s - loss: 13.2596 - acc: 0.0476
 4096/15998 [======>.......................] - ETA: 665s - loss: 13.3113 - acc: 0.0474
 4224/15998 [======>.......................] - ETA: 658s - loss: 13.3693 - acc: 0.0464
 4352/15998 [=======>......................] - ETA: 650s - loss: 13.3765 - acc: 0.0464
 4480/15998 [=======>......................] - ETA: 644s - loss: 13.4133 - acc: 0.0473
 4608/15998 [=======>......................] - ETA: 637s - loss: 13.4664 - acc: 0.0473
 4736/15998 [=======>......................] - ETA: 629s - loss: 13.5120 - acc: 0.0471
 4864/15998 [========>.....................] - ETA: 622s - loss: 13.5630 - acc: 0.0469
 4992/15998 [========>.....................] - ETA: 615s - loss: 13.6006 - acc: 0.0467
 5120/15998 [========>.....................] - ETA: 608s - loss: 13.6405 - acc: 0.0469
 5248/15998 [========>.....................] - ETA: 601s - loss: 13.6793 - acc: 0.0467
 5376/15998 [=========>....................] - ETA: 593s - loss: 13.7067 - acc: 0.0467
 5504/15998 [=========>....................] - ETA: 586s - loss: 13.7421 - acc: 0.0467
 5632/15998 [=========>....................] - ETA: 579s - loss: 13.7734 - acc: 0.0463
 5760/15998 [=========>....................] - ETA: 572s - loss: 13.7957 - acc: 0.0469
 5888/15998 [==========>...................] - ETA: 565s - loss: 13.8287 - acc: 0.0467
 6016/15998 [==========>...................] - ETA: 557s - loss: 13.8638 - acc: 0.0464
 6144/15998 [==========>...................] - ETA: 550s - loss: 13.8783 - acc: 0.0474
 6272/15998 [==========>...................] - ETA: 543s - loss: 13.9009 - acc: 0.0478
 6400/15998 [===========>..................] - ETA: 536s - loss: 13.9181 - acc: 0.0484
 6528/15998 [===========>..................] - ETA: 529s - loss: 13.9439 - acc: 0.0486
 6656/15998 [===========>..................] - ETA: 522s - loss: 13.9714 - acc: 0.0484
 6784/15998 [===========>..................] - ETA: 516s - loss: 14.0021 - acc: 0.0481
 6912/15998 [===========>..................] - ETA: 509s - loss: 14.0235 - acc: 0.0482
 7040/15998 [============>.................] - ETA: 502s - loss: 14.0357 - acc: 0.0486
 7168/15998 [============>.................] - ETA: 494s - loss: 14.0391 - acc: 0.0497
 7296/15998 [============>.................] - ETA: 487s - loss: 14.0547 - acc: 0.0499
 7424/15998 [============>.................] - ETA: 480s - loss: 14.0646 - acc: 0.0505
 7552/15998 [=============>................] - ETA: 473s - loss: 14.0823 - acc: 0.0507
 7680/15998 [=============>................] - ETA: 465s - loss: 14.1015 - acc: 0.0507
 7808/15998 [=============>................] - ETA: 458s - loss: 14.1164 - acc: 0.0508
 7936/15998 [=============>................] - ETA: 451s - loss: 14.1253 - acc: 0.0513
 8064/15998 [==============>...............] - ETA: 444s - loss: 14.1456 - acc: 0.0510
 8192/15998 [==============>...............] - ETA: 437s - loss: 14.1575 - acc: 0.0510
 8320/15998 [==============>...............] - ETA: 429s - loss: 14.1627 - acc: 0.0513
 8448/15998 [==============>...............] - ETA: 422s - loss: 14.1770 - acc: 0.0514
 8576/15998 [===============>..............] - ETA: 415s - loss: 14.1894 - acc: 0.0515
 8704/15998 [===============>..............] - ETA: 408s - loss: 14.1992 - acc: 0.0515
 8832/15998 [===============>..............] - ETA: 401s - loss: 14.2184 - acc: 0.0512
 8960/15998 [===============>..............] - ETA: 393s - loss: 14.2356 - acc: 0.0510
 9088/15998 [================>.............] - ETA: 386s - loss: 14.2518 - acc: 0.0508
 9216/15998 [================>.............] - ETA: 379s - loss: 14.2667 - acc: 0.0508
 9344/15998 [================>.............] - ETA: 372s - loss: 14.2743 - acc: 0.0510
 9472/15998 [================>.............] - ETA: 365s - loss: 14.2897 - acc: 0.0509
 9600/15998 [=================>............] - ETA: 358s - loss: 14.3023 - acc: 0.0507
 9728/15998 [=================>............] - ETA: 350s - loss: 14.3081 - acc: 0.0510
 9856/15998 [=================>............] - ETA: 343s - loss: 14.3137 - acc: 0.0512
 9984/15998 [=================>............] - ETA: 336s - loss: 14.3317 - acc: 0.0509
10112/15998 [=================>............] - ETA: 329s - loss: 14.3323 - acc: 0.0513
10240/15998 [==================>...........] - ETA: 322s - loss: 14.3440 - acc: 0.0513
10368/15998 [==================>...........] - ETA: 314s - loss: 14.3486 - acc: 0.0516
10496/15998 [==================>...........] - ETA: 307s - loss: 14.3574 - acc: 0.0517
10624/15998 [==================>...........] - ETA: 300s - loss: 14.3625 - acc: 0.0519
10752/15998 [===================>..........] - ETA: 293s - loss: 14.3714 - acc: 0.0520
10880/15998 [===================>..........] - ETA: 286s - loss: 14.3831 - acc: 0.0519
11008/15998 [===================>..........] - ETA: 278s - loss: 14.3939 - acc: 0.0519
11136/15998 [===================>..........] - ETA: 271s - loss: 14.3996 - acc: 0.0521
11264/15998 [====================>.........] - ETA: 264s - loss: 14.4119 - acc: 0.0519
11392/15998 [====================>.........] - ETA: 257s - loss: 14.4216 - acc: 0.0518
11520/15998 [====================>.........] - ETA: 250s - loss: 14.4219 - acc: 0.0523
11648/15998 [====================>.........] - ETA: 243s - loss: 14.4263 - acc: 0.0525
11776/15998 [=====================>........] - ETA: 236s - loss: 14.4305 - acc: 0.0526
11904/15998 [=====================>........] - ETA: 228s - loss: 14.4380 - acc: 0.0528
12032/15998 [=====================>........] - ETA: 221s - loss: 14.4491 - acc: 0.0526
12160/15998 [=====================>........] - ETA: 214s - loss: 14.4601 - acc: 0.0525
12288/15998 [======================>.......] - ETA: 207s - loss: 14.4695 - acc: 0.0524
12416/15998 [======================>.......] - ETA: 200s - loss: 14.4776 - acc: 0.0524
12544/15998 [======================>.......] - ETA: 193s - loss: 14.4892 - acc: 0.0521
12672/15998 [======================>.......] - ETA: 185s - loss: 14.4971 - acc: 0.0521
12800/15998 [=======================>......] - ETA: 178s - loss: 14.5119 - acc: 0.0516
12928/15998 [=======================>......] - ETA: 171s - loss: 14.5224 - acc: 0.0514
13056/15998 [=======================>......] - ETA: 164s - loss: 14.5286 - acc: 0.0514
13184/15998 [=======================>......] - ETA: 157s - loss: 14.5367 - acc: 0.0514
13312/15998 [=======================>......] - ETA: 150s - loss: 14.5436 - acc: 0.0513
13440/15998 [========================>.....] - ETA: 142s - loss: 14.5456 - acc: 0.0516
13568/15998 [========================>.....] - ETA: 135s - loss: 14.5557 - acc: 0.0514
13696/15998 [========================>.....] - ETA: 128s - loss: 14.5609 - acc: 0.0515
13824/15998 [========================>.....] - ETA: 121s - loss: 14.5683 - acc: 0.0514
13952/15998 [=========================>....] - ETA: 114s - loss: 14.5733 - acc: 0.0515
14080/15998 [=========================>....] - ETA: 107s - loss: 14.5782 - acc: 0.0516
14208/15998 [=========================>....] - ETA: 99s - loss: 14.5875 - acc: 0.0514 
14336/15998 [=========================>....] - ETA: 92s - loss: 14.5989 - acc: 0.0511
14464/15998 [==========================>...] - ETA: 85s - loss: 14.6077 - acc: 0.0510
14592/15998 [==========================>...] - ETA: 78s - loss: 14.6147 - acc: 0.0508
14720/15998 [==========================>...] - ETA: 71s - loss: 14.6247 - acc: 0.0505
14848/15998 [==========================>...] - ETA: 64s - loss: 14.6260 - acc: 0.0507
14976/15998 [===========================>..] - ETA: 57s - loss: 14.6345 - acc: 0.0505
15104/15998 [===========================>..] - ETA: 49s - loss: 14.6404 - acc: 0.0505
15232/15998 [===========================>..] - ETA: 42s - loss: 14.6434 - acc: 0.0506
15360/15998 [===========================>..] - ETA: 35s - loss: 14.6498 - acc: 0.0505
15488/15998 [============================>.] - ETA: 28s - loss: 14.6545 - acc: 0.0506
15616/15998 [============================>.] - ETA: 21s - loss: 14.6624 - acc: 0.0504
15744/15998 [============================>.] - ETA: 14s - loss: 14.6688 - acc: 0.0503
15872/15998 [============================>.] - ETA: 7s - loss: 14.6775 - acc: 0.0501 
15998/15998 [==============================] - 990s - loss: 14.6848 - acc: 0.0499 - val_loss: 15.2475 - val_acc: 0.0503
[Finished in 1035.8s]
'''



