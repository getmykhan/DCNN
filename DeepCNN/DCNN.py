## Importing all the Dependencies

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

## Image Preprocessing
## Introduce some Noise/Image Augmentation 

train_data_preprocess = ImageDataGenerator(
	rescale = 1./255,
	shear_range = 0.2,
	zoom_range = 0.2,
	horizontal_flip = True)

test_data_preprocess = (1./255)

train = train_data_preprocess.flow_from_directory(
	'dataset/training',
	target_size = (128,128),
	batch_size = 32,
	class_mode = 'binary')

test = train_data_preprocess.flow_from_directory(
	'dataset/test',
	target_size = (128,128),
	batch_size = 32,
	class_mode = 'binary')

## Initialize the Convolutional Neural Net

cnn = Sequential()

cnn.add(Convolution2D(32, 3, 3, input_shape = (128,128,3), activation = 'relu'))
cnn.add(MaxPooling2D())
cnn.add(Flatten())

## Deep Fully connected
cnn.add(Dense(output_dim = 128, activation = 'relu'))
cnn.add(Dense(output_dim = 128, activation = 'relu'))
cnn.add(Dense(output_dim = 1, activation='sigmoid'))

cnn.compile(optimizer = 'adam', loss='binary_crossentropy', metrics =['accuracy'])

cnn.fit_generator(
	train,
	samples_per_epoch = 8000,
	nb_epoch = 50,
	validation_data = test,
	nb_val_samples = 2000)