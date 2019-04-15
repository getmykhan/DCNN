## Importing all the Dependencies
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

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

# Initialising the CNN
cnn = Sequential()

# Step 1 - Convolution
# Step 2 - Pooling
cnn.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
cnn.add(Flatten())

# Step 4 - Full connection
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = cnn.fit_generator(train,
                         steps_per_epoch = 250,
                         epochs = 25,
                         validation_data = test,
                         validation_steps = 2000)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test_image = image.load_img('\\dataset\\single_prediction\\9. what-does-it-mean-when-cat-wags-tail.jpg', target_size=(128,128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
print(result)

if result[0][0] == 1:
	print('dog')
else:
	print('cat')