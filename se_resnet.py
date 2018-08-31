# Author: Md. Ibrahim Khan

from keras import optimizers, losses
from keras.layers import *
from keras.models import Model
from keras.backend import int_shape
from keras.utils import to_categorical, plot_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def se_block(block_input, num_filters, ratio=8):                             # Squeeze and excitation block

	'''
		Args:
			block_input: input tensor to the squeeze and excitation block
			num_filters: no. of filters/channels in block_input
			ratio: a hyperparameter that denotes the ratio by which no. of channels will be reduced
			
		Returns:
			scale: scaled tensor after getting multiplied by new channel weights
	'''

	pool1 = GlobalAveragePooling2D()(block_input)
	flat = Reshape((1, 1, num_filters))(pool1)
	dense1 = Dense(num_filters//ratio, activation='relu')(flat)
	dense2 = Dense(num_filters, activation='sigmoid')(dense1)
	scale = multiply([block_input, dense2])
	
	return scale

def resnet_block(block_input, num_filters):                                  # Single ResNet block

	'''
		Args:
			block_input: input tensor to the ResNet block
			num_filters: no. of filters/channels in block_input
			
		Returns:
			relu2: activated tensor after addition with original input
	'''

	if int_shape(block_input)[3] != num_filters:
		block_input = Conv2D(num_filters, kernel_size=(1, 1))(block_input)
	
	conv1 = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(block_input)
	norm1 = BatchNormalization()(conv1)
	relu1 = Activation('relu')(norm1)
	conv2 = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(relu1)
	norm2 = BatchNormalization()(conv2)
	
	se = se_block(norm2, num_filters=num_filters)
	
	sum = Add()([block_input, se])
	relu2 = Activation('relu')(sum)
	
	return relu2

def se_resnet14():
	
	''' 
		Squeeze and excitation blocks applied on an 14-layer adapted version of ResNet18.
		Adapted for MNIST dataset.
		Input size is 28x28x1 representing images in MNIST.
		Output size is 10 representing classes to which images belong.
	'''

	input = Input(shape=(28, 28, 1))
	conv1 = Conv2D(64, kernel_size=(7, 7), activation='relu', padding='same', kernel_initializer='he_normal')(input)
	pool1 = MaxPooling2D((2, 2), strides=2)(conv1)
	
	block1 = resnet_block(pool1, 64)
	block2 = resnet_block(block1, 64)
	
	pool2 = MaxPooling2D((2, 2), strides=2)(block2)
	
	block3 = resnet_block(pool2, 128)
	block4 = resnet_block(block3, 128)
	
	pool3 = MaxPooling2D((3, 3), strides=2)(block4)
	
	block5 = resnet_block(pool3, 256)
	block6 = resnet_block(block5, 256)
	
	pool4 = MaxPooling2D((3, 3), strides=2)(block6)
	flat = Flatten()(pool4)
	
	output = Dense(10, activation='softmax')(flat)
	
	model = Model(inputs=input, outputs=output)
	return model
	

if __name__=='__main__':

	model = se_resnet14()
	print(model.summary())
	
	# Training configuration
	model.compile(loss=losses.categorical_crossentropy,
	              optimizer=optimizers.Adam(),
	              metrics=['accuracy'])
	
	# Data preparation
	train = pd.read_csv('mnist_train.csv')
	test = pd.read_csv('mnist_test.csv')
	
	input_shape = (28, 28, 1)
	
	X_train = np.array(train.iloc[:, 1:])
	y_train = to_categorical(np.array(train.iloc[:, 0]))
	
	X_test = np.array(test.iloc[:, 1:])
	y_test = to_categorical(np.array(test.iloc[:, 0]))
	
	X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
	X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
	
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	
	X_train /= 255
	X_test /= 255
	
	# Training
	train_history = model.fit(X_train, y_train,
                              batch_size=128,
                              epochs=20,
                              verbose=1)
	
	# Evaluation
	result = model.evaluate(X_test, y_test, verbose=0)
	
	print("Test Loss", result[0])
	print("Test Accuracy", result[1])
	
	# Plotting loss and accuracy metrics
	accuracy = train_history.history['acc']
	loss = train_history.history['loss']
	epochs = range(len(accuracy))
	
	plt.plot(epochs, accuracy, 'b', label='Training accuracy')
	plt.title('Training accuracy')
	plt.savefig('train_acc.jpg')
	plt.figure()
	
	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.title('Training loss')
	plt.savefig('train_loss.jpg')
	plt.show()