import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join, isfile
from os import listdir
import theano.tensor as T
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.models import Model
from keras.models import load_model
import read_one_image_and_convert_to_vector as readimg

INPUT_FILE1 = "img_vector.npy"
INPUT_FILE2 = "img_vector2.npy"

SET_PATH1 = "train" # 20k image dataset path
SET_PATH2 = "set_2k" # 2k image dataset path
SET_PATH3 = "set_100" # 100 image dataset path

def read_image_values(fileloc):
    filelist = [join(fileloc,f) for f in listdir(fileloc) if isfile(join(fileloc,f))]
    # create a new vector for reading the set of images
    image_vector = np.empty(shape=(784,0))
    # reading the files present in the training as well as testing data
    for flname in filelist:
        image_vector = np.hstack((image_vector, readimg.read_one_image_and_convert_to_vector(flname)))
    return image_vector.T


## checking to see if the input image vector is stored locally
#print "Checking to see if the 20k input vector"
#if isfile(INPUT_FILE1):
	#img_vector = np.load(INPUT_FILE1)
	#print img_vector.shape
#else:
	## reading the image values for set 1 getting shape=(20k,784)
	#img_vector = read_image_values(SET_PATH1)
	## now saving the file locally for faster load
	#np.save(INPUT_FILE1,img_vector)

#print "Finished loading the 20k input set"

## checking and loading the input image vector set2
#if isfile(INPUT_FILE2):
	#img_vector2 = np.load(INPUT_FILE2)
	#print img_vector2.shape
#else:
	## reading the image values from set2 getting shape = (2k,784)
	#img_vector2 = read_image_values(SET_PATH2)
	## now saving the file locally for faster load
	#np.save(INPUT_FILE2, img_vector2)


choice = int(raw_input("Enter 1 for Task1 \nEnter 2 for Task2 \nEnter 3. for Task3 \nEnter 4. for Task4\nEnter 5. for Task5\n"))

if choice == 1:
	# creating the model for auto-encoding
	# initalizing a hidden layer
	hidden_layer = 100
	input_dimension = 784

	# creating the model
	model = Sequential()
	model.add(Dense(hidden_layer, input_shape=(input_dimension,), activation='relu'))
	model.add(Dense(input_dimension, activation='linear'))

	# setting the hyper parameters
	batch_size = 100
	nb_epoch = 50
	optimizer = 'RMSprop' 
	loss = 'MSE'
	metrics = ['accuracy']

	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
		
	#print "fit has completed for set1 (20k) "
	#print "Now printing the loss"
	#print history_img1.history['loss']
	
	# now evaluating the model
	score_1_lt = []
	score_2_lt = []
	
	for indx in range(nb_epoch):
		# now fitting the model on the image set
		history_img1 = model.fit(img_vector,img_vector, nb_epoch=1,shuffle=True,batch_size=batch_size, verbose=2)
		# now training the data
		score_1 = model.evaluate(img_vector, img_vector, verbose=2)
		score_1_lt.append(score_1[0])
		score_2 = model.evaluate(img_vector2, img_vector2, verbose=2)
		score_2_lt.append(score_2[0])
	# saving the model 
	model.save("m1_100_nn1.h5", overwrite=True)

	set_1, = plt.plot(score_1_lt,label='20k')
	set_2, = plt.plot(score_2_lt,label='2k')
	plt.ylabel("MSE")
	plt.xlabel("Number of Epochs")
	plt.legend(handles=[set_1,set_2])
	plt.show()
	
elif choice == 2:		
	# ------------------------------------------- Task2 ----------------------------------------
	# creating the model for auto-encoding
	# initalizing a hidden layer
	hidden_layer_set = [20,40,60,80,100]
	input_dimension = 784
	# setting the hyper parameters
	batch_size = 120
	nb_epoch = 50
	optimizer = 'RMSprop' 
	loss = 'MSE'
	metrics = ['accuracy']

	score_1_lt = []
	score_2_lt = []

	for hidden_layer in hidden_layer_set:	
		# creating the model
		#print "hidden layer = " + str(hidden_layer)
		model = Sequential()
		model.add(Dense(hidden_layer, input_shape=(input_dimension,), activation='relu'))
		model.add(Dense(input_dimension, activation='linear'))
		model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
		history_bt1 = model.fit(img_vector,img_vector, nb_epoch=nb_epoch,shuffle=True,batch_size=batch_size, verbose=2)
		score_1 = model.evaluate(img_vector, img_vector, verbose=2)
		score_2 = model.evaluate(img_vector2, img_vector2, verbose=2)
		score_1_lt.append(score_1[0])
		score_2_lt.append(score_2[0])


	set_1, = plt.plot(score_1_lt,label='20k')
	set_2, = plt.plot(score_2_lt,label='2k')
	plt.ylabel("MSE")
	plt.xlabel("Number of nodes in hidden layer")
	plt.legend(handles=[set_1,set_2])
	plt.show()
	
elif choice == 3:
	hidden_layer = 100
	input_dimension = 784

	# creating the model
	model1 = Sequential()
	model1.add(Dense(hidden_layer, input_shape=(input_dimension,), activation='relu'))
	model1.add(Dense(input_dimension, activation='linear'))
	
	# setting the hyper parameters
	batch_size = 100
	nb_epoch = 100
	optimizer = 'RMSprop' 
	loss = 'MSE'
	metrics = ['accuracy']

	model1.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	
	# now fitting the model
	history_bt1 = model1.fit(img_vector,img_vector, nb_epoch=nb_epoch,shuffle=True,batch_size=batch_size, verbose=2)
	
	model1.save("m2_100_nn100.h5",overwrite=True)
	
	# getting the weights of the model
	model_weights, model_biases = model1.layers[1].get_weights()
	print model_weights.shape
	
	# Creating a grid
	fig, axes = plt.subplots(10, 10, figsize=(12, 12))
	# now printing all the images present in the weight
	for i in range(100):
		row, column = divmod(i, 10)
		axes[row, column].imshow(model_weights[i, :].reshape(28, 28), cmap=plt.cm.gray)
		axes[row, column].axis('off') # get rid of tick marks/labels
	plt.savefig('Task3_1.png')
	
elif choice == 4:
	# fetching the load model which was trained in Task3
	print "Now loading the previous model from Task3"
	model = load_model('m2_100_nn100.h5')
	# now loading the 100 image dataset
	input_vector = read_image_values(SET_PATH3)
	print "Input for 100 images is read its shape is"
	print input_vector.shape
	# now let us get all the output images
	output_images = np.empty(shape=(0,784))
	for indx in range(input_vector.shape[0]):
		output_image = model.predict(input_vector[indx].reshape(1,784))
		output_images = np.vstack((output_images,output_image))
	print "output images shape is "
	print output_images.shape
	# now generating the subplots
	fig, axes = plt.subplots(10,20, figsize=(12, 12), sharey=True)
	offset = 10 # using this to display output images eg. input is in (0,0) then ouput is in (0,10)
	# now displaying the images in matplotlib.pyplot
	for i in range(100):
		row, column = divmod(i, 10)
		axes[row, column].imshow(input_vector[i,:].reshape(28,28), cmap=plt.cm.gray)
		axes[row, column+offset].imshow(output_images[i,:].reshape(28,28), cmap=plt.cm.gray)
		axes[row, column].axis('off') # get rid of tick marks/labels
		axes[row, column+offset].axis('off') # get rid of tick marks/labels
	plt.savefig('Task4.png')
	
	
	
	
