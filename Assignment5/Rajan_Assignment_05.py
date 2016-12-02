# Rajan, Rohin
# 1001-154-037
# 2016-11-15
# Assignment_05

import numpy as np
import read_one_image_and_convert_to_vector as readimg
import theano
import theano.tensor as T
from os import listdir
from os.path import isfile,join
import matplotlib.pyplot as plt
import display_confusion_matrix as displayConfusionMatrix
import sys
#sys.stdout = open('test1.txt','w')

class clCifarDataSet:
		# read the dataset and generate the train and test values
		def __init__(self):
			mytrain_path = join("cifar_data_1000_100","train")
			mytest_path = join("cifar_data_1000_100","test")
			self.trainInputDataSet, self.trainTargetDataSet = self.read_data(mytrain_path)
			print "done reading the training data"
			self.testInputDataSet,self.testTargetDataSet = self.read_data(mytest_path)
			print "done reading the testing data"
			
		# function that would generate the train data and targets for that input
		def read_data(self, mypath):
			#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
			# get the list of files in the folder
			# small cifar dataset
			files_in_dataset = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
			# taking the first image to get the dimension of the image
			input_file_dimension = self.read_cifar_images(files_in_dataset[0]).shape
			# creating the new empty array 
			inputDataSet = np.empty(shape=(input_file_dimension[0],0))
			# now generating the output values for image
			targetDataSet = np.empty(shape=(10,0))
			target_values = [f.split("_")[0] for f in listdir(mypath) if isfile(join(mypath, f))]
			# iterating through the list of files 
			for file_indx in range(len(files_in_dataset)):
				# appending the values to train data set
				inputDataSet = np.hstack((inputDataSet, self.read_cifar_images(files_in_dataset[file_indx])))
				new_value = np.zeros(shape=(10,1))
				new_value[target_values[file_indx]] = 1
				targetDataSet = np.hstack((targetDataSet, new_value))
			# shuffling both the training as well as test data first transposing but returing the orignal dataset dimensions
			inputDataSet, targetDataSet = self.shuffle_in_unison(inputDataSet.T, targetDataSet.T)
			return inputDataSet, targetDataSet
			
		# function that reads the cifar images	
		def read_cifar_images(self,filename):
			return readimg.read_one_image_and_convert_to_vector(filename)
			
		# method to shuffle both the input samples and target matrix dimensions are input_samples 1000 * 3072
		# target is also in the dimensions 1000 * 10 in this case they should be in the same length
		def shuffle_in_unison(self,input_sample, target):
			shuffled_input = np.empty(input_sample.shape, dtype=input_sample.dtype)
			shuffled_target = np.empty(target.shape, dtype=target.dtype)
			permutation = np.random.permutation(len(input_sample))
			for old_index, new_index in enumerate(permutation):
				shuffled_input[new_index] = input_sample[old_index]
				shuffled_target[new_index] = target[old_index]
				# return the transposed values
			return shuffled_input.T, shuffled_target.T
			

def model(X, w1, w2):
	# layer 1 hidden layer
	net1 = T.dot(w1,X)
	#net1 = T.nnet.sigmoid(net1)
	net1 = T.nnet.relu(net1)
	
	# layer 2 output layer
	net2 = T.dot(w2,net1)
	net2 = T.nnet.softmax(net2.T)
	return net2.T,net1
	
	# function that updates the weights based on the learning rate as well as the gradient calculated
def update_weights(W1,W2, dw1, dw2, learning_rate):
	return_list= []
	return_list.append((W1, W1 - learning_rate*dw1))
	return_list.append((W2, W2 - learning_rate*dw2))
	return return_list
			
if __name__ =="__main__":
	
	# defining the hyper parameters and reading the train and targets matrix for it
	number_of_hidden_layer_nodes = 75
	output_layer_size = 10
	clobj = clCifarDataSet()
	train_data = clobj.trainInputDataSet
	target_output = clobj.trainTargetDataSet
	num_of_inputs = train_data.shape[0]
	alpha = 0.01
	num_of_epochs = 100
	initial_weights = 0.0001
	lambda_value = 0.1
	
	# initalizing the weights for both the layers
	W1 = theano.shared(np.random.uniform(-initial_weights,initial_weights,(number_of_hidden_layer_nodes, num_of_inputs)), name="w1")
	W2 = theano.shared(np.random.uniform(-initial_weights,initial_weights,(output_layer_size, number_of_hidden_layer_nodes)), name="w2")
	
	# declaring the input (X) and output (Y)
	X = T.matrix('X',dtype='float64')
	Y = T.matrix('Y',dtype='float64')
	
	# calculating the output of the neurons
	net_value, layer1_output = model(X,W1,W2)
	# getting the output of the neuron
	y_pred = T.argmax(net_value)
	
	cost = T.mean(T.nnet.categorical_crossentropy(net_value, Y))
	# regularization
	cost += lambda_value * (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2)))
	
	# calculating the gradient value wrt to the weights for the hidden as well as the output layer
	dw1,dw2 = T.grad(cost, wrt=[W1,W2])
	# now updating the shared weights based on the gradient descent
	updates = update_weights(W1,W2,dw1,dw2,alpha)
	
	# creating the train function for the entire network
	train = theano.function(inputs=[X,Y],outputs=[cost,dw1,dw2,net_value], updates=updates)
	# predicting the target value 
	predict = theano.function(inputs=[X], outputs=y_pred)
	
	# now calling the train network
	transpose_train = train_data.T
	transpose_target = target_output.T	
	
	total = len(transpose_train)
	print "total length of train is = " + str(total)
	error_rate_list = []
	cost_list = []
	for iteration in range(num_of_epochs):
		print "iteration = " + str(iteration)
		count = 0
		inner_cost = []
		for indx in range(total):
			train_value = transpose_train[indx].reshape(num_of_inputs,1)
			target_value = transpose_target[indx].reshape(output_layer_size,1)
			c1, c_dw1,c_dw2, netvalue = train(train_value,target_value)
			pred_output = predict(train_value)
			actual_output = np.argmax(target_value)
			if pred_output != actual_output:
				count +=1
			inner_cost.append(c1)
		error_rate = (count/float(total))*100.0
		error_rate_list.append(error_rate)
		cost_list.append(np.mean(inner_cost))
		print "Error Rate = " + str(error_rate)
		print "Cost Mean = " + str(np.mean(inner_cost))
	
	# now using the test data for calculating the error rate and generating the confusion matrix
	# first fetching the test data and targets vector
	test_input_data = clobj.testInputDataSet
	test_target_data = clobj.testTargetDataSet
	
	transposed_test_input_data = test_input_data.T
	transposed_test_target_data = test_target_data.T
	
	# confusion matrix initialization
	confusion_matrix = np.zeros(shape=(output_layer_size,output_layer_size))
	count_test = 0
	total_test = len(transposed_test_input_data)
	print "Now iterating through the test data and getting the predicted values"
	print "total len for test is = " + str(total_test)
	for indx in range(total_test):
		test_input = transposed_test_input_data[indx].reshape(num_of_inputs,1)
		test_target = transposed_test_target_data[indx].reshape(output_layer_size,1)
		pred_output_test = predict(test_input)
		actual_output_test = np.argmax(test_target)
		print "pred output"
		print pred_output_test
		print "actual output"
		print actual_output_test
		if pred_output_test != actual_output_test:
			count_test+=1
		confusion_matrix[actual_output_test,pred_output_test] +=1
	# calculating the error rate for the test data
	error_rate_test = (count_test/float(total_test))*100.0
	print "test count"
	print count_test
	print "The error rate for the test data is"
	print error_rate_test
	
	print "Printing the confusion matrix for the test data"
	print confusion_matrix
	displayConfusionMatrix.display_confusion_matrix(confusion_matrix)
	
	
	# plotting the graph for error rate 
	iteration = range(num_of_epochs)
	
	# plotting the Error rate vs Epochs
	plt.plot(iteration, error_rate_list)
	plt.xlabel('Epochs')
	plt.ylabel('Error Rate')
	plt.grid()
	plt.show()
	
	# plotting the loss vs epochs
	plt.plot(iteration, cost_list)
	plt.xlabel('Epochs')
	plt.ylabel('Loss function')
	plt.grid()
	plt.show()
	
	
	
	
	
	
	
	#--------------------------------------------------------------------------Debugging
	#for indx in range(2):
		#train_value = transpose_train[indx].reshape(num_of_inputs,1)
		#target_value = transpose_target[indx].reshape(10,1)
		#a1,b1,c1,d1,d2 = train(train_value,target_value)
		#print "cost"
		#print a1
		#print "dw1"
		#print b1.shape
		#print "dw2"
		#print c1.shape
		#print "output of layer2"
		#print d1
		#print "pre output of layer2"
		#print d2
		#print "actual output"
		#print target_value
		#print "argmax of layer2"
		#print predict(train_value)
		#print "argmax of actual output"
		#print np.argmax(target_value)
	
	#print "target matrix"
	#print target_output
	
	#train_value = readimg.read_one_image_and_convert_to_vector("cifar_data_100_10/train/2_72.png")
	#train_value = np.hstack((train_value, readimg.read_one_image_and_convert_to_vector("cifar_data_100_10/train/1_72.png")))
	#test_value = np.zeros(shape=(10,1))
	#test_value[2] = 1
	#test_1 = np.zeros(shape=(10,1))
	#test_1[1] = 1
	#test_value = np.hstack((test_value, test_1))
	#for indx in range(2):
		#a1,b1,c1,d1,d2,w1,w2 = train(train_value.T[indx].reshape(3072,1),test_value.T[indx].reshape(10,1))
		#print "cost"
		#print a1
		#print "dw1"
		#print b1
		#print "dw2"
		#print c1
		#print "output of layer2"
		#print d1
		##print "softmax value of layer2"
		##print softmax(np.dot(w2,d2))
		#print "output of layer1"
		#print d2
		#print "W1"
		#print w1
		#print "W2"
		#print w2
		#print "input"
		#print train_value.T[indx].reshape(3072,1)
		#print "target"
		#print test_value.T[indx].reshape(10,1)
