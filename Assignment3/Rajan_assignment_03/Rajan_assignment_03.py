# Rajan, Rohin
# 1001-154-037
# 2016-09-29
# Assignment_03

import numpy as np
import Tkinter as Tk
import matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import read_one_image_and_convert_to_vector as readImage
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import colorsys
from os import listdir
from os.path import isfile , join

class ClDataSet:
    # This class encapsulates the data set
    # The data set includes input samples and targets
    def __init__(self, samples=[[0., 0., 1., 1.], [0., 1., 0., 1.]], targets=[[0., 1., 1., 0.]]):
        # Note: input samples are assumed to be in column order.
        # This means that each column of the samples matrix is representing
        # a sample point
        # The default values for samples and targets represent an exclusive or
        # Farhad Kamangar 2016_09_05
        self.samples = np.array(samples)

        if targets != None:
            self.targets = np.array(targets)
        else:
            self.targets = None


class ClMnistDataSet:
    # This class is used to read the mnist dataset and create the input vector
    # as well as the target vector
    def __init__(self):
        # reading the images from the mnist dataset
        mnist_db_location = "mnist_images"
        mnist_db_files = [ join(mnist_db_location,f) for f in listdir(mnist_db_location) if (isfile(join(mnist_db_location,f)))]
        # extracting the target values for each of the input values
        mnist_targets = [ f.split("_")[0] for f in listdir(mnist_db_location)]
        # reading the input images and generating an input matrix
        sample_vector_dimentions = readImage.read_one_image_and_convert_to_vector(mnist_db_files[0]).shape[0]
        self.samples = np.empty(shape=[sample_vector_dimentions,0]) # getting the dimensions from the first image
        for path in mnist_db_files:
            image_vector = readImage.read_one_image_and_convert_to_vector(path)
            self.samples = np.hstack([self.samples,image_vector])
        self.targets = self.generate_target_vector(mnist_targets)

        # now shuffling the data set
        self.samples, self.targets = self.shuffle_in_unison(self.samples.T,self.targets.T)

    # method to generate the target matrix
    def generate_target_vector(self, target_values):
        target_matrix = np.zeros((10,len(target_values)))
        target_matrix  = target_matrix.T
        for indx,val in enumerate(target_values):
            target_matrix[indx][int(val)] = 1
        return target_matrix.T

    # method to shuffle both the input samples and target matrix dimensions are input_samples 1000 * 784
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





nn_experiment_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs": 784,  # number of inputs to the network
    "learning_rate": 0.001,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 10, "activation_function": "linear"}],  # list of dictionaries
    "data_set": ClMnistDataSet(),
    'number_of_classes': 10,
    'number_of_samples_in_each_class': 3
}


class ClNNExperiment:
    """
    This class presents an experimental setup for a single layer Perceptron
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, settings={}):
        self.__dict__.update(nn_experiment_default_settings)
        self.__dict__.update(settings)
        # Set up the neural network
        settings = {"min_initial_weights": self.min_initial_weights,  # minimum initial weight
                    "max_initial_weights": self.max_initial_weights,  # maximum initial weight
                    "number_of_inputs": self.number_of_inputs,  # number of inputs to the network
                    "learning_rate": self.learning_rate,  # learning rate
                    "layers_specification": self.layers_specification
                    }
        self.neural_network = ClNeuralNetwork(self, settings)
        # Make sure that the number of neurons in the last layer is equal to number of classes
        self.neural_network.layers[-1].number_of_neurons = self.number_of_classes

    def run_forward_pass(self, display_input=True, display_output=True,
                         display_targets=True, display_target_vectors=True,
                         display_error=True):
        self.neural_network.calculate_output(self.data_set.samples)

        if display_input:
            print "Input : ", self.data_set.samples
        if display_output:
            print 'Output : ', self.neural_network.output
        if display_targets:
            print "Target (class ID) : ", self.target
        if display_target_vectors:
            print "Target Vectors : ", self.desired_target_vectors
        if self.desired_target_vectors.shape == self.neural_network.output.shape:
            self.error = self.desired_target_vectors - self.neural_network.output
            if display_error:
                print 'Error : ', self.error
        else:
            print "Size of the output is not the same as the size of the target.", \
                "Error cannot be calculated."


    def adjust_weights(self):
        self.neural_network.adjust_weights(self.data_set.samples,
                                           self.data_set.targets, self.neural_network.output)

    def calculate_error_rate(self):
        return self.neural_network.calculate_error_rate(self.neural_network.calculate_output(self.data_set.samples).T,
                                                        self.data_set.targets.T)

class ClNNGui2d:
    """
    This class presents an experiment to demonstrate
    Perceptron learning in 2d space.
    Farhad Kamangar 2016_09_02
    """

    def __init__(self, master, nn_experiment):
        self.master = master
        #
        self.nn_experiment = nn_experiment
        self.number_of_classes = self.nn_experiment.number_of_classes
        self.xmin = 1
        self.xmax = 1000
        self.ymin = 0
        self.ymax = 100
        self.master.update()
        self.number_of_samples_in_each_class = self.nn_experiment.number_of_samples_in_each_class
        self.number_of_epooch = 10
        self.learning_rate = self.nn_experiment.learning_rate
        self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class
        self.step_size = 0.02
        self.current_sample_loss = 0
        self.sample_points = []
        self.target = []
        self.sample_colors = []
        self.weights = np.array([])
        self.class_ids = np.array([])
        self.output = np.array([])
        self.desired_target_vectors = np.array([])
        self.xx = np.array([])
        self.yy = np.array([])
        self.loss_type = ""
        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")
        self.error_rate_counter = 0
        self.error_list = []
        self.index_list = []
        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = self.figure.add_subplot(111)
        plt.title("Hebb Rule Error Rate")
        plt.scatter(0, 0)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # Create sliders frame
        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='s1')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='s1')
        # Create buttons frame
        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='b1')
        # Set up the sliders
        ivar = Tk.IntVar()
        self.learning_rate_slider_label = Tk.Label(self.sliders_frame, text="Learning Rate")
        self.learning_rate_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_rate_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.0001, to_=1, resolution=0.0001, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.number_of_classes_slider_label = Tk.Label(self.sliders_frame, text="Number of Classes")
        self.number_of_classes_slider_label.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.number_of_classes_slider = Tk.Scale(self.sliders_frame, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                                 from_=2, to_=15, bg="#DDDDDD",
                                                 activebackground="#FF0000",
                                                 highlightcolor="#00FFFF", width=10)
        self.number_of_classes_slider.set(self.number_of_classes)
        self.number_of_classes_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_classes_slider_callback())
        self.number_of_classes_slider.grid(row=1, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.number_of_epooch_slider_label = Tk.Label(self.sliders_frame, text="Number of Epooch")
        self.number_of_epooch_slider_label.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.number_of_epooch_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1, to_=1000, resolution=10, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.number_of_epooch_slider_callback())
        self.number_of_epooch_slider.set(self.number_of_epooch)
        self.number_of_epooch_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_epooch_slider_callback())
        self.number_of_epooch_slider.grid(row=2, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.reset_button = Tk.Button(self.buttons_frame, text="Reset", bg="yellow", fg="red",
                                      command=lambda: self.reset_button_callback())
        self.reset_button.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.randomize_weights_button = Tk.Button(self.buttons_frame,
                                                  text="Randomize Weights",
                                                  bg="yellow", fg="red",
                                                  command=lambda: self.randomize_weights_button_callback())
        self.randomize_weights_button.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Learn)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.print_nn_parameters_button = Tk.Button(self.buttons_frame,
                                                    text="Print NN Parameters",
                                                    bg="yellow", fg="red",
                                                    command=lambda: self.print_nn_parameters_button_callback())
        self.print_nn_parameters_button.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # creating a new property to store the selected hebb learning rule
        self.hebb_learning_variable = Tk.StringVar()
        # creating or setting up the dropdown list for hebb learning rule
        self.hebb_learning_rules_dropdown = Tk.OptionMenu(self.buttons_frame,self.hebb_learning_variable,
                "Filtered Learning","Delta Rule","Unsupervised Hebb",
                command=lambda event: self.hebb_learning_rules_dropdown_callback())
        self.hebb_learning_variable.set("Filtered Learning")
        self.hebb_learning_rules_dropdown.grid(row=4, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.initialize()
        self.refresh_display()

    def initialize(self):
        self.nn_experiment.neural_network.randomize_weights()

    def display_samples_on_image(self):
        # Display the samples for each class
        for class_index in range(0, self.number_of_classes):
            self.axes.scatter(self.nn_experiment.data_set.samples[0, class_index * self.number_of_samples_in_each_class: \
                (class_index + 1) * self.number_of_samples_in_each_class],
                              self.nn_experiment.data_set.samples[1, class_index * self.number_of_samples_in_each_class: \
                                  (class_index + 1) * self.number_of_samples_in_each_class],
                              c=self.sample_points_colors(class_index * (1.0 / self.number_of_classes)),
                              marker=(3 + class_index, 1, 0), s=50)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas.draw()

    def refresh_display(self):
        self.nn_experiment.neural_network.calculate_output(self.nn_experiment.data_set.samples)
        self.canvas.draw()

    def reset_button_callback(self):
        self.error_list = []
        self.index_list = []
        self.error_rate_counter = 0
        self.axes.cla()
        plt.title("Hebb Rule Error Rate")
        plt.scatter(0, 0)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas.draw()

    def learning_rate_slider_callback(self):
        self.nn_experiment.learning_rate = self.learning_rate
        self.nn_experiment.neural_network.learning_rate = self.learning_rate
        self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class
        self.refresh_display()

    def number_of_classes_slider_callback(self):
        self.number_of_classes_slider.set(10)
        self.number_of_classes = self.number_of_classes_slider.get()

    def number_of_epooch_slider_callback(self):
        self.number_of_epooch = self.number_of_epooch_slider.get()

    def adjust_weights_button_callback(self):
        temp_text = self.adjust_weights_button.config('text')[-1]
        self.adjust_weights_button.config(text='Please Wait')
        for k in range(self.number_of_epooch):
            self.nn_experiment.adjust_weights()
            error_rate = self.nn_experiment.calculate_error_rate()
            self.index_list.append(self.error_rate_counter+k)
            self.error_list.append(error_rate)
            self.axes.plot(self.index_list,self.error_list)
            self.canvas.draw()
        self.error_rate_counter +=self.number_of_epooch
        self.adjust_weights_button.config(text=temp_text)
        self.adjust_weights_button.update_idletasks()

    def randomize_weights_button_callback(self):
        temp_text = self.randomize_weights_button.config('text')[-1]
        self.randomize_weights_button.config(text='Please Wait')
        self.randomize_weights_button.update_idletasks()
        self.nn_experiment.neural_network.randomize_weights()
        self.refresh_display()
        self.randomize_weights_button.config(text=temp_text)
        self.randomize_weights_button.update_idletasks()

    def print_nn_parameters_button_callback(self):
        temp_text = self.print_nn_parameters_button.config('text')[-1]
        self.print_nn_parameters_button.config(text='Please Wait')
        self.print_nn_parameters_button.update_idletasks()
        self.nn_experiment.neural_network.display_network_parameters()
        self.refresh_display()
        self.print_nn_parameters_button.config(text=temp_text)
        self.print_nn_parameters_button.update_idletasks()

    def hebb_learning_rules_dropdown_callback(self):
        # setting the hebb learning rule
        self.nn_experiment.neural_network.hebb_learning_variable = self.hebb_learning_variable.get()


neural_network_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs": 784,  # number of inputs to the network
    "learning_rate": 0.001,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire training set as a batch
    "layers_specification": [{"number_of_neurons": 10,
                              "activation_function": "linear"}]  # list of dictionaries
}


class ClNeuralNetwork:
    """
    This class presents a multi layer neural network
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, experiment, settings={}):
        self.__dict__.update(neural_network_default_settings)
        self.__dict__.update(settings)
        # create nn
        self.experiment = experiment
        self.layers = []
        #  default value is filtered learning it would be set based on the input given by the user
        self.hebb_learning_variable = "Filtered Learning"
        for layer_index, layer in enumerate(self.layers_specification):
            if layer_index == 0:
                layer['number_of_inputs_to_layer'] = self.number_of_inputs
            else:
                layer['number_of_inputs_to_layer'] = self.layers[layer_index - 1].number_of_neurons
            self.layers.append(ClSingleLayer(layer))

    def randomize_weights(self, min=-0.1, max=0.1):
        # randomize weights for all the connections in the network
        for layer in self.layers:
            layer.randomize_weights(self.min_initial_weights, self.max_initial_weights)

    def display_network_parameters(self, display_layers=True, display_weights=True):
        for layer_index, layer in enumerate(self.layers):
            print "\n--------------------------------------------", \
                "\nLayer #: ", layer_index, \
                "\nNumber of Nodes : ", layer.number_of_neurons, \
                "\nNumber of inputs : ", self.layers[layer_index].number_of_inputs_to_layer, \
                "\nActivation Function : ", layer.activation_function, \
                "\nWeights : ", layer.weights

    # method to change the actual output / normalize the actual data
    # In this only the max value index and set only that value 1 and remaining 0
    def adjust_output(self, output_transpose):
        adjust_shape_value = output_transpose.shape[1]
        new_output = np.zeros(shape=[adjust_shape_value,0])
        for indx in range(len(output_transpose)):
            current_output = np.zeros(shape=[output_transpose.shape[1],1])
            maxval = np.argmax(output_transpose[indx].reshape(output_transpose.shape[1],1))
            # checking if the elements in the output are all equal or not
            if not self.check_if_vector_equality(output_transpose[indx].reshape(output_transpose.shape[1],1)):
                current_output[maxval] = 1
                # checking to see if output elements are all greater than 0 then setting the values to 1 else the
                # current_output would be a zero matrix that would be appened
            elif output_transpose[indx][maxval] > 0:
                current_output = np.ones(shape=[output_transpose.shape[1],1])
            new_output = np.hstack([new_output,current_output])
        return new_output

    def calculate_output(self, input_values):
        # Calculate the output of the network, given the input signals
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                output = layer.calculate_output(input_values)
            else:
                output = layer.calculate_output(output)
        self.output = self.adjust_output(output.T)
        return self.output

    # function to check if the vector are equal or not
    def check_if_vector_equality(self, vector_matrix):
        # get the min value and the max value if both are the same then the elements of the vector are the same
        minvalue = np.argmin(vector_matrix)
        maxvalue = np.argmax(vector_matrix)
        return minvalue == maxvalue

    # calculating the error rate for the given input_samples transpose as well as the actual result transpose
    def calculate_error_rate(self, actual_result_transpose, target_matrix_transpose):
        error = 0
        self.error_rate = []
        for indx in range(len(actual_result_transpose)):
            # getting the index of the max value which the actual result is present
            actual_max_indx = np.argmax(actual_result_transpose[indx])
            # getting the findx of the max value which the target result is present
            target_max_indx = np.argmax(target_matrix_transpose[indx])
            # check to see if the elements in the matrix are equal or not
            if not self.check_if_vector_equality(actual_result_transpose[indx]):
                # checking to see if the indexes match if not then get the error
                if actual_max_indx != target_max_indx:
                    error += 1
            else:
                # if all the elements are equal then its an error
                error += 1
        # calculating error rate
        error_float = error / float(len(actual_result_transpose))
        error_float *= 100
        return error_float

    def adjust_weights(self, input_samples, targets, actual_output):
        # Appending 1 to the input_samples in order to accommodate the bias
        if len(input_samples.shape) == 1:
            tmp_samples = np.append(input_samples, 1)
        else:
            tmp_samples = np.vstack([input_samples, np.ones((1, input_samples.shape[1]), float)])

        print "actual output"
        print actual_output

        print "weights"
        print self.layers[0].weights

        # checking to see if any of the hebb rule is selected
        if self.hebb_learning_variable == "Filtered Learning" or self.hebb_learning_variable == "Delta Rule" \
                or self.hebb_learning_variable == "Unsupervised Hebb":
            # calculating the error
            error = targets - actual_output
            # fetching the total length of the dataset
            for indx in range(tmp_samples.shape[1]):
                ind_target = targets.T[indx].reshape(10,1)
                # transpose of the input sample in [ 1 * 785]
                ind_input_sample = tmp_samples.T[indx].reshape(1,785)
                # getting individual actual output
                ind_actual_output = actual_output.T[indx].reshape(10,1)
                # fetching individual actual outputs and re-adjusting the dataset
                if self.hebb_learning_variable == "Filtered Learning":
                    gamma_value = 1 - self.learning_rate
                    for layer in self.layers:
                        correction = self.learning_rate * np.dot(ind_target,ind_input_sample)
                        layer.weights = (gamma_value * layer.weights) + correction
                elif self.hebb_learning_variable == "Delta Rule":
                    ind_error = error.T[indx].reshape(10,1)
                    for layer in self.layers:
                        layer.weights = layer.weights + self.learning_rate * np.dot(ind_error,ind_input_sample)
                # in this the else would be unsupervised Hebb
                else:
                    for layer in self.layers:
                        layer.weights = layer.weights + self.learning_rate * np.dot(ind_actual_output, ind_input_sample)

single_layer_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs_to_layer": 784,  # number of input signals
    "number_of_neurons": 10,  # number of neurons in the layer
    "activation_function": "linear"  # default activation function
}


class ClSingleLayer:
    """
    This class presents a single layer of neurons
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, settings):
        self.__dict__.update(single_layer_default_settings)
        self.__dict__.update(settings)
        self.randomize_weights()

    def randomize_weights(self, min_initial_weights=None, max_initial_weights=None):
        if min_initial_weights == None:
            min_initial_weights = self.min_initial_weights
        if max_initial_weights == None:
            max_initial_weights = self.max_initial_weights
        # self.weights = np.random.uniform(min_initial_weights, max_initial_weights,
        #                                  (self.number_of_neurons, self.number_of_inputs_to_layer + 1))
        self.weights = np.ones([self.number_of_neurons, self.number_of_inputs_to_layer + 1],dtype=float)
        # self.weights = np.zeros([self.number_of_neurons, self.number_of_inputs_to_layer+1],dtype=float)

    def calculate_output(self, input_values):
        # Calculate the output of the layer, given the input signals
        # NOTE: Input is assumed to be a column vector. If the input
        # is given as a matrix, then each column of the input matrix is assumed to be a sample
        # Farhad Kamangar Sept. 4, 2016
        if len(input_values.shape) == 1:
            net = self.weights.dot(np.append(input_values, 1))
        else:
            net = self.weights.dot(np.vstack([input_values, np.ones((1, input_values.shape[1]), float)]))
        if self.activation_function == 'linear':
            self.output = net
        if self.activation_function == 'sigmoid':
            self.output = sigmoid(net)
        if self.activation_function == 'hardlimit':
            np.putmask(net, net > 0, 1)
            np.putmask(net, net <= 0, 0)
            self.output = net
        return self.output


if __name__ == "__main__":
    nn_experiment_settings = {
        "min_initial_weights": -0.1,  # minimum initial weight
        "max_initial_weights": 0.1,  # maximum initial weight
        "number_of_inputs": 784,  # number of inputs to the network
        "learning_rate": 0.001,  # learning rate
        "layers_specification": [{"number_of_neurons": 10, "activation_function": "linear"}],  # list of dictionaries
        "data_set": ClMnistDataSet(),
        'number_of_classes': 10,
        'number_of_samples_in_each_class': 3
    }
    np.random.seed(1)
    ob_nn_experiment = ClNNExperiment(nn_experiment_settings)
    main_frame = Tk.Tk()
    main_frame.title("Hebb Learning Rule")
    main_frame.geometry('640x480')
    ob_nn_gui_2d = ClNNGui2d(main_frame, ob_nn_experiment)
    main_frame.mainloop()
