# Rajan, Rohin
# 1001-154-037
# 2016-10-16
# Assignment_04

import numpy as np
import Tkinter as Tk
import matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import read_csv_data_and_convert_to_vector as readcsv
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import colorsys
from os import listdir
from os.path import isfile , join

class ClStockMarketDataSet:
    def __init__(self):
        # read the stock data set and fetch the information about the data
        stock_db_location = "stock_data.csv"
        stock_data_set = readcsv.read_csv_as_matrix(stock_db_location)
        stock_data_set = readcsv.normalize_dataset(stock_data_set)
        stock_db_matrix_dimentions = stock_data_set.shape[1]
        self.samples = np.empty(shape=[stock_db_matrix_dimentions,0])
        for data_value in stock_data_set:
            self.samples = np.hstack([self.samples,data_value.reshape(stock_db_matrix_dimentions,1)])
        self.selected_samples = self.samples

nn_experiment_default_settings = {
    # Optional settings
    "number_of_inputs": 2,  # number of inputs to the network
    "learning_rate": 0.01,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire training set as a batch
    "layers_specification": [{"number_of_neurons": 2, "activation_function": "linear"}],  # list of dictionaries
    "data_set": ClStockMarketDataSet(),
    'number_of_classes': 2,
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
        settings = {"number_of_inputs": self.number_of_inputs,  # number of inputs to the network
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


    def adjust_weights(self,learning_rate):
        self.neural_network.adjust_weights(self.data_set.input,
                                           self.data_set.target, self.neural_network.output,learning_rate)

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
        self.xmax = 10
        self.ymin = 0
        self.ymax = 10
        self.master.update()
        self.number_of_samples_in_each_class = self.nn_experiment.number_of_samples_in_each_class
        self.batch_size = 1
        self.sample_size_percentage = 100
        self.number_of_iterations = 1
        self.number_of_delayed_elements = 1
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
        self.error_price_list = []
        self.error_volume_list = []
        self.error_max_price_list = []
        self.error_max_volume_list = []
        self.index_list = []
        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = self.figure.add_subplot(111)
        plt.title("LMS Stock Prediction ")
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
                                             from_=0.01, to_=1, resolution=0.01, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.number_of_delayed_elements_slider_label = Tk.Label(self.sliders_frame, text="Number of delayed elements")
        self.number_of_delayed_elements_slider_label.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.number_of_delayed_elements_slider = Tk.Scale(self.sliders_frame, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                                 from_=2, to_=15, bg="#DDDDDD",
                                                 activebackground="#FF0000",
                                                 highlightcolor="#00FFFF", width=10)
        self.number_of_delayed_elements_slider.set(self.number_of_delayed_elements)
        self.number_of_delayed_elements_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_delayed_elements_slider_callback())
        self.number_of_delayed_elements_slider.grid(row=1, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.sample_size_percentage_slider_label = Tk.Label(self.sliders_frame, text="Sample size(%)")
        self.sample_size_percentage_slider_label.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.sample_size_percentage_slider = Tk.Scale(self.sliders_frame, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                                 from_=0, to_=100, bg="#DDDDDD",
                                                 activebackground="#FF0000",
                                                 highlightcolor="#00FFFF", width=10)
        self.sample_size_percentage_slider.set(self.sample_size_percentage)
        self.sample_size_percentage_slider.bind("<ButtonRelease-1>", lambda event: self.sample_size_percentage_slider_callback())
        self.sample_size_percentage_slider.grid(row=2, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.batch_size_slider_label = Tk.Label(self.sliders_frame, text="Batch size")
        self.batch_size_slider_label.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.batch_size_slider = Tk.Scale(self.sliders_frame, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                                 from_=10, to_=1000, bg="#DDDDDD",
                                                 activebackground="#FF0000",
                                                 highlightcolor="#00FFFF", width=10)
        self.batch_size_slider.bind("<ButtonRelease-1>", lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=3, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.number_of_iterations_slider_label = Tk.Label(self.sliders_frame, text="Number of iterations")
        self.number_of_iterations_slider_label.grid(row=4, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.number_of_iterations_slider= Tk.Scale(self.sliders_frame, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                                 from_=1, to_=10, bg="#DDDDDD",
                                                 activebackground="#FF0000",
                                                 highlightcolor="#00FFFF", width=10)
        self.number_of_iterations_slider.set(self.number_of_iterations)
        self.number_of_iterations_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_iterations_slider_callback())
        self.number_of_iterations_slider.grid(row=4, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.reset_button = Tk.Button(self.buttons_frame, text="Reset", bg="yellow", fg="red",
                                      command=lambda: self.reset_button_callback())
        self.reset_button.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.reset_weights_button = Tk.Button(self.buttons_frame,
                                                  text="Set Weights to Zero",
                                                  bg="yellow", fg="red",
                                                  command=lambda: self.reset_weights_button_callback())
        self.reset_weights_button.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
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
        self.initialize()
        self.refresh_display()

    def initialize(self):
        self.nn_experiment.neural_network.reset_weights()

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
        # self.nn_experiment.neural_network.calculate_output(self.nn_experiment.data_set.samples)
        self.canvas.draw()

    #  function to fetch the batch size based on the slider selected by the user
    def batch_size_slider_callback(self):
        self.batch_size = self.batch_size_slider.get()
        # creating a list of batches of the input samples
        if self.batch_size == 0:
            number_of_batches = 1
        else:
            number_of_batches = self.nn_experiment.data_set.selected_samples.shape[1] / self.batch_size
        # creating a batch size input and placing the input samples in them
        self.nn_experiment.data_set.batch_inputs = []
        starting_indx = 0
        ending_indx = starting_indx+ self.batch_size
        for indx in range(number_of_batches):
            new_batch = self.nn_experiment.data_set.selected_samples.T[starting_indx:ending_indx+1]
            starting_indx = ending_indx+1
            ending_indx = ending_indx+self.batch_size
            self.nn_experiment.data_set.batch_inputs.append(new_batch.T)

    # function to fetch the number of iterations based on the slider value selected by the user
    def number_of_iterations_slider_callback(self):
        self.number_of_iterations = self.number_of_iterations_slider.get()

    def reset_button_callback(self):
        self.error_price_list= []
        self.error_volume_list = []
        self.index_list = []
        # self.axes.cla()
        plt.title("LMS Error Rate")
        plt.scatter(0, 0)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas.draw()

    def learning_rate_slider_callback(self):
        self.nn_experiment.learning_rate = self.learning_rate
        self.nn_experiment.neural_network.learning_rate = self.learning_rate
        self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class
        self.refresh_display()

    # function to set the delayed elements as well as create the input dimensions needed
    def number_of_delayed_elements_slider_callback(self):
        self.number_of_delayed_elements = self.number_of_delayed_elements_slider.get()
        # calculating the input dimension
        self.input_dimension = self.number_of_delayed_elements * 2 + 2
        # generating the weight matrix based on the input dimension
        self.nn_experiment.neural_network.randomize_weights(self.input_dimension)

    # function that calculates the percentage of the sample size
    def sample_size_percentage_slider_callback(self):
        # converting the value to float data type
        sample_size_percentage = self.sample_size_percentage_slider.get() / float(100)
        # calculating the new sample size of the samples needed
        new_sample_size = int(sample_size_percentage * self.nn_experiment.data_set.samples.shape[1])
        # fetching the new sample size based on the above percentage taken
        # first transposing in order to get both the values correctly i.e, from [2 x sample_size] to [sample_size x 2]
        self.nn_experiment.data_set.selected_samples = self.nn_experiment.data_set.samples.T[0:new_sample_size]
        # setting it back to the original dimensions needed [2 x delayed_elements_size]
        self.nn_experiment.data_set.selected_samples = self.nn_experiment.data_set.selected_samples.T
        # resetting the scale dimenstion based on the input samples
        self.resetting_batch_size_slider_range(new_sample_size)

    #  function to reset the range for batch slider when the sample_size changes
    def resetting_batch_size_slider_range(self, new_sample_size):
        self.batch_size_slider = Tk.Scale(self.sliders_frame, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                                 from_=10, to_=new_sample_size, bg="#DDDDDD",
                                                 activebackground="#FF0000",
                                                 highlightcolor="#00FFFF", width=10)
        self.batch_size_slider.set(self.batch_size)
        self.batch_size_slider.bind("<ButtonRelease-1>", lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=3, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.refresh_display()

    def adjust_weights_button_callback(self):
        temp_text = self.adjust_weights_button.config('text')[-1]
        self.adjust_weights_button.config(text='Please Wait')
        # Iterate through the number of iterations
        for itr in range(self.number_of_iterations):
            # iterating through the batch inputs
            for batch_no, batch in enumerate(self.nn_experiment.data_set.batch_inputs):
                # 1st iteration is for adjusting the weights for the entire batch
                # 2nd iteration is for calculating the error and plotting the mean square error
                for indicator in range(2):
                    if  indicator == 0:
                        for indx in range(self.batch_size):
                            if (indx + self.number_of_delayed_elements+1) >= self.batch_size or \
                                    (indx + self.number_of_delayed_elements+1) >= len(batch):
                                break
                            else:
                                self.generate_input_sample(indx,self.input_dimension,batch)
                                self.nn_experiment.neural_network.calculate_output(self.nn_experiment.data_set.input)
                                self.nn_experiment.adjust_weights(self.learning_rate)
                                self.canvas.draw()
                    else:
                        error_list = []
                        error_sqaured_list = []
                        for indx in range(self.batch_size):
                            if indx + self.number_of_delayed_elements+1 >= self.batch_size:
                                break
                            else:
                                self.generate_input_sample(indx,self.input_dimension,batch)
                                self.nn_experiment.neural_network.calculate_output(self.nn_experiment.data_set.input)
                                error = self.nn_experiment.neural_network.calculate_error(self.nn_experiment.data_set.target)
                                error_list.append(error)
                                error_sqaured_list.append(pow(error,2))
                                self.canvas.draw()
                # calculating the mean squared error
                error_sqaured_sum = sum(error_sqaured_list)
                # calculate the max error
                # error_max = (error_list.sort())[-1]
                self.error_price_list.append((float(error_sqaured_sum[0])/len(error_sqaured_list)))
                self.error_volume_list.append((float(error_sqaured_sum[1])/len(error_sqaured_list)))
                if batch_no == 0 and itr == 0:
                    self.index_list.append(batch_no)
                else:
                    self.index_list.append(self.index_list[-1]+1)
                # self.error_max_list.append(error_max)
                self.axes.cla()
                self.axes.plot(self.index_list,self.error_price_list)
                self.axes.plot(self.index_list,self.error_volume_list)
                self.canvas.draw()
            print "Error Price list"
            print self.error_price_list
            print "Error Volume list"
            print self.error_volume_list
            # self.axes.plot(self.index_list,self.error_max_list)

        self.adjust_weights_button.config(text=temp_text)
        self.adjust_weights_button.update_idletasks()

# function to generate the input sample based on the number of delayed elements and starting indx value
    def generate_input_sample(self,starting_indx, input_dimension, batch_input):
        # calculate the end indx value for the input sample
        input_end_indx =  starting_indx+self.number_of_delayed_elements
        # calculate the target indx value
        target_indx = input_end_indx+1
        input_sample = batch_input.T[starting_indx: input_end_indx]
        # appending the current input to input sample
        input_sample = np.vstack((input_sample,batch_input.T[input_end_indx]))
        # now reshaping to the corresponding input dimension [input_dimension x 1]
        self.nn_experiment.data_set.input = input_sample.reshape(input_dimension,1)
        # reshaping the corresponding target value to the corresponding dimension
        self.nn_experiment.data_set.target = batch_input.T[target_indx].reshape(2,1)


    def reset_weights_button_callback(self):
        temp_text = self.reset_weights_button.config('text')[-1]
        self.reset_weights_button.config(text='Please Wait')
        self.reset_weights_button.update_idletasks()
        self.nn_experiment.neural_network.randomize_weights()
        self.refresh_display()
        self.reset_weights_button.config(text=temp_text)
        self.reset_weights_button.update_idletasks()

    def print_nn_parameters_button_callback(self):
        temp_text = self.print_nn_parameters_button.config('text')[-1]
        self.print_nn_parameters_button.config(text='Please Wait')
        self.print_nn_parameters_button.update_idletasks()
        self.nn_experiment.neural_network.display_network_parameters()
        self.refresh_display()
        self.print_nn_parameters_button.config(text=temp_text)
        self.print_nn_parameters_button.update_idletasks()



neural_network_default_settings = {
    # Optional settings
    "number_of_inputs": 2,  # number of inputs to the network
    "learning_rate": 0.01,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire training set as a batch
    "layers_specification": [{"number_of_neurons": 2,
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
        for layer_index, layer in enumerate(self.layers_specification):
            if layer_index == 0:
                layer['number_of_inputs_to_layer'] = self.number_of_inputs
            else:
                layer['number_of_inputs_to_layer'] = self.layers[layer_index - 1].number_of_neurons
            self.layers.append(ClSingleLayer(layer))

    def reset_weights(self):
        # resetting the weights for all the connections in the network to zero
        for layer in self.layers:
            layer.reset_weights()

    def randomize_weights(self,input_dimension):
        for layer in self.layers:
            layer.randomize_weights(input_dimension)

    def display_network_parameters(self, display_layers=True, display_weights=True):
        for layer_index, layer in enumerate(self.layers):
            print "\n--------------------------------------------", \
                "\nLayer #: ", layer_index, \
                "\nNumber of Nodes : ", layer.number_of_neurons, \
                "\nNumber of inputs : ", self.layers[layer_index].number_of_inputs_to_layer, \
                "\nActivation Function : ", layer.activation_function, \
                "\nWeights : ", layer.weights

    def calculate_output(self, input_values):
        # Calculate the output of the network, given the input signals
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                output = layer.calculate_output(input_values)
            else:
                output = layer.calculate_output(output)
        self.output = output
        return self.output

    def calculate_error(self, target):
        return target - self.output


    def adjust_weights(self, input_samples, targets, actual_output, learning_rate):
        # Appending 1 to the input_samples in order to accommodate the bias
        error = targets - actual_output
        if len(input_samples.shape) == 1:
            tmp_samples = np.append(input_samples, 1)
        else:
            tmp_samples = np.vstack([input_samples, np.ones((1, input_samples.shape[1]), float)])
        for layer in self.layers:
            layer.weights = layer.weights + 2 * learning_rate * np.dot(error,tmp_samples.T)

single_layer_default_settings = {
    # Optional settings
    "number_of_inputs_to_layer": 2,  # number of input signals
    "number_of_neurons": 2,  # number of neurons in the layer
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
        self.reset_weights()

    def reset_weights(self):
        # self.weights = np.ones([self.number_of_neurons, self.number_of_inputs_to_layer + 1],dtype=float)
        self.weights = np.zeros([self.number_of_neurons, self.number_of_inputs_to_layer+1],dtype=float)

    def randomize_weights(self,input_dimensions):
        self.weights = np.ones([self.number_of_neurons,input_dimensions+1],dtype=float)

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
        "number_of_inputs": 2,  # number of inputs to the network
        "learning_rate": 0.01,  # learning rate
        "layers_specification": [{"number_of_neurons": 2, "activation_function": "linear"}],  # list of dictionaries
        "data_set": ClStockMarketDataSet(),
        'number_of_classes': 2,
        'number_of_samples_in_each_class': 3
    }
    np.random.seed(1)
    ob_nn_experiment = ClNNExperiment(nn_experiment_settings)
    main_frame = Tk.Tk()
    main_frame.title("LMS Learning")
    main_frame.geometry('640x520')
    ob_nn_gui_2d = ClNNGui2d(main_frame, ob_nn_experiment)
    main_frame.mainloop()
