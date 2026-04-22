
from colorsys import hsv_to_rgb, rgb_to_hsv
from multiprocessing import connection
import random

import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import * 
from tkinter import ttk
from tkinter.ttk import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk



gamma = 0.01

#output is between 0 and 1, so we need to multiply it by the max age of a crab to get the predicted age
max_age = 30

#math functions

def const_function(predicted, actual):
    return (actual - predicted) ** 2

def derivative_const_function(predicted, actual):
    return 2 * (actual - predicted)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0




class neuron:
    def __init__(self, numWeights, bias, activation_function=sigmoid):
        self.weights = [random.random() for _ in range(numWeights)]
        self.bias = bias
        self.activation_function = activation_function
        self.derivativeActivation = sigmoid_derivative if activation_function == sigmoid else relu_derivative
        self.errorderivative = 0
        self.weightderivative = [0 for _ in range(numWeights)]

    def feedforward(self, inputs):
        total = 0
        for w, i in zip(self.weights, inputs):
            total += w * i
        total += self.bias
        return self._activate(total)
    
    def _activate(self, x):
        return self.activation_function(x)
    
    def update_weights(self, inputs, last_layer_weights, next_layer_error):


        for i in range(len(self.weights)):
            self.weightderivative[i] = self.derivativeActivation(self.feedforward(inputs)) * self.weights[i] * next_layer_error
            self.weights[i] -= gamma * self.errorderivative * inputs[i]

        self.bias -= gamma * self.errorderivative

        self.errorderivative = sum(last_layer_weights * next_layer_error) * self.derivativeActivation(self.feedforward(inputs))
         


    

#create layer of neurons
class layer:
    def __init__(self, neurons, activation_function=sigmoid):
        self.neurons = neurons
        self.activation_function = activation_function
        
        for neuron in self.neurons:
            neuron.activation_function = activation_function

    def feedforward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            output = neuron.feedforward(inputs)
            outputs.append(output)
        return outputs
    
    def backpropagate(self, inputs, errors):
        for neuron, error in zip(self.neurons, errors):
            neuron.update_weights(inputs, error)

class inputLayer:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.neurons = [neuron(0, 0) for _ in range(num_inputs)]

    def feedforward(self, inputs):
        return inputs

#neural network base functions


def predict(inputs):
    for l in neural_network:
        inputs = l.feedforward(inputs)
    return inputs

def backpropagation(network, inputs, actual):
    # Forward pass
    outputs = [inputs]  # Store outputs of each layer
    for layer in network:
        inputs = layer.feedforward(inputs)
        outputs.append(inputs)

    # Calculate error at the output layer
    predicted = sum(outputs[-1]) * max_age
    error = const_function(predicted, actual)
    derivative_error = derivative_const_function(predicted, actual)

    print(f"Cost: {error}")

    # Backward pass
    for i in range(len(network) - 1, 0, -1):
        layer = network[i]
        prev_outputs = outputs[i - 1]
        errors = [derivative_error] * len(layer.neurons)  # Simplified error distribution
        layer.backpropagate(prev_outputs, errors)

def train():
    for index in range(len(train_data)):
        row = train_data.iloc[index].tolist()
        inputs = row[:-1]
        actual = int(row[-1])
        backpropagation(neural_network, inputs, actual)
        
    draw_network()

def batch_train(batch_size=10):
    for index in range(0, len(train_data), batch_size):
        batch = train_data.iloc[index:index + batch_size]
        for _, row in batch.iterrows():
            inputs = row[:-1].tolist()
            actual = int(row.tolist()[-1])
            backpropagation(neural_network, inputs, actual)
        
        
    draw_network()


#save & load functions

def save_model():
    print_weights()

def load_model():
    pass

#debug functions

def print_weights():
    for l in neural_network:
        for n in l.neurons:
            print(n.weights)
            print(n.bias)

#button functions

def on_test():

    confidence_threshold = 0.1

    accuracy = 0
    for index in range(len(test_data)):
        row = test_data.iloc[index].tolist()
        inputs = row[:-1]
        actual = int(row[-1])

        predicted = sum(predict(inputs)) * max_age

        cost = const_function(predicted, actual)
        print(f"Predicted: {predicted}, Actual: {actual}, cost: {cost}")
        accuracy += 1 if abs(predicted - actual) < confidence_threshold else 0

    accuracy /= len(test_data)
    print("----------------------------")
    print(f"Accuracy: {accuracy}")

#get data from csv file

data = pd.read_csv("CrabAgePrediction.csv")

#reformat data into only numbers

data["Sex"] = data["Sex"].map({"M": 1, "F": 2, "I": 3})

#normalize data
for column in data.columns:
    if column != "Age":
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

#split data into training and testing sets
train_data = data.sample(frac=0.7, random_state=42)
test_data = data.drop(train_data.index)

#create a neural network

num_inputs = len(data.columns) - 1 # Number of input features (excluding the target variable)

def add_layer(neural_network, num_neurons, activation_function=sigmoid):
    if len(neural_network) == 0:
        neural_network.append(inputLayer(num_inputs))
    else:
        neural_network.append(layer([neuron(len(neural_network[-1].neurons), random.random()) for _ in range(num_neurons)], activation_function=activation_function))

neural_network = [inputLayer(num_inputs)]
add_layer(neural_network, 6, activation_function=sigmoid)
add_layer(neural_network, 5, activation_function=sigmoid)
add_layer(neural_network, 1, activation_function=sigmoid)


#display neural network


background_color = "#1e1e1e" # Set the background color to a dark shade
button_color = "#6F6881" # Set the button color to a purple shade

root = tk.Tk()
root.title("Crab Network") # Set the window title
root.geometry("1000x700") # Set the initial window size (width x height)
root.configure(bg=background_color) 

frame = tk.Frame(root, bg=background_color)
frame.pack(pady=20,expand=True, fill=BOTH, side=BOTTOM)



buttonsFrame = tk.Frame(root, bg=background_color)
buttonsFrame.pack(padx=50,pady=20,expand=False, side=TOP)

trainButton = tk.Button(buttonsFrame, text="Train", command=train, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to test the neural network
trainButton.pack(padx=50, pady=10,side=LEFT) # Add some padding around the button

trainButton = tk.Button(buttonsFrame, text="Batch Train", command=batch_train, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to test the neural network
trainButton.pack(padx=50, pady=10,side=LEFT) # Add some padding around the button

testButton = tk.Button(buttonsFrame, text="Test", command=on_test, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to test the neural network
testButton.pack(padx=50, pady=10,side=LEFT) # Add some padding around the button

saveButton = tk.Button(buttonsFrame, text="Save Model", command=save_model, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to save the model
saveButton.pack(padx=50, pady=10,side=LEFT) # Add some padding around the button

loadButton = tk.Button(buttonsFrame, text="Load Model", command=load_model, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to load the model
loadButton.pack(padx=50, pady=10,side=LEFT) # Add some padding around the button



neuron_canvas = tk.Canvas(frame, bg=background_color) # Create a canvas to draw the neural network
#make canvas fill the frame
neuron_canvas.pack(pady=5,padx=200,expand=True, fill=BOTH,side=LEFT)



#drawing functions

def change_hex_brightness(hex_color, brightness):

    # Remove '#' if present
    hex_color = hex_color.lstrip('#')

    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    h, s, v = rgb_to_hsv(r, g, b)

    r, g, b = hsv_to_rgb(h, s, brightness)

    r, g, b = int(r * 255), int(g * 255), int(b * 255)

    return f'#{r:02x}{g:02x}{b:02x}'


def centerX(total_width, layer_width, total_layers, layer_index):
    middle = total_width / 2

    if total_layers % 2 == 0:
        return middle - ((total_layers / 2 - layer_index) * layer_width)
    else:
        return middle - ((total_layers // 2 - layer_index) * layer_width)



def centerY(total_height, layer_height, total_neurons, neuron_index):
    middle = total_height / 2

    if total_neurons % 2 == 0:
        return middle - ((total_neurons / 2 - neuron_index) * layer_height) + (layer_height / 2)
    else:
        return middle - ((total_neurons // 2 - neuron_index) * layer_height)

def draw_network(layer_width=neuron_canvas.winfo_width()/len(neural_network), layer_height=neuron_canvas.winfo_height()/len(neural_network[0].neurons)):

    neuron_canvas.delete("all") # Clear the canvas before redrawing

    neuron_radius = 20
    
    for l in range(len(neural_network)):

        neuronColor = "#526bbc"
        outlineColor = "#8ea5ef"

        standardLineColor = "#8ea5ef"


        if l == 0:
            neuronColor = "#7b7e11"
            outlineColor = "#72720b"

        #create neurons & connections 
        for i in range(len(neural_network[l].neurons)):
            
            if l < len(neural_network) - 1: # Check if it's not the last layer
                for j in range(len(neural_network[l + 1].neurons)):
                    #get weight of the connection

                    if(l == 0):
                        weight = 200
                    else:
                        weight = neural_network[l].neurons[i].weights[j]

                    brightness = max(0.1, min(1.0, abs(weight))) # Normalize weight to 0.1-1.0

                    lineColor = change_hex_brightness(standardLineColor, brightness)

                    neuron_canvas.create_line(centerX(neuron_canvas.winfo_width(), layer_width, len(neural_network), l), 
                                              centerY(neuron_canvas.winfo_height(), layer_height, len(neural_network[l].neurons), i), 
                                              centerX(neuron_canvas.winfo_width(), layer_width, len(neural_network), l + 1), 
                                              centerY(neuron_canvas.winfo_height(), layer_height, len(neural_network[l + 1].neurons), j) , fill=lineColor, width=2) # Draw a line to represent the connection

            neuron_canvas.create_oval( centerX(neuron_canvas.winfo_width(), layer_width, len(neural_network), l) - neuron_radius,
                                        centerY(neuron_canvas.winfo_height(), layer_height, len(neural_network[l].neurons), i) - neuron_radius,
                                        centerX(neuron_canvas.winfo_width(), layer_width, len(neural_network), l) + neuron_radius,
                                        centerY(neuron_canvas.winfo_height(), layer_height, len(neural_network[l].neurons), i) + neuron_radius,
                                        fill=neuronColor,outline=outlineColor,width=5) # Draw a circle to represent the neuron
            neuron_canvas.pack(pady=0)
            

draw_network()


neuron_canvas.bind("<Configure>", lambda event: draw_network(neuron_canvas.winfo_width()/len(neural_network), neuron_canvas.winfo_height()/len(neural_network[0].neurons) )) # Redraw the network when the canvas is resized

root.mainloop()


#change canvas dimensions
