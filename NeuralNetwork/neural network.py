
from colorsys import hsv_to_rgb, rgb_to_hsv
from multiprocessing import connection
import random
import json

import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import * 
from tkinter import ttk
from tkinter.ttk import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


#TODO train good model --
#     fix batch train --
#     add attention
#     optimize gamma/other parts of process
#     refactor code to multiple files to make it easier to use
#     add a graph/function showing accuracy improvements after x epochs 



gamma = 0.01

random_factor = 1

neural_network = []

#output is between 0 and 1, so we need to multiply it by the max age of a crab to get the predicted age
max_age = 30

#math functions

def cost_function(predicted, actual):
    return (actual - predicted) ** 2

def derivative_cost_function(predicted, actual):
    return (actual - predicted)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

#Training progress bar

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



class neuron:
    def __init__(self, numWeights, bias, activation_function=sigmoid):
        self.weights = [random_factor * (random.random() - 0.5) for _ in range(numWeights)]
        self.bias = bias

        self.activation = 0
        self.activation_function = activation_function
        self.derivativeActivation = sigmoid_derivative if activation_function == sigmoid else relu_derivative
        self.error_derivative = 0
        self.weightsderivative = [0 for _ in range(numWeights)]

    def feedforward(self, inputs):
        total = 0
        for w, i in zip(self.weights, inputs):
            total += w * i
            
        total += self.bias

        self.activation = self.activation_function(total)


        return self.activation
        
    def update_weights(self, inputs, next_layer_errors):

        self.error_derivative = sum(next_layer_errors) * self.derivativeActivation(self.activation)

        #print(f"error {self.error_derivative}, actder {self.derivativeActivation(self.activation)}, nextlerr {sum(next_layer_errors)}")
       # print(f'wchange {[self.error_derivative * i for i in inputs]}')


        self.weights = [w + gamma * self.error_derivative * i for w, i in zip(self.weights, inputs)]
        self.bias += gamma * self.error_derivative

        self.weightsderivative = [weight * self.error_derivative for weight in self.weights]

        #print(f"newweights {self.weights}")
              
class finalNeuron:
    def __init__(self, numWeights, bias, activation_function=sigmoid):

        self.weights = [random_factor * (random.random() - 0.5) for _ in range(numWeights)]
        self.bias = bias

        self.activation = 0
        self.activation_function = activation_function
        self.derivativeActivation = sigmoid_derivative if activation_function == sigmoid else relu_derivative

        self.weightsderivative = [0 for _ in range(numWeights)]
        self.error_derivative = 0

    def feedforward(self, inputs):
        total = 0
        for w, i in zip(self.weights, inputs):
            total += w * i
            
        total += self.bias

        self.activation = self._activate(total)
        return self.activation
    
    def _activate(self, x):
        return self.activation_function(x)
    
    def update_weights(self, inputs, predicted, actual):

        self.error_derivative = derivative_cost_function(predicted, actual) * self.derivativeActivation(self.activation)

        #print(f"Pred {predicted}, Act {actual}, error {self.error_derivative}, actder {self.derivativeActivation(self.activation)}")
        #print(f'wchange {[self.error_derivative * i for w, i in zip(self.weights, inputs)]}')

        self.weights = [w + gamma * self.error_derivative * i for w, i in zip(self.weights, inputs)]

        self.bias += gamma * self.error_derivative

        #totalderivative = sum(self.weights) * self.error_derivative
        self.weightsderivative = [weight * self.error_derivative for weight in self.weights]

       # print(f"newweights {self.weights}")

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
    
    def backpropagate(self, inputs, next_layer_errors):
        for i, neuron in enumerate(self.neurons):
            # print([neuronError[i] for neuronError in next_layer_errors])

            neuron.update_weights(inputs, [neuronError[i] for neuronError in next_layer_errors])
        
class inputLayer:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.neurons = [neuron(1, 0) for _ in range(num_inputs)]

    def feedforward(self, inputs):
        for i,neuron in enumerate(self.neurons):
            neuron.activation = inputs[i]
        return inputs
    
    def backpropagate(self):
        pass

class finalLayer:
    def __init__(self, num_inputs, activation_function=sigmoid):
        self.activation_function = activation_function
        self.neurons = [finalNeuron(num_inputs, random.random(), activation_function=activation_function)]

    def feedforward(self, inputs):
        return [self.neurons[0].feedforward(inputs)]
    
    def backpropagate(self, inputs, actual):
        predicted = self.neurons[0].activation
        self.neurons[0].update_weights(inputs, predicted, actual)

#neural network base functions


def predict(inputs):
    for l in neural_network:
        inputs = l.feedforward(inputs)
    return inputs

def backpropagation(network, inputs, actual):

    # Forward pass
    for layer in network:
        inputs = layer.feedforward(inputs)


    # Calculate error at the output layer
    predicted = sum(inputs)
    error = cost_function(predicted, actual)

    # Backward pass
    for i in range(len(network) - 1, 0, -1):
        layer = network[i]
        prev_outputs = [neuron.activation for neuron in network[i - 1].neurons]
        if i == len(network) - 1:
            layer.backpropagate(prev_outputs, actual)
        else:
            next_layer_errors = [neuron.weightsderivative for neuron in network[i + 1].neurons]
            layer.backpropagate(prev_outputs, next_layer_errors)
    
    return error

def backpropagation_test():

    global neural_network

    network = neural_network
    inputs = [0.35,0.7]
    actual = 0.5

    for layer in network:
        inputs = layer.feedforward(inputs)
    
    predicted = sum(inputs)
    error = cost_function(predicted, actual)

    print(f"Cost: {error}")

    for i in range(len(network) - 1, 0, -1):
        layer = network[i]

        prev_outputs = [neuron.activation for neuron in network[i - 1].neurons]
        print(f"l {i} prev outp {prev_outputs}")
        if i == len(network) - 1:
            layer.backpropagate(prev_outputs, actual)
        else:
            next_layer_errors = [neuron.weightsderivative for neuron in network[i + 1].neurons]
            layer.backpropagate(prev_outputs, next_layer_errors)
 
def train():
    for index in range(len(train_data)):
        row = train_data.iloc[index].tolist()
        inputs = row[:-1]
        actual = float(row[-1]) / max_age
        backpropagation(neural_network, inputs, actual)
        if index % 100 == 0:
            draw_network( neuron_canvas.winfo_width()/len(neural_network), neuron_canvas.winfo_height()/len(neural_network[0].neurons) )

        
    draw_network( neuron_canvas.winfo_width()/len(neural_network), neuron_canvas.winfo_height()/len(neural_network[0].neurons) )

def train_epoch(epoch = 100000):

    for i in range(epoch):
        row = train_data.iloc[int(random.random() * len(train_data))].tolist()
        inputs = row[:-1]
        actual = float(row[-1]) / max_age

        

        if(i%100 == 0):
            error = backpropagation(neural_network, inputs, actual)
            printProgressBar(i,epoch,suffix=f"\n Cost:{error}")
        else:
            error = backpropagation(neural_network, inputs, actual)





    draw_network( neuron_canvas.winfo_width()/len(neural_network), neuron_canvas.winfo_height()/len(neural_network[0].neurons) )
    test()

def train_once():
    index = random.randint(0, len(train_data) - 1)
    row = train_data.iloc[index].tolist()
    inputs = row[:-1]
    actual = float(row[-1]) / max_age
    backpropagation(neural_network, inputs, actual)
    draw_network( neuron_canvas.winfo_width()/len(neural_network), neuron_canvas.winfo_height()/len(neural_network[0].neurons) )

def batch_train(batch_size=10):
    for index in range(0, len(train_data), batch_size):
        batch = train_data.iloc[index:index + batch_size]
        for _, row in batch.iterrows():
            inputs = row[:-1].tolist()
            actual = int(row.tolist()[-1])
            backpropagation(neural_network, inputs, actual)
        
        draw_network( neuron_canvas.winfo_width()/len(neural_network), neuron_canvas.winfo_height()/len(neural_network[0].neurons) )
        
    draw_network( neuron_canvas.winfo_width()/len(neural_network), neuron_canvas.winfo_height()/len(neural_network[0].neurons) )


#save & load functions

def save_model(filename="NeuralNetwork/model.net"):

    #debug
    print_weights()


    with open(filename, "w") as f:
        # get num neurons in each layer

        f.write(f"{[len(l.neurons) for l in neural_network]}\n")

        for l in neural_network[1:]: #don't store 1st layer

            f.write(f"{l.activation_function.__name__}\n")
            for n in l.neurons:
                floatWeights = [float(x) for x in n.weights]
                f.write(f"{floatWeights}\n")
                f.write(f"{n.bias}\n")

def load_model(filename="NeuralNetwork/model.net"):

    global neural_network

    print("Loading Model")

    with open(filename, "r") as f:
        layer_sizes = eval(f.readline().strip())

        neural_network = initialize_network(layer_sizes[0])

        print(layer_sizes)


        for size in layer_sizes[1:-1]:
            activation_function = f.readline().strip()
            add_layer(neural_network, size, activation_function=sigmoid if activation_function == "sigmoid" else relu)
            print(f"New layer size: {size}")
            print(f"New Layer Function: {activation_function}")
        
            for i, n in enumerate(neural_network[-1].neurons):
                print(f"Neuron {i+1}")
                weights = f.readline().strip()
                n.weights = json.loads(weights)
                print(f"Weights:{n.weights}" )
                n.bias = float(f.readline().strip())
                print(f"Bias: {n.bias}")
                n.activation_function = sigmoid if neural_network[-1].activation_function == sigmoid else relu

        #add final layers
        activation_function = f.readline().strip()
        finalize_network(neural_network, activation_function=sigmoid if activation_function == "sigmoid" else relu)
        for n in neural_network[-1].neurons:
            weights = f.readline().strip()
            n.weights = json.loads(weights)
            n.bias = float(f.readline().strip())
            n.activation_function = sigmoid if neural_network[-1].activation_function == sigmoid else relu
    
    print(" ----------- Finished loading model ------------ \n\n")
    print_weights()
    draw_network()

#debug functions

def print_weights():
    for l in neural_network:
        print(f"-------------- Layer {neural_network.index(l)} -------------- \n")
        for n in l.neurons:
            print(f"  Neuron {l.neurons.index(n)}:")
            print(n.weights)
            print(n.bias)
            print(f"  Activation: {n.activation} \n"
                  f"  Error Derivative: {n.error_derivative} \n")

#button functions

def test():

    confidence_threshold = 1

    accuracy = 0

    for index in range(len(test_data)):
        row = test_data.iloc[index].tolist()
        inputs = row[:-1]
        actual = float(row[-1])

        predicted = sum(predict(inputs))

        cost = cost_function(predicted * max_age, actual)

        print(f"Predicted: {predicted * max_age}, Actual: {actual}, cost: {cost}")

        accuracy += 1 if abs(predicted * max_age - actual) < confidence_threshold else 0

    print("----------------------------")
    print(f"Confidence Threshold: {confidence_threshold} | Num Accurate: {accuracy} | Total num {len(test_data)} | Accuracy: {100 * accuracy / len(test_data) :5f} %")
    

#get data and preprocess it

data = pd.read_csv("CrabAgePrediction.csv")
data["Sex"] = data["Sex"].map({"M": 1, "F": 2, "I": 3})

# #normalize data
# for column in data.columns:
#      if column != "Age":
#          data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

# split data into training and testing sets
train_data = data.sample(frac=0.7, random_state=42)
test_data = data.drop(train_data.index)

#testing stuff

#data = pd.read_csv("testFile.csv")

#train_data = data
#test_data = data

#create a neural network

num_inputs = len(data.columns) - 1 # Number of input features (excluding the target variable)

print(num_inputs)

def initialize_network(num_inputs=1):
    network = [inputLayer(num_inputs)]
    num_inputs = len(data.columns) - 1
    return network

def add_layer(neural_network, num_neurons, activation_function=sigmoid):
    if len(neural_network) == 0:
        neural_network.append(inputLayer(num_inputs))
    else:
        neural_network.append(layer([neuron(len(neural_network[-1].neurons), random.random()) for _ in range(num_neurons)], activation_function=activation_function))

def finalize_network(neural_network, activation_function=sigmoid):
    neural_network.append(finalLayer(len(neural_network[-1].neurons), activation_function=activation_function))


neural_network = initialize_network(num_inputs)

add_layer(neural_network, 6, activation_function=relu)
add_layer(neural_network, 5, activation_function=relu)
add_layer(neural_network, 5, activation_function=relu)

finalize_network(neural_network, activation_function=sigmoid)

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

trainButton2 = tk.Button(buttonsFrame, text="Epoch Train", command=train_epoch, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to test the neural network
trainButton2.pack(padx=50, pady=10,side=LEFT) # Add some padding around the button

testButton = tk.Button(buttonsFrame, text="Test", command=test, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to test the neural network
testButton.pack(padx=50, pady=10,side=LEFT) # Add some padding around the button

saveButton = tk.Button(buttonsFrame, text="Save Model", command=save_model, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to save the model
saveButton.pack(padx=50, pady=10,side=LEFT) # Add some padding around the button

loadButton = tk.Button(buttonsFrame, text="Load Model", command=load_model, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to load the model
loadButton.pack(padx=50, pady=10,side=LEFT) # Add some padding around the button

# loadTestButton = tk.Button(buttonsFrame, text="Train Test", command=backpropagation_test, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to load the model
# loadTestButton.pack(padx=50, pady=10,side=LEFT) # Add some padding around the button



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


    finalcolor = f'#{r:02x}{g:02x}{b:02x}'
    S
    return finalcolor[0:7]

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

    layer_width = neuron_canvas.winfo_width()/len(neural_network)
    layer_height = neuron_canvas.winfo_height()/len(max(neural_network, key=lambda l: len(l.neurons)).neurons)
    neuron_canvas.delete("all") # Clear the canvas before redrawing

    neuron_radius = 20
    
    for l in range(len(neural_network)):

        neuronColor = "#526bbc"
        outlineColor = "#8ea5ef"

        standardLineColor = "#003cff"
        stadardNegLineColor = "#ff0000"


        if l == 0:
            neuronColor = "#7b7e11"
            outlineColor = "#72720b"

        #create neurons & connections 
        for i in range(len(neural_network[l].neurons)):
            
            if l > 0: # Check if it's not the first layer
                for j in range(len(neural_network[l-1].neurons)):
                    #get weight of the connection


                    weight = neural_network[l].neurons[i].weights[j]

                    brightness = abs(weight)/random_factor # Normalize weight to 0.1-1.0

                    if weight < 0:
                        lineColor = change_hex_brightness(stadardNegLineColor, brightness)
                    else:
                        lineColor = change_hex_brightness(standardLineColor, brightness)


                    neuron_canvas.create_line(centerX(neuron_canvas.winfo_width(), layer_width, len(neural_network), l), 
                                              centerY(neuron_canvas.winfo_height(), layer_height, len(neural_network[l].neurons), i), 
                                              centerX(neuron_canvas.winfo_width(), layer_width, len(neural_network), l - 1), 
                                              centerY(neuron_canvas.winfo_height(), layer_height, len(neural_network[l - 1].neurons), j) , fill=lineColor, width=2) # Draw a line to represent the connection

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
