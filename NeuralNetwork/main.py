#TODO train good model --
#     fix batch train --
#     add attention
#     optimize gamma/other parts of process
#     refactor code to multiple files to make it easier to use -||-
#     add a graph/function showing accuracy improvements after x epochs 

#display neural network

import network_core as net
import window_helper as wh

import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import * 


#TODO train good model --
#     fix batch train --
#     add attention
#     optimize gamma/other parts of process
#     refactor code to multiple files to make it easier to use
#     add a graph/function showing accuracy improvements after x epochs 



gamma = 0.01

confidenceThreshold = 1

neural_network = []

max_weight = 1

#output is between 0 and 1, so we need to multiply it by the max age of a crab to get the predicted age
max_age = 30


#button functions

def test_network():

    test_results = net.test(neural_network,test_data,confidenceThreshold/max_age)

    print(f"Confidence Threshold: {confidenceThreshold} | Num Accurate: {test_results} | Total num {len(test_data)} | Accuracy: {100 * test_results / len(test_data) :5f} %")

def train():
    global neural_network
    neural_network = net.train(neural_network,train_data,gamma=gamma)

def epochTrain():
    global neural_network
    neural_network = net.train_epoch(neural_network,train_data,gamma=gamma)    

    print(f"nnf {neural_network}")
    
    test_network()
    
def save():
    global neural_network
    net.save_model(neural_network)
    
def load():
    global neural_network
    neural_network = net.load_model()

def drawNetwork(canvas):

    global max_weight
    global neural_network

    max_weight = wh.draw_network(canvas, neural_network,max_weight)






#get data and preprocess it

data = pd.read_csv("CrabAgePrediction.csv")
data["Sex"] = data["Sex"].map({"M": 1, "F": 2, "I": 3})

#normalize output

data["Age"] /= max_age

# split data into training and testing sets
train_data = data.sample(frac=0.7, random_state=42)
test_data = data.drop(train_data.index)



#create a neural network
num_inputs = len(data.columns) - 1 # Number of input features (excluding the target variable)

neural_network = net.initialize_network(num_inputs)

#net.print_weights(neural_network)

net.add_layer(neural_network, 6, activation_function=net.relu)
net.add_layer(neural_network, 5, activation_function=net.relu)
net.add_layer(neural_network, 5, activation_function=net.relu)

net.finalize_network(neural_network, activation_function=net.sigmoid)

#net.print_weights(neural_network)



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

trainButton2 = tk.Button(buttonsFrame, text="Epoch Train", command=epochTrain, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to test the neural network
trainButton2.pack(padx=50, pady=10,side=LEFT) # Add some padding around the button

testButton = tk.Button(buttonsFrame, text="Test", command=test_network, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to test the neural network
testButton.pack(padx=50, pady=10,side=LEFT) # Add some padding around the button

saveButton = tk.Button(buttonsFrame, text="Save Model", command=save, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to save the model
saveButton.pack(padx=50, pady=10,side=LEFT) # Add some padding around the button

loadButton = tk.Button(buttonsFrame, text="Load Model", command=load, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to load the model
loadButton.pack(padx=50, pady=10,side=LEFT) # Add some padding around the button

neuron_canvas = tk.Canvas(frame, bg=background_color) # Create a canvas to draw the neural network

#make canvas fill the frame

neuron_canvas.pack(pady=5,padx=200,expand=True, fill=BOTH,side=LEFT)

neuron_canvas.bind("<Configure>", lambda event: drawNetwork) # Redraw the network when the canvas is resized

drawNetwork(neuron_canvas)

root.mainloop()

