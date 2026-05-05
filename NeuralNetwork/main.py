#TODO train good model --
#     fix batch train --
#     add attention
#     optimize gamma/other parts of process
#     refactor code to multiple files to make it easier to use -||-
#     add a graph/function showing accuracy improvements after x epochs 


import network_core as net
import window_helper as wh

import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import * 


gamma = 0.01

confidenceThreshold = 1

neural_network = []

max_weight = 1

#output is between 0 and 1, so we need to multiply it by the max age of a crab to get the predicted age
max_age = 30


#button functions

def test_network():

    train_results = net.test(neural_network,train_data,confidenceThreshold/max_age)
    test_results = net.test(neural_network,data,confidenceThreshold/max_age)

    train_accuracy = 100 * train_results / len(train_data)
    test_accuracy = 100 * test_results / len(data)

    print(f"TRAINING DATA || Confidence Threshold: {confidenceThreshold} | Num Accurate: {train_results} | Total num {len(train_data)} | Accuracy: { train_accuracy:5f} %")
    print(f"TEST DATA     || Confidence Threshold: {confidenceThreshold} | Num Accurate: {test_results} | Total num {len(test_data)} | Accuracy: { test_accuracy:5f} %")

    return train_accuracy,test_accuracy

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

def plot_epochs():
    numepochs = 10000
    numnumepochs = 50
    global neural_network

    results = []
    epochs = []

    ax1.clear()

    ax1.set_title('Model Accuracy')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")

    ax1.yaxis.label.set_color(axis_color)
    ax1.xaxis.label.set_color(axis_color)
    ax1.title.set_color(axis_color)

    ax1.legend()
    fig.tight_layout()
    best_acc = 0
    
    for i in range(numnumepochs):
        
        epochs.append(i * numepochs)
        results.append(test_network())
        
        #change train and test data each epoch

        train_data = data.sample(frac=0.7, random_state=i)
        test_data = data.drop(train_data.index)

        neural_network = net.train_epoch(neural_network,train_data,gamma=gamma,epoch=numepochs)


        current_acc = results[-1][1]

        if(current_acc > best_acc):
            best_acc = current_acc

            #save best network (cannot check if its better than the one already saved, so we save it to a temp file)
            net.save_model(neural_network,filename="NeuralNetwork/tempmodel.net")


        ax1.plot(epochs,[result[0] for result in results], '.-', markersize=10, label='Training Accuracy',c="#C7842C")
        ax1.plot(epochs,[result[1] for result in results], '.-', markersize=10, label='Test Accuracy',c="#7166E9")
        ax1.set_ybound(0,100)

        graph1.draw()
        drawNetwork(neuron_canvas)
        
        textFrame.delete("1.0", tk.END) 
        textFrame.insert("1.0", f"Accuracy: {current_acc:2f} \nBest Accuracy: {best_acc:2f}")

        root.update_idletasks()
        root.update()


#get data and preprocess it

data = pd.read_csv("CrabAgePrediction.csv")
data["Sex"] = data["Sex"].map({"M": 1, "F": 2, "I": 3})

#normalize data

for column in data.columns:
    if column != "Age":
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

data["Age"] /= max_age

# split data into training and testing sets
train_data = data.sample(frac=0.7, random_state=42)
test_data = data.drop(train_data.index)

#create a neural network
num_inputs = len(data.columns) - 1 # Number of input features (excluding the target variable)

neural_network = net.initialize_network(num_inputs)

net.add_layer(neural_network, 16, activation_function=net.relu)
net.add_layer(neural_network, 8, activation_function=net.relu)
net.add_layer(neural_network, 5, activation_function=net.relu)

net.finalize_network(neural_network, activation_function=net.sigmoid)

#display neural network

background_color = "#1e1e1e" # Set the background color to a dark shade
button_color = "#6F6881" # Set the button color to a purple shade
axis_color = "#BDB6CF"

root = tk.Tk()
root.title("Crab Network") # Set the window title
root.geometry("1300x700") # Set the initial window size (width x height)
root.configure(bg=background_color) 

frame = tk.Frame(root, bg=background_color)
frame.pack(pady=20,expand=True, fill=BOTH, side=BOTTOM)

buttonsFrame = tk.Frame(root, bg=background_color)
buttonsFrame.pack(padx=50,pady=20,expand=False, side=TOP,fill=BOTH)

trainButton = tk.Button(buttonsFrame, text="Train", command=epochTrain, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to test the neural network
trainButton.pack(padx=50, pady=10,side=LEFT)

trainButton2 = tk.Button(buttonsFrame, text="Epoch Train", command=plot_epochs, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to test the neural network
trainButton2.pack(padx=50, pady=10,side=LEFT)

testButton = tk.Button(buttonsFrame, text="Test", command=test_network, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to test the neural network
testButton.pack(padx=50, pady=10,side=LEFT)

saveButton = tk.Button(buttonsFrame, text="Save Model", command=save, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to save the model
saveButton.pack(padx=50, pady=10,side=LEFT)

loadButton = tk.Button(buttonsFrame, text="Load Model", command=load, bg=button_color, fg="white", font=("Arial", 12), relief=FLAT, bd=3) # Create a button to load the model
loadButton.pack(padx=50, pady=10,side=LEFT)


neuron_canvas = tk.Canvas(frame, bg=background_color) # Create a canvas to draw the neural network
neuron_canvas.pack(pady=5,padx=10,expand=True, fill=BOTH,side=LEFT)

neuron_canvas.bind("<Configure>", lambda event: drawNetwork(neuron_canvas)) # Redraw the network when the canvas is resized

drawNetwork(neuron_canvas)


graph_canvas = tk.Canvas(frame, bg=background_color)
graph_canvas.pack(pady=5,padx=10,expand=False, fill=BOTH,side=RIGHT)

graphFrame = tk.Frame(graph_canvas, bg=background_color)
graphFrame.pack(side=tk.TOP,fill = BOTH, expand = False, pady=5,padx=5)

fig = Figure(figsize=(6,3), dpi=100)
fig.patch.set_facecolor(background_color)

ax1 = fig.add_subplot(111)
ax1.set_facecolor(background_color)

ax1.spines['bottom'].set_color(axis_color)
ax1.spines['top'].set_color(axis_color) 
ax1.spines['right'].set_color(axis_color)
ax1.spines['left'].set_color(axis_color)

ax1.tick_params(axis='x', colors=axis_color)
ax1.tick_params(axis='y', colors=axis_color)

ax1.yaxis.label.set_color(axis_color)
ax1.xaxis.label.set_color(axis_color)
ax1.title.set_color(axis_color)





graph1 = FigureCanvasTkAgg(fig, master=graphFrame)
graph1_widget = graph1.get_tk_widget()
graph1_widget.grid(row=0, column=0, sticky="nsew",padx=2,pady=2)

textFrame = tk.Text(graph_canvas,height=5, width=50,bg=background_color,fg=axis_color,wrap=tk.NONE)
textFrame.pack(side=tk.TOP, expand = False, pady=5,padx=5)

root.mainloop()

