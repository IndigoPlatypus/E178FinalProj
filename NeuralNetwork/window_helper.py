
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

#display neural network

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

def draw_network(neuron_canvas, neural_network, max_weight):

    if(neural_network == None):
        print("Invalid Neural Network for Draw Function")
        return


    layer_width = neuron_canvas.winfo_width()/len(neural_network)
    layer_height = neuron_canvas.winfo_height()/len(max(neural_network, key=lambda l: len(l.neurons)).neurons)

    neuron_canvas.delete("all") # Clear the canvas before redrawing

    neuron_radius = 20
    
    newMaxWeight = max_weight

    for l in range(len(neural_network)):

        standardLineColor = "#003cff"
        stadardNegLineColor = "#ff0000"

        layer_height = neuron_canvas.winfo_height()/len(neural_network[l].neurons)

        #create connections 
        for i in range(len(neural_network[l].neurons)):
            if l > 0: # Check if it's not the first layer
                prev_layer_height = neuron_canvas.winfo_height()/len(neural_network[l-1].neurons)

                for j in range(len(neural_network[l-1].neurons)):
                    #get weight of the connection

                    weight = neural_network[l].neurons[i].weights[j]

                    if(abs(weight) > max_weight):
                        newMaxWeight = abs(weight)
                    brightness = abs(weight)/max_weight

                    if weight < 0:
                        lineColor = change_hex_brightness(stadardNegLineColor, brightness)
                    else:
                        lineColor = change_hex_brightness(standardLineColor, brightness)


                    neuron_canvas.create_line(centerX(neuron_canvas.winfo_width(), layer_width, len(neural_network), l), 
                                              centerY(neuron_canvas.winfo_height(), layer_height, len(neural_network[l].neurons), i), 
                                              centerX(neuron_canvas.winfo_width(), layer_width, len(neural_network), l - 1), 
                                              centerY(neuron_canvas.winfo_height(), prev_layer_height, len(neural_network[l - 1].neurons), j) , fill=lineColor, width=2) # Draw a line to represent the connection

            
    for l in range(len(neural_network)): #create neurons
        neuronColor = "#526bbc"
        outlineColor = "#8ea5ef"

        layer_height = neuron_canvas.winfo_height()/len(neural_network[l].neurons)

        if l == 0:
            neuronColor = "#7b7e11"
            outlineColor = "#72720b"

        for i in range(len(neural_network[l].neurons)):

            neuron_canvas.create_oval( centerX(neuron_canvas.winfo_width(), layer_width, len(neural_network), l) - neuron_radius,
                                        centerY(neuron_canvas.winfo_height(), layer_height, len(neural_network[l].neurons), i) - neuron_radius,
                                        centerX(neuron_canvas.winfo_width(), layer_width, len(neural_network), l) + neuron_radius,
                                        centerY(neuron_canvas.winfo_height(), layer_height, len(neural_network[l].neurons), i) + neuron_radius,
                                        fill=neuronColor,outline=outlineColor,width=5) # Draw a circle to represent the neuron
            neuron_canvas.pack(pady=0)

    
    return newMaxWeight
        