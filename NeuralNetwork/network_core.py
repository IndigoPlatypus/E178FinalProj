import random
import json

import pandas as pd
import numpy as np


#KEEP IN MIND: Model ONLY outputs between 0-1 -> NORMALIZE FINAL PARAMETER


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

#main components

class neuron:
    def __init__(self, numWeights, bias, activation_function=sigmoid):
        self.weights = [2*(random.random() - 0.5) for _ in range(numWeights)] #between -1 and 1
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
        
    def update_weights(self, inputs, next_layer_errors,gamma):

        self.error_derivative = sum(next_layer_errors) * self.derivativeActivation(self.activation)

        #print(f"error {self.error_derivative}, actder {self.derivativeActivation(self.activation)}, nextlerr {sum(next_layer_errors)}")
       # print(f'wchange {[self.error_derivative * i for i in inputs]}')


        self.weights = [w + gamma * self.error_derivative * i for w, i in zip(self.weights, inputs)]
        self.bias += gamma * self.error_derivative

        self.weightsderivative = [weight * self.error_derivative for weight in self.weights]

        #print(f"newweights {self.weights}")
              
class finalNeuron:
    def __init__(self, numWeights, bias, activation_function=sigmoid):

        self.weights = [2 * (random.random() - 0.5) for _ in range(numWeights)] #between -1 and 1
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
    
    def update_weights(self, inputs, predicted, actual, gamma):

        self.error_derivative = derivative_cost_function(predicted, actual) * self.derivativeActivation(self.activation)

        #print(f"Pred {predicted}, Act {actual}, error {self.error_derivative}, actder {self.derivativeActivation(self.activation)}")
        #print(f'wchange {[self.error_derivative * i for w, i in zip(self.weights, inputs)]}')

        self.weights = [w + gamma * self.error_derivative * i for w, i in zip(self.weights, inputs)]

        self.bias += gamma * self.error_derivative

        #totalderivative = sum(self.weights) * self.error_derivative
        self.weightsderivative = [weight * self.error_derivative for weight in self.weights]

       # print(f"newweights {self.weights}")


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
    
    def backpropagate(self, inputs, next_layer_errors,gamma):
        for i, neuron in enumerate(self.neurons):
            # print([neuronError[i] for neuronError in next_layer_errors])

            neuron.update_weights(inputs, [neuronError[i] for neuronError in next_layer_errors],gamma)

class finalLayer:
    def __init__(self, num_inputs, activation_function=sigmoid):
        self.activation_function = activation_function
        self.neurons = [finalNeuron(num_inputs, random.random(), activation_function=activation_function)]

    def feedforward(self, inputs):
        return [self.neurons[0].feedforward(inputs)]
    
    def backpropagate(self, inputs, actual, gamma):
        predicted = self.neurons[0].activation
        self.neurons[0].update_weights(inputs, predicted, actual,gamma)

#neural network creation functions

def initialize_network(num_inputs=1):
    return [inputLayer(num_inputs)]

def add_layer(network, num_neurons, activation_function=sigmoid):
    if len(network) == 0:
        print("Inputs not initialized")
        return
    else:
        return network.append(layer([neuron(len(network[-1].neurons), random.random()) for _ in range(num_neurons)], activation_function=activation_function))
    
def finalize_network(neural_network, activation_function=sigmoid):
    return neural_network.append(finalLayer(len(neural_network[-1].neurons), activation_function=activation_function))


#neural network base functions

def predict(neural_network, inputs):

    if(neural_network == None):
        print("Please insert a valid network")
        return

    for l in neural_network:
        inputs = l.feedforward(inputs)
    return inputs

def backpropagation(network, inputs, actual, gamma):

    if(network == None):
        print("Please insert a valid network")
        return

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
            layer.backpropagate(prev_outputs, actual, gamma)
        else:
            next_layer_errors = [neuron.weightsderivative for neuron in network[i + 1].neurons]
            layer.backpropagate(prev_outputs, next_layer_errors, gamma)
    
    return error

def backpropagation_test(neural_network, gamma):

    neural_network

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
            layer.backpropagate(prev_outputs, actual, gamma)
        else:
            next_layer_errors = [neuron.weightsderivative for neuron in network[i + 1].neurons]
            layer.backpropagate(prev_outputs, next_layer_errors, gamma)
 
#neural network train functions

def train(network, train_data, gamma=0.1):

    for index in range(len(train_data)):
        row = train_data.iloc[index].tolist()
        inputs = row[:-1]
        actual = float(row[-1])

        error = backpropagation(network, inputs, actual, gamma)
        printProgressBar(index,len(train_data),suffix=f"\n Cost:{error}")

    return network

def train_epoch(neural_network, train_data, epoch = 100000, gamma=0.1):

    for i in range(epoch):
        row = train_data.iloc[int(random.random() * len(train_data))].tolist()
        inputs = row[:-1]
        actual = float(row[-1])

        if(i%100 == 0):
            error = backpropagation(neural_network, inputs, actual, gamma)
            printProgressBar(i,epoch,suffix=f"\n Cost:{error}")
        else:
            error = backpropagation(neural_network, inputs, actual, gamma)
    
    return neural_network

def train_once(neural_network,train_data,gamma=0.1):
    index = random.randint(0, len(train_data) - 1)
    row = train_data.iloc[index].tolist()
    inputs = row[:-1]
    actual = float(row[-1])
    backpropagation(neural_network, inputs, actual,gamma)

def batch_train(neural_network, train_data, batch_size=10,gamma=0.1): #DOES NOT WORK

    for index in range(0, len(train_data), batch_size):
        batch = train_data.iloc[index:index + batch_size]
        for _, row in batch.iterrows():
            inputs = row[:-1].tolist()
            actual = int(row.tolist()[-1])
            backpropagation(neural_network, inputs, actual,gamma)

#neural network test functions

def test(neural_network, test_data, confidence_threshold):

    accuracy = 0

    for index in range(len(test_data)):

        row = test_data.iloc[index].tolist()
        inputs = row[:-1]
        actual = float(row[-1])

        predicted = sum(predict(neural_network, inputs))

        cost = cost_function(predicted, actual)

        print(f"Predicted: {predicted}, Actual: {actual}, cost: {cost}")

        accuracy += 1 if abs(predicted - actual) < confidence_threshold else 0

    return (accuracy)
 
#save & load functions

def save_model(neural_network, filename="NeuralNetwork/model.net"):

    #debug
    print_weights(neural_network)


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
    print_weights(neural_network)

    return neural_network

#debug functions

def print_weights(neural_network):

    if(type(neural_network) == None):
        print("Neural network is empty!!!")

    for l in neural_network:
        print(f"-------------- Layer {neural_network.index(l)} -------------- \n")
        for n in l.neurons:
            print(f"  Neuron {l.neurons.index(n)}:")
            print(n.weights)
            print(n.bias)
            print(f"  Activation: {n.activation} \n"
                  f"  Error Derivative: {n.error_derivative} \n")


