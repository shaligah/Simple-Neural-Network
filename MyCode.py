#importing the necessary libraries
import math
import csv
import random

#creating the neuron class
class Neuron():
    def __init__(self, num_of_weights,weights_positioning, gradient):
        self.num_of_weights = num_of_weights
        self.weights = [0,0]
        self.weights_positioning = weights_positioning
        self.gradient = gradient
        self.activation_value = 0
        self.delta_weights = [0]*num_of_weights
        
        for i in range(num_of_weights):
            self.weights[i] =random.random()
            
    
    def Matrixmult(self, layer_number):
        neurons = 0
        for i in range(0,len(layer_number)):
            neurons += layer_number[i].activation_value * layer_number[i].weights[self.weights_positioning]
            self.activation_value = 1/(1 + (math.exp(-0.8 * neurons)))

            
first_layer = [Neuron(2,0,0),Neuron(2,1,0),Neuron(2,2,0)]
second_layer = [Neuron(2,0,0), Neuron(2,1,0)]
third_layer = [Neuron(2,0,0), Neuron(2,1,0)]

#creating the feedforward function for implementation
def feedforwarding(inputs):
    for i in range(len(inputs)+1):
        if (i==len(inputs)):
            first_layer[i].activation_value = 1
        else:
            first_layer[i].activation_value = inputs[i]
    
    for m in range(len(second_layer)):
        second_layer[m].Matrixmult(first_layer)
            
    for n in range(len(third_layer)):
            third_layer[n].Matrixmult(second_layer)
    
    return [third_layer[0].activation_value,third_layer[1].activation_value]

#creating the backpropagation function
def backpropagation(targets):
    momentum = 0.1
    learning_rate = 0.9
    error = []
    
    for i in range(len(targets)):
        m = targets[i] - third_layer[i].activation_value
        error.append(m)
    
    for i in range(len(third_layer)):
        third_layer[i].gradient =  learning_rate * third_layer[i].activation_value * (1.0 - third_layer[i].activation_value) * error[i]
    
    for j in range(len(second_layer)):
        second_layer[j].gradient = learning_rate * second_layer[j].activation_value * (1.0 - second_layer[j].activation_value) * ((third_layer[0].gradient *second_layer[j].weights[0]) + (third_layer[1].gradient *second_layer[j].weights[1]))
    
    for k in range(len(second_layer)):
        for l in range(len(third_layer)):
            second_layer[k].delta_weights[l] = learning_rate * third_layer[l].gradient * second_layer[k].activation_value + (momentum * second_layer[k].delta_weights[l])
            
    for m in range(len(first_layer)):
        for n in range(len(second_layer)):
            first_layer[m].delta_weights[n] = learning_rate * second_layer[n].gradient * first_layer[m].activation_value + (momentum * first_layer[m].delta_weights[n])
    
    for o in range(len(second_layer)):
        for p in range(len(third_layer)):
            second_layer[o].weights[p] += second_layer[o].delta_weights[p]
    
    for q in range(len(first_layer)):
        for r in range(len(second_layer)):
            first_layer[q].weights[r] = first_layer[q].weights[r] + first_layer[q].delta_weights[r]

#creating the training function for learning
def training():
    epochs = 800
    i = 0
    while i < epochs:
        count = 0
        total_error1 = 0
        with open('Training Data.csv', newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in data:
                inputs = [row[0],row[1]]
                outputs = [row[2],row[3]]
                feedforwarding(inputs)
                backpropagation(outputs)
                error1 = (row[2] - third_layer[0].activation_value)
                error2 = (row[3] - third_layer[1].activation_value)
                total_error = (error1 + error2)**2
                rms = math.sqrt(total_error/2)
                count = count + 1
                total_error1 = total_error1 + rms
                real = total_error1 / count
            print(real)
            while i>200:                                #creating a stopping criteria
                if real -real<=0.00000000000000012:
                    break
        i = i + 1

#creating a test function to learn whether the network has learned
def validation():
    count = 0
    total_error1 = 0
    with open('NN V Data.csv', newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in data:
            inputs = [row[0],row[1]]
            outputs = [row[2],row[3]]
            feedforwarding(inputs)
            error1 = (row[2] - third_layer[0].activation_value)
            error2 = (row[3] - third_layer[1].activation_value)
            total_error = (error1 + error2)**2
            rms = math.sqrt(total_error/2)
            count = count + 1
            total_error1 = total_error1 + rms
            real = total_error1 / count
        print(real)

#create a function that saves the updated weights into a csv file
def saveweights():
    weights = []
    for m in range(len(first_layer)):
        weights.append(first_layer[m].weights)
    for n in range(len(second_layer)):
        weights.append(second_layer[n].weights)
    with open('Final weights.csv', 'w', newline= '') as file:
        p = csv.writer(file,delimiter=',')
        p.writerows(weights)
   
#creating a function tha calls the saved weights from the NN for implementation into Lunar Landing game 
def weightsfunc():
    with open('Final weights.csv', 'r', newline= '') as filename:
        data = list(csv.reader(filename, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
        for i in range(len(first_layer)):
            first_layer[i].weights = data[i]
            for j in range(len(second_layer)):
                second_layer[j].weights = data[len(first_layer)+j]




