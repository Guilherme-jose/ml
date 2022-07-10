from random import randrange
from PIL import Image
import numpy as np
import neuralNetwork as nn
import activationFunctions

def by255(x):
    return x/255
        
print("initializing weights")
nn = nn.NeuralNetwork(840)
nn.addDenseLayer(1024, activationFunctions.sigmoid, activationFunctions.sigmoidD)
nn.addDenseLayer(2, activationFunctions.sigmoid, activationFunctions.sigmoidD)

trainingSet = []
trainingOutput = []
for i in range(1,4001):
    image = Image.open('training_set_small/cats/cat.' + str(i) + '.jpg')
    data = list(image.getdata())
    image.close()
    data = np.reshape(data, 840)
    data = [by255(i) for i in data]
    trainingSet.append(data)
    trainingOutput.append([1,0])
    
    image = Image.open('training_set_small/dogs/dog.' + str(i) + '.jpg')
    data = list(image.getdata())
    image.close()
    data = np.reshape(data, 840)
    data = [by255(i) for i in data]
    trainingSet.append(data)
    trainingOutput.append([0,1])
    
print("training")

nn.train(trainingSet,trainingOutput,10, "classifier")
    
print("finished")

