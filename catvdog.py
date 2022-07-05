from random import randrange
from PIL import Image
import numpy as np
import neuralNetwork as nn

def by255(x):
    return x/255
        
print("initializing weights")
nn = nn.NeuralNetwork([5250, 5000, 3000, 1000, 200,2])

trainingSet = []
trainingOutput = []

print("training")
for i in range(1000):
    if(randrange(0,2) == 0):
        image = Image.open('training_set_small/cats/cat.' + str(randrange(1,4001)) + '.jpg')
        data = np.asarray(image)
        data = np.reshape(data, 5250)
        data = data.tolist()
        for it in data:
            it = by255(it)
        nn.train([data], [[1,0]])
        print(i)
    else:
        image = Image.open('training_set_small/dogs/dog.' + str(randrange(1,4001)) + '.jpg')
        image = Image.eval(image, by255)
        data = np.asarray(image)
        data = np.reshape(data, 5250)
        data = data.tolist()
        for it in data:
            it = by255(it)
        nn.train([data], [[0,1]])
        print(i)

temp = nn.weights
s = ''.join(str(x) for x in temp)
print(s)

for i in range(100):
    if(randrange(0,2) == 0):
        image = Image.open('training_set_small/cats/cat.' + str(randrange(1,4001)) + '.jpg')
        data = np.asarray(image)
        data = np.reshape(data, 5250)
        data = data.tolist()
        print(nn.guess(data), "1 0")
    else:
        image = Image.open('training_set_small/dogs/dog.' + str(randrange(1,4001)) + '.jpg')
        data = np.asarray(image)
        data = np.reshape(data, 5250)
        data = data.tolist()
        print(nn.guess(data), ("0 1"))
    
print("finished")

