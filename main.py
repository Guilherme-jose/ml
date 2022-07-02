from random import random, randrange
import neuralNetwork as nn

nn = nn.NeuralNetwork(2,10,1)

input = [[0,0],[0,1],[1,0],[1,1]]
output = [[0],[1],[1],[0]]

print(nn.guess(input[0]))
print(nn.guess(input[1]))
print(nn.guess(input[2]))
print(nn.guess(input[3]))

for i in range(100000):
    k = randrange(0,4)
    nn.train([input[k]], [output[k]])
    #if(i%1000==0):
        #print(i/1000)
    
print(nn.guess(input[0]))
print(nn.guess(input[1]))
print(nn.guess(input[2]))
print(nn.guess(input[3]))