from random import random, randrange
import neuralNetwork as nn

nn = nn.NeuralNetwork([2,100,100,100,100,1])

def print2p(x):
    print("%.2f" % x)
    
input = [[0,0],[0,1],[1,0],[1,1]]
output = [[0],[1],[1],[0]]

print2p(nn.guess(input[0])[0][0])
print2p(nn.guess(input[1])[0][0])
print2p(nn.guess(input[2])[0][0])
print2p(nn.guess(input[3])[0][0])
print("--------------------")

for i in range(100000):
    k = randrange(0,4)
    nn.train([input[k]], [output[k]])
    #if(i%1000==0):
        #print(i/1000)
    
print2p(nn.guess(input[0])[0][0])
print2p(nn.guess(input[1])[0][0])
print2p(nn.guess(input[2])[0][0])
print2p(nn.guess(input[3])[0][0])