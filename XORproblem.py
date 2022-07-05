from math import floor
from random import random, randrange
import sys
import neuralNetwork as nn
import pygame

nn = nn.NeuralNetwork([2,2,1])

def print2p(x):
    print("%.2f" % x)
    
input = [[0,0],[0,1],[1,0],[1,1]]
output = [[0],[1],[1],[0]]
print(nn.bias)
print2p(nn.guess(input[0])[0][0])
print2p(nn.guess(input[1])[0][0])
print2p(nn.guess(input[2])[0][0])
print2p(nn.guess(input[3])[0][0])
print("--------------------")

for i in range(10000):
    k = randrange(0,4)
    nn.train([input[k]], [output[k]])
    if(i%1000==0):
        print(i/1000)
    
pygame.init()

size = width, height = 960, 960
speed = [2, 2]
black = 0, 0, 0

screen = pygame.display.set_mode(size)

screen.fill(black)
for i in range(100):
    for j in range(100):
        v = nn.guess([float(i)/100,float(j)/100])
        k = 255*v[0][0]
        k = max(0, k)
        #print(k)
        pygame.draw.circle(screen,[55,0,k,k],[5*i+1,5*j+1],3)
            
print2p(nn.guess(input[0])[0][0])
print2p(nn.guess(input[1])[0][0])
print2p(nn.guess(input[2])[0][0])
print2p(nn.guess(input[3])[0][0])

print(nn.bias)
while(True):
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
    
    pygame.display.flip()

