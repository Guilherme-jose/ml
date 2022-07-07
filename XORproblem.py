from math import floor
from random import random, randrange
import sys
import neuralNetwork as nn
import pygame

def print2p(x):
    print("%.2f" % x)
    
def showSpace(resolution):
    screen.fill(black)
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
    for i in range(resolution):
        for j in range(resolution):
            v = nn.guess([float(i)/resolution,float(j)/resolution])
            k = 255*v[0][0]
            k = max(0, k)
            k = min(255, k)
            pygame.draw.circle(screen,[55,0,k,255],[(500/resolution)*i+1,(500/resolution)*j+1],300/resolution)
    pygame.display.flip()

def solveXOR(nn):
    nn.reinit([2,2,1])
    input = [[0,0],[0,1],[1,0],[1,1]]
    output = [[0],[1],[1],[0]]
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
            showSpace(20)
                
    print2p(nn.guess(input[0])[0][0])
    print2p(nn.guess(input[1])[0][0])
    print2p(nn.guess(input[2])[0][0])
    print2p(nn.guess(input[3])[0][0])

    showSpace(100)
    
pygame.init()

size = width, height = 500, 500
speed = [2, 2]
black = 0, 0, 0

screen = pygame.display.set_mode(size)

nn = nn.NeuralNetwork([2,2,1])
solveXOR(nn)
    

while(True):
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
        if event.type == pygame.KEYDOWN: solveXOR(nn)
    
    pygame.display.flip()

