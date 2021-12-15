import math
import numpy as np
from Layer import *
from GradDescType import *
from sklearn.utils import shuffle 

class Network(object):
    '''
    The Network class contains a list of layers
    and provides the train and evaluate functions
    '''
    def __init__(self, X,Y, numLayers, dropOut=1.0, activationF=ActivationType.SIGMOID,
                 lastLayerAF=ActivationType.SIGMOID):
        self.X = X
        self.Y = Y
        self.numLayers = numLayers 
        self.Layers = [] # network has multiple layers 
        self.lastLayerAF = lastLayerAF
        for i in range(len(numLayers)):
            if (i == 0):    # First Layer 
                layer = Layer(numLayers[i], X.shape[1], False, dropOut,activationF)
            elif (i == len(numLayers) - 1):   # last Layer
                layer = Layer(Y.shape[1], numLayers[i-1], True, dropOut, lastLayerAF)
            else:   # intermediate layer
                layer = Layer(numLayers[i], numLayers[i-1], False, dropOut, activationF)
            self.Layers.append(layer)


    def Evaluate(self, indata, decision_plotting=0):   # evaluate all layers
        self.Layers[0].Evaluate(indata)
        for i in range(1,len(self.numLayers)):
            self.Layers[i].Evaluate(self.Layers[i-1].a)
        return self.Layers[len(self.numLayers)-1].a 

    def Train(self, epochs, learningRate, lambda1, gradDescType, batchSize=1):
        for j in range(epochs):
            error = 0
            self.X, self.Y = shuffle(self.X, self.Y, random_state=0)
            for i in range(self.X.shape[0]):
                self.Evaluate(self.X[i])
                if (self.lastLayerAF == ActivationType.SOFTMAX):
                    error += -(self.Y[i] * np.log(self.Layers[len(self.numLayers)-1].a+0.001)).sum()
                else:
                    error += ((self.Layers[len(self.numLayers)-1].a - self.Y[i]) * \
                        (self.Layers[len(self.numLayers)-1].a - self.Y[i])).sum()
                lnum = len(self.numLayers) -1  # last layer number 

                #compute deltas, grads on all layers 
                while(lnum >= 0):
                    if (lnum == len(self.numLayers)-1):   # last layer
                        if (self.lastLayerAF == ActivationType.SOFTMAX):
                            self.Layers[lnum].delta = -self.Y[i]+self.Layers[lnum].a
                        else:
                            self.Layers[lnum].delta = -(self.Y[i]-self.Layers[lnum].a) * self.Layers[lnum].derivAF
                    else: # intermediate layer
                        self.Layers[lnum].delta = np.dot(self.Layers[lnum+1].W.T, self.Layers[lnum+1].delta) * self.Layers[lnum].derivAF

                    if (lnum > 0):   # previous output 
                        prevOut = self.Layers[lnum-1].a
                    else:
                        prevOut = self.X[i]

                    self.Layers[lnum].WGrad += np.dot(self.Layers[lnum].delta,prevOut.T)
                    self.Layers[lnum].bGrad += self.Layers[lnum].delta
                    lnum = lnum-1 

                if (gradDescType == GradDescType.MINIBATCH):
                    if (i % batchsize == 0):
                        self.UpdateGradsBiases(learningRate, lambda1, batchSize)
                
                if (gradDescType == GradDescType.STOCHASTIC):
                  self.UpdateGradsBiases(learningRate,lambda1,1)

                if (gradDescType == GradDescType.BATCH):
                    self.UpdateGradsBiases(learningRate, lambda1, self.X.shape[0])
                print("Iter = " + str(j) + " Error = "+ str(error))

    def UpdateGradsBiases(self, learningRate, lambda1, batchSize):
        # update weight and biases for all layers
        for ln in range(len(self.numLayers)):
            self.Layers[ln].W = self.Layers[ln].W - learningRate * (1/batchSize) * self.Layers[ln].WGrad - \
                learningRate * lambda1 * self.Layers[ln].W
            self.Layers[ln].b = self.Layers[ln].b - learningRate * (1/batchSize) * self.Layers[ln].bGrad
            self.Layers[ln].ClearWBGrads()




     