# -*- coding: utf-8 -*-
"""
Created on Sun Feb 02 10:46:48 2014

@author: ryan
"""

import pybrain.datasets
import pybrain.tools.neuralnets
import pybrain.structure.networks
import pybrain.structure.modules
import pybrain.structure.connections
import pybrain.supervised.trainers
import numpy as np

def convertDataNeuralNetwork(x, y):
    colx = 1 if len(np.shape(x))==1 else np.size(x, axis=1)
    coly = 1 if len(np.shape(y))==1 else np.size(y, axis=1)
    
    fulldata = pybrain.datasets.SupervisedDataSet(colx, coly)

    for d, v in zip(x, y):
        fulldata.addSample(d, v)
    
    return fulldata

def simpleneuralnet(x, y, layers, nodes, epochstillshuffel = 200, maxEpochs = 5000, learningrate = 0.01, lrdecay = 1.0, momentum = 0., epochsperstep = 100):
    """Uses the pybrain library to train a new neural network
    
    Args:
        x (ndarray):             2d array of input values for training
        y (ndarray):             2d array of output values for training
        layers (int):            number of hidden layers in neural network
        nodes (int):             number of nodes in each hidden layer
        epochstillshuffel (int): number of training cycles before the training and validating sets are reshuffeled. Defaults to 200.
        maxEpochs (int):         total number of training cycles. Defaults to 5000
        learningrate (float):    how fast training occurs. Defaults to 0.01
        lrdecay (float):         decay in learning rate over time. Defaults to 1.0
        momentum (float):        resistance to changes in network learning
        epochsperstep (int):     step in training cycles
        
    Returns:
        pybrain neural network: trained network that can be activated with new x values
    
    TODO: make this a class
    """
    
    print 'layers = '+str(layers)+', nodes = '+str(nodes)
    fulldata = convertDataNeuralNetwork(x, y)
    
    regressionTrain, regressionTest = fulldata.splitWithProportion(.75)
    fnn = pybrain.structure.networks.FeedForwardNetwork()
    
    inLayer = pybrain.structure.modules.LinearLayer(regressionTrain.indim)
    outLayer = pybrain.structure.modules.LinearLayer(regressionTrain.outdim)
    
    hiddenlayers = []
    for l in range(layers):
        hiddenlayers.append(pybrain.structure.modules.SigmoidLayer(nodes))

    fnn.addInputModule(inLayer)
    fnn.addOutputModule(outLayer)
    
    for hiddenLayer in hiddenlayers:
        fnn.addModule(hiddenLayer)
    
    in_to_hidden = pybrain.structure.connections.FullConnection(inLayer, hiddenlayers[0])
    hidden_connections = []
    for l in range(1, layers):
        hidden_connections.append(pybrain.structure.connections.FullConnection(hiddenlayers[l-1], hiddenlayers[l]))
    hidden_to_out = pybrain.structure.connections.FullConnection(hiddenlayers[-1], outLayer)
    
    fnn.addConnection(in_to_hidden)
    for connection in hidden_connections:
        fnn.addConnection(connection)
    fnn.addConnection(hidden_to_out)
    
    fnn.sortModules()
    
    trainer =  pybrain.supervised.trainers.BackpropTrainer(fnn, dataset=regressionTrain, verbose=False, learningrate = learningrate, lrdecay = lrdecay, momentum = momentum)
    
    epochcount = 0
    while epochcount<maxEpochs:
        trainer.trainUntilConvergence(maxEpochs = epochsperstep)
        epochcount+=epochsperstep
        if epochcount%epochstillshuffel==0:
            print str(epochcount) + ' epochs done of ' + str(maxEpochs)
            yhat = np.transpose(fnn.activateOnDataset(fulldata))[0]
            if np.any(np.isnan(yhat)):
                print "NaN detected - training stopped"
                break
            print finderrors(y, yhat)
            print 'reshuffeling..'
            regressionTrain, regressionTest = fulldata.splitWithProportion(.75)
            trainer.setData(regressionTrain)
    
    return fnn

def finderrors(y, output):
    errors = {}
    errors['mse'] = sum((y-output)**2)/len(y)
    errors['max'] = max(abs(y-output))
    errors['p50'] = np.percentile(abs(y-output), 50)
    errors['p90'] = np.percentile(abs(y-output), 90)
    return errors
