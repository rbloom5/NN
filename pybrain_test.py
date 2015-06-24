#!/usr/bin/python

import pybrain
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

# from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import numpy as np
# reload(pybrain)
# from pybrain.tools.shortcuts import buildNetwork
print '##################################################################'
print '##################################################################'
print '##################################################################'

fnn = pybrain.structure.networks.FeedForwardNetwork()
inLayer = pybrain.structure.LinearLayer(4)
fnn.addInputModule(inLayer)

outLayer = pybrain.structure.SoftmaxLayer(3)
fnn.addOutputModule(outLayer)

hiddenLayer = pybrain.structure.SigmoidLayer(2)
fnn.addModule(hiddenLayer)

hidden_to_out = pybrain.structure.connections.FullConnection(hiddenLayer,outLayer)
fnn.addConnection(hidden_to_out)


# fnn.addConnection(pybrain.structure.connections.FullConnection(inLayer,hiddenLayer))#,\


fnn.addConnection(pybrain.structure.connections.CustomFullConnection(inLayer,hiddenLayer,\
                                        # inSliceFrom=0, inSliceTo=2, \
                                        inSliceIndices=[0,1], outSliceIndices=[0]       ))
fnn.addConnection(pybrain.structure.connections.CustomFullConnection(inLayer,hiddenLayer,\
                                        # inSliceFrom=2, inSliceTo=4, \
                                        inSliceIndices=[2,3], outSliceIndices=[1]       ))

fnn.sortModules()

print fnn.activate([2, .5, .9, .1])
print fnn.activate([5, 3, 9, .1])


means = [(-1,0),(2,4),(3,1)]
cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
alldata = ClassificationDataSet(4, 1, nb_classes=3)
for n in xrange(3):
    for klass in range(3):
        i1 = multivariate_normal(means[klass],cov[klass])
        i2 = multivariate_normal(means[klass],cov[klass])
        input = np.concatenate((i1,i2),axis=1)
        # print input
        alldata.addSample(input, [klass])

alldata._convertToOneOfMany( )  

trainer = BackpropTrainer( fnn, dataset=alldata, momentum=0.1, verbose=True, weightdecay=0.01)

       




trainer.trainEpochs( 1000 )
















