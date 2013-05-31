import sys
import numpy

def sigmoid(mat):
    mat = numpy.exp(-1*mat)
    mat = 1 + mat
    mat = 1/mat
    return mat

def logistic_regression(theta,x,y):        
    m = x.shape[0]
    tmp = sigmoid(numpy.dot(x,theta))
    tmp = tmp - y
    tmp = numpy.dot((numpy.transpose(x)),tmp)
    return tmp/m
    
