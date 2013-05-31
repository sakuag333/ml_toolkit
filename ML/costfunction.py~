import numpy
import sys

def sigmoid(mat):
    mat = numpy.exp(-1*mat)
    mat = 1 + mat
    mat = 1/mat
    return mat

def logistic_regression(theta,x,y,regularisation,parameter):
    m = x.shape[0]
    tmp = sigmoid(numpy.dot(x,theta))
    tmp = -1*(numpy.dot(numpy.transpose(numpy.log(tmp)),y) + numpy.dot(numpy.transpose(numpy.log(1-tmp)),1-y))
    if regularisation:
        var = 0
        for i in range(1,theta.shape[0]):
            var += (theta[i])*(theta[i])
        tmp = tmp + (parameter/2)*(var)
    return tmp/m

