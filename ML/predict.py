import sys
import numpy

def logistic_regression(theta,x):
    prediction = numpy.zeros((x.shape[0],1))
    tmp = numpy.dot(x,theta)
    for i in range(0,tmp.shape[0]):
        if tmp[i][0] >= 0:
            prediction[i][0] = 1
        else:
            prediction[i][0] = 0
    return prediction
        
