import config
import readcsv
import datacleaning
import featurescaling
import algorithm
import predict
import writecsv
import numpy

data = readcsv.readcsv(config.trainingset)
config.data = data
config.row_ = data.shape[0]
config.col_ = data.shape[1]

datacleaning.cabin(config.data,10)                   #Problem specific data cleaning
datacleaning.gender(config.data,3)                   #Problem specific data cleaning
feature = config.allowed_feature
config.cdata = datacleaning.reduce_features(config.row_,len(feature),feature,config.data)
config.cdata = datacleaning.polynomial_feature_cubic(config.cdata)
crow_ = (config.cdata).shape[0]
ccol_ = (config.cdata).shape[1]

featurescaling.featurescaling(config.cdata)

feature = config.outputlabel
y = datacleaning.get_output_label(config.row_,feature,config.data)
config.y  =y
theta = numpy.zeros((ccol_,1))

print "training....."

theta,cost = algorithm.logistic_regression(config.repeat,theta,config.cdata,y,config.alpha,True,config.regularisation)
config.cost = cost
config.theta = theta
print config.cost

data = readcsv.readcsv(config.testset)
config.data = data
config.row_ = data.shape[0]
config.col_ = data.shape[1]

datacleaning.cabin(config.data,9)       #Problem specific data cleaning 
datacleaning.gender(config.data,2)      #Problem specific data cleaning 
feature = config.allowed_feature_testdata
config.cdata = datacleaning.reduce_features(config.row_,len(feature),feature,config.data)
config.cdata = datacleaning.polynomial_feature_cubic(config.cdata)
crow_ = config.cdata.shape[0]
ccol_ = config.cdata.shape[1]


featurescaling.featurescaling(config.cdata)

print "predicting......"

config.prediction = predict.logistic_regression(theta,config.cdata)

writecsv.writecsv_modified(config.prediction,config.outputfile,config.testset)  

