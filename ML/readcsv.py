import numpy
import csv
import sys

def readcsv(filename):
    data = []
    csv_file_object = csv.reader(open(filename, 'rb'))     
    header = csv_file_object.next() 

    for row in csv_file_object: 
        data.append(row) 

    data = numpy.array(data)
    return data






