# central.py
# central class definition
# Toranova mailto: chia_jason96@live.com
# May 2019

# This is a module for the central
# defines a class which uses the ridge regressor
# the class may also read locally for testing
# purposes

import pyplasma.pyplasma as pyplasma
from aux import read_csvfile
import numpy

# allow this module to doubles as a python script
# shows certain example how to use it locally

# this allows declarations of abstract methods/classes
from abc import ABC, abstractmethod
# This is the abstract base class for all central objects
# it implements some shared and common function across
# all central objects and declares some methods that
# should be implemented (ABC - abstract base class)
class ConceptCentral(ABC):
    pass


    


if __name__ == "__main__":
    pyplasma.test()
    pyplasma.ridge_test()

    lval = 1.0 #lambda

    csvfile = '/home/cjason/library/guides/python/pysample/datagen/data0.csv'
    x_mat, y_vct, rowcount = read_csvfile(csvfile) 
    xrowsz = x_mat.shape[0]
    xcolsz = x_mat.shape[1]
    yelmsz = y_vct.shape[0]
    ydimsz = y_vct.shape[1] if len(y_vct.shape) > 1 else 1 

    print("Performing Sanity checks")
    print("Read :",csvfile)
    print("Row count :",rowcount)
    print("xmat dim (x,y) : ", x_mat.shape) 
    print("yvct dim (x,y) : ", y_vct.shape)
    # perform some conversion

    alpha = xrowsz * ydimsz
    pyplasma.ridge_solve( 
            x_mat.flatten(), xrowsz, xcolsz,
            y_vct.flatten(), yelmsz, ydimsz,
            lval, 0,
            alpha
            )
            
