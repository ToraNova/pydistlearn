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

if __name__ == "__main__":
    pyplasma.test()
    pyplasma.pputil_test()

    lval = 1.0 #lambda

    csvfile = '/home/cjason/library/python/pysample/datagen/data0.csv'
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
    pyplasma.pputil_ridge( 
            x_mat.flatten(), xrowsz, xcolsz,
            y_vct.flatten(), yelmsz, ydimsz,
            lval,
            alpha
            )
            
