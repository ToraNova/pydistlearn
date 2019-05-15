# Plasma_test script
# Toranova mailto: chia_jason96@live.com
# May 2019

# This is a module for the central
# defines a class which uses the ridge regressor
# the class may also read locally for testing
# purposes

import pyplasma.pyplasma as pyplasma
from preproc.aux import read_csvfile
import numpy

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

    alpha = pyplasma.ridge_solve( 
            x_mat.flatten(), xrowsz, xcolsz,
            y_vct.flatten(), yelmsz, ydimsz,
            lval, 0,
            xrowsz * ydimsz
            )
    print(alpha)
            
