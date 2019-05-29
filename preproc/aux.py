# Highly specialized lib file, do not use for other dataset.
# This file is meant for lookup tables for categorical value quantisation.
import csv, numpy

import pyioneer.support.gpam as gpam
pam = gpam.getgpam()
# This function is used to decide on the batches
def evaluate_bsize( esize, bfactor ):
    if(bfactor > esize):
        # bad user again
        pam.warn("Bad user. bfactor > esize!")
        return int( esize )
    if(bfactor > 1):
        # this uses bfactor as a literal numeric
        return int( bfactor )
    elif(bfactor > 0):
        # this uses bfactor as a proportion to esize
        return int( esize * bfactor )
    else:
        # this means the user has made a mistake.
        pam.warn("Bad user. bfactor < 0")
        return int(-bfactor)

def find_minesz( negformlist ):
    '''finds the minimum entry size amongst the lists to
    allow batching nicely without errors'''
    try:
        mincand = negformlist[0].primary.get('esize')
        for f in negformlist:
            if( f.primary.get('esize') < mincand ):
                mincand = f.primary.get('esize')
        pam.debug("minimum candidate is",mincand)
        return mincand
    except Exception as e:
        pam.expt(str(e))



# This function is used to read
def read_csvfile(filename,compd=numpy.double,limiter=0,skips=0):
    # dataset preparation
    with open(filename) as dataset:

        reader = csv.reader(dataset,delimiter=';', quotechar='"')
        for index in range(skips):
            next(reader)
        rin_list = list(reader) #read in the csv file, skipping the first
        if(limiter>0):
            rowcount = len( rin_list ) if len( rin_list ) <= limiter else limiter
        else:
            rowcount = len( rin_list )
        x_mat = []
        y_vct = []
        #x_mat = numpy.zeros(dtype=compd,shape=(rowcount,f_size)) #specifically choose 13 vectors only
        #y_vct = numpy.zeros(dtype=numpy.bool,shape=(rowcount,1))

        for index,row in enumerate(rin_list):
            #print("Registering row",index,row)
            if(index>=limiter and limiter > 0):
                break

            x_mat.append([float(i) for i in row[:-1]])
            y_vct.append(float(row[-1]))

        x_mat = numpy.array(x_mat, dtype=compd)
        y_vct = numpy.array(y_vct, dtype=compd)

    return x_mat,y_vct,rowcount
