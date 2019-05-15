# the controller class controls data preprocessing
# to reduce code sizes on the central/donor class
# definition and improve maintainability

from pyioneer.variable import HomoCSVDataController
from sklearn import preprocessing
import numpy
from . import dist_matrix as dm

class NPDController(HomoCSVDataController):
    '''inhertis from the HomoCSVDataController class. deals with homogeneous data (naive)'''
    _compd = numpy.double
    _default_readkey = "__mm"
    _typevar0 = "__d"
    _typevar1 = "__t"
    _batchvar0 = "__b0"
    _batchvar1 = "__b1"

    _kernel_readkey = "__k"

    _typeTdim = 1

    hasTarget = False

    def __init__(self, verbose=False, debug=False, owarn=False):
        super().__init__(verbose=verbose,debug=debug, owarn=owarn)

    def read(self,filename, hasTarget, htype=float, dname=_default_readkey, dtype = _compd,
            skipc = 0, adelimiter=';', aquotechar ='"'):
        '''overrides the read from the base class, allows partitioning
        and ensures final results are numpy arrays'''
        self.debug("Reading file",filename)
        super().read(filename,htype,dname,skipc,adelimiter,aquotechar)
        self.hasTarget = hasTarget
        # converts to python obj
        if( self.isRead(dname) ):
            try:
                #no need to split
                self.load(dname, numpy.array( super().get(dname) ))#numpy array transformation
                if( hasTarget ):
                    #perform splitting
                    self._mp[ dname+self._typevar0 ], self._mp[ dname+self._typevar1 ] =\
                            dm.vsplit_mat( self._mp[ dname ], -self._typeTdim )
                else:
                    self.load( dname+self._typevar0, super().get(dname))
            except Exception as e:
                self.expt(str(e))
                super().unload( dname )

    def size(self, dname = _default_readkey):
        '''obtains the size of dname and it's batched data, returns a dictionary of sizes'''
        out = {"data": super().get( dname+self._typevar0 ).shape}
        if( self.isBatched(dname) ):
            out["data0"]= super().get( dname+self._typevar0+self._batchvar0 ).shape
            out["data1"]= super().get( dname+self._typevar0+self._batchvar1 ).shape
        if( self.hasTarget ):
            out["target"] = self._mp[ dname+self._typevar1].shape
            if( self.isBatched(dname) ):
                out["target0"]=super().get( dname+self._typevar1+self._batchvar0).shape
                out["target1"]=super().get( dname+self._typevar1+self._batchvar1).shape
        return out

    def show(self, dname = _default_readkey):
        '''display the loaded datasets in the controller'''
        if( self.isRead(dname) ):
            if( self.isBatched(dname)):
                self.verbose("Displaying batched up data")
                self.info("Shape of {}/{}".format(self._typevar0+self._batchvar0,
                    self._typevar0+self._batchvar1),
                    super().get( dname+self._typevar0+self._batchvar0).shape,"/",
                    super().get( dname+self._typevar0+self._batchvar1).shape)
                self.info(dname,"mean of",self._typevar0) 
                self.raw(super().get( dname+self._typevar0).mean(axis=0) )
                self.info(dname,"stdv of",self._typevar0) 
                self.raw(super().get( dname+self._typevar0).std(axis=0) )
                if( self.hasTarget ):
                    self.info("Shape of {}/{}".format(self._typevar1+self._batchvar0,
                        self._typevar1+self._batchvar1),
                        super().get( dname+self._typevar1+self._batchvar0).shape,"/",
                        super().get( dname+self._typevar1+self._batchvar1).shape)
                    self.info(dname,"mean of",self._typevar1) 
                    self.raw(super().get(dname+self._typevar1).mean(axis=0) )
                    self.info(dname,"stdv of",self._typevar1) 
                    self.raw(super().get(dname+self._typevar1).std(axis=0) )
            else:
                self.verbose("Displaying read data")
                self.info("Shape of",self._typevar0,
                    super().get(dname+self._typevar0).shape)
                #self.display(dname+self._typevar0,raw=True)
                self.info(dname,"mean of",self._typevar0) 
                self.raw( super().get(dname+self._typevar0).mean(axis=0) )
                self.info(dname,"stdv of",self._typevar0) 
                self.raw( super().get(dname+self._typevar0).std(axis=0) )
                if( self.hasTarget ):
                    self.info("Shape of",self._typevar1,
                        super().get(dname+self._typevar1).shape)
                    #self.display(dname+self._typevar1,raw=True)
                    self.info(dname,"mean of",self._typevar1) 
                    self.raw(super().get(dname+self._typevar1).mean(axis=0) )
                    self.info(dname,"stdv of",self._typevar1) 
                    self.raw(super().get(dname+self._typevar1).std(axis=0) )
        else:
            self.warn(dname,"not read.")

    def stdnorm(self, dname = _default_readkey):
        '''performs standard normalization on the data i.e, mean = 0 std = 1'''
        if( self.isRead(dname)) :
            self.load( dname+self._typevar0,\
                    preprocessing.scale( super().get( dname+self._typevar0 ) ))
            if( self.hasTarget ):
                self.load( dname+self._typevar1,\
                        preprocessing.scale( super().get( dname+self._typevar1 ) ))

    def batch(self, s_point, dname= _default_readkey ):
        '''splits the read data onto different quadrants (batches) to allow
        machine learning (test/train) sets to be used'''
        if( self.isRead(dname) ):
            # batches up the dname
            self._mp[ dname+self._typevar0+self._batchvar0 ],\
            self._mp[ dname+self._typevar0+self._batchvar1 ] = \
                dm.hsplit_mat( self._mp[ dname+self._typevar0 ], s_point)
            if( self.hasTarget ):
                self._mp[ dname+self._typevar1+self._batchvar0 ],\
                self._mp[ dname+self._typevar1+self._batchvar1 ] = \
                    dm.hsplit_mat( self._mp[ dname+self._typevar1 ], s_point)

    def isBatched(self,dname):
        '''attempts to test if the dset is batched or not'''
        return self.isLoaded( dname+self._typevar0 + self._batchvar0 )

    def get(self, dname, side='', batch=''):
        '''overrides the ancestral base class to obtains the data from the dictionary
        @dname- name of the data,
        @side - if it is the data, or target. use _typevar0 or _typevar1 to specify
        @side - if it is the train/test data, use _batchvar0 or _batchvar1 to specify
        '''
        #easy typing mapping TODO: a implementable function for subclasses ?
        if(side == "data"):
            side = self._typevar0
        elif(side == "target"):
            side = self._typevar1
        if(batch == "test"):
            side = self._batchvar0
        elif(batch == "train"):
            side = self._batchvar1
        return super().get( dname+side+batch )






