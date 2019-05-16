# donor package
# donor class definition
# Toranova mailto: chia_jason96@live.com
# May 2019

# This is a module for the donor
# the class may also read locally for testing
# purposes

# default
from abc import ABC, abstractmethod

# sklearn
from sklearn import preprocessing
# pyioneer (external library)
from pyioneer.support import Pam

class ConceptDonor(Pam):

    _mdistalpha = None
    _mnegform = None
    _npdc = None # This should be a data controller

    hasTarget = False
    hasAlpha = False
    kernel = None

    def __init__(self,verbose=False,debug=False,owarn=False):
        super().__init__(verbose=verbose,debug=debug)

    # required implementations (the constructor must read in to fill up _mDmat, and
    # possibly _mTvct if it hasTarget. the hasTarget must also be set to True if
    # the donor truly possess the targets
    @abstractmethod
    def conntrain(self):
        '''conntrain should begin connection with the central, send in the
        _mDmat_train and (if it has the target) _mTvct_train. await response
        from server and fill in the received alpha to _mdist_alpha'''
        pass

    @abstractmethod
    def conntest(self):
        '''conntest should begin connection with the central, send in the 
        _mDmat_train and (if it has the target) _mTvct_train. await response
        from server which it will return the error rating of the model'''
        pass

    @abstractmethod
    def negotiate(self,ahostaddr):
        '''start negotation, first sends in the donor's own prepared negform to inform the
        central about the number of entries/features, it is expected that _mDmat is read
        from a file/stdin before this 
        @params ahostaddr - a tuple ('localhost',portnumber i.e 8000)'''
        pass

    @abstractmethod
    def connpred(self):
        #TODO: figure out how to implement this
        pass

    @abstractmethod
    def recover_weights(self, colmajor=False):
        '''please implement a weight recovery function. vanilla uses numpy'''
        pass

    ##############################################################################################
    # These are common throughout almost all implementation and thus are implemented in the ABC
    ##############################################################################################
    def display_internals(self):
        '''invokes a display command to display the internal content using any data controllers'''
        if self._npdc != None:
            self._npdc.show()
        if self._mnegform != None:
            self._mnegform.display()

    def partition_internals(self, s_point):
        '''invokes a partition command to perform splitting of the data set into the train/test'''
        if self._npdc != None:
            self._npdc.batch(s_point)
        else:
            self.error("Failed to partition. NPDC is null!")

    def normalize_internals(self):
        '''perform normalization on the internal dataset, please call partition again'''
        if self._npdc != None:
            self._npdc.stdnorm()
        else:
            self.error("Failed to normalize. NPDC is null!")

    def sizeof_internals(self):
        if self._npdc != None:
            return self._npdc.size()
        else:
            self.error("Failed to obtain sizes. NPDC is null!")

    def isTrained(self):
        '''check if the donor is trained properly. (donor must negotatie before training)'''
        return self._npdc != None and self._npdc.isLoaded( self._npdc_distkey )


    def hasNegotiated(self):
        '''check if the donor has negotiated. this is the first round of comm. that
        the donor and the central must undergo before training/testing phase'''
        return (self._mnegform.isfilled()) and (type(self.kernel) != None)
