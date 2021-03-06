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
    _npdc = None # This should be a data controller DEPRECATED (now uses a dataframe)

    hasTarget = False
    hasAlpha = False
    isTrained = False
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

    def hasNegotiated(self):
        '''check if the donor has negotiated. this is the first round of comm. that
        the donor and the central must undergo before training/testing phase'''
        if( self._mnegform is None ):
            return False
        return (self._mnegform.isfilled()) and (type(self.kernel) is not None)
