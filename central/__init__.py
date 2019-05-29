# central package
# central class definition
# Toranova mailto: chia_jason96@live.com
# May 2019

# default
from abc import ABC, abstractmethod

# sklearn
from sklearn import preprocessing
# pyioneer (external library)
from pyioneer.support import Pam

class ConceptCentral(Pam):

    _hostnum = None
    _mnegformlist = []            # list of negforms recovered from the negotiations
    _mdmlconnlist = []            # list of connections accepted during DML
    _mdmlklist = []
    _mdmltlist = []
    _npdc = None # This should be a data controller

    donorTrained = False

    def __init__(self,verbose=False,debug=False):
        super().__init__(verbose=verbose,debug=debug)

    # required implementations (the constructor must read in to fill up _mDmat, and
    # possibly _mTvct if it hasTarget. the hasTarget must also be set to True if
    # the donor truly possess the targets
    @abstractmethod
    def dmltrain(self):
        '''dmltrain should start listen on the ahostaddr bounded during negotations,
        and attempts to receive the numpy arrays obtained from the various donors.
        it then has to generate the distributed alpha from the operations'''
        pass

    @abstractmethod
    def dmltest(self):
        '''dmltest should start to listen on the _npdc_hostadr bounded during negotiations,
        and attempts to receive the numpy arrays (TEST BATCH) from the same donors, 
        it then returns the evaluation metrics'''
        pass

    @abstractmethod
    def host_negotiations(self,asrvaddr, batchfactor=0.1, rrlambda=1.0):
        '''begins negotiations by listening on ahostaddr. receives the negform and syncs
        them up with the donors
        @params ahostaddr - a tuple ('localhost',portnumber i.e 8000)'''
        pass

    @abstractmethod
    def dmlpred(self):
        #TODO: figure out how to implement this
        pass

    ##############################################################################################
    # These are common throughout almost all implementation and thus are implemented in the ABC
    ##############################################################################################

    def hasNegotiated(self):
        '''check if the donor has negotiated. this is the first round of comm. that
        the donor and the central must undergo before training/testing phase'''
        return len(self._mnegformlist) > 0 and len(self._mdmlconnlist) > 0


