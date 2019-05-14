# vanilla.py (DONOR)
# vanilla donor is the vanilla version of a donor
# that is, it uses unencrypted sockets, no SSL/TLS
# and performs the most basic operations (generic)

from . import ConceptDonor
import csv
import numpy

# local
import preproc.negotiate as negotiate
import preproc.controller as controller
NegForm = negotiate.NegForm

class VanillaDonor(ConceptDonor):
        
    # dimension of the target. for multivariate, this > 1
    tdim = 1
    compd = numpy.double

    def __init__(self,filename,ahasTarget, htype, skipc = 0, adelimiter=';', aquotechar ='"',
            verbose=False, debug=False):
        super().__init__(verbose=verbose,debug=debug)
        '''creates the vanilla donor by reading in a file, filles the file
        up and will read based on what the donor is created as (hasTarget or no?)'''
        self._npdc = controller.NPDController(verbose,debug) 
        self.hasTarget = ahasTarget
        self._npdc.read( filename, ahasTarget, htype, skipc = skipc, adelimiter = adelimiter,
                aquotechar = aquotechar)

    # required implementations (the constructor must read in to fill up _mDmat, and
    # possibly _mTvct if it hasTarget. the hasTarget must also be set to True if
    # the donor truly possess the targets
    def conntrain(self):
        '''conntrain should begin connection with the central, send in the
        _mDmat_train and (if it has the target) _mTvct_train. await response
        from server and fill in the received alpha to _mdist_alpha'''
        pass

    def conntest(self):
        '''conntest should begin connection with the central, send in the 
        _mDmat_train and (if it has the target) _mTvct_train. await response
        from server which it will return the error rating of the model'''
        pass

    def connpred(self):
        #TODO: figure out how to implement this
        pass

    # common functions
    def negotiate(self,ahostaddr):
        '''start negotation, first sends in the donor's own prepared negform to inform the
        central about the number of entries/features, it is expected that _mDmat is read
        from a file/stdin before this 
        @params ahostaddr - a tuple ('localhost',portnumber i.e 8000)'''
        _mnegform = Negform(self) #creates the negotiation form
        self.verbose("Negform created. Beginning Negotation...")
        _mnegform.display()
        try:
            negstr = json.dumps(_mnegform.primary) #obtains the primary neg data
            _msocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #creates the sock obj
            _msocket.connect( ahostaddr ) #attempting to connect to the host (central)
            debug("Host connected. Sending negstr over (%s)" % negstr)

        except Exception as e:
            self.error("Exception has occurred.",str(e))


