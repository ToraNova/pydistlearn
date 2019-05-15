# vanilla.py (CENTRAL)
# vanilla central is the vanilla version of a central
# that is, it uses unencrypted sockets, no SSL/TLS
# and performs the most basic operations (generic)

# default
import csv, json
import numpy
import socket
import traceback

# sklearn
from sklearn import preprocessing
from . import ConceptCentral

# local
import preproc.negotiate as negotiate
import preproc.controller as controller
NegForm = negotiate.NegForm

import pyioneer.network.tcp.smsg as smsg

class VanillaCentral(ConceptCentral):

    # dimension of the target. for multivariate, this > 1
    tdim = 1
    compd = numpy.double

    def __init__(self, hostnum,verbose=False, debug=False,owarn=False):
        super().__init__(verbose=verbose,debug=debug)
        '''creates the vanilla donor by reading in a file, filles the file
        up and will read based on what the donor is created as (hasTarget or no?)'''
        self._npdc = controller.NPDController(verbose,debug,owarn) 
        self._hostnum = hostnum

    def dmltrain(self):
        '''dmltrain should start listen on the ahostaddr bounded during negotations,
        and attempts to receive the numpy arrays obtained from the various donors.
        it then has to generate the distributed alpha from the operations'''
        pass
    def dmltest(self):
        '''dmltest should start to listen on the _npdc_hostadr bounded during negotiations,
        and attempts to receive the numpy arrays (TEST BATCH) from the same donors, 
        it then returns the evaluation metrics'''
        pass

    def dmlpred(self):
        #TODO: figure out how to implement this
        pass

    def host_negotiations(self,asrvaddr):
        '''begins negotiations by listening on ahostaddr. receives the negform and syncs
        them up with the donors
        @params ahostaddr - a tuple ('localhost',portnumber i.e 8000)'''
        # the host are indexed based on the order in which they connect
        self._mnegformlist = []
        self._mdmlconnlist = []
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind( asrvaddr )
            self.sock.listen( self._hostnum )
            for i in range( self._hostnum ):
                self.debug("Listening on address:",asrvaddr)
                connection, rcliaddr = self.sock.accept()
                self.debug("Connection established with",rcliaddr)
                # receives the json obj
                negstr = smsg.recv( connection )
                negjs = json.loads( negstr ) 
                self.debug("Received negotiation forms...",negstr)
                self.debug("NEGJS type",type(negjs))
                negf = NegForm( negjs )
                negf.display()
                self._mnegformlist.append( negf )
                self._mdmlconnlist.append( connection )
        except Exception as e:
            self.expt(str(e),traceback.format_exc())

    def shutdown_connections(self):
        for c in self._mdmlconnlist:
            try:
                c.close()
            except Exception as e:
                self.expt(str(e))
        try:
            self.sock.close()
        except Exception as e:
            self.expt(str(e))
