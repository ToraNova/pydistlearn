# vanilla.py (DONOR)
# vanilla donor is the vanilla version of a donor
# that is, it uses unencrypted sockets, no SSL/TLS
# and performs the most basic operations (generic)

from . import ConceptDonor
import csv, json
import numpy, pandas
import socket
import traceback
from io import StringIO
import traceback

# sklearn
from sklearn import preprocessing

# local
from preproc.aux import serialize, deserialize
import preproc.negotiate as negotiate
import preproc.controller as controller
NPDController = controller.NPDController
NegForm = negotiate.NegForm

import pyioneer.network.tcp.smsg as smsg

class VanillaDonor(ConceptDonor):
        
    # dimension of the target. for multivariate, this > 1
    tdim = 1
    compd = numpy.double

    def __init__(self,filename,ahasTarget, htype, skipc = 0, adelimiter=';', aquotechar ='"',
            verbose=False, debug=False,owarn=False):
        super().__init__(verbose=verbose,debug=debug)
        '''creates the vanilla donor by reading in a file, filles the file
        up and will read based on what the donor is created as (hasTarget or no?)'''
        self._npdc = NPDController(verbose,debug,owarn) 
        self.hasTarget = ahasTarget
        self._npdc.read( filename, ahasTarget,htype, skipc = skipc, adelimiter = adelimiter,
                aquotechar = aquotechar)

    # required implementations (the constructor must read in to fill up _mDmat, and
    # possibly _mTvct if it hasTarget. the hasTarget must also be set to True if
    # the donor truly possess the targets
    def conntrain(self):
        '''conntrain should begin connection with the central, send in the
        _mDmat_train and (if it has the target) _mTvct_train. await response
        from server and fill in the received alpha to _mdist_alpha'''
        if( self.hasNegotiated() ):
            self.verbose("Sending kernel to central")
            #dumped = json.dumps( self.kernel.tolist() ) # THIS line is crashing the system (for size 10k)

            dumped = serialize( self.kernel )
            self.verbose("Total serial dump: {} bytes".format(len(dumped)))
            smsg.send( self._msocket, dumped ) #json dump and send the array
            # await confirmation
            repmsg = self._msocket.recv(4)
            if( repmsg.decode('utf-8') == "ACKN" ):
                # proceed
                if( self.hasTarget ):
                    # dump the target_train to bytes and send it on over socket
                    dumped = serialize( self._npdc.get(side='target',batch='train'))
                    smsg.send( self._msocket, dumped) 
                # await for alpha
                self.info("All Kernels sent. Awaiting central response.")

                rcv = smsg.recv( self._msocket )
                if(rcv != None):
                    try:
                        if( rcv.decode('utf-8') == 'ABRT'):
                            self.error("Abort request by central.")
                            self.hasAlpha = False
                    except UnicodeDecodeError :
                        self.verbose("Unicode decode failed. Proceeding with deserialization")
                        self._mdistalpha = deserialize(rcv)
                        self.info("Distributed alpha received.")
                        self.hasAlpha=True
                        self.recover_weights() # perform weight recovery
                else:
                    self.error("rcv is null. Receiving error _mdistalpha")
                    self.hasAlpha = False
            else:
                #failed
                self.error("Failed to receive ACKN from host. Terminating conntrain")
                self.hasAlpha = False
        else:
            self.error("This donor has not synchronized the params with the central,\
                    please run negotiate( addr ) first !")
            self.hasAlpha = False
        return self.hasAlpha

    def conntest(self):
        '''conntest should begin connection with the central, send in the 
        _mDmat_train and (if it has the target) _mTvct_train. await response
        from server which it will return the error rating of the model
        RETURNS True upon No errors. False otherwise'''
        if( self.isTrained ):
            aggregate = self._npdc.get( side="data",batch="test").dot( self._mweights )
            self.verbose("Sending test prediction to central",aggregate.shape)
            #self.raw( aggregate )
            dumped = serialize( aggregate )
            smsg.send( self._msocket, dumped )
            repmsg = self._msocket.recv(4)
            if( repmsg.decode('utf-8') == "ACKN" ):
                #proceed
                if( self.hasTarget ):
                    # dump the target_test to bytes and send it on over socket
                    dumped = serialize( self._npdc.get(side='target',batch='test'))
                    smsg.send( self._msocket, dumped )
                    #await for test results
                    self.info("All Aggregates sent. Awaiting results.") 

                rcv = smsg.recv( self._msocket )
                if(rcv != None):
                    if( rcv.decode('utf-8') == 'ABRT'):
                        self.error("Abort request by central.")
                    else:
                        self._mres = json.loads(rcv)
                        self.verbose("Received DML test results:")
                        self.info("MSE:", self._mres.get("mse"))
                        self.info("R2S:", self._mres.get("r2s"))
                        return True
                else:
                    self.error("rcv is null. Receiving error on _mres")
            else:
                self.error("Failed to receive ACKN from host. Terminating conntest")
        else:
            self.error("Weights not available. Is the donor trained ?")
        return False

    def connpred(self):
        #TODO: figure out how to implement this
        pass

    def recover_weights(self, colmajor=False):
        '''recovers the weight'''
        if( self.hasAlpha ):
            ool = (1/self._mnegform.primary['rrlambda'])
            self.debug("OOL (lval):",ool)
            if( type(ool) != float and type(ool) != int):
                self.warn("OOL not a float or int")
            if( not colmajor ):
                self._mweights = ool*self._npdc.get(\
                        side="data",batch="train").transpose().dot(\
                        self._mdistalpha)
            else:
                self._mweights = ool*self._npdc.get(\
                        side="data",batch="train").dot(\
                        self._mdistalpha)
            if( type(self._mweights) == numpy.ndarray ):
                self.isTrained = True
                self.info("Weights recovered successfully",self._mweights.shape)
                self.debug("Weights array:")
                self.raw( self._mweights )
            else:
                self.isTrained = False
        else:
            self.isTrained = False

    # common functions
    def negotiate(self,ahostaddr):
        '''start negotation, first sends in the donor's own prepared negform to inform the
        central about the number of entries/features, it is expected that _mDmat is read
        from a file/stdin before this 
        @params ahostaddr - a tuple ('localhost',portnumber i.e 8000)'''
        _mnegform = NegForm(self) #creates the negotiation form
        self.verbose("Negform created. Beginning Negotation...")
        try:
            negstr = json.dumps(_mnegform.primary) #obtains the primary neg data
            self._msocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._msocket.connect( ahostaddr ) #attempting to connect to the host (central)
            self.debug("Host connected. Sending negstr")
            smsg.send( self._msocket, negstr )
            self.debug("Negotiation form sent to central. Awaiting synchronization")
            self._mnegform = NegForm( json.loads( smsg.recv( self._msocket ) ) )
            self.info("Synchronized form:")
            self._mnegform.display()
            self.partition_internals( self._mnegform.primary["bsize"] )
            self.kernel = self._npdc.computeKernel()
            if not type(self.kernel) == numpy.ndarray:
                self.warn("Kernel computational error!")
            else:
                self.verbose("Partitioned and computed the kernel",self.kernel.shape)
        except Exception as e:
            self.expt(str(e),traceback.format_exc())
        finally:
            return self.hasNegotiated()

    ##############################################################################################
    # These are common throughout almost all implementation and thus are implemented in the ABC
    # updated: migrated from the conceptual class to this.
    ##############################################################################################
    def display_internals(self):
        '''invokes a display command to display the internal content using any data controllers'''
        if self._npdc is not None:
            self._npdc.show()
        if self._mnegform is not None:
            self._mnegform.display()

    def partition_internals(self, s_point):
        '''invokes a partition command to perform splitting of the data set into the train/test'''
        if self._npdc is not None:
            self._npdc.batch(s_point)
        else:
            self.error("Failed to partition. NPDC is null!")

    def normalize_internals(self):
        '''perform normalization on the internal dataset, please call partition again'''
        if self._npdc is not None:
            self._npdc.stdnorm()
        else:
            self.error("Failed to normalize. NPDC is null!")

    def sizeof_internals(self):
        if self._npdc is not None:
            return self._npdc.size()
        else:
            self.error("Failed to obtain sizes. NPDC is null!")

    def shutdown_connections(self):
        try:
            self._msocket.close()
        except Exception as e:
            self.expt(str(e))


