# pandframe.py (DONOR)
# pandframe donor is a donor version based on pandas
# dataframe
# that is, it uses unencrypted sockets, no SSL/TLS
# but it features a richer set of data manipulation
# functionalities

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
from preproc.aux import computeKernel
from preproc.aux import serialize, deserialize
import preproc.negotiate as negotiate
import preproc.neoctl as neoctl
NeoNPDController = neoctl.NeoNPDController
NegForm = negotiate.NeoNegForm

import pyioneer.network.tcp.smsg as smsg

class PandFrameDonor(ConceptDonor):
        
    # dimension of the target. for multivariate, this > 1
    tdim = 1
    compd = numpy.double
    _msocket = None

    def __init__(self, ahasTarget = False, verbose=False, debug=False):
        super().__init__(verbose=verbose,debug=debug)
        '''creates the vanilla donor by reading in a file, filles the file
        up and will read based on what the donor is created as (hasTarget or no?)'''
        self.hasTarget = ahasTarget

    def load_datafile( self, filename, skipc=0, adelimiter =';', aquotechar ='"'):
        self.verbose("Donor datafile:",filename)
        self._npdc = NeoNPDController( filename, 0, adelimiter,\
                self._pam_vflag,self._pam_dflag)
        return True

    # required implementations (the constructor must read in to fill up _mDmat, and
    # possibly _mTvct if it hasTarget. the hasTarget must also be set to True if
    # the donor truly possess the targets
    def conntrain(self,targetname=None):
        '''conntrain should begin connection with the central, send in the
        _mDmat_train and (if it has the target) _mTvct_train. await response
        from server and fill in the received alpha to _mdist_alpha'''
        if( self.hasNegotiated() ):
            self.targetname = targetname

            if(self.targetname is not None):
                self.train_x, self.test_x, self.train_y, self.test_y = \
                        self._npdc.obtrain( self._mnegform.primary["bsize"] , targetname )
            else:
                self.train_x, self.test_x = self._npdc.obtrain( self._mnegform.primary["bsize"] )
            self.kernel = computeKernel( self.train_x )
            if not type(self.kernel) == numpy.ndarray:
                self.warn("Kernel computational error!")
            else:
                self.verbose("Partitioned and computed the kernel",self.kernel.shape)
            self.verbose("Sending kernel to central")
            #dumped = json.dumps( self.kernel.tolist() ) # THIS line is crashing the system (for size 10k)
            self.debug("Training headers (include targets) :")
            self.raw( self._npdc.hlist )
            dumped = serialize( self.kernel )
            self.verbose("Total serial dump: {} bytes".format(len(dumped)))
            smsg.send( self._msocket, dumped ) #json dump and send the array
            # await confirmation
            repmsg = self._msocket.recv(4)
            if( repmsg.decode('utf-8') == "ACKN" ):
                # proceed
                if( self.targetname is not None ):
                    # dump the target_train to bytes and send it on over socket
                    dumped = serialize( self.train_y )
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
            aggregate = self.test_x.dot( self._mweights )
            self.verbose("Sending test prediction to central",aggregate.shape)
            #self.raw( aggregate )
            dumped = serialize( aggregate )
            smsg.send( self._msocket, dumped )
            repmsg = self._msocket.recv(4)
            if( repmsg.decode('utf-8') == "ACKN" ):
                #proceed
                if( self.targetname is not None ):
                    # dump the target_test to bytes and send it on over socket
                    dumped = serialize( self.test_y )
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
                        self.info("MAE:", self._mres.get("mae"))
                        self.info("MAX:", self._mres.get("max"))
                        self.info("EVS:", self._mres.get("evs"))
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
        if( self.isTrained ):
            aggregate = self._npdc.df.dot( self._mweights )
            self.verbose("Sending test prediction to central",aggregate.shape)
            #self.raw( aggregate )
            dumped = serialize( aggregate )
            smsg.send( self._msocket, dumped )
            repmsg = self._msocket.recv(4)
            if( repmsg.decode('utf-8') == "ACKN" ):
                #proceed
                self.info("All Aggregates sent. Awaiting results.") 

                rcv = smsg.recv( self._msocket )
                if(rcv != None):
                    try:
                        if( rcv.decode('utf-8') == 'ABRT'):
                            self.error("Abort request by central.")
                    except UnicodeDecodeError:
                        self._mres = deserialize( rcv )
                        self.verbose("Received DML pred results:")
                        self.info( self._mres )
                        return True
                else:
                    self.error("rcv is null. Receiving error on _mres")
            else:
                self.error("Failed to receive ACKN from host. Terminating conntest")
        else:
            self.error("Weights not available. Is the donor trained ?")
        return False
    
    def save_weights( self, wfilename):
        '''saves the weight as a json format file'''
        if(self.isTrained):
            tlist = self._npdc.hlist # make copy
            if( self.targetname is not None ):
                del tlist[ self._npdc.hlist.index( self.targetname ) ] #delete target
            wdat = pandas.DataFrame( self._mweights, index = tlist, columns=["Weights"])
            wdat.to_csv( wfilename, index=True,index_label="Features")

            self.debug("Weight saved successfully")
            return True
        else:
            self.error("Unable to save weights. Donor not even trained.")
            return False

    def load_weights( self, wfilename,mmdrop = True):
        # Please run this AFTER preprocessing !
        wdat = pandas.read_csv( wfilename, index_col="Features" )
        #verify weights are correct with the predicting columns

        # Presence Test
        absentlist = []
        for w in wdat.index.values.tolist():
            if w not in self._npdc.hlist:
                absentlist.append(w)
        self.debug("Absent on dfile:",absentlist)
        if(len(absentlist) > 0):
            if(mmdrop):
                wdat.drop( absentlist, axis= self._npdc._constant_axis_ROWS, inplace=True)
            else:
                return False
        # Alignment Test  TODO: automatically realign if mismatch
        for h,w in zip( self._npdc.hlist, wdat.index.values.tolist() ):
            if( h != w ):
                self.error("Mismatch! h/w:",h,w)
                return False
        self._mweights = wdat["Weights"].values #obtain the weight
        self.isTrained = True
        return True


    def recover_weights(self, colmajor=False):
        '''recovers the weight'''
        if( self.hasAlpha and isinstance(self.train_x, numpy.ndarray )):
            ool = (1/self._mnegform.primary['rrlambda'])
            self.debug("OOL (lval):",ool)
            if( type(ool) != float and type(ool) != int):
                self.warn("OOL not a float or int")
            if( not colmajor ):
                self._mweights = ool*self.train_x.transpose().dot(\
                        self._mdistalpha)
            else:
                self._mweights = ool*self.train_x.dot(\
                        self._mdistalpha)
            if( type(self._mweights) == numpy.ndarray ):
                self.isTrained = True
                self.debug("Weights recovered successfully",self._mweights.shape)
                self.debug("Weights array:")
                for h,w in zip( self._npdc.hlist, self._mweights):
                    self.raw( h, w )
            else:
                self.isTrained = False
        else:
            self.isTrained = False

    def preprocess(self, preprocdir):
        operres = {}
        with open( preprocdir ) as f:
            d = json.load(f)
        for p in d["mainlist"]:
            for key,direct in p.items():
                if(key== "__comment"):
                    #ignore the comment
                    continue
                ores = []
                mlist = None
                if( key.startswith("__list__") ):
                    #special list keyword
                    # find the list name
                    listname = key[ len("__list__"):]
                    mlist = d.get(listname)
                    if mlist is None:
                        print("Error:",listname,"used in __list__ but not defined")
                        exit(1)
                    else:
                        key = mlist
                        print("Applying json based processing for Columns:",key)
                elif( key=="*"):
                    key=None
                    print("Applying json based processing for all columns")
                else:
                    print("Applying json based processing for Columns:",key)
                for dp in direct:
                    print("Performing '{} {}'".format(
                        dp.get("process"),
                        dp.get("mets")))
                    ores.append(
                        self._npdc.preproc( dp.get("process"),
                                    key, 
                                    dp.get("mets"),
                                    dp.get("arglist")
                        )
                    )
                if key==None:
                    operres["*"] = ores
                elif mlist is not None:
                    operres[listname] = ores
                    del mlist
                else:
                    operres[key] = ores

        #mc.fillempty_withvalue()
        #mc.apply_mapper("LabelEncoder")
        for k,v in operres.items():
            self.verbose(k,v)
            if( not all( v ) ):
                return False
        return True

    def display_internals(self):
        self._npdc.head(5)

    # common functions
    def negotiate(self,ahostaddr,train=True):
        '''start negotation, first sends in the donor's own prepared negform to inform the
        central about the number of entries/features, it is expected that _mDmat is read
        from a file/stdin before this 
        @params ahostaddr - a tuple ('localhost',portnumber i.e 8000)'''
        if( train ):
            # TRAINING MODE
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
            except Exception as e:
                self.expt(str(e),traceback.format_exc())
            finally:
                return self.hasNegotiated()
        else:
            # PREDICTION MODE
            try:
                self._msocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._msocket.connect( ahostaddr )
                return True
            except Exception as e:
                self.expt(str(e),traceback.format_exc())
                return False

    def shutdown_connections(self):
        try:
            if(self._msocket is not None):
                self._msocket.close()
        except Exception as e:
            self.expt(str(e))


