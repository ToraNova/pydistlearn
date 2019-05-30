# vanilla.py (CENTRAL)
# vanilla central is the vanilla version of a central
# that is, it uses unencrypted sockets, no SSL/TLS
# and performs the most basic operations (generic)

from . import ConceptCentral
import csv, json
import numpy, pandas
import socket, select
from io import StringIO
import traceback

import hashlib

# sklearn
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score,\
                            mean_absolute_error, max_error,\
                            explained_variance_score


# local
from preproc.aux import serialize, deserialize
import preproc.negotiate as negotiate
import preproc.controller as controller
import preproc.aux as aux
NPDController = controller.NPDController
NegForm = negotiate.NegForm

import pyioneer.network.tcp.smsg as smsg

import pyplasma.pyplasma as pyplasma

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
        self._mdmlklist = []
        self._mdmltlist = []
        out = True
        if( self.hasNegotiated() ):
            f = StringIO()
            lval = self._mnegformlist[0].primary['rrlambda']
            self.verbose("Hosting DML training lval:",lval,"bsize:",
                    self._mnegformlist[0].primary['bsize'])
            for i in range( self._hostnum ):
                conn = self._mdmlconnlist[i]
                self.verbose("Receiving kernel from donor",i)

                rcv = smsg.recv( conn )
                if( rcv == None ):
                    self.error("kernel rcv has failed from donor",i)
                    out = False
                else:
                    rk = deserialize( rcv )
                    #rk = numpy.array(json.loads( rcv ))
                    self.verbose("kernel rcv has succeeded from donor",i)
                    self._mdmlklist.append( rk )
                # sends the ACK
                conn.send('ACKN'.encode('utf-8'))
                if( self._mnegformlist[i].primary['dflag']):
                    # this donor possess target, receives it
                    rcv = smsg.recv( conn )
                    if( rcv == None):
                        self.error("train targets rcv has failed from donor",i)
                        out = False
                    else:
                        rt = deserialize( rcv )
                        #rt = numpy.array(json.loads( rcv ))
                        self._mdmltlist.append( rt )

            if( len(self._mdmlklist) < 1):
                self.error("No kernels received")
                out = False
                self.__abortAll()
            elif( len(self._mdmltlist) < 1):
                self.error("No Targets. Please check if target donor is specified correctly")
                out = False
                self.__abortAll()
            else:
                # all kernels and targets received
                # sum all kernels
                ksum = numpy.zeros( self._mdmlklist[0].shape, dtype = self.compd )
                for i,k in enumerate(self._mdmlklist):
                    self.info("kernel",i,k.shape)
                    ksum += k
                
                # only train against the first TODO : selectable training
                targ = self._mdmltlist[0]
                self.verbose("Starting Ridge Solver") 
                if( len(targ.shape) < 2 ):
                    tdim = 1
                else:
                    tdim = targ.shape[1]
                alpha = pyplasma.ridge_solve(
                        ksum.flatten(), ksum.shape[0], ksum.shape[1],
                        targ.flatten(), targ.shape[0], tdim,
                        lval, 1,
                        ksum.shape[0]*tdim)
                self.info("Alpha meta:",type(alpha),alpha.shape)

                dumped = serialize( alpha )
                for i in range( self._hostnum ):
                    conn = self._mdmlconnlist[i]
                    smsg.send( conn, dumped ) #dump to bytes and send
                self.info("DML training complete")
        self.donorTrained = out
        return out

    def dmltest(self):
        '''dmltest should start to listen on the _npdc_hostadr bounded during negotiations,
        and attempts to receive the numpy arrays (TEST BATCH) from the same donors, 
        it then returns the evaluation metrics
        RETURNS Ture upon No errors. False otherwise'''
        self._mdmlalist = []
        self._mdmlvlist = []
        out = True
        if( self.hasNegotiated() and self.donorTrained ):
            self.verbose("Hosting DML test")
            for i in range( self._hostnum ):
                conn = self._mdmlconnlist[i]
                rcv = smsg.recv( conn )
                if( rcv == None ):
                    self.error("aggregate rcv has failed from donor",i)
                    out = False #Error occurred
                else:
                    ra = deserialize( rcv )
                    #ra = numpy.array( json.loads(rcv))
                    self._mdmlalist.append( ra )
                conn.send('ACKN'.encode('utf-8'))
                if( self._mnegformlist[i].primary['dflag']):
                    rcv = smsg.recv( conn )
                    if( rcv == None):
                        self.error("test targets rcv has failed from donor",i)
                        out = False #Error occurred
                    else:
                        rt = deserialize( rcv )
                        #rt = numpy.array(json.loads(rcv))
                        self._mdmlvlist.append( rt )
                        self.verbose("received test targets from donor",i)

            if( len(self._mdmlalist) < 1):
                self.error("No aggregates received, Aborting")
                out = False #Error occurred
                self.__abortAll()
            if( len(self._mdmlvlist) < 1):
                self.error("No verifying targets. Is target donor specified correctly?")
                out = False
                self.__abortAll()
            else:
                asum = numpy.zeros( self._mdmlalist[0].shape, dtype = self.compd)
                for i,a in enumerate(self._mdmlalist):
                    self.info("aggregate",i,a.shape)
                    asum = a + asum

                # only test against the first TODO: selectable testing

                varg = self._mdmlvlist[0]
                jrep = {
                        "mse": mean_squared_error( asum, varg),
                        "mae": mean_absolute_error( asum, varg),
                        "max": max_error( asum, varg),
                        "evs": explained_variance_score( asum, varg),
                        "r2s": r2_score( asum, varg)
                        }
                self.verbose("mse",jrep.get("mse"))
                self.verbose("mae",jrep.get("mae"))
                self.verbose("max",jrep.get("max"))
                self.verbose("evs",jrep.get("evs"))
                self.verbose("r2s",jrep.get("r2s"))

                for i in range(self._hostnum):
                    conn = self._mdmlconnlist[i]
                    smsg.send( conn, json.dumps( jrep ) )
                self.verbose("DML done. JREP pushed to donors")
        else:
            self.error("Not negotiated or donor is not trained yet, unable to test")
            out = False
        return out

    def dmlpred(self):
        self._mdmlalist = []
        out = True
        self.verbose("Hosting DML test")
        for i in range( self._hostnum ):
            conn = self._mdmlconnlist[i]
            rcv = smsg.recv( conn )
            if( rcv == None ):
                self.error("aggregate rcv has failed from donor",i)
                out = False #Error occurred
            else:
                ra = deserialize( rcv )
                #ra = numpy.array( json.loads(rcv))
                self._mdmlalist.append( ra )
            conn.send('ACKN'.encode('utf-8'))

        if( len(self._mdmlalist) < 1):
            self.error("No aggregates received, Aborting")
            out = False #Error occurred
            self.__abortAll()
        else:
            asum = numpy.zeros( self._mdmlalist[0].shape, dtype = self.compd)
            for i,a in enumerate(self._mdmlalist):
                self.info("aggregate",i,a.shape)
                asum = a + asum

            # only test against the first TODO: selectable testing
            for i in range( self._hostnum ):
                conn = self._mdmlconnlist[i]
                dumped = serialize(asum)
                smsg.send( conn, dumped )
            self.verbose("DML done. ASUM pushed to donors")
        return out

    def host_negotiations(self,asrvaddr, batchfactor=0.1, rrlambda=1.0,train=True):
        '''begins negotiations by listening on ahostaddr. receives the negform and syncs
        them up with the donors
        @params ahostaddr - a tuple ('localhost',portnumber i.e 8000)
        @params batchfactor - the value to batch up the train/test set
        @params rrlamdda - the ridge regressor's hyperparam'''
        # the host are indexed based on the order in which they connect
        self._mdmlconnlist = []
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if(train):
            # TRAINING MODE
            self.verbose("Negotiation for DML TRAIN")
            self._mnegformlist = []
            try:
                self.sock.bind( asrvaddr )
                self.sock.listen( self._hostnum )
                # Negotiations round 1
                for i in range( self._hostnum ):
                    self.debug("Listening on address:",asrvaddr)
                    connection, rcliaddr = self.sock.accept()
                    self.debug("Connection established with",rcliaddr)
                    # receives the json obj
                    negf = NegForm( json.loads( smsg.recv( connection ) ) )
                    negf.display()
                    self._mnegformlist.append( negf )
                    self._mdmlconnlist.append( connection )

                # Process to formulate the batchsizes
                bsize = aux.evaluate_bsize(
                        aux.find_minesz( self._mnegformlist ), batchfactor)
                self.info("bsize decided on",bsize)

                # Negotiations round 2
                for i in range( self._hostnum ):
                    self._mnegformlist[i].primary['bsize'] = bsize
                    self._mnegformlist[i].primary['rrlambda'] = rrlambda
                    smsg.send( self._mdmlconnlist[i],
                            json.dumps( self._mnegformlist[i].primary ))
                    self.debug("Negotiation forms pushed back to host",i)
            except Exception as e:
                self.expt(str(e),traceback.format_exc())
            finally:
                return self.hasNegotiated()
        else:
            # PREDICTION MODE
            self.verbose("Negotiation for DML PRED")
            try:
                self.sock.bind( asrvaddr )
                self.sock.listen( self._hostnum )
                for i in range( self._hostnum ):
                    self.debug("Listening on address:",asrvaddr)
                    connection, rcliaddr = self.sock.accept()
                    self.debug("Connection established with",rcliaddr)
                    self._mdmlconnlist.append( connection )
                return True
            except Exception as e:
                self.expt(str(e),traceback.format_exc())
                return False

    def __abortAll(self):
        for i in range(self._hostnum):
            conn = self._mdmlconnlist[i]
            smsg.send( conn, "ABRT") #send the abort message to all host

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
