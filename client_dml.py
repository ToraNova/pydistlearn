# pydistlearn basic test script
# Toranova mailto: chia_jason96@live.com
# May 2019

import pyplasma.pyplasma as pyplasma
from preproc import read_csvfile
from donor.pandframe import PandFrameDonor
import numpy
import argparse
import pyioneer.support.gpam as gpam

if __name__ == "__main__":

    mp = argparse.ArgumentParser() #creates the argparser obj

    mp.add_argument("datafile", type=str, help="Input datafile for clients (can be found in sample_dsets)")
    mp.add_argument("-a","--hostname", type=str, help="Hostname of the central (i.e localhost)", default="localhost")
    mp.add_argument("-p","--port",type=int, help="Port number (default 48960)",default=48960)
    mp.add_argument("-t","--target",type=str, help="Specify the Linear Regression Training target")
    mp.add_argument("-d","--predir",type=str, help="Specify the preprocessing directive file")
    mp.add_argument("-s","--save",type=str, help="Save the model as <modelname>")
    mp.add_argument("-m","--model", type=str,\
            help="Loads a model file. Donor will attempt to predict outcomes rather than train/test")
    args = mp.parse_args() #preliminary parsing to see if it is client or not

    #client 
    hasTarget = args.target is not None
    donor = PandFrameDonor(  hasTarget, verbose=True, debug=True )
    rc = donor.load_datafile( args.datafile, skipc =1, adelimiter = ',')
    if(not rc):
        donor.error("Load datafile failed")
        exit(1)

    if( args.model is not None and args.save is not None):
        donor.error("Please specify EITHER model or save. not both.")
        exit(1)

    if( args.predir is not None ):
        rc = donor.preprocess( args.predir )
        if( rc != True ):
            donor.error("Preprocessing error")
            exit(1)
    else:
        donor.warn("No preprocessing file loaded. This could lead to some errors !")

    rc = donor.negotiate( (args.hostname, args.port), train= args.model is None )
    if( not rc ):
        donor.error("Negotiations failed")
        exit(1)

    if( args.model is not None ):
        rc = donor.load_weights( args.model )
        if( not rc):
            donor.error("Load weights failed")
            exit(1)

        rc = donor.connpred()
        if( not rc):
            donor.error("Prediction failed")
            exit(1)
    else:
        if(donor._pam_dflag):
            donor.display_internals()

        rc = donor.conntrain( args.target )
        if( not rc ):
            donor.error("Conntrain failed")
            exit(1)

        rc = donor.conntest()
        if( not rc ):
            donor.error("Conntest failed")
            exit(1)

    if( args.save is not None):
        rc = donor.save_weights( args.save )
        if( not rc ):
            donor.error("Weights saved failed")
            exit(1)

    donor.shutdown_connections()
    exit(0)

