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
    args = mp.parse_args() #preliminary parsing to see if it is client or not

    #client 
    clihasTar = args.target is not None
    donor = PandFrameDonor( args.datafile, args.predir , clihasTar,  skipc=1 , adelimiter=',',
            debug=True, verbose=True,owarn=True)

    donor.display_internals()
    donor.negotiate( (args.hostname, args.port) )

    donor.conntrain( args.target )
    donor.conntest()

    donor.shutdown_connections()

    exit(0)

