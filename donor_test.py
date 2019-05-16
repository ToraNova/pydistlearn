# pydistlearn basic test script
# Toranova mailto: chia_jason96@live.com
# May 2019

import pyplasma.pyplasma as pyplasma
from preproc import read_csvfile
from donor.vanilla import VanillaDonor
import numpy
import argparse
import pyioneer.support.gpam as gpam

if __name__ == "__main__":

    mp = argparse.ArgumentParser() #creates the argparser obj

    mp.add_argument("datafile", type=str, help="Input datafile for clients (can be found in sample_dsets)")
    mp.add_argument("-a","--hostname", type=str, help="Hostname of the central (i.e localhost)", default="localhost")
    mp.add_argument("-p","--port",type=int, help="Port number (default 48960)",default=48960)
    mp.add_argument("-t","--hasTarget",action='store_true',default=False,help="Specify if the client has targets or not")
    args = mp.parse_args() #preliminary parsing to see if it is client or not

    gpam.disable() #disable some verboses

    #client 
    donor = VanillaDonor( args.datafile, args.hasTarget , float, debug=True, verbose=True,owarn=True)
    donor.normalize_internals()
    donor.display_internals()
    donor.negotiate( (args.hostname, args.port) )
    donor.display_internals()

    donor.conntrain()
    donor.conntest()

    donor.shutdown_connections()


