# pydistlearn basic test script
# Toranova mailto: chia_jason96@live.com
# May 2019

import pyplasma.pyplasma as pyplasma
from preproc import read_csvfile
from central.vanilla import VanillaCentral
import numpy
import argparse
import pyioneer.support.gpam as gpam

if __name__ == "__main__":

    mp = argparse.ArgumentParser() #creates the argparser obj

    mp.add_argument("-a","--hostname", type=str, help="Hostname of the central (i.e localhost)", default="localhost")
    mp.add_argument("-p","--port",type=int, help="Port number (default 48960)",default=48960)
    mp.add_argument("-n","--dcount",type=int, help="Number of donors (default 2)",default=2)
    mp.add_argument("-r","--rlambda",type=float, help="Ridge Regression hyperparameter (default 1.0)",default=1.0)
    mp.add_argument("-b","--batch",type=float, help="Batching Factor (default 0.1)",default=0.1)
    args = mp.parse_args() #preliminary parsing to see if it is client or not

    gpam.disable() #disblae some verboses

    #central
    central = VanillaCentral( args.dcount , verbose=True, debug=True, owarn=True)
    central.host_negotiations( ('0.0.0.0',args.port), batchfactor= args.batch, rrlambda=args.rlambda)

    central.dmltrain()
    central.dmltest()

    central.shutdown_connections()

