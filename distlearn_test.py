# pydistlearn basic test script
# Toranova mailto: chia_jason96@live.com
# May 2019

import pyplasma.pyplasma as pyplasma
from preproc import read_csvfile
from donor.vanilla import VanillaDonor
import numpy

if __name__ == "__main__":

    # creates a donor tvda that reads in the file sample_dsets/xda.csv and hasTarget = false
    tvda = VanillaDonor("sample_dsets/xda.csv",False, float,debug=True,verbose=True)
    tvda.display_internals()
    # creates a donor tvda that reads in the file sample_dsets/xdb_wtar.csv and hasTarget = true
    tvdb = VanillaDonor("sample_dsets/xdb_wtar.csv",True, float,debug=True,verbose=True)
    tvdb.display_internals()


    
            
