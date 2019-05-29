# Test the preprocessor

import csv, numpy, pandas
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, LabelEncoder
from preproc.neoctl import NeoNPDController
import argparse, json

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("datafile", type=str, help="Input datafile for clients (can be found in sample_dsets)")
    ap.add_argument("--preproc","-p", type=str, help="Preproessing directives (.json file)")
    ap.add_argument("--outfile","-o", type=str ,help="Save the processed output file as")
    ap.add_argument("--showcase","-s",action="store_true",help="show the data loaded")
    ap.add_argument("--coldisp","-c",type=str, help="Display a certain column")

    args = ap.parse_args()
    mc = NeoNPDController( args.datafile ,verbose=True,debug=True)
    if( args.showcase ):
        mc.head()
        print()
        print()
        mc.emptyrows()
        print()
        print()
        mc.emptycols()

    elif( args.coldisp is not None):
        mc.printcertain( args.coldisp )

    elif( args.preproc is not None):
        if( args.outfile is None):
            print("If preprocess specified, please specify also outfile")
            exit(1)
        operres = {}
        with open( args.preproc ) as f:
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
                        mc.preproc( dp.get("process"),
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
            print(k,v)

        print("Performing rewrite")
        mc.rewrite( args.outfile )
        exit(0)



