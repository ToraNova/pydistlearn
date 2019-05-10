#!/bin/bash
# preloads some shared library
if [ -z $MKLROOT ]
then
        echo "Warning: MKLROOT not set!"
        exit 0
fi

if [ -z $1 ]
then
	# Intel bug mitigation
	export LD_PRELOAD="$MKLROOT/lib/intel64/libmkl_core.so $MKLROOT/lib/intel64/libmkl_sequential.so"
        echo LD_PRELOADS:
        echo $LD_PRELOAD
        echo "Use . preload_env.sh <any> to unload"
else
        export LD_PRELOAD=""
        echo "Clearing PRELOADS"
fi
