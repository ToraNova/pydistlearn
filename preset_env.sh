
# setups the environment variable for runtime loading of shared libraries
# use . present_env.sh to source it
# project pyplasma - ToraNova 2019
export PYTHV=python3.6
export MKLROOT=/opt/intel/mkl
export PRTROOT=/home/cjason/library/prodtools
export PLSROOT=$(pwd)/pyplasma/plasma-17.1
LD_LIBRARY_PATH=$PRTROOT/lib
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MKLROOT/lib/intel64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/pyplasma
echo LD_LIBRARY_PATH current environment :
echo $LD_LIBRARY_PATH
echo MKL/PRT/PLS ROOTS:
echo $MKLROOT $PRTROOT $PLSROOT
