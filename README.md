# pydistlearn

### A project to provide distributed machine learning

### Uses pyplasma, a python wrapped PLASMA

### Dependencies:
* numpy
* prodtools
* pyioneer
* Intel's math kernel library (MKL)
* CMAKE 3.14.x

### Math Kernel Library

Please install intel's math kernel library (https://software.seek.intel.com/performance-libraries) 
run install.sh (if you run it with root permission, it installs onto /opt, else it will install on 
the user's home directory). The installation path is essential as we need to link the library later

#### Installation:

Install pre-requisites

	apt-get install swig

Compile the custom library prodtools and pyioneer

	git clone https://github.com/ToraNova/library
	cd library/prodtools && ./cmake_build.sh && cd ../pyioneer && sudo ./install.sh

Clone this repository and initialize pyplasma's repo

	git clone https://github.com/ToraNova/pydistlearn
	cd pydistlearn && git submodule init

Edit the preset_env file to point to the roots for MKL (use your favourite editor), afterwhich, 
source the preset_env.sh file to your environment.

	. preset_env.sh

Compile

	make

To run, please load the shared object first

	. preload_env.sh

### Run some tests
PLASMA tests are available on this file : [plasma_test.py](plasma_test.py)

	python3 plasma_test.py

expect true on last line

### Examples
Examples are found with [donor_test.py](donor_test.py) and [central_test.py](central_test.py)

##### TODO: 
* fix the readme
* scripts to install the whole lib
* setup example for client/server
* preprocessing functionalities on python sklearn
* 	data - normalization (mean = 0 , std - 1)
* 	categotical - numerical
* 	cleaning data (what to do with missing data) - user entry/ mean-median-mode
* 	calculate correlation btwn diff column and datasets (on donor's datatset)
* 	calculating the distribution of each column (statistics for each column)
* 	train/test
* evaluation of models ( method of eval, RMSC root mean square error)
* real world use cases
* predicting the length of stay ( insurance )
* cost of hospitalization
