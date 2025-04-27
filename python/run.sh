#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################
E_BADARGS=65
if [ $# -ne 1 ]
then
	echo "Usage: `basename $0` <input>"
	exit $E_BADARGS
fi

input=$1
source p5_venv/bin/activate || exit 1

install_loc=/home/alanxw/school/csci2951o/ubuntu-docker/CPLEX_Studio2211
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${install_loc}/cplex/bin/x86-64_linux/cplex
export DOCPLEX_COS_LOCATION=${install_loc}

# run the solver
python3.9 src/main.py $input
