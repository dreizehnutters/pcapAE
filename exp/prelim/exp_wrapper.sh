#!/bin/bash

# wrapper to test sklean dim redu alogs 
# ./exp_wrapper.sh 2>&1 | tee all.log

StringVal="dummy pca kpca ica sPCA lle se iso"

# ARG1 data set
# ARG2 fraction of data
# ARG3 log name

echo -e "dim redu exp" | tee exp_$3.log
for val in $StringVal; do
	echo "[*] testing: "$val
	/usr/bin/free 2>&1 | tee -a exp_$3.log
	echo
	/usr/bin/time -f 'real, user, sys:  %E, %U, %S' python3 dim_redu.py --modus $val --data $1 --fraction $2 2>&1 | tee -a exp_$3.log
	echo
	echo
	echo
done

