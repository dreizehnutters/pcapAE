#!/bin/bash

./test_install.py


python3 -m torch.utils.collect_env

echo
echo "success"
awk 'BEGIN{
    s="/\\/\\/\\/\\/\\"; s=s s s s s s s s;
    for (colnum = 0; colnum<77; colnum++) {
        r = 255-(colnum*255/76);
        g = (colnum*510/76);
        b = (colnum*255/76);
        if (g>255) g = 510-g;
        printf "\033[48;2;%d;%d;%dm", r,g,b;
        printf "\033[38;2;%d;%d;%dm", 255-r,255-g,255-b;
        printf "%s\033[0m", substr(s,colnum+1,1);
    }
    printf "\n";
}'


rm -rf pcaps
rm -rf data

DEBUG="debug"
NAME="$DEBUG"
CUDA=$2
VERBOSE="-V"
NOBANNER="--no_banner"
WORKERS="--workers 0"
rev=$(tput rev)
blk=$(tput blink)
bld=$(tput bold)
clr=$(tput sgr0)

if [ -z "$1" ]; then
	# download pcaps
	mkdir pcaps
	wget -O pcaps/vali.pcap https://www.wireshark.org/download/automated/captures/randpkt-2020-09-06-16170.pcap
	wget -O pcaps/train.pcap https://www.wireshark.org/download/automated/captures/randpkt-2020-04-02-31746.pcap
	wget -O pcaps/fit.pcap https://www.wireshark.org/download/automated/captures/randpkt-2020-03-04-18423.pcap
	wget -O pcaps/predict.pcap https://www.wireshark.org/download/automated/captures/randpkt-2020-03-04-18423.pcap
	wget -O pcaps/random.pcap https://www.wireshark.org/download/automated/captures/fuzz-2007-12-17-17357.pcap

	# convert pcaps
	echo -e "$blk================================================================ $clr"
	echo "001$rev ${bld}pcap2ds.py -p pcaps/vali.pcap -o data/vali -m byte --chunk 1024 --threads 2 $clr"
	echo "$blk================================================================ $clr"
	python3 ../pcap2ds.py -p pcaps/vali.pcap -o data/vali -m byte --chunk 1024 --threads 2 --force
	ret=$?
	if [ $ret -ne 0 ]; then
		echo "Script 01 exited with error." >&2
		exit 0
	fi
	echo -e "\n$blk================================================================ $clr"
	echo "002$rev ${bld}pcap2ds.py -p pcaps/train.pcap -o data/train -m byte --chunk 1024 --threads 2 $clr"
	echo "$blk================================================================ $clr"
	python3 ../pcap2ds.py -p pcaps/train.pcap -o data/train -m byte --chunk 1024 --threads 2 --force
	ret=$?
	if [ $ret -ne 0 ]; then
		echo "Script 02 exited with error." >&2
		exit 0
	fi

	echo -e "\n$blk================================================================ $clr"
	echo "003$rev ${bld}pcap2ds.py -p pcaps/fit.pcap -o data/fit -m byte --chunk 1024 --threads 2 $clr"
	echo "$blk================================================================ $clr"
	python3 ../pcap2ds.py -p pcaps/fit.pcap -o data/fit -m byte --chunk 1024 --threads 2 --force
	ret=$?
	if [ $ret -ne 0 ]; then
		echo "Script 03 exited with error." >&2
		exit 0
	fi
fi

# *PCAPAE* API warapper
echo -e "\n$blk================================================================ $clr"
echo -e "${bld}training compressor $clr"
echo "004$rev ${bld}main.py -t data/train -v data/vali --verbose --epochs 2 $VERBOSE --name $NAME $WORKERS $CUDA $clr"
echo "$blk================================================================ $clr"
python3 ../main.py -t data/train -v data/vali --epochs 2 $VERBOSE --name $NAME $CUDA $WORKERS $NOBANNER
ret=$?
if [ $ret -ne 0 ]; then
	echo "Script 04 exited with error." >&2
	exit 0
fi

# ## retrain example
# #provide --[m]odel path
# echo -e "\n$blk================================================================ $clr"
# echo -e "${bld}re-training compressor  $clr"
# echo "004$rev ${bld}main.py -t data/train -v data/vali --verbose --epochs 2 --retrain --model runs/save_model/$DEBUG/best $CUDA $WORKERS $clr"
# echo "$blk================================================================ $clr"
# python3 ../main.py -t data/train -v data/vali --verbose --epochs 2 --retrain --model runs/save_model/$DEBUG/best $CUDA $WORKERS $NOBANNER
# ret=$?
# if [ $ret -ne 0 ]; then
# 	echo "Script 04 exited with error." >&2
# 	exit 0
# fi


## eval
echo -e "\n$blk================================================================ $clr"
echo "006$rev ${bld}pcaps2ds.py -p pcaps/predict.pcap -o data/predict -m byte --chunk 1024 --threads 2 $clr"
echo "$blk================================================================ $clr"
python3 ../pcap2ds.py -p pcaps/predict.pcap -o data/predict -m byte --chunk 1024 --threads 2 --force
ret=$?
if [ $ret -ne 0 ]; then
	echo "Script 06 exited with error." >&2
	exit 0
fi


echo -e "\n$blk================================================================ $clr"
echo "008$rev ${bld}pcaps2ds.py -p pcaps/random.pcap -o data/anomalie -m byte --chunk 1024 --threads 2 --bad $clr"
echo "$blk================================================================ $clr"
python3 ../pcap2ds.py -p pcaps/random.pcap -o data/anomalie -m byte --chunk 1024 --threads 2 --bad --force
ret=$?
if [ $ret -ne 0 ]; then
	echo "Script 08 exited with error." >&2
	exit 0
fi

echo -e "\n$blk================================================================ $clr"
echo -e "${bld}reducing fit and predict data $clr"
echo "009$rev ${bld}main.py --model runs/save_model/$DEBUG/best --fit data/fit --predict data/predict --name $NAME $WORKERS $CUDA $clr"
echo "$blk================================================================ $clr"
python3 ../main.py --model runs/save_model/$DEBUG/best --fit data/fit --predict data/predict --name $NAME $CUDA $WORKERS $NOBANNER
ret=$?
if [ $ret -ne 0 ]; then
	echo "Script 09 exited with error." >&2
	exit 0
fi

### train AD with reduced data
echo -e "\n$blk================================================================ $clr"
echo -e "${bld}training anomalie detection with reduced data $clr"
echo "010$rev ${bld}main.py --fit runs/redu_data/$DEBUG/data_fit_cpd --model ../ad/blueprints/base_lof.yaml --predict runs/redu_data/$DEBUG/data_predict_cpd --AD --verbose --name $NAME $WORKERS $CUDA $clr"
echo "$blk================================================================ $clr"
python3 ../main.py --fit runs/redu_data/$DEBUG/data_fit_cpd --model ../ad/blueprints/base_lof.yaml --predict runs/redu_data/$DEBUG/data_predict_cpd --AD --verbose $NOBANNER --name $NAME $WORKERS $CUDA
ret=$?
if [ $ret -ne 0 ]; then
	echo "Script 10 exited with error." >&2
	exit 0
fi

echo -e "\n$blk================================================================ $clr"
echo -e "${bld}compressing new data with saved model $clr"
echo "011$rev ${bld}main.py --model runs/save_model/$DEBUG/best --predict anomalie --name $NAME $WORKERS $CUDA $clr"
echo "$blk================================================================ $clr"
python3 ../main.py --model runs/save_model/$DEBUG/best --predict data/anomalie --name $NAME $CUDA $WORKERS $NOBANNER
ret=$?
if [ $ret -ne 0 ]; then
	echo "Script 11 exited with error." >&2
	exit 0
fi

echo -e "\n$blk================================================================ $clr"
echo -e "${bld}predicting AD model with new & compressed data $clr"
echo "012$rev ${bld}main.py --model runs/save_model/$DEBUG/AD/aba7718a_LocalOutlierFactor_model.jlib --predict runs/redu_data/$DEBUG/data_all_anomalie_cpd --verbose --name $NAME $WORKERS $CUDA $clr"
echo "$blk================================================================ $clr"
python3 ../main.py --model runs/save_model/$DEBUG/AD/aba7718a_LocalOutlierFactor_model.jlib --predict runs/redu_data/$DEBUG/data_anomalie_cpd --verbose --name $NAME $CUDA $WORKERS $NOBANNER 
ret=$?
if [ $ret -ne 0 ]; then
	echo "Script 12 exited with error." >&2
	exit 0
fi


echo -e "\n$blk================================================================ $clr"
echo -e "${bld}baseline: only pcapAE for Anomaly Detection loss based on training data $clr"
echo "013$rev ${bld}main.py --model runs/save_model/$DEBUG/best --predict data/predict --thr 0-0.5 --baseline pcapAE  --verbose --name $NAME $WORKERS $CUDA $clr"
echo "$blk================================================================ $clr"
python3 ../main.py --model runs/save_model/$DEBUG/best --predict data/predict --thr 0-0.5 --baseline pcapAE --verbose --name $NAME $CUDA $WORKERS $NOBANNER 
ret=$?
if [ $ret -ne 0 ]; then
	echo "Script 13 exited with error." >&2
	exit 0
fi


echo -e -"\n$blk================================================================ $clr"
echo -e "${bld}baseline: only pcapAE for Anomaly Detection loss based all anomalie data $clr"
echo "014$rev ${bld}main.py --model runs/save_model/$DEBUG/best --predict data/anomalie --thr 0-0.5 --baseline pcapAE  --verbose --name $NAME $WORKERS $CUDA $clr"
echo "$blk================================================================ $clr"
python3 ../main.py --model runs/save_model/$DEBUG/best --predict data/anomalie --thr 0-0.5 --baseline pcapAE --verbose --name $NAME $CUDA $WORKERS $NOBANNER
ret=$?
if [ $ret -ne 0 ]; then
	echo "Script 14 exited with error." >&2
	exit 0
fi
