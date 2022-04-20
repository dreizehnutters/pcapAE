#!/bin/bash

# options menu
DO_PRE="y"
DO_FL="y"
DO_RE=""
DO_MIN="y"
DO_EVAL="y"
TRAIN_AD="y"
TEST_AD="y"
BASELINE="y"

# hyperparameters controler
SET="SWAT_A6"
#####
EPOCHS="1"
MODI="byte"
FINPUT="1"
LRATE="0.00005"
BSIZE="2"
CELL="GRU" #--cell
LOSS="MSE" #--loss
SHED="cycle" #--scheduler
DROPOUT="0.1" #--dropout
FOUTPUT="0" #--foutput
NAME_POFI="${SET}/${MODI}_F${FINPUT}_L${LRATE}_S${BSIZE}_C${CELL}"
#NAME_POFI="${SET}/${MODI}_F${FINPUT}_O${FOUTPUT}_L${LRATE}_S${BSIZE}_C${CELL}"

# env paths
PROD=""
THREADS="1"
CUDA=""
PY_BIN=/usr/bin/python3.7
SRC_DIR="/home/geb/Workspace/gits/thesis/src"
if [[ $PROD ]]; then
	THREADS="4"
	CUDA="--CUDA"
	PY_BIN=/usr/bin/python3.7
	SRC_DIR="/home/reach_fmk/thesis/src"
fi
WORK_DIR=$SRC_DIR/../exp/$SET
echo "================="
echo -e "WORKING DIR: ${WORK_DIR}"
echo "================="


# AD settings
ALGO_PRINTS="${SRC_DIR}/ad/blueprints"
#SALT="2a0c083c";ALGO="OneClassSVM"
SALT="aba7718a";ALGO="LocalOutlierFactor"
#SALT="b9053c7c";ALGO="IsolationForest"
THE_MODEL=${SALT}_${ALGO}

# pcap paths
PCAP_PATH="../pcaps"
TRAIN_PCAP="train.pcap"
VALI_PCAP="vali.pcap"
FIT_PCAP="fit.pcap"
PREDICT_PCAP="predict.pcap"
EVAL_PCAP="eval.pcap"

# terminal output
VERBOSE="--verbose"
rev=""
blk=""
bld=""
clr=""
FANCY="yes"
if [[ $FANCY ]]; then
	rev=$(tput rev)
	blk=$(tput blink)
	bld=$(tput bold)
	clr=$(tput sgr0)
fi
#%#%#%#%#%#%#%#%#%#%#%


## prepare data
if [[ $DO_PRE ]]; then
	echo -e "\n$blk================================================================ $clr"
	echo "01$rev ${bld}pcaps2ds.py -p ${PCAP_PATH}/${TRAIN_PCAP} -o ${SET}/train_${MODI} -m ${MODI} --threads $THREADS $clr"
	echo "$blk================================================================ $clr"
	$PY_BIN ${SRC_DIR}/pcap2ds.py -p ${PCAP_PATH}/${TRAIN_PCAP} -o ${SET}/train_${MODI} -m ${MODI} --chunk 1024 --threads $THREADS --force
	ret=$?
	if [ $ret -ne 0 ]; then
		echo "Script 01 exited with error." >&2
		exit 0
	fi

	echo -e "\n$blk================================================================ $clr"
	echo "02$rev ${bld}pcaps2ds.py -p ${PCAP_PATH}/${VALI_PCAP} -o ${SET}/vali_${MODI} -m ${MODI} --threads $THREADS $clr"
	echo "$blk================================================================ $clr"
	$PY_BIN  ${SRC_DIR}/pcap2ds.py -p ${PCAP_PATH}/${VALI_PCAP} -o ${SET}/vali_${MODI} -m ${MODI} --chunk 1024 --threads $THREADS --force
	ret=$?
	if [ $ret -ne 0 ]; then
		echo "Script 02 exited with error." >&2
		exit 0
	fi

	echo -e "\n$blk================================================================ $clr"
	echo "03$rev ${bld}pcaps2ds.py -p ${PCAP_PATH}/${FIT_PCAP} -o ${SET}/fit_${MODI} -m ${MODI} --threads $THREADS $clr"
	echo "$blk================================================================ $clr"
	$PY_BIN ${SRC_DIR}/pcap2ds.py -p ${PCAP_PATH}/${FIT_PCAP} -o ${SET}/fit_${MODI} -m ${MODI} --chunk 1024 --threads $THREADS --force
	ret=$?
	if [ $ret -ne 0 ]; then
		echo "Script 03 exited with error." >&2
		exit 0
	fi

	echo -e "\n$blk================================================================ $clr"
	echo "04$rev ${bld}pcaps2ds.py -p ${PCAP_PATH}/${PREDICT_PCAP} -o ${SET}/predict_${MODI} -m ${MODI} --threads $THREADS $clr"
	echo "$blk================================================================ $clr"
	$PY_BIN  ${SRC_DIR}/pcap2ds.py -p ${PCAP_PATH}/${PREDICT_PCAP} -o ${SET}/predict_${MODI} -m ${MODI} --chunk 1024 --threads $THREADS --force
	ret=$?
	if [ $ret -ne 0 ]; then
		echo "Script 04 exited with error." >&2
		exit 0
	fi

	# ## gen ground truth
	# echo -e "\n$blk================================================================ $clr"
	# echo "05$rev ${bld}pcap2GT.py -p ${PCAP_PATH}/${EVAL_PCAP} -r ${PCAP_PATH}/${VALI_PCAP} $VERBOSE $clr"
	# echo "$blk================================================================ $clr"
	# $PY_BIN ${SRC_DIR}/pcap2GT.py -p ${PCAP_PATH}/${EVAL_PCAP} -r ${PCAP_PATH}/${VALI_PCAP} $VERBOSE
	# ret=$?
	# if [ $ret -ne 0 ]; then
	# 	echo "Script 05 exited with error." >&2
	# 	exit 0
	# fi

	echo -e "\n$blk================================================================ $clr"
	echo "06$rev ${bld}pcaps2ds.py -p ${PCAP_PATH}/${EVAL_PCAP} -o ${SET}/eval_${MODI} -m ${MODI} --threads $THREADS -g ${PCAP_PATH}/${EVAL_PCAP}_GT.csv $clr"
	echo "$blk================================================================ $clr"
	$PY_BIN ${SRC_DIR}/pcap2ds.py -p ${PCAP_PATH}/${EVAL_PCAP} -o ${SET}/eval_${MODI} -m ${MODI} --chunk 1024 --threads $THREADS -g ${PCAP_PATH}/${EVAL_PCAP}_GT.csv --force
	ret=$?
	if [ $ret -ne 0 ]; then
		echo "Script 06 exited with error." >&2
		exit 0
	fi
fi


## machine learning
if [[ $DO_FL ]]; then

	if [[ $DO_RE ]]; then
		### RETRAIN 
		echo -e "\n$blk================================================================ $clr"
		echo -e "${bld}REtraining compressor $clr"
		echo "07a$rev ${bld}main.py -t ${SET}/train_$MODI -v ${SET}/vali_$MODI --retrain --model ${WORK_DIR}/save_model/${NAME_POFI}/best --epochs $EPOCHS $VERBOSE --name ${NAME_POFI} $CUDA --dir $WORK_DIR --learn_rate $LRATE --batch_size $BSIZE --finput $FINPUT $clr"
		echo "$blk================================================================ $clr"
		$PY_BIN ${SRC_DIR}/main.py -t ${SET}/train_$MODI -v ${SET}/vali_$MODI --retrain --model ${WORK_DIR}/save_model/SWAT/byte_F1_L0.05_S2/LAST_checkpoint_XXXXXXX.pth.tar --epochs $EPOCHS $VERBOSE --name ${NAME_POFI} $CUDA --dir ${WORK_DIR} --learn_rate $LRATE --batch_size $BSIZE --finput $FINPUT
		ret=$?
		if [ $ret -ne 0 ]; then
			echo "Script 07a exited with error." >&2
			exit 0
		fi

	else
		### TRAIN 
		echo -e "\n$blk================================================================ $clr"
		echo -e "${bld}training compressor $clr"
		echo "07$rev ${bld}main.py -t ${SET}/train_$MODI -v ${SET}/vali_$MODI --epochs 2 $VERBOSE --name ${NAME_POFI} $CUDA --dir $WORK_DIR --learn_rate $LRATE --dropout $DROPOUT --batch_size $BSIZE --finput $FINPUT --foutput $FOUTPUT $clr"
		echo "$blk================================================================ $clr"
		$PY_BIN ${SRC_DIR}/main.py -t ${SET}/train_$MODI -v ${SET}/vali_$MODI --epochs $EPOCHS $VERBOSE --name ${NAME_POFI} $CUDA --dir ${WORK_DIR} --learn_rate $LRATE --dropout $DROPOUT --batch_size $BSIZE --finput $FINPUT --foutput ${FOUTPUT}
		ret=$?
		if [ $ret -ne 0 ]; then
			echo "Script 07 exited with error." >&2
			exit 0
		fi
	fi
fi

if [[ $DO_MIN ]]; then
	### GEN REDU DATA WITH TRAINED MODEL
	echo -e "\n$blk================================================================ $clr"
	echo -e "${bld}reducing fit and predict data $clr"
	echo "08$rev ${bld}main.py --model ${WORK_DIR}/save_model/${NAME_POFI}/best --fit ${SET}/fit_$MODI --predict ${SET}/predict_$MODI --name ${NAME_POFI} $CUDA --dir $WORK_DIR $clr"
	echo "$blk================================================================ $clr"
	$PY_BIN  ${SRC_DIR}/main.py --model ${WORK_DIR}/save_model/${NAME_POFI}/best --fit ${SET}/fit_$MODI --predict ${SET}/predict_$MODI --name ${NAME_POFI} $CUDA --dir ${WORK_DIR}
	ret=$?
	if [ $ret -ne 0 ]; then
		echo "Script 08 exited with error." >&2
		exit 0
	fi
fi

if [[ $DO_EVAL ]]; then
	echo -e "\n$blk================================================================ $clr"
	echo -e "${bld}compressing new data with saved model $clr"
	echo "09$rev ${bld}main.py --model ${WORK_DIR}/save_model/${NAME_POFI}/best --predict ${SET}/eval_$MODI --name ${NAME_POFI} $CUDA --dir $WORK_DIR $clr"
	echo "$blk================================================================ $clr"
	$PY_BIN  ${SRC_DIR}/main.py --model ${WORK_DIR}/save_model/${NAME_POFI}/best --predict ${SET}/eval_$MODI --name ${NAME_POFI} $CUDA --dir ${WORK_DIR}
	ret=$?
	if [ $ret -ne 0 ]; then
		echo "Script 09 exited with error." >&2
		exit 0
	fi
fi


## handle shallow learning
if [[ $TRAIN_AD ]]; then
	### train AD with reduced data
	echo -e "\n$blk================================================================ $clr"
	echo -e "${bld}training anomalie detection with reduced data $clr"
	echo "10$rev ${bld}main.py --fit ${WORK_DIR}/redu_data/${NAME_POFI}/${SET}_fit_${MODI}_cpd --model ${ALGO_PRINTS}/${ALGO}.yaml --predict ${WORK_DIR}/redu_data/${NAME_POFI}/${SET}_predict_${MODI}_cpd --AD $VERBOSE --name AD_${SET}_${MODI}_finput${FINPUT} $CUDA --dir $WORK_DIR $clr"
	echo "$blk================================================================ $clr"
	$PY_BIN  ${SRC_DIR}/main.py --fit ${WORK_DIR}/redu_data/${NAME_POFI}/${SET}_fit_${MODI}_cpd --model ${ALGO_PRINTS}/${ALGO}.yaml --predict ${WORK_DIR}/redu_data/${NAME_POFI}/${SET}_predict_${MODI}_cpd --AD $VERBOSE --name AD_${SET}_${MODI}_finput${FINPUT} $CUDA --dir ${WORK_DIR}
	ret=$?
	if [ $ret -ne 0 ]; then
		echo "Script 10 exited with error." >&2
		exit 0
	fi
fi


### eval AD
if [[ $TEST_AD ]]; then
	echo -e "\n$blk================================================================ $clr"
	echo -e "${bld}predicting AD model with new & compressed data $clr"
	echo "11$rev ${bld}main.py --model ${WORK_DIR}/save_model/AD_${SET}_${MODI}_finput${FINPUT}/AD/${THE_MODEL}_model.jlib --predict ${WORK_DIR}/redu_data/${NAME_POFI}/${SET}_eval_${MODI}_cpd $VERBOSE --name EVAL_${SET}_${MODI}_finput${FINPUT}_${ALGO} $CUDA --dir $WORK_DIR $clr"
	echo "$blk================================================================ $clr"
	$PY_BIN  ${SRC_DIR}/main.py --model ${WORK_DIR}/save_model/AD_${SET}_${MODI}_finput${FINPUT}/AD/${THE_MODEL}_model.jlib --predict ${WORK_DIR}/redu_data/${NAME_POFI}/${SET}_eval_${MODI}_cpd $VERBOSE --name EVAL_${SET}_${MODI}_finput${FINPUT}_${ALGO} $CUDA --dir ${WORK_DIR} $NOBANNER 
	ret=$?
	if [ $ret -ne 0 ]; then
		echo "Script 11 exited with error." >&2
		exit 0
	fi
fi


## base line
if [[ $BASELINE ]]; then
	echo -e -"\n$blk================================================================ $clr"
	echo -e "${bld}baseline: only pcapAE for Anomaly Detection (loss based) all normal data $clr"
	echo "014$rev ${bld}main.py --model ${WORK_DIR}/save_model/${NAME_POFI}/best --predict ${SET}/predict_$MODI --baseline pcapAE --thr 0.05-0.9 --verbose --name ${NAME_POFI}_baseline $CUDA --dir $WORK_DIR $clr"
	echo "$blk================================================================ $clr"
	$PY_BIN ${SRC_DIR}/main.py --model ${WORK_DIR}/save_model/${NAME_POFI}/best --predict ${SET}/predict_$MODI --baseline pcapAE --thr 0.05-0.9 --verbose --name ${NAME_POFI}_baseline $DEBUG $CUDA --dir ${WORK_DIR} $WORKERS $NOBANNER
	ret=$?
	if [ $ret -ne 0 ]; then
		echo "Script 14 exited with error." >&2
		exit 0
	fi

	echo -e -"\n$blk================================================================ $clr"
	echo -e "${bld}baseline: only pcapAE for Anomaly Detection (loss based) all anomalie data $clr"
	echo "014$rev ${bld}main.py --model ${WORK_DIR}/save_model/${NAME_POFI}/best --predict ${SET}/eval_$MODI --baseline pcapAE --thr 0.05-0.9 --verbose --name ${NAME_POFI}_eval_baseline $CUDA --dir $WORK_DIR $clr"
	echo "$blk================================================================ $clr"
	$PY_BIN ${SRC_DIR}/main.py --model ${WORK_DIR}/save_model/${NAME_POFI}/best --predict ${SET}/eval_$MODI --baseline pcapAE --thr 0.05-0.9 --verbose --name ${NAME_POFI}_eval_baseline $DEBUG $CUDA --dir ${WORK_DIR} $WORKERS $NOBANNER
	ret=$?
	if [ $ret -ne 0 ]; then
		echo "Script 14 exited with error." >&2
		exit 0
	fi
fi
