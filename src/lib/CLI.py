from os import path, getcwd
from sys import argv
from multiprocessing import cpu_count
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from lib.utils import BOLD, CLR, ACTION

#######
# CLI #
#######
PARSER = ArgumentParser(description="[+] pcapAE wrapper", formatter_class=ArgumentDefaultsHelpFormatter)
PARSER.add_argument('-t',
                    '--train',
                    help=f"{BOLD}<path>{CLR} to dataset to learn",
                    metavar='',
                    default=None,
                    required=False)
PARSER.add_argument('-v',
                    '--vali',
                    metavar='',
                    default=None,
                    help=f"{BOLD}<path>{CLR} to dataset to validate",
                    required=False)
PARSER.add_argument('-f','--fit',
                    help=f"{BOLD}<path>{CLR} to data set to fit AD",
                    metavar='',
                    default='',
                    required=False)
PARSER.add_argument('-p','--predict',
                    help=f"{BOLD}<path>{CLR} to data to make a predict on",
                    metavar='',
                    default='',
                    required=False)
PARSER.add_argument('--eval',
                    help=f"{BOLD}<path>{CLR} to data to make a eval on",
                    metavar='',
                    default='',
                    required=False)
PARSER.add_argument('-m','--model',
                    help=f"{BOLD}<path>{CLR} to model to retrain or evaluate",
                    metavar='',
                    type=str,
                    default='',
                    required=False)
PARSER.add_argument('-b',
                    '--batch_size',
                    metavar='',
                    choices=[1]+[2**n for n in range(1,12)],
                    default=128,
                    type=int,
                    help=f'{BOLD}<number>{CLR} of samples per pass')
PARSER.add_argument('-lr',
                    '--learn_rate',
                    metavar='',
                    default=1e-3,
                    type=float,
                    help=f'starting learning {BOLD}<rate>{CLR} between | [1,0)')
PARSER.add_argument('-fi',
                    '--finput',
                    default=1,
                    metavar='',
                    type=int,
                    help=f'{BOLD}<number>{CLR} input frames')
PARSER.add_argument('-fo',
                    '--foutput',
                    metavar='',
                    default=0,
                    type=int,
                    help=f'{BOLD}<number>{CLR} predict frames. 0 - predict input')
PARSER.add_argument('-o',
                    '--optim',
                    default='adamW',
                    choices=['adam', 'adamW' ,'sgd'],
                    metavar='',
                    type=str,
                    help=f'gradient decent {BOLD}<strategy>{CLR} | [adam | adamW | sgd]')
PARSER.add_argument('-c',
                    '--clipping',
                    metavar='',
                    type=float,
                    default=10.0,
                    help=f"gradient clip {BOLD}<value>{CLR} | [0,10]",
                    required=False)
PARSER.add_argument('--fraction',
                    type=float,
                    default=1,
                    metavar='',
                    help=f"{BOLD}<fraction>{CLR} of data to process | (0, 1]",
                    required=False)
PARSER.add_argument('-w',
                    '--workers',
                    default=0,
                    metavar='',
                    type=int,
                    help=f'{BOLD}<number>{CLR} of data loader worker threads | [0, 8]')
PARSER.add_argument('--loss',
                    default='MSE',
                    metavar='',
                    choices=['MSE', 'BCE'],
                    type=str,
                    help=f'loss {BOLD}<criterion>{CLR} | [MSE | BCE]')
PARSER.add_argument('--scheduler',
                    default='cycle',
                    metavar='',
                    choices=['step', 'cycle', 'plateau'],
                    type=str,
                    help=f'learn rate {BOLD}<scheduler>{CLR} | [step | cycle | plateau]')
PARSER.add_argument('--epochs',
                    default=144,
                    type=int,
                    metavar='',
                    help=f'{BOLD}<number>{CLR} of epochs')
PARSER.add_argument('--cell',
                    default='GRU',
                    choices=['GRU', 'LSTM'],
                    metavar='',
                    type=str,
                    help=f'network cell {BOLD}<type>{CLR} | [GRU | LSTM]')
PARSER.add_argument('--no_bn',
                    default=False,
                    help='disable batch norm layers',
                    action='store_true')
PARSER.add_argument('--dropout',
                    type=float,
                    default=0,
                    metavar='',
                    help=f"{BOLD}<dropout>{CLR} value | [0, 1]",
                    required=False)
PARSER.add_argument('--seed',
                    metavar='',
                    help=f'{BOLD}<seed>{CLR} to fixing randomness',
                    default=1994,
                    type=int)
PARSER.add_argument('--noTensorboard',
                    action='store_true',
                    help="do not start tensorboard")
PARSER.add_argument('-d','--dir',
                    default=f"{getcwd()}/runs",
                    type=str,
                    help=f'experiment working directory {BOLD}<path>{CLR}')
PARSER.add_argument('--cuda',
                    '--CUDA','--GPU',
                    help='enable GPU support',
                    action='store_true')
PARSER.add_argument('-V',
                    '--verbose',
                    default=False,
                    help='print verbose messages',
                    action='store_true')
PARSER.add_argument('--noCache',
                    default=False,
                    help='disable caching',
                    action='store_true')
PARSER.add_argument('--retrain',
                    default=False,
                    help='retrain given model',
                    action='store_true')
PARSER.add_argument('--name',
                    default=None,
                    type=str,
                    help=f'experiment {BOLD}<name>{CLR} prefix')
PARSER.add_argument('--AD',
                    default=False,
                    help='use anomaly detection module',
                    action='store_true')
PARSER.add_argument('--baseline',
                    choices=[False, 'pcapAE' ,'noDL'],
                    default=False,
                    help='compute raw baseline')
PARSER.add_argument('--grid_search',
                    default=False,
                    help='do grid search',
                    action='store_true')
PARSER.add_argument('--n_jobs',
                    metavar='',
                    choices=range(1, cpu_count()-2),
                    default=1,
                    type=int,
                    help=f'{BOLD}<number>{CLR} of CPU cores to use [only for LOF & IF]')
PARSER.add_argument('-thr',
                    '--threshold',
                    default='',
                    metavar='',
                    type=str,
                    help=f'{BOLD}<min_thr-max_thr>{CLR} threshold  for normal data | default None')
PARSER.add_argument('--no_banner', action='store_true')
PARSER.usage = f"""
{BOLD}# AE training {CLR}
python3 main.py --train <TRAIN_SET_PATH> --vali <VALI_SET_PATH> [--cuda]

{BOLD}# AE data compression (pcap -> _codes_) {CLR}
python3 main.py --model <PCAPAE_MODEL> --fit <FIT_SET_PATH> --predict <PREDICT_SET_PATH> [--cuda]

{BOLD}# shallow ML anomaly detection training {CLR}
python3 main.py --AD --model *.yaml --fit <REDU_FIT_SET_PATH> [--predict <REDU_PREDICT_SET_PATH>] [--grid_search]

{BOLD}# test triang AD on new data {CLR}
python3 main.py --model <AD_MODLE_PATH> --predict <REDU_SET_PATH>

{BOLD}# only deep learning baseline {CLR}
python3 main.py --baseline pcapAE --model <PCAPAE_MODEL> --predict <PREDICT_SET_PATH>

{BOLD}# no deep leanring baseline {CLR}
python3 main.py --baseline noDL --AD --model ../test/blueprints/base_if.yaml --fit <FIT_SET_PATH> --vali <VALI_SET_PATH> --predict <PREDICT_SET_PATH> 

{BOLD}====={CLR}
"""


# verify input patterns
ARGS = PARSER.parse_args()
help_text = ""
from os import path
if len(argv) == 1:
    help_text = "[!] no arguments passed! :<"
else:
    if all([ARGS.retrain,\
            ARGS.model == '']):
        help_text = "[!] --retrain needs a --[m]odel <path>"

    if all([ARGS.model,
            ARGS.retrain is False,
            ARGS.fit == '']) and all([x not in ARGS.model for x in ['save_model', 'AD']]):
        help_text = "[!] --[m]odel option needs either --retrain flag or --[f]it data to prepare for AD"

    if ARGS.baseline != 'noDL' and all([ARGS.train is None,\
            ARGS.vali]):
        if ARGS.eval is '':
            help_text = "[!] no --[t]rain set specified"

    if all([ARGS.predict == '',\
            ARGS.fit == '',\
            ARGS.train is None,\
            ARGS.vali is None]):
        help_text = "[!] no data provided"

    if all([ARGS.train is None,\
            ARGS.vali is None,\
            ARGS.model == '',\
            ARGS.AD is False,\
            'redu_data' not in ARGS.predict]):
        help_text = "[!] no --[t]rain set specified"

    if all([ARGS.AD,\
            ARGS.fit == '']):
        help_text = "[!] no --[f]it data provided!"

    if ARGS.model != '':
        if ('yaml' not in ARGS.model) and ('.pth.tar' not in ARGS.model) and ('_checkpoint_' not in ARGS.model) and all([x not in ARGS.model for x in ['save_model', 'AD']]):
            exit(f"[!] '{ARGS.model}' is not a valid path for a pytorch model")

        if ARGS.AD and (ARGS.fit != '' or ARGS.predict != '') and ARGS.baseline != 'noDL':
            from glob import glob
            for ds in [ARGS.fit, ARGS.predict]:
                test = (glob(ARGS.fit+'/*'))
                for t in test:
                    if '.hdf5' in t:
                        help_text = "[!] provided data set is no suitable for anomaly detection! Transform the data first."
    elif all([ARGS.model == '',\
              'redu_data' not in ARGS.fit,\
              'redu_data' not in ARGS.predict,\
              ARGS.AD == True,\
              ARGS.baseline == False]):
        # TODO relax rule for baseline compression
        help_text = "[!] --[f]it data must be in reduced form since no --model was provided"
    
if help_text != "":
    PARSER.print_help()
    print()
    exit(help_text)

if not ARGS.no_banner:
    print("""  ______  ______  ______  ______     ______  ______    
 /\  __ \/\  ___\/\  __ \/\  __ \   /\  __ \/\  ___\   
 \ \  __/\ \ \___\ \  __ \ \  __/   \ \  __ \ \  _\_   
  \ \_\   \ \_____\ \_\ \_\ \_\      \ \_\ \_\ \_____\ 
   \/_/    \/_____/\/_/\/_/\/_/       \/_/\/_/\/_____/""")
