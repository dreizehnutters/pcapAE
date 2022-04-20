#! /usr/bin/python3
# main
from sys import exit, argv, path as spath
from signal import SIGTERM
from subprocess import Popen, DEVNULL, check_output, PIPE
from os import path, getcwd, setsid, environ, killpg, getpgid

import torch
import numpy as np
from coolname import generate_slug
spath.insert(0, '../')
from lib.utils import *
from lib.pcapAE import PcapAE
from lib.earlystopping import EarlyStopping
# pcap2ds
import os
import shutil
import multiprocessing as mp
from sys import exit
from csv import reader
from glob import glob
from ipaddress import ip_address
from subprocess import check_output
from time import sleep, process_time
from socket import  inet_ntop, AF_INET, AF_INET6
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import h5py
import dpkt
import numpy as np
from tqdm import tqdm
from dpkt import pcap, pcapng


# utils
import re
from math import log, e
from datetime import datetime
from collections import OrderedDict
from os import path, walk, makedirs, getcwd

import matplotlib.pyplot as plt
from torch import nn, no_grad, float32
from PIL import Image
from io import BytesIO
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from spatialentropy import altieri_entropy
from scipy.stats import entropy as scipy_entropy
from numpy import unique, arange, array, ndindex, count_nonzero, zeros, argmax, maximum, mean, random, linspace


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc

import h5py
import glob

from lib.H5Dataset import H5Dataset
from rich import box
from rich.table import Table
from rich import print as rprint

# pcapAE
from sys import exit
from os import path, walk, makedirs
import numpy as np
import torch
from torch import nn, no_grad
import torch.optim as optim
from torch.optim import lr_scheduler

from tqdm import tqdm
from lib.utils import *
from lib.encoder import Encoder
from lib.decoder import Decoder
from lib.model import AutoEncoder
from lib.H5Dataset import H5Dataset

# CLI.py
from os import path
from sys import argv
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# AD
from ad.utils import *
from ad.ad import *

for x in range(11):rprint(f"{color_switch(x/10)}{x/10}",end='/')
print('\nif no colors appear check terminal config / tmux (set -g default-terminal "screen-256color")')
print(f"{ACTION} {BOLD}bold text{CLR} normal text")
print(f"{ACTION} ANOMALIE/POSITIVE LABLE: {BOLD}{ANOMALY}{CLR}")
print(f"{ACTION} NORMAL/NEGATIVE LABLE: {BOLD}{NORMAL}{CLR}")
table = Table(title=f"[b]DEMO", box=box.ASCII)
table.add_column("unicode arrow down",justify="center")
table.add_column("unicode arrow up",justify="center")
table.add_row(f"↓",\
              f"↑")
rprint(table)

print("if no unicode chars are shown -> check 'dpkg -l locales' config!")

from time import sleep
for i in tqdm(range(10)):
    sleep(.1)

import scipy
import rich
import coolname
import matplotlib
import tensorboard
import sklearn
import spatialentropy

for x in [dpkt, h5py, scipy, np, coolname, matplotlib, tensorboard, sklearn, ]:print(f"{x}{BOLD}=={x.__version__}{CLR}")

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)


print("plotting to X11")
fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

plt.show()
