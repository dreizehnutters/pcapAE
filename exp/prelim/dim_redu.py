import sys
sys.path.append('../src/pcapAE')
from H5Dataset import H5Dataset
import numpy as np
import psutil
from argparse import ArgumentParser

from math import log, e
def entropy2(labels, base=None):
  """ Computes entropy of label distribution. """

  n_labels = len(labels)

  if n_labels <= 1:
    return 0

  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
    return 0

  ent = 0.

  # Compute entropy
  base = e if base is None else base
  for i in probs:
    ent -= i * log(i, base)

  return ent


from resource import getrusage as resource_usage, RUSAGE_SELF
from time import time as timestamp


def unix_time(function, args=tuple(), kwargs={}):
    '''Return `real`, `sys` and `user` elapsed time, like UNIX's command `time`
    You can calculate the amount of used CPU-time used by your
    function/callable by summing `user` and `sys`. `real` is just like the wall
    clock.
    Note that `sys` and `user`'s resolutions are limited by the resolution of
    the operating system's software clock (check `man 7 time` for more
    details).
    '''
    start_time, start_resources = timestamp(), resource_usage(RUSAGE_SELF)
    a = function(*args, **kwargs)
    end_resources, end_time = resource_usage(RUSAGE_SELF), timestamp()

    print(f'real: {end_time - start_time}\nsys: {end_resources.ru_stime - start_resources.ru_stime}\nuser: {end_resources.ru_utime - start_resources.ru_utime}')

    return a







PARSER = ArgumentParser(description="")
PARSER.add_argument('-m', '--modus', default='pca', help="algo: pca, kpca, ics, sPCA, iso, lle, se")
PARSER.add_argument('-d', '--data', required=True, help="path to data")
PARSER.add_argument('-f', '--fraction', default=1, help="fraction of data to process")
PARSER.add_argument('-t', '--target', default=64, help="target number of dimensions")
ARGS = PARSER.parse_args()

# number of dimension to reduce to
target_dim = int(ARGS.target)
# fraction of data to load
fraction_of_data = float(ARGS.fraction)
print(fraction_of_data)
# this may take some time / subject to change
raw_data = H5Dataset(ARGS.data, train=True,
                        n_frames_input=1,
                        n_frames_output=0,
                        fraction=fraction_of_data,
                        verbose=True)
tmp = []

for x in range(len(raw_data)):
    tmp.append(np.array(raw_data[x][1]))
    

arr = np.array(tmp)
arr = arr.reshape((len(raw_data)*(raw_data.n_frames_input+raw_data.n_frames_output),1,32,32))
data = arr[:,0]
data = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
data = data.astype(np.float32)
print(f"[*] dimensions of test data: {data.shape} ~{fraction_of_data*100}% of the data")

from scipy.stats import iqr
from scipy.stats import entropy
print(f"[*] mean: {data.mean()}")
print(f"[*] median: {np.median(data)}")
print(f"[*] average: {np.average(data)}")
print(f"[*] max: {data.max()}")
print(f"[*] min: {data.min()}")
print(f"[*] std: {data.std()}")
print(f"[*] variance: {data.var()}")
print(f"[*] interquartile range: {iqr(data)}")
print(f"[*] 50% quantile: {np.quantile(data, .5)}")

avg_entropy = []
for x in range(data.shape[0]):
    avg_entropy.append(entropy2(data[x]))
    
print(f"[*] avg entropy: {np.average(avg_entropy)}")
avg_entropy = []

del tmp
del arr
del raw_data


selector = ARGS.modus
# print(f"[*] memory\n{psutil.virtual_memory()}")

print("[*]", selector)
if selector == 'dummy':
    print("------ done with", selector, "----------")
    print("global time")
    from sys import exit
    exit()

if selector == 'pca':
    #########
    ## PCA ##
    #########
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=target_dim)
    print(reducer)
    print((reducer.get_params()))

    print("fit")
    (unix_time(reducer.fit, (data, )))
    print("transform")
    new_data = (unix_time(reducer.transform, (data, )))
    

if selector == 'kpca':
    ################
    ## KERNEL PCA ##
    ################
    from sklearn.decomposition import KernelPCA
    reducer = KernelPCA(n_components=target_dim, n_jobs=-2)
    print(reducer)
    print((reducer.get_params()))
    
    print("fit")
    (unix_time(reducer.fit, (data, )))
    print("transform")
    new_data = (unix_time(reducer.transform, (data, )))

if selector == 'ica':
    ################
    ##  FAST ICA  ##
    ################
    from sklearn.decomposition import FastICA
    reducer = FastICA(n_components=target_dim)
    print(reducer)
    print((reducer.get_params()))
    
    print("fit")
    (unix_time(reducer.fit, (data, )))
    print("transform")
    new_data = (unix_time(reducer.transform, (data, )))

if selector == 'sPCA':
    ################
    ## sparse PCA ##
    ################
    from sklearn.decomposition import SparsePCA
    reducer = SparsePCA(n_components=target_dim, verbose=0, n_jobs=-2)
    print(reducer)
    print((reducer.get_params()))
    
    print("fit")
    (unix_time(reducer.fit, (data, )))
    print("transform")
    new_data = (unix_time(reducer.transform, (data, )))

if selector == 'iso':
    ############
    ## ISOMAP ##
    ############
    from sklearn.manifold import Isomap
    reducer = Isomap(n_components=target_dim, n_jobs=-2)
    print(reducer)
    print((reducer.get_params()))
    
    print("fit")
    (unix_time(reducer.fit, (data, )))
    print("transform")
    new_data = (unix_time(reducer.transform, (data, )))

if selector == 'lle':
    ############################
    ## LocallyLinearEmbedding ##
    ############################
    from sklearn.manifold import LocallyLinearEmbedding
    reducer = LocallyLinearEmbedding(n_components=target_dim, n_jobs=-2)
    print(reducer)
    print((reducer.get_params()))
    
    print("fit")
    (unix_time(reducer.fit, (data, )))
    print("transform")
    new_data = (unix_time(reducer.transform, (data, )))

if selector == 'se':
    #######################
    ## SpectralEmbedding ##
    #######################
    from sklearn.manifold import SpectralEmbedding
    reducer = SpectralEmbedding(n_components=target_dim, n_jobs=-2)
    print(reducer)
    print((reducer.get_params()))
    
    print("fit")
    (unix_time(reducer.fit, (data, )))
    print("transform")
    new_data = (unix_time(reducer.transform, (data, )))


print(f"[new] mean: {new_data.mean()}")
print(f"[new] median: {np.median(new_data)}")
print(f"[new] average: {np.average(new_data)}")
print(f"[new] max: {new_data.max()}")
print(f"[new] min: {new_data.min()}")
print(f"[new] std: {new_data.std()}")
print(f"[new] variance: {new_data.var()}")
print(f"[new] interquartile range: {iqr(new_data)}")
print(f"[new] 50% quantile: {np.quantile(new_data, .5)}")
avg_entropy = []
for x in range(new_data.shape[0]):
    avg_entropy.append(entropy2(new_data[x]))
    
print(f"[new] avg entropy: {np.average(avg_entropy)}")
avg_entropy = []

print("------ done with", selector, "----------")
print("global time")