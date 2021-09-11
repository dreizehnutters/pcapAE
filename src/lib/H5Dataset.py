from os import path
from sys import exit
import multiprocessing as mp

from glob import glob
from tqdm import tqdm
from numpy import vstack, array, array_split
from h5py import File as h5py_File
from torch import from_numpy, tensor
from torch.utils.data import Dataset
from lib.utils import ACTION


class H5Dataset(Dataset):
    def __init__(self,
                path:str,
                train:bool=False,
                n_frames_input:int=10,
                n_frames_output:int=10,
                stride:int=1,
                fraction:int=1,
                shape:tuple=(-1, -1),
                use_cache:bool=False,
                do_norm:bool=True,
                do_preload:bool=False,
                verbose:bool=False,
                device:str='cpu',
                save_path:str=''):
        self.file_path = path
        self.train = train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.stride = stride
        self.fraction = fraction
        self.verbose = verbose
        self.use_cache = False#use_cache
        self.do_norm = do_norm

        if self.use_cache:
            self._out_cache = dict()
            self._in_cache = dict()
            self._gt_cache = dict()
        else:
            self._in_cache = dict()
        self.cached_out_data = []
        self.cached_in_data = []
        self.device = device
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        assert self.n_frames_input >= 1, "[!] no input"
        assert self.stride >= 1, "[!] stride cant be smaller then 1"
        assert self.stride <= self.n_frames_total, f"[!] stride cant be bigger then {n_frames_input} \
                                                                                 + {n_frames_output}"
        self.dataset_len = shape[0]
        self.size = shape[1]
        self.shape = (self.__len__(), self.size**2)

        if self.verbose:
           print(f"{ACTION} loading {self.file_path} for {'training' if self.train else 'testing'} with shape {self.shape}")

        if False:
            self.preload()


    def save_log(self, save_path: str, id: str):
        info_str = '\n'.join('{}={}'.format(k, v) for k, v in vars(self).items())
        with open(path.join(save_path, f"{id}_dataset.log"), 'w') as fd:
            fd.write("H5Dataset Attributes\n")
            fd.write(info_str + '\n')
            fd.write("-------\n")


    def __getitem__(self, index:int):
        """Summary
        Args:
            index (int): Description
        Returns:
            tuple: (index, out_seq, in_seq, ground_truth)
        Raises:
            IndexError: Description
        """
        if not hasattr(self, 'dataset'):
            self.dataset = h5py_File(self.file_path, 'r', swmr=True)
        save_index = index
        if save_index not in self._in_cache:
            datas = []
            records = []
            ground_truth = []
            for idx in range(index, index + self.n_frames_total):
                if idx >= self.dataset_len:
                    raise IndexError
                record = self.dataset[str(idx)]
                datas.append(record['data'][()])
                if not self.train:
                    ground_truth.append(record['label'][()])
                    """ ground_truth layout: (tensor of shape (# number of packets in fragment, 2))
                    <packet_id, ano_id>
                           [[5, 1], <-> packet 5 -> 1 (ANOMALY)  fragment_567 -> (ANOMALY_0.75/normal_0.25)
                            [6,-1], <-> packet 6 ->-1 (normal)
                            [6,-1]] <-> packet 6 ->-1 (normal)
                            [7,-1]] <-> packet 7 ->-1 (normal)
                    """
            data = array(datas)
            in__seq = data[:self.n_frames_input]
            if not self.train:
                gts_seq = array(ground_truth)[:self.n_frames_input]
                gts_seq = vstack(gts_seq[::-1])
                gts_seq = from_numpy(gts_seq)
            else:
                gts_seq = ground_truth
            if self.n_frames_output > 0:
                out_seq = data[self.n_frames_input:self.n_frames_total]
            else:
                out_seq = in__seq

            in__seq = from_numpy(in__seq)
            out_seq = from_numpy(out_seq)
            if self.device.__str__() != 'cpu':
                in__seq = in__seq.to(self.device, non_blocking=True)
                out_seq = out_seq.to(self.device, non_blocking=True)

            if self.do_norm:
                in__seq = (in__seq / 255)
                out_seq = (out_seq / 255)
            if self.use_cache:
                self._gt_cache[save_index]  = gts_seq
                self._out_cache[save_index] = out_seq
                self._in_cache[save_index]  = in__seq
            else:
                return out_seq, in__seq, gts_seq
        return self._out_cache[save_index], self._in_cache[save_index], self._gt_cache[save_index]


    def __len__(self):
        return int((1 + self.dataset_len//self.stride -\
                     self.n_frames_input//self.stride -\
                     self.n_frames_output//self.stride) *\
                     self.fraction)

    def convert_to_numpy(self):
        data = []
        gts = []
        reshape_dim = 256 if self.size != 32 else 1024
        for _, in_, gt in tqdm(self,\
                                desc=f"converting {('/'.join(self.file_path.split('/')[-2:]))} to numpy data set in RAM...",
                                total=self.__len__(),\
                                unit=' fragments',\
                                leave=False,\
                                dynamic_ncols=True):
            data.append(in_.reshape(reshape_dim,).cpu().detach().numpy())
            gts.append(gt)

        return array(data), array(gts)


    def task(self, job):
        name = mp.current_process().name
        level = name.split('-')[-1]
        for x in tqdm(job,
                      desc=f"{ACTION} {level}@preloading {self.file_path.split('/')[-1]} to RAM",
                      total=len(job),
                      unit=' fragments',
                      leave=True,
                      dynamic_ncols=True,
                      position=int(level)):
            self.__getitem__(x)

    def preload(self, NUM_OF_THREADS=2):
        print(f"{ACTION} starting {NUM_OF_THREADS} threads")
        for job in array_split(range(self.__len__()), NUM_OF_THREADS):
            p = mp.Process(target=self.task, args=(job,))
            p.start()
        p.join()
        print(f"{ACTION} preload done")
