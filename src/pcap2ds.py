#! /usr/bin/env python3

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


BOLD = '\033[1m'
CLR = '\033[0m'
ACTION = f"[{BOLD}*{CLR}]"

BYTE_MODE = 'byte'
PACKET_MODE = 'packet'
FLOW_MODE = 'flow'

SIXTYFOR = 64
THIRTYTWO = 32

PARSER = ArgumentParser(description=f"{BOLD}[+] generate h5 data set from a set of PCAPs{CLR}", formatter_class=ArgumentDefaultsHelpFormatter)
PARSER.add_argument('-p',
                    '--pcap',
                    type=str,
                    metavar='',
                    help=f"{BOLD}<path>{CLR} to pcap or list of pcaps to process",
                    required=True)
PARSER.add_argument('-o',
                    '--out',
                    type=str,
                    metavar='',
                    help=f"{BOLD}<path>{CLR} to output dir",
                    required=True)
PARSER.add_argument('-m',
                    '--modus',
                    choices=[BYTE_MODE, PACKET_MODE ,FLOW_MODE],
                    metavar='',
                    type=str,
                    required=True,
                    help=f'gradient decent {BOLD}<strategy>{CLR} | [{BYTE_MODE} | {PACKET_MODE} | {FLOW_MODE}]')
PARSER.add_argument('-g',
                    '--ground',
                    type=str,
                    metavar='',
                    help=f"{BOLD}<path>{CLR} to optional evaluation packet level ground truth .csv",
                    default=None)
PARSER.add_argument('-n',
                    '--name',
                    metavar='',
                    type=str,
                    help=f"data set optional {BOLD}<name>{CLR}",
                    default=None)
PARSER.add_argument('-t',
                    '--threads',
                    choices=([-1] + list(range(1, mp.cpu_count()-1))),
                    metavar='',
                    type=int,
                    help=f"{BOLD}<number>{CLR} of threads | {BOLD}-1{CLR} to use maximum",
                    default=1)
PARSER.add_argument('--chunk',
                    default=1024,
                    type=int,
                    metavar='',
                    required=False,
                    help=f"square number {BOLD}<fragment>{CLR} size")
PARSER.add_argument('--force',
                    action='store_true',
                    help=f"{BOLD}force{CLR} to delete output dir path")
PARSER.add_argument('--bad',
                    action='store_true',
                    help=f"if set all lables to: '1'=<ANOMALIE> else '-1'=<NORMAL>")
PARSER.usage = "python3 pcap2ds.py -p some.pcap -o out_dir --chunk 1024 --modus byte"
ARGS = PARSER.parse_args()

if ARGS.chunk is None:
    exit("set --chunk option [1024]")

print(f"[{BOLD}+{CLR}] {BOLD}{ARGS.modus}{CLR} preprocessing")

NUM_OF_THREADS = mp.cpu_count()-2 if ARGS.threads == -1 else ARGS.threads

def load_cap(pcap_path: str, modus=None):
    """Summary

    Args:
        pcap_path (str): Description

    Returns:
        TYPE: Description
    """
    global_flows = {'tcp':{}, 'udp':{}}
    max_len = int((check_output([f"capinfos {pcap_path} -Mc | grep Number | tr -d ' ' | cut -d ':' -f 2"], shell=True)).decode("utf-8").strip())
    if modus == FLOW_MODE:
        for transport_layer in ['tcp', 'udp']:
            global_flows[transport_layer] = dict()
    try:
        fd = open(pcap_path, 'rb')
        cap = pcap.Reader(fd)
    except ValueError:
        try:
            fd = open(pcap_path, 'rb')
            cap = pcapng.Reader(fd)
        except ValueError as error:
            raise error

    return cap, max_len, [fd], global_flows


def is_binary_string(pcap_path: str):
    """test if file is a txt list or binary

    Args:
        pcap_path (str): Description

    Returns:
        bool: Description
    """
    textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
    binary_string = lambda bytes: bool(bytes.translate(None, textchars))
    return binary_string(open(pcap_path, 'rb').read(64))


def inet_to_str(inet):
    """Convert inet object to a string

        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    try:
        return inet_ntop(AF_INET, inet)
    except ValueError:
        return inet_ntop(AF_INET6, inet)


def sort_IP(src, sport, dst, dport):
    sorted_keys = sorted(ip_address(x).compressed for x in [src, dst])
    if sorted_keys[0] == sorted_keys[1]:
        if sport > dport:
            return f"{sorted_keys[0]}:{sport}", f"{sorted_keys[1]}:{dport}"
        elif sport <= dport:
            return f"{sorted_keys[1]}:{dport}", f"{sorted_keys[0]}:{sport}"
        else:
            print("unreachable code")
    else:
        if sorted_keys[0] == src:
            return f"{sorted_keys[0]}:{sport}", f"{sorted_keys[1]}:{dport}"
        elif sorted_keys[0] == dst:
            return f"{sorted_keys[0]}:{dport}", f"{sorted_keys[1]}:{sport}"
        else:
            print("unreachable code")


class PcapLoader():
    def __init__(self,
                 pcap_path,
                 ground_truth=None,
                 datatype='',
                 chunk_size=1024,
                 modus=None):
        """
        Args:
            pcap_path (string): Path to the pcap file
            chunk_size (int): maximum transfer unit / size of output
            ground_truth (string): path to ground truth csv for data [optional]
            datatype (string): type of data set
        """
        self.fd = []
        self.name = pcap_path
        self.chunk_size = chunk_size
        self.modus = modus
        self.datatype = datatype
        self.q = mp.Queue(maxsize=NUM_OF_THREADS*2)
        self.train = True if ground_truth is None else False
        self.x_dim = int(np.sqrt(chunk_size))
        self.y_dim = int(np.sqrt(chunk_size))
        assert self.x_dim * self.y_dim == chunk_size, f"[!] {chunk_size} is not a perfect square"

        task = f"{ACTION} {BOLD}calculating{CLR} maximum number of packets..."
        if is_binary_string(pcap_path):
            print(task)
            self.cap, self.cap_len, self.fd, self.flow_dict = load_cap(self.name, self.modus)
            raw_cap = None
            self.save_chunks = 1
        else:
            self.cap = []
            self.cap_len = 0
            with open(pcap_path, 'r') as file:
                cap_pathes = file.readlines()

                self.save_chunks = len(cap_pathes)
                pbar = tqdm(total=len(cap_pathes), desc=task, unit=' pcaps', ncols=50, dynamic_ncols=True)
                for cap_path in cap_pathes:

                    raw_cap, tmp_len, fd, tmp_dict = load_cap(cap_path.rstrip("\n\r"), self.modus)
                    pbar.update(1)

                    self.flow_dict = tmp_dict
                    self.cap += raw_cap
                    self.cap_len += tmp_len
                    self.fd += fd

        if ground_truth:
            with open(ground_truth, 'r') as path:
                self.ground_truth = [[(f"{packet_id + 1}->{int(rec[0])}")] for packet_id, rec in enumerate(reader(path, delimiter=','))]
            assert len(self.ground_truth) == self.cap_len, f"[!] ground truth does not match cap! {len(self.ground_truth)} \
                                                                                                !={ self.cap_len}"
        else:
            lable = "1" if ARGS.bad else "-1"
            self.ground_truth = [[f"{x+1}->{lable}"] for x in range(self.cap_len)]

    def run_byte(self):
        tail = b''
        gt_tail = []
        idx = 0
        for (_, packet), gt in tqdm(zip(self.cap, self.ground_truth),
                                        total=self.cap_len,
                                        unit=' packets',
                                        ncols=100, dynamic_ncols=True):

            gt = gt_tail + (gt*len(packet))
            packet = tail + packet


            if len(packet) >= self.chunk_size:
                # get head
                head = packet[:self.chunk_size]
                gt_head = set(gt[:self.chunk_size])
                self.q.put(([idx], head, gt_head))
                idx +=1

                # save rest
                tail = packet[self.chunk_size:]
                gt_tail = gt[self.chunk_size:]

                # pro
                while len(tail) > self.chunk_size:
                    head = tail[:self.chunk_size]
                    gt_head = set(gt_tail[:self.chunk_size])
                    if len(head) >= self.chunk_size:
                        self.q.put(([idx], head, gt_head))
                        idx +=1
                    tail = tail[self.chunk_size:]
                    gt_tail = gt_tail[self.chunk_size:]
            else:
                # save rest
                tail = packet
                gt_tail = gt

        # epilog
        while len(tail) > self.chunk_size:
            head = tail[:self.chunk_size]
            gt_head = set(gt[:self.chunk_size])
            if len(head) >= self.chunk_size:
                self.q.put(([idx], head, gt_head))
                idx +=1
            tail = tail[self.chunk_size:]
            gt = gt[self.chunk_size:]


        for tmp_fd in self.fd:
            tmp_fd.close()

        del self.fd, self.cap

        self.max_chunks = idx


    def run_packet(self):
        idx = 0
        for (_, packet), gt in tqdm(zip(self.cap, self.ground_truth),
                                        total=self.cap_len,
                                        unit=' packets',
                                        ncols=100, dynamic_ncols=True):

            fragment =  packet[:self.chunk_size]
            if len(fragment) < self.chunk_size:
                fragment += b'\x00'*(self.chunk_size - len(fragment))
            self.q.put(([idx], fragment, gt))
            idx += 1

        for tmp_fd in self.fd:
            tmp_fd.close()

        del self.fd, self.cap
        self.max_chunks = idx


    def run_flow(self):
        for (_, packet), gt in tqdm(zip(self.cap, self.ground_truth),
                                        total=self.cap_len,
                                        unit=' packets',
                                        ncols=100, dynamic_ncols=True):
            eth = dpkt.ethernet.Ethernet(packet)
            # only process IP packets
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
            # drop ICMP packets
            elif isinstance(eth.data, dpkt.icmp.ICMP) or isinstance(eth.data.data, dpkt.icmp.ICMP):
                continue
            else:
                ip = eth.data
                IP_PAYLOAD = ip.data
                src_ip, dst_ip = inet_to_str(ip.src), inet_to_str(ip.dst)
                transport_layer = IP_PAYLOAD.__class__.__name__
                if transport_layer not in ['TCP', 'UDP']:
                    continue
                src_port, dst_port = IP_PAYLOAD.sport, IP_PAYLOAD.dport

                larger_IP, smaller_IP = sort_IP(src_ip, int(src_port), dst_ip, int(dst_port))
                test = f"{larger_IP}<->{smaller_IP}"
                try:
                    bucket = self.flow_dict[transport_layer.lower()][test]
                except KeyError:
                    self.flow_dict[transport_layer.lower()][test] = []
                    bucket = self.flow_dict[transport_layer.lower()][test]

                bucket.append((gt, IP_PAYLOAD.__bytes__()[:SIXTYFOR]))
        for tmp_fd in self.fd:
            tmp_fd.close()
        del packet, gt, self.cap, self.ground_truth
        print(f"{ACTION} fragmentizing...")
        idx = 0
        meta_info = []
        for transport_layer in ['tcp', 'udp']:
            flow_keys_for_layer = list(self.flow_dict[transport_layer].keys())
            max_layer_flows = len(flow_keys_for_layer)
            meta_info.append([max_layer_flows, transport_layer, ])
            tmp_flow_count = 0
            for key in flow_keys_for_layer:

                flow_lst = self.flow_dict[transport_layer][key]
                buff = b''
                gt_buff = []
                pbar = tqdm(flow_lst, total=len(flow_lst), unit=' fragments', leave=False, dynamic_ncols=True)
                for gt, packet in pbar:
                    pbar.set_postfix({'flows': f"{tmp_flow_count}/{max_layer_flows}",
                                      'protocol': transport_layer})
                    if (len(packet) != THIRTYTWO) and (len(packet) != SIXTYFOR):
                        packet += b'\x00'*((SIXTYFOR if len(packet) > THIRTYTWO else THIRTYTWO) - len(packet))

                    buff += packet
                    gt_buff += gt*len(packet)
                    if len(buff) >= self.chunk_size:
                        # get head
                        head = buff[:self.chunk_size]
                        gt_head = set(gt_buff[:self.chunk_size])
                        gt_buff = gt_buff[self.chunk_size:]

                        self.q.put(([idx], head, gt_head))
                        idx +=1

                        # carry rest
                        buff = buff[self.chunk_size:]
                        gt = gt[self.chunk_size:]
                # dropping processed flows to save memory
                del self.flow_dict[transport_layer][key]
                tmp_flow_count += 1
        self.max_chunks = idx
        # TODO add to self.info
        print(f"{ACTION} processing {meta_info[0][0]} {meta_info[0][1]} flows")
        print(f"{ACTION} processing {meta_info[1][0]} {meta_info[1][1]} flows")


    def do_write(self):
        with h5py.File(os.path.join(self.save_path, f"{self.datatype}_{mp.current_process().name}.hdf5"),
                       'w',
                        libver='latest') as writer:
            while True:
                idx, chunk, gt = self.q.get()
                if None in idx:
                    break
                idx = idx[0]
                # convert to dec and reshape
                writer[f"{idx}/data"] = [np.array(list(map(int, chunk)), dtype=np.uint8).reshape(self.x_dim, self.y_dim)]

                # [[5, 1], <-> packet 5 -> 1 (ANOMALY)  fragment_567 -> (ANOMALY_0.75/normal_0.25)
                #  [6,-1], <-> packet 6 ->-1 (normal) 
                #  [6,-1]] <-> packet 6 ->-1 (normal)
                #  [7,-1]] <-> packet 7 ->-1 (normal)
                writer[f"{idx}/label"] = np.array([list(map(int, x.split('->'))) for x in gt], dtype=np.int32)


    def save_pcap(self, save_path: str):
        """Summary
        Args:
            save_path (str): Description
        """
        self.save_path = save_path
        print(f"{ACTION} saving to '{BOLD}file://{self.save_path}{CLR}'")

        print(f"{ACTION} starting {BOLD}{NUM_OF_THREADS}{CLR} consumer threads")
        pool = mp.Pool(NUM_OF_THREADS, initializer=self.do_write)

        # start producer
        if self.modus == BYTE_MODE:
            self.run_byte()
        elif self.modus == PACKET_MODE:
            self.run_packet()
        elif self.modus == FLOW_MODE:
            self.run_flow()
        else:
            exit("unreachable code")

        # end consumer after production is done
        for _ in range(1, NUM_OF_THREADS+1):
            self.q.put(([None], None, None))
        pool.close()
        pool.join()
        del self.q

        # merge parts into single file
        if NUM_OF_THREADS > 1:
            files = glob(f"{self.save_path}/*.hdf5")
            writer_path = files.pop()
            with h5py.File(writer_path, mode='r+', libver='latest') as h5fw:
                max_files = len(files)
                for part_id, h5name in enumerate(files):
                    h5fr = h5py.File(h5name,'r')
                    keys = h5fr.keys()
                    key_bar = tqdm(keys, total=len(keys), desc=f"{ACTION} merging", unit=' keys', leave=True, dynamic_ncols=True)
                    for obj in key_bar:
                        h5fr.copy(obj, h5fw)
                        key_bar.set_postfix({'part': f"{part_id}/{max_files}"})
                    h5fr.close()
                    os.unlink(h5name)
            os.rename(writer_path, os.path.join(self.save_path, f"{self.datatype}.hdf5"))
        else:
            os.rename(os.path.join(self.save_path, f"{self.datatype}_ForkPoolWorker-1.hdf5"), os.path.join(self.save_path, f"{self.datatype}.hdf5"))

        info = (f"[!] extracted {BOLD}{'{:,}'.format(self.max_chunks)}{CLR} {self.modus} fragments (a {self.chunk_size} bytes) from {'{:,}'.format(self.cap_len)} packets")
        print(f"{'---'*10}\n{info}\n{'---'*10}")
        self.info = info


    def save_log(self, save_path: str, duration: str):
        """Summary
        Args:
            save_path (str): Description
            duration (str): Description
        """
        with open(os.path.join(save_path, 'log.txt'), 'w') as fd:
            info_str = '\n'.join('{}={}'.format(k, v) for k, v in vars(PARSER.parse_args()).items())
            fd.write("PARSER_ARGS\n")
            fd.write(info_str + '\n')
            fd.write("-------\n")
            fd.write("CLASS_ARGS\n")
            info_str = '\n'.join('{}={}'.format(k, v) for k, v in vars(self).items() if k != "ground_truth" and k != 'flow_dict')
            fd.write(info_str + '\n')
            fd.write(f"saving in: {save_path}\n")
            fd.write(f"\n===\n{duration} s\n===\n")



if __name__ == '__main__':

    out_dir = os.path.abspath(ARGS.out)
    if os.path.isdir(out_dir):
        if not ARGS.force:
            exit(f"{ACTION} {out_dir} already exists...")
        print(f"{ACTION} {BOLD}removing{CLR} {out_dir}...")
        sleep(1)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    start = process_time()
    loadedCap = PcapLoader(pcap_path=os.path.abspath(ARGS.pcap),
                           ground_truth=os.path.abspath(ARGS.ground) if ARGS.ground is not None else None,
                           datatype=ARGS.pcap.split('/')[-1].replace('.','_') if ARGS.name is None else ARGS.name,
                           chunk_size=ARGS.chunk,
                           modus=ARGS.modus)

    loadedCap.save_pcap(out_dir)
    loadedCap.save_log(out_dir, process_time() - start)
