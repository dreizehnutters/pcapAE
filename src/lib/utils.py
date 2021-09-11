from sys import argv
from glob import glob
from io import BytesIO
from collections import OrderedDict
from datetime import datetime as time
from os import path, walk, makedirs, getcwd, environ

from PIL import Image
import matplotlib.pyplot as plt
from rich.console import Console
from torchvision.utils import make_grid
from spatialentropy import altieri_entropy
from torchvision.transforms import ToTensor
from scipy.ndimage.filters import gaussian_filter1d
from numpy import unique, arange, array, ndindex, count_nonzero, random, where
from torch import nn, no_grad, float32, load as t_load, save as t_save, manual_seed, device, cuda, backends, _C, version

environ['PYTHONHASHSEED'] = '42'
BOLD = '\033[1m'
CLR = '\033[0m'
ACTION = f"[*]"
TENSORBOARD_PORT = 9001

def roll_name():
    from coolname import generate_slug
    end = False
    while not end:
        name = f"{generate_slug(3).split('-')[1]}"
        if len(name) > 3:
            end = True

    return f"{name}{''.join(map(str,random.randint(0,9,2)))}"


def register_writer(no_tensorboard, DATASET, abs_work_dir):
    print(f"{ACTION} logging dir: {BOLD}{abs_work_dir}{CLR}")
    if no_tensorboard:
        writer = None
        log_dir = abs_work_dir
        if not path.isdir(log_dir):
            makedirs(log_dir)
        return writer

    from torch.utils.tensorboard import SummaryWriter
    from subprocess import Popen, PIPE
    from os import setsid
    from torch import  __file__ as torch_path

    import socket
    host = socket.gethostname()

    writer = SummaryWriter(abs_work_dir)
    out, _ = Popen([f"lsof -i:{TENSORBOARD_PORT}"], shell=True, stdout=PIPE).communicate()
    if len(str(out)) <= 3:
        site_packages = torch_path.split("torch")[0]
        TENSORBOARD_PATH = f"{site_packages}/tensorboard"
        assert path.isdir(TENSORBOARD_PATH), exit(f"[!] tensorboard not installed @ {TENSORBOARD_PATH}")
        print(f"{ACTION} starting {BOLD}tensorboard{CLR}")
        TENSORBOARD_PROC = Popen([f"python3 {TENSORBOARD_PATH}/main.py --logdir={abs_work_dir} --host localhost --port {TENSORBOARD_PORT} --tag {DATASET} --window_title 'pcapAE - Tensorboard [{host}]'"],
                                    shell=True,
                                    preexec_fn=setsid)
    else:
        print(f"{ACTION} {BOLD}tensorboard{CLR} is running at {BOLD}http://localhost:{TENSORBOARD_PORT}{CLR}")
    return writer


def prolog():
    # args
    from lib.CLI import PARSER
    ARGS = PARSER.parse_args()
    extra = ''
    if ARGS.retrain:
        EXP_NAME = ARGS.model.split("/save_model/")[1].split('/')[0]
    else:
        EXP_NAME = roll_name()
    if ARGS.name is not None:
        EXP_NAME, extra = clean_expName(ARGS.name)
        ARGS.name = EXP_NAME
    if ARGS.dir == '' or ARGS.dir == '.':
        ARGS.dir = f"{getcwd()}/runs"
    print(f"{ACTION} experiment tag: {BOLD}{EXP_NAME}{CLR}")
    
    # fix random
    RANDOM_SEED = ARGS.seed
    print(f"{ACTION} set seed to {BOLD}{RANDOM_SEED}{CLR}")
    random.seed(RANDOM_SEED)
    manual_seed(RANDOM_SEED)
    
    # handle CUDA
    DEVICE = device("cuda" if (cuda.is_available() and ARGS.cuda) else "cpu")
    if ARGS.cuda and DEVICE == 'cuda':
        environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
        environ["CUDA_VISIBLE_DEVICES"] = "0"
        if cuda.device_count() > 1:
            cuda.manual_seed_all(RANDOM_SEED)
        else:
            cuda.manual_seed(RANDOM_SEED)
        # enable cudnn beckend
        backends.cudnn.enabled = True
        # may slow down the model
        backends.cudnn.deterministic = True
        # ignore reproducibility for performance
        backends.cudnn.benchmark = True
        cudnn_version = backends.cudnn.version()
        cuda_version_c = _C._cuda_getCompiledVersion()
        cuda_version = version.cuda
        GPU_name = cuda.get_device_name()
        print(f"[!]{BOLD} GPU support via {GPU_name}{CLR} with:\n{BOLD}CUDA:{CLR}{cuda_version}\n{BOLD}CUDA compiled:{CLR}{cuda_version_c}\n{BOLD}cudNN:{CLR}{cudnn_version}")
    else:
        print(f"[!]{BOLD} NO GPU support enabled{CLR}")

    if ARGS.verbose:
        print(f"{ACTION} pytorch git hash: {version.git_version}")

    return {'ARGS':ARGS,
            'PARSER':PARSER,
            'EXP_NAME':EXP_NAME,
            'EXTRA':extra,
            'DEVICE':DEVICE,
            'WRITER':register_writer(ARGS.noTensorboard, EXP_NAME.split('/')[0], path.abspath(ARGS.dir)),
            'CALL_STRING':" ".join(argv[:])}


def shannon_entropy(image, base=2):
    from scipy.stats import entropy as scipy_entropy
    _, counts = unique(image, return_counts=True)
    return scipy_entropy(counts, base=base)


def sort_human(l):
    from re import split as re_split
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re_split('([-+]?[0-9]*\.?[0-9]*)', key)]
    l.sort(key=alphanum)
    return l


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data)
        nn.init.normal_(m.bias.data)


def get_net(cell='GRU', device='cuda:0', size=32, no_bn=False, dropout=0, verbose=False):
    from lib.ConvRNN import CGRU_cell, CLSTM_cell
    """Summary
    Args:
        cell (str, optional): Description
        device (str, optional): Description
        size (int, optional): Description
    Returns:
        TYPE: Description
    """
    if verbose:
        print(f"{ACTION} building the {BOLD}{cell}{CLR} network")

    CELL_TYPE = CGRU_cell if cell == 'GRU' else CLSTM_cell
    big_net = True if size == 32 else False
    enc_name = f"conv{cell}_encoder_params_{device}_{size}"
    enc = [
        [
            # in_channels,out_channels,kernel_size,stride,padding
            OrderedDict({'conv1_leaky_1': [1, 2, 3, 1, 1]}),
            OrderedDict({'conv2_leaky_1': [4, 4, 3, 2, 1]}),
            OrderedDict({'conv3_leaky_1': [4, 4, 3, 2, 1]}),
            OrderedDict({'conv4_leaky_1': [4, 4, 3, 2, 1]}) if big_net else None,
        ],
        [
            CELL_TYPE(shape=(32,32),\
                        input_channels=2,\
                        filter_size=5,\
                        num_features=4,\
                        bias=True,\
                        no_bn=no_bn,\
                        dropout=dropout,\
                        device=device) if big_net else None,
            CELL_TYPE(shape=(16,16),\
                        input_channels=4 if big_net else 2,\
                        filter_size=5,\
                        num_features=4,\
                        bias=True,\
                        no_bn=no_bn,\
                        dropout=dropout,\
                        device=device),
            CELL_TYPE(shape=(8,8),\
                        input_channels=4,\
                        filter_size=5,\
                        num_features=4,\
                        bias=True,\
                        no_bn=no_bn,\
                        dropout=dropout,\
                        device=device),
            CELL_TYPE(shape=(4,4),\
                        input_channels=4,\
                        filter_size=5,\
                        num_features=4,\
                        bias=True,\
                        no_bn=no_bn,\
                        dropout=dropout,\
                        device=device),
        ],
        enc_name
    ]

    dec_name = f"conv{cell}_decoder_params_{device}_{size}"
    dec = [
        [
            OrderedDict({'deconv0_leaky_1': [4, 4, 4, 2, 1]}) if big_net else None,
            OrderedDict({'deconv1_leaky_1': [4, 4, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [4, 4, 4, 2, 1]}),
            OrderedDict({'conv3_leaky_1':   [4, 2, 3, 1, 1], #stage1
                         'conv4_leaky_1':   [2, 1, 1, 1, 0]}),
        ],
        [
            CELL_TYPE(shape=(4,4),\
                        input_channels=4,\
                        filter_size=5,\
                        num_features=4,\
                        bias=True,\
                        no_bn=no_bn,\
                        dropout=dropout,\
                        device=device),#rnn4
            CELL_TYPE(shape=(8,8),\
                        input_channels=4,\
                        filter_size=5,\
                        num_features=4,\
                        bias=True,\
                        no_bn=no_bn,\
                        dropout=dropout,\
                        device=device),
            CELL_TYPE(shape=(16,16),\
                        input_channels=4,\
                        filter_size=5,\
                        num_features=4,\
                        bias=True,\
                        no_bn=no_bn,\
                        dropout=dropout,\
                        device=device),
            CELL_TYPE(shape=(32,32),\
                        input_channels=4,\
                        filter_size=5,\
                        num_features=4,\
                        bias=True,\
                        no_bn=no_bn,\
                        dropout=dropout,\
                        device=device)  if big_net else None, #rnn1
        ],
        dec_name
    ]
    for blob in [enc, dec]:
        for idx, _ in enumerate(blob):
            if type(blob[idx]) != str:
                blob[idx] = list(filter(None, blob[idx]))

    return enc, dec


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))


def savelog(save_path,
             exp_name,
             parser,
             net,
             encoder_params,
             decoder_params,
             dropout,
             input_arguements,
             lossfunction,
             optimizer,
             teacher,
             gradient_clip_value,
             show_print=False):

    acc = ""
    tmp_dict = dict(vars(parser).items())
    lr = tmp_dict['learn_rate']
    batch_size = tmp_dict['batch_size']
    CELL = tmp_dict['cell']
    finput = tmp_dict['finput']
    foutput = tmp_dict['foutput']
    cell = tmp_dict['cell']
    batch_norm = not tmp_dict['no_bn']
    scheduler = tmp_dict['scheduler']
    scheduler = tmp_dict['scheduler']
    size = int(encoder_params[2][-2:])**2
    loss = tmp_dict['loss']
    param_dict={
                'input_size':size,
                'finput':finput,
                'learn_rate':lr,
                'batch_size':batch_size,
                'dropout':dropout,
                'cell':cell,
                'loss':loss,
                'scheduler':scheduler,
                'batch_norm':batch_norm,
                'optim':optimizer,
                'foutput':foutput,
                }
    rest = "".join(f"{v}_" for _, v in param_dict.items())[:-1]
    legend = "".join(f"{k}_" for k, _ in param_dict.items())[:-1]

    exp_name = f"{exp_name}_{rest}"

    if not path.isdir(f'{save_path}/logs/{exp_name}'):
        makedirs(f'{save_path}/logs/{exp_name}')

    with open(f'{save_path}/logs/{exp_name}/log.txt', 'w+') as fd:
        fd.write(f"python3 {input_arguements}\n\n")
        fd.write("")
        fd.write(f'saving in: {save_path}/logs/{exp_name}\n')
        fd.write("")
        now = time.now()
        fd.write(f'Current date and time : {now.strftime("%Y-%m-%d %H:%M:%S")}\n')
        fd.write("\n=========\n")
        fd.write("")

        fd.write('\n' + 'legend' + '\n')
        fd.write(f'name_{legend}' + '\n')
        fd.write("-------\n")

        info_str = '\n'.join('{}={}\n'.format(k, v) for k, v in vars(parser).items())
        fd.write('PARSER_ARGS\n\n')
        fd.write(info_str + '\n')
        fd.write("-------\n")
        acc += '\n'.join('\t[§] {}={}{}{}'.format(k, BOLD, v, CLR) for k, v in vars(parser).items())

        fd.write('encoder_params\n')
        fd.write(encoder_params[2].__str__() + '\n')
        acc += f'\n\t[§] encoder_params={BOLD}{encoder_params[2].__str__()}{CLR}'

        fd.write(str(encoder_params[0]) + '\n')
        fd.write(encoder_params[1].__str__() + '\n')
        fd.write("-------\n")

        fd.write('decoder_params\n')
        fd.write(decoder_params[2].__str__() + '\n')
        acc += f'\n\t[§] encoder_params={BOLD}{decoder_params[2].__str__()}{CLR}'

        fd.write(str(decoder_params[0]) + '\n')
        fd.write(decoder_params[1].__str__() + '\n')
        fd.write("-------\n")

        fd.write('net\n')
        fd.write(net + '\n')
        fd.write("-------\n")

        fd.write('lossfunction\n')
        fd.write(lossfunction.__str__() + '\n')
        fd.write("-------\n")
        acc += f"\n\t[§] loss function={BOLD}{lossfunction.__str__()}{CLR}\n"

        fd.write('dropout\n')
        fd.write(dropout.__str__() + '\n')
        fd.write("-------\n")
        acc += f"\t[§] dropout={BOLD}{dropout}{CLR}\n"

        fd.write('gradient_clip_value\n')
        fd.write(str(gradient_clip_value) + '\n')
        fd.write("-------\n")
        acc += f"\t[§] grad clip={BOLD}{gradient_clip_value}{CLR}\n"

        fd.write('optimizer\n')
        fd.write(optimizer.__str__() + '\n')
        fd.write("-------\n")
        acc += f"\t[§] optimizer={BOLD}{optimizer.__str__()}{CLR}\n"

        fd.write('teacher\n')
        fd.write(teacher.__str__() + '\n')
        fd.write("-------\n")

        if show_print:
            print(net)
            print(f"\n{ACTION} Hyperparameters\n{acc}")


def plot_images(net, inputVar, pred, device, idx, loss, targetVar):
    with no_grad():
        code = net.encode(inputVar)
        if device.__str__() != 'cpu':
            code = code.cpu().numpy()
        else:
            code = code.detach().numpy()

        code = code.reshape(8, 8)
        pred = pred.type(float32)
        if device.__str__() != 'cpu':
            predIMG = pred.cpu().numpy()[0][0][0]
        else:
            predIMG = pred.detach().numpy()[0][0][0]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    MAX = inputVar.shape[-1]

    #plt.grid(False)
    ax1.imshow(inputVar[0][0][0].cpu().numpy(), cmap=plt.cm.gray, extent=(0, MAX, MAX, 0))
    ax1.set_title(f"input ID: {idx}")
    ax1.set_xticks([])
    ax1.set_yticks(arange(0, MAX+1, MAX/2))

    # denormilize from [-1,1] to [0,255]
    new_code = ((code - code.min()) * (1/(code.max() - code.min()) * 255.99)).astype('uint8')
    entropy = altieri_entropy(points=array([x for x in ndindex(8,8)]), types=new_code.flatten(), base=2).entropy

    ax2.imshow(new_code, cmap='jet', extent=(0, 8, 8, 0))
    ax2.set_title(f"code entropy: {entropy:.6f}")
    ax2.set_xticks([])
    ax2.set_yticks(arange(0, 9, 4))

    ax3.imshow(predIMG, cmap=plt.cm.gray, extent=(0, MAX, MAX, 0))
    ax3.set_title(f"loss:\n{loss:.9f}")
    ax3.set_yticks(arange(0, MAX+1, MAX/2))
    ax3.set_xticks([])

    ax4.imshow(targetVar[0][0][0].cpu().numpy(), cmap=plt.cm.gray, extent=(0, MAX, MAX, 0))
    ax4.set_title(f"target ID: {idx}")
    ax4.set_yticks(arange(0, MAX+1, MAX/2))
    ax4.set_xticks([])

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-.3, hspace=.3)

    # Save the plot to a PNG in memory.
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return make_grid(ToTensor()(Image.open(buf)))


def pre_callback(WRITER, EXP_NAME, param_dict, loss, log_file_path, net=None, data=None):
    if net:
        # view computation graph
        WRITER.add_graph(net, data)
    WRITER.add_hparams(hparam_dict=param_dict,
                        metric_dict={f'hparam/loss':loss},
                        run_name=EXP_NAME)

    WRITER.add_text(EXP_NAME, open(log_file_path, "r").read(), global_step=None, walltime=None)


def in_epoch_callback(WRITER, EXP_NAME, index, learning_rate, loss, validation_loss, images=None):
    if images is not None:
        WRITER.add_image(f"{EXP_NAME.replace('/','_')}/insight", images, index)

    # individual plot
    WRITER.add_scalars(EXP_NAME,
                       {"learning_rate": learning_rate,
                       "loss": loss,
                       "validation_loss": validation_loss},
                       index)


def post_epoch_callback(WRITER, EXP_NAME, index, loss, validation_loss, param_dict):
    WRITER.add_scalars("_GLOBAL/loss", {EXP_NAME: loss}, index)
    WRITER.add_scalars("_GLOBAL/vali", {EXP_NAME: validation_loss}, index)

    WRITER.add_hparams(
                hparam_dict=param_dict,
                metric_dict={f'hparam/loss':loss},
                run_name=EXP_NAME)


def infer_size(train_path):
    from h5py import File
    file_ = None
    for x in glob(f"{train_path}/*.hdf5"):
            file_ = x

    assert file_ is not None, exit(f"{train_path} does not exist")
    with File(file_, 'r') as file:
            test = file["0"]['data'].shape
    return test[1]


def get_raw_data(path_, ARGS):
    from lib.H5Dataset import H5Dataset
    in_file = assert_file(path.abspath(path_))
    return H5Dataset(path=in_file,
                     train=False,
                     n_frames_input=ARGS.finput,
                     n_frames_output=ARGS.foutput,
                     shape=get_shape(in_file),
                     use_cache=False,
                     device='cpu',
                     verbose=ARGS.verbose).convert_to_numpy()


def assert_file(path):
    file = None
    for x in glob(f"{path}/*.hdf5"):
        file = x
    assert file is not None, exit(f"{path} does not exist")
    return file


def get_shape(file_path):
    from h5py import File as h5py_File
    with h5py_File(file_path, 'r') as file:
        test = file["0"]['data'].shape
        assert "label" in file["0"].keys(), f"[!] dataset {file_path} has no ground truth!"
        return len(file), test[1]


def load_compressed_data(path):
    from torch import load, stack
    data = None
    console = Console()
    with console.status("[bold green]loading data...", spinner='dots') as status:
        for file in glob(f"{path}/*.pt"):
            if 'compressed' in file:
                data = load(file)

        assert data is not None, f"{path} is bad!"
        return stack(data).cpu().numpy()


def sani(input_):
    if input_ == 'done':
        return -1, -1
    if '-' in input_:
        min_input, max_input = input_.split('-')
        try:
            tmp_min = float(min_input)
            tmp_max = float(max_input)
        except ValueError:
            print("[!] no correct input! :v")
            tmp_min, tmp_max = 0, 1
        return tmp_min, tmp_max
    else:
        print("[!] no correct input! 'minValue-maxValue' (0.05-0.9)")
        return 0, 1


def make_discrete(raw, min_norm_cutoff, max_norm_cutoff, NORMAL_LABEL=-1, ANOMALY_LABEL=1):
    return where(((raw >= min_norm_cutoff) & (raw < max_norm_cutoff)), NORMAL_LABEL, ANOMALY_LABEL)


def clean_expName(name):
    return ((name if '*' in name else name+'*').split('*'))


def plot_predict(predict_preds, ground_truth, lines, tag, **kwargs):

    log_dir = kwargs['log_dir']+'/logs'
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000

    data_set_name = kwargs['ds_name'].split('/')[0]

    sigma = 1
    do_smoothing = True

    # baselines
    avg_line, std = lines
    min_norm_cutoff = avg_line - std
    max_norm_cutoff = avg_line + std

    # make figure
    fig, ax = plt.subplots(1,1)

    predict_preds_d = make_discrete(predict_preds, min_norm_cutoff, max_norm_cutoff)
    ax.plot(range(predict_preds.shape[0]), gaussian_filter1d(predict_preds, sigma=sigma),
                color='steelblue',
                marker='.',
                label=kwargs['lable'],
                linestyle="None",
                alpha=1)
    del predict_preds

    gt_len = range(len(ground_truth))
    fixed_gt = [ground_truth[x][:,:,1].max().item() for x in gt_len]
    del ground_truth
    from ad.utils import get_metrics
    metric_dict = get_metrics(fixed_gt, predict_preds_d, show=False, NORMAL_LABEL=-1, ANOMALY_LABEL=1)
    
    # add to legend
    ax.plot([], [], ' ', label=f"PR: {metric_dict['pr']:.2f}")
    ax.plot([], [], ' ', label=f"RC: {metric_dict['re']:.2f}")
    ax.plot([], [], ' ', label=f"$F_1$: {metric_dict['f1']:.2f}")
    ax.plot([], [], ' ', label=f"FPR: {metric_dict['fpr']:.2f} ")
    # plot gt
    ax.bar(gt_len, [-0.005 if x >= 1 else 0 for x in fixed_gt], color='red',label=f"positive")
    ax.bar(gt_len, [0 if x >= 1 else -0.005 for x in fixed_gt], color='green',label=f"negative")
    del fixed_gt
    # plot max_norm_cutoff
    ax.axhline(y=max_norm_cutoff, color='black', alpha=1, linestyle='-',label=f"{kwargs['tol']}$\sigma$")

    # set axis
    finput = f"S_{kwargs['finput']}"
    data_type = "flow" if 'flow' in kwargs['file_type'] else 'byte'
    ax.set_title(f"{data_set_name} {data_type} ${finput}$")
    ax.set_xscale('linear')
    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    ax.legend(loc='upper left', prop=dict(size=8), bbox_to_anchor=(1.01, 1))
    ax.grid(linestyle='dashed')
    
    format_ext = 'png'
    save_dir = log_dir
    abs_save_dir = f"{save_dir}/{tag.split('/')[0]}"
    if not path.isdir(abs_save_dir):
        makedirs(abs_save_dir)
    plot_name = f"{tag}_{data_set_name}_{data_type}_F{kwargs['finput']}"
    save_path = f"{save_dir}/{plot_name}.{format_ext}"
    if len(glob(f"{save_dir}/{plot_name}*")) >= 1:
        suffix = len(glob(f"{save_dir}/{plot_name}*"))
        save_path = f"{save_dir}/{plot_name}_{suffix}.{format_ext}"
    print(f"{ACTION} saving figure to {save_path}")
    fig.savefig(fname=save_path,
                dpi=600,
                format=format_ext,
                facecolor='w',
                edgecolor='w',
                orientation='portrait',
                pad_inches=0.05,
                bbox_inches='tight',)

    return predict_preds_d


def plot_baseline(datas, ground_truth, lines, tag="dummy", **kwargs):
    log_dir = kwargs['log_dir']+'/logs'

    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    eval_lable = kwargs['lables'][-1]
    data_set_name = eval_lable.split('/')[0]

    # baselines
    avg_line, std = lines
    min_norm_cutoff = avg_line - std
    max_norm_cutoff = avg_line + std

    train_preds = t_load(datas[0])
    vali_preds = t_load(datas[1])
    eval_preds = datas[2]

    sigma = 1
    do_smoothing = True
    
    # make figure
    fig, ax = plt.subplots(1,1)

    train_preds_d = make_discrete(train_preds, min_norm_cutoff, max_norm_cutoff)
    if train_preds is not None:
        train_lable = "$Train_{AD}$"
        if do_smoothing:
            train_preds = gaussian_filter1d(train_preds, sigma=sigma)
        ax.plot(range(train_preds.shape[0]), train_preds,
                    color='navy',
                    marker='.',
                    label=train_lable,
                    linestyle="None",
                    alpha=1)
    del train_preds
    
    eval_preds_d = make_discrete(eval_preds, min_norm_cutoff, max_norm_cutoff)
    if do_smoothing:
        eval_preds = gaussian_filter1d(eval_preds, sigma=sigma)
    ax.plot(range(eval_preds.shape[0]), eval_preds,
                color='orange',
                marker='.',
                label="$Eval_p$" if 'VO' in data_set_name else "$Eval$",
                linestyle="None",
                alpha=0.3)
    del eval_preds

    vali_preds_d = make_discrete(vali_preds, min_norm_cutoff, max_norm_cutoff)
    if do_smoothing:
        vali_preds = gaussian_filter1d(vali_preds, sigma=sigma)
    ax.plot(range(vali_preds.shape[0]), vali_preds,
                color='m',
                marker='.',
                label="$Validation_{AD}$",
                linestyle="None",
                alpha=0.3)
    del vali_preds

    gt_len = range(len(ground_truth))
    fixed_gt = [ground_truth[x][:,:,1].max().item() for x in gt_len]
    del ground_truth
    from ad.utils import get_metrics
    metric_dict = get_metrics(fixed_gt, eval_preds_d, show=False, NORMAL_LABEL=-1, ANOMALY_LABEL=1)
    # add to legend
    ax.plot([], [], ' ', label=f"PR: {metric_dict['pr']:.2f}")
    ax.plot([], [], ' ', label=f"RC: {metric_dict['re']:.2f}")
    ax.plot([], [], ' ', label=f"$F_1$: {metric_dict['f1']:.2f}")
    ax.plot([], [], ' ', label=f"FPR: {metric_dict['fpr']:.2f} ")
    # plot gt
    ax.bar(gt_len, [-0.005 if x >= 1 else 0 for x in fixed_gt], color='red',label=f"positive")
    ax.bar(gt_len, [0 if x >= 1 else -0.005 for x in fixed_gt], color='green',label=f"negative")
    del fixed_gt
    # plot max_norm_cutoff
    ax.axhline(y=max_norm_cutoff, color='black', alpha=1, linestyle='-',label=f"{kwargs['tol']}$\sigma$")

    # set axis
    finput = f"S_{kwargs['finput']}"
    data_type = "flow" if 'flow' in kwargs['file_type'] else 'byte'
    ax.set_title(f"{data_set_name} {data_type} ${finput}$")
    ax.set_xscale('linear')
    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    ax.legend(loc='upper left', prop=dict(size=8), bbox_to_anchor=(1.01, 1))
    ax.grid(linestyle='dashed')
    
    WRITER = kwargs['WRITER']
    format_ext = 'png'
    save_dir = log_dir
    abs_save_dir = f"{save_dir}/{tag.split('/')[0]}"
    if not path.isdir(abs_save_dir):
        makedirs(abs_save_dir)
    plot_name = f"{tag}_{data_set_name}_{data_type}_F{kwargs['finput']}"
    save_path = f"{save_dir}/{plot_name}.{format_ext}"
    if len(glob(f"{save_dir}/{plot_name}*")) >= 1:
        suffix = len(glob(f"{save_dir}/{plot_name}*"))
        save_path = f"{save_dir}/{plot_name}_{suffix}.{format_ext}"
    print(f"{ACTION} saving figure to {save_path}")
    fig.savefig(fname=save_path,
                dpi=600,
                format=format_ext,
                facecolor='w',
                edgecolor='w',
                orientation='portrait',
                pad_inches=0.05,
                bbox_inches='tight',)
    if WRITER is not None:
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        WRITER.add_image(f"baseline/{tag}", make_grid(ToTensor()(Image.open(buf))), 1)

    return train_preds_d, vali_preds_d, eval_preds_d
