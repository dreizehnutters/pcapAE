from glob import glob
from csv import DictWriter
from os import path, getcwd, setsid
from datetime import datetime as time

from rich import box
from rich.table import Table
from rich import print as rprint

from torch import load as t_load
from sklearn.metrics import confusion_matrix
from numpy import unique, arange, zeros, argmax, maximum, mean, random, linspace, where, array, intersect1d

BOLD = '\033[1m'
CLR = '\033[0m'
ACTION = f"[{BOLD}*{CLR}]"

# normal/inlier/NEGATIV
NORMAL = -1
# anomaly/outlir/POSITIVE
ANOMALY = 1

CLASSES = 1

from rich.console import Console
console = Console()


# fixed implementaton of -> https://pypi.org/project/emmv/
def emmv_scores(anomaly_score, trained_model, df, n_generated=100000, alpha_min=0.9, alpha_max=0.999, t_max=0.9):
    from sklearn.metrics import auc
    """Get Excess-Mass (EM) and Mass Volume (MV) scores for unsupervised ML AD models.

    :param trained_model: Trained ML model with a 'decision_function' method
    :param df: Pandas dataframe of features (X matrix)
    :param n_generated: , defaults to 10000
    :param alpha_min: Min value for alpha axis, defaults to 0.9
    :param alpha_max: Max value for alpha axis, defaults to 0.999
    :param t_max: Min EM value required, defaults to 0.9
    :return: A dictionary of two scores ('em' and 'mv')
    """

    # Get limits and volume support.
    lim_inf = df.min(axis=0)
    lim_sup = df.max(axis=0)
    offset = 1e-60 # to prevent division by 0
    volume_support = (lim_sup - lim_inf).prod() + offset

    # Determine EM and MV parameters
    t = arange(0, 100 / volume_support, 0.01 / volume_support)
    axis_alpha = arange(alpha_min, alpha_max, 0.0001)
    unif = random.uniform(lim_inf, lim_sup, size=(n_generated, df.shape[1]))

    # Get anomaly scores
    s_unif = trained_model.decision_function(unif)

    # Get EM and MV scores
    AUC_em, em, amax = excess_mass(t, t_max, volume_support, s_unif, anomaly_score, n_generated, auc)
    AUC_mv, mv = mass_volume(axis_alpha, volume_support, s_unif, anomaly_score, n_generated, auc)

    # Return a dataframe containing EMMV information
    return mean(em), mean(mv)

def excess_mass(t, t_max, volume_support, s_unif, s_X, n_generated, auc):
    EM_t = zeros(t.shape[0])
    n_samples = s_X.shape[0]
    s_X_unique = unique(s_X)
    EM_t[0] = 1.

    for u in s_X_unique:
        EM_t = maximum(EM_t, 1. / n_samples * (s_X > u).sum() -
                        t * (s_unif > u).sum() / n_generated
                        * volume_support)
    amax = argmax(EM_t <= t_max) + 1
    if amax == 1:
        amax = -1 # failed to achieve t_max
    AUC = auc(t[:amax], EM_t[:amax])
    return AUC, EM_t, amax


def mass_volume(axis_alpha, volume_support, s_unif, s_X, n_generated, auc):
    n_samples = s_X.shape[0]
    s_X_argsort = s_X.argsort()
    mass = 0
    cpt = 0
    u = s_X[s_X_argsort[-1]]
    mv = zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        while mass < axis_alpha[i]:
            cpt += 1
            u = s_X[s_X_argsort[-cpt]]
            mass = 1. / n_samples * cpt  # sum(s_X > u)
        mv[i] = float((s_unif >= float(u)).sum()) / n_generated * volume_support
    return auc(axis_alpha, mv), mv


def color_switch(value):
    if value >= 0.9:
        color = "[green1]"
    elif value >= 0.8:
        color = "[green3]"
    elif value >= 0.7:
        color = "[orange1]"
    elif value >= 0.6:
        color = "[orange3]"
    elif value >= 0.5:
        color = "[green]"
    elif value >= 0.4:
        color = "[orange4]"
    elif value >= 0.3:
        color = "[red]"
    elif value >= 0.2:
        color = "[red3]"
    else:
        color = "[red1]"
    return color


def plot_cm_to_console(TN, FP, FN, TP, EXP_NAME):
    all_data = TP + FP + FN + TN
    print(f"{' '*85}")
    table = Table(title=f"[b]{EXP_NAME} confusion-matrix", box=box.ASCII)
    table.add_column("pred\\con")
    table.add_column("1",justify="center")
    table.add_column("-1",justify="center")
    table.add_row("positive", f"[green1]TP: [b]{TP}[/b]\n[r]~{TP/all_data*100:.2f}%", f"[red1]FP: [b]{FP}[/b]\n[r]~{FP/all_data*100:.2f}%")
    table.add_row()
    table.add_row("negative", f"[red3]FN: [b]{FN}[/b]\n[r]~{FN/all_data*100:.2f}%", f"[green3]TN: [b]{TN}[/b]\n[r]~{TN/all_data*100:.2f}%")
    rprint(table)


def plot_metrics_to_console(pr, re, f1, fpr, em, mv, EXP_NAME):
    print(f"{' '*85}")
    table = Table(title=f"[b]{EXP_NAME} metrics", box=box.ASCII)
    table.add_column("precision",justify="center")
    table.add_column("recall",justify="center")
    table.add_column("F_1",justify="center")
    table.add_column("FPR",justify="center")
    #table.add_column("em ↓",justify="center")
    #table.add_column("mv ↑",justify="center")
    table.add_row(f"{color_switch(pr)}{pr:.5f}",\
                  f"{color_switch(re)}{re:.5f}",\
                  f"{color_switch(f1)}{f1:.5f}",\
                  f"{color_switch(1-fpr)}{fpr:.5f}",)
                  #f"{em:.5f}",\
                  #f"{mv:.5f}")
    rprint(table)


# ISO PLOT
def plot_iso_curves(ground_truth, pred_score):
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    precision = dict()
    recall = dict()
    for i in range(CLASSES):
        precision[i], recall[i], thresholds = precision_recall_curve(ground_truth, pred_score)

    plt.figure(figsize=(7, 8))
    f_scores = linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')

    for i, color in zip(range(CLASSES), ['green', 'turquoise']):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.title('Precision-Recall curve')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.show()


def save_results(save_path="runs/logs/results.csv", **blob):
    csv_columns = list(blob.keys())
    if not path.exists(save_path):
        with open(save_path, 'w') as csvfile:
            writer = DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerow(blob)
            writer.writerow({})
    else:
        with open(save_path, 'a+') as csvfile:
            writer = DictWriter(csvfile, fieldnames=csv_columns)
            writer.writerow(blob)
            writer.writerow({})


def get_metrics(truth, predicted, show=False, NORMAL_LABEL=-1, ANOMALY_LABEL=1):
    start_time = time.now()
    try:
        if truth[0].shape[1] == 2:
            truth = [truth[x][:,1].max().item() for x in range(len(truth))]
        else:
            truth = [truth[x][:,:,1].max().item() for x in range(len(truth))]
    except Exception:
        pass
    with console.status(f"{ACTION}[bold green] calculating confusion matrix", spinner='dots') as status:
        TN, FP, FN, TP = confusion_matrix(truth, predicted, labels=[NORMAL_LABEL, ANOMALY_LABEL], normalize=None).ravel()
    print(f"{ACTION} metric calculation took: {time.now()-start_time}")
    pr  = TP / (TP + FP) if (TP + FP) != 0 else 1
    re  = TP / (TP + FN) if (TP + FN) != 0 else 1
    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0
    f1  = 2*((pr * re)/(pr + re)) if (pr + re) != 0 else 0.0
    if show:
        print(f"\n§§§§§§\npr: {pr}\nre: {re}\nf1: {f1}\n§§§§§§")
    return {'TN':TN,'FP':FP,'FN':FN,'TP':TP,'fpr':fpr,'re':re,'pr':pr,'f1':f1}


def f1_scorer(clf, data, ground):
    y_pred = where(clf.decision_function(data) >= 0, NORMAL, ANOMALY)
    print(clf.get_params())
    _, _, f1 = get_metrics(ground, y_pred, show=True, NORMAL_LABLE=NORMAL, ANOMALY_LABLE=ANOMALY)
    return f1

def spawn_tensorboard(prefix):
    from torch.utils.tensorboard import SummaryWriter
    from torch import __file__
    from subprocess import Popen, PIPE
    WRITER = SummaryWriter(f"{getcwd()}/gs_{prefix}")
    LOG_DIR = WRITER.log_dir
    TENSORBOARD_PORT = 6000 + int(f"{hash(prefix)}"[-1])
    out, _ = Popen([f"lsof -i:{TENSORBOARD_PORT}"], shell=True, stdout=PIPE).communicate()
    if len(str(out)) <= 3:
        site_packages = __file__.split("torch")[0]
        TENSORBOARD_PATH = f"{site_packages}/tensorboard"
        assert path.isdir(TENSORBOARD_PATH), exit(f"[!] tensorboard not installed @ {TENSORBOARD_PATH}")
        print(f"{ACTION} starting {BOLD}tensorboard{CLR}")
        TENSORBOARD_PROC = Popen([f"python3 {TENSORBOARD_PATH}/main.py --logdir={LOG_DIR} --host localhost --port {TENSORBOARD_PORT}"],
                                    shell=True,
                                    preexec_fn=setsid)
    print(f"{ACTION} {BOLD}{prefix}{CLR}-tensorboard is running at {BOLD}http://localhost:{TENSORBOARD_PORT}{CLR}")
    return WRITER


def get_gt(path, skip=False):
    if isinstance(path, str):
        if 'tmp' in path:
            if skip:
                return t_load(path)
            return get_gt(t_load(path))
        else:
            for file in glob(f"{path}/*.pt"):
                if 'ground' in file:
                    blob = t_load(file)
            return blob
    else:
        try:
            array(path[0][0])[:,1].max().item()
        except IndexError:
            return [array(path[x])[:,1].max().item() for x in range(len(path))]
        else:
            return [array(path[x][0])[:,1].max().item() for x in range(len(path))]


def get_ano_pids(blob, preds, ANOMALY_LABEL=1, NORMAL_LABEL=-1):
    from rich.console import Console
    console = Console()
    with console.status(f"[bold]{ACTION}[/bold][green] calculating PIDs for confusion matrix...", spinner='dots') as status:
        blob_range = range(len(blob))
        try:
            agg_gt = array([blob[x][:,:,1].max().item() for x in blob_range])
            pids = array([(x, list(blob[x][:,:,0].cpu().numpy()[0])) for x in blob_range],dtype=object)
        except IndexError:
            agg_gt = array([blob[x][:,1].max().item() for x in blob_range])
            pids = array([(x, list(blob[x][:,0].cpu().numpy())) for x in blob_range],dtype=object)

        PRED_POS = (pids[where(preds==ANOMALY_LABEL)[0]])
        PRED_POS_F = array([fid_[0] for fid_ in PRED_POS])

        COND_POS = (pids[where(agg_gt==ANOMALY_LABEL)[0]])
        COND_POS_F = array([fid_[0] for fid_ in COND_POS])

        if len(PRED_POS_F) == 0:
            return_object = ([], [])
            PRED_NEG_F = array([])
            COND_NEG_F = array([])
        else:
            PRED_NEG = (pids[where(preds==NORMAL_LABEL)[0]])
            PRED_NEG_F = array([fid_[0] for fid_ in PRED_NEG], dtype='int64')

            COND_NEG = (pids[where(agg_gt==NORMAL_LABEL)[0]])
            COND_NEG_F = array([fid_[0] for fid_ in COND_NEG], dtype='int64')

            TP_idx = intersect1d(PRED_POS_F,COND_POS_F, return_indices=True)[0]
            if len(TP_idx) == 0:
                TP = array([])
            else:
                TP = pids[TP_idx]
                del TP_idx
            FP_idx = intersect1d(PRED_POS_F,COND_NEG_F, return_indices=True)[0]
            if len(FP_idx) == 0:
                FP = array([])
            else:
                FP = pids[FP_idx]
                del FP_idx

            return_object = (TP, FP,)

        if len(COND_POS_F) == 0:
            return_object += ([],)
        else:
            if len(PRED_NEG_F) == 0:
                return_object += ([],)
            else:
                return_object += (pids[intersect1d(COND_POS_F,PRED_NEG_F, return_indices=True)[0]],) # FN

        # (case@(FID, PIDs),)
        return return_object
