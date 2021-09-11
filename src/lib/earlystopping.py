from os import path, makedirs, walk ,remove, scandir, unlink

from numpy import inf
from torch import save as t_save

from lib.utils import sort_human, BOLD, CLR


class EarlyStopping:
    def __init__(self, log_path, patience=7, model=None, verbose=False, exp_tag=""):
        """Early stops the training if validation loss doesn't improve after a given patience.
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = inf
        self.global_min_loss = inf
        save_dir = f"{log_path}/save_model/{exp_tag}"
        self.save_path = save_dir
        if not path.isdir(save_dir):
            makedirs(save_dir)
        save_dir = f"{self.save_path}/best/"
        if not path.isdir(save_dir):
            makedirs(save_dir)
        if model is not None:
            self.meta_info = {'meta':(model.encoder_params,\
                                      model.decoder_params,\
                                      model.n_frames_input,\
                                      model.n_frames_output)}
        else:
            self.meta_info = {}


    def __str__(self):
        return '\n'.join(f"{k}={v}" for k, v in vars(self).items())

    def __call__(self, val_loss, model, epoch, step=0):
        """Summary
        Args:
            val_loss (TYPE): Description
            model (TYPE): Description
            epoch (TYPE): Description
        """
        score = -val_loss

        model.update(self.meta_info)

        if step != 0:
            self.save_checkpoint(val_loss, model, epoch, step)
        else:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, epoch, step)
            elif score < self.best_score:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    print(f"{BOLD}[*] early stopping at epoch {epoch} !{CLR}")
                else:
                    print(f"[*] early stopping counter: {BOLD}{self.counter}/{self.patience}{CLR}")
                # self.del_old_models()
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, epoch, step)
                self.counter = 0
            t_save(model, f"{self.save_path}/LAST_checkpoint_{epoch}_{step}_{val_loss:.6f}.pth.tar")

    def del_old_models(self, keep=10):
        _, _, files = next(walk(self.save_path))
        file_count = len(files)
        if file_count > keep:
            for old_model in sort_human(files)[:keep//2]:
                remove(path.join(self.save_path, old_model))

    def save_checkpoint(self, val_loss, model, epoch, step=0):
        """Saves model when validation loss decrease
        Args:
            val_loss (TYPE): Description
            model (TYPE): Description
            epoch (TYPE): Description
        """
        # save best model
        if step != 0:
            save_flag = "IE"
            print(f"[$] saveing model at step: {step} in epoch {epoch}")
            self.del_old_models()
            t_save(model, f"{self.save_path}/{save_flag}checkpoint_{epoch}_{step}_{val_loss}.pth.tar")
        else:
            if val_loss < self.global_min_loss:
                if self.verbose:
                    print(f"[*] validation loss record {BOLD}{val_loss}{CLR} in epoch: {BOLD}{epoch}{CLR}@{step}")
                self.global_min_loss = val_loss
                save_flag = "best/"
                for file in scandir(f"{self.save_path}/{save_flag}"):
                    unlink(file.path)
            else:
                save_flag = ""

            #self.del_old_models()

            t_save(model, f"{self.save_path}/{save_flag}checkpoint_{epoch}_{step}_{val_loss}.pth.tar")
            self.val_loss_min = val_loss
