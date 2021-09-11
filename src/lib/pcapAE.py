from sys import exit
from os import path, walk, makedirs
from numpy import inf, array, where, partition, std as np_std
from torch.optim import lr_scheduler, SGD, Adam, AdamW
from torch import nn, no_grad, cuda, flatten, utils, cat, stack

from torch.multiprocessing import set_start_method

from tqdm import tqdm
from lib.utils import *
from lib.encoder import Encoder
from lib.decoder import Decoder
from lib.model import AutoEncoder
from lib.H5Dataset import H5Dataset


class PcapAE(object):
    """Summary
    Attributes:
        arg (TYPE): Description
    """
    # set_start_method('spawn')
    def __init__(
        self,
        device,
        log_dir,
        encoder_params=None,
        decoder_params=None,
        n_frames_input=None,
        n_frames_output=None,
        verbose=True,
        model_path=None,
        name="DEBUG"
    ):
        super(PcapAE, self).__init__()
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.device = device
        self.device_type = self.device.__str__()
        self.log_dir = log_dir
        self.verbose = verbose
        self.name = name
        self.param_dict = {}
        if model_path is None:
            self.from_load = False
            self._cell = 'GRU' if 'GRU' in encoder_params[1][-1].__class__.__name__ else 'LSTM'
            self.model = AutoEncoder(encoder=Encoder(encoder_params[0], encoder_params[1], seq_len=self.n_frames_input).to(self.device, non_blocking=True),
                                     decoder=Decoder(decoder_params[0], decoder_params[1], seq_len=self.n_frames_input).to(self.device, non_blocking=True))
            self.model.train()
            self.encoder_params = encoder_params
            self.decoder_params = decoder_params

            self.tmp_epoch = 0
        else:
            if path.isdir(model_path):
                dir_path_file = list(walk(model_path))[0][2][-1]
                model_path = f"{model_path}/{dir_path_file}"

            load_dict = t_load(model_path)
            encoder_params,\
            decoder_params,\
            n_frames_input,\
            n_frames_output = load_dict['meta']
            self.model_path = model_path
            print(f"{ACTION} loading model -> {self.model_path}")
            self.encoder_params = encoder_params
            self.decoder_params = decoder_params

            self.n_frames_input = n_frames_input
            self.n_frames_output = n_frames_output
            self._cell = 'GRU' if 'GRU' in encoder_params[1][-1].__class__.__name__ else 'LSTM'
            self.model = AutoEncoder(encoder=Encoder(encoder_params[0], encoder_params[1], seq_len=self.n_frames_input).to(self.device, non_blocking=True),
                                     decoder=Decoder(decoder_params[0], decoder_params[1], seq_len=self.n_frames_input).to(self.device, non_blocking=True))
            self.model.load_state_dict(load_dict['state_dict'])
            self.optim = load_dict['optimizer']
            self.from_load = True
            self.tmp_epoch = load_dict['epoch'] + 1
            del load_dict
            self.model.eval()
        try:
            self.dropout = self.encoder_params[1][-1].dropout.p
        except AttributeError:
            self.dropout = 0
        self.input_size = int(self.encoder_params[2].split('_')[-1])**2
        self.has_batch_norm = not self.encoder_params[1][-1].no_bn
        self.model.to(self.device, non_blocking=True)

    def __repr__(self):
        suffix = ("\n".join(f"{k}: {BOLD}{v}{CLR}" for k, v in self.param_dict.items()))
        return f"{BOLD}{self.name}{CLR}\n{'='*10}\n{self.encoder_params[2]}\n{self.decoder_params[2]}\nlogging -> {self.log_dir}\n{suffix}"
    def __str__(self):
        return self.__repr__()

    def fit(
        self,
        train_set,
        vali_set,
        criterion,
        optimizer,
        scheduler,
        early_stopping,
        log_dir,
        dropout=0,
        epochs=144,
        batch_size=4,
        learn_rate=.1,
        gradient_clip_value=10,
        num_workers=0,
        fraction=.5,
        no_cache=False,
        writer=None,
        return_net=False
    ):
        """Summary
        Args:
            data (TYPE): Description
        """
        assert self.tmp_epoch < epochs, exit(f"[!] number of epochs ({epochs}) smaller then last epoch ({self.tmp_epoch})")

        self.model.train()

        do_cache = not no_cache

        WRITER = writer

        self.param_dict={
                        'input_size':self.input_size,
                        'finput':self.n_frames_input,
                        'learn_rate':learn_rate,
                        'batch_size':batch_size,
                        'dropout':self.dropout,
                        'cell':self._cell,
                        'loss':criterion,
                        'scheduler':scheduler,
                        'batch_norm':self.has_batch_norm,
                        'optim':optimizer,
                        'foutput':self.n_frames_output,
                        }
        log_path = f"{self.log_dir}/logs/{self.name}_"
        log_path += "".join(f"{v}_" for _, v in self.param_dict.items())[:-1]
        log_path += f"/log.txt"

        # load data
        train_file = assert_file(path.abspath(train_set))
        trainFolder = H5Dataset(path=train_file,
                                train=True,
                                n_frames_input=self.n_frames_input,
                                n_frames_output=self.n_frames_output,
                                shape=get_shape(train_file),
                                use_cache=do_cache,
                                do_preload=True,
                                fraction=fraction,
                                device=self.device,
                                verbose=self.verbose)
        vali_file = assert_file(path.abspath(vali_set))
        validFolder = H5Dataset(path=(vali_file),
                                train=True,
                                n_frames_input=self.n_frames_input,
                                n_frames_output=self.n_frames_output,
                                shape=get_shape(vali_file),
                                use_cache=do_cache,
                                do_preload=True,
                                device=self.device,
                                verbose=self.verbose)

        trainLoader = utils.data.DataLoader(trainFolder,
                                            batch_size=batch_size,
                                            pin_memory=False,
                                            drop_last=True,
                                            num_workers=num_workers,
                                            shuffle=False)
        validLoader = utils.data.DataLoader(validFolder,
                                            batch_size=batch_size,
                                            pin_memory=False,
                                            drop_last=True,
                                            num_workers=num_workers,
                                            shuffle=False)
        assert trainFolder.size == validFolder.size, exit("[!] Dataset mismatch!")
        assert len(trainLoader) > 0, exit("[!] training dataset to small, decrease the --batch_size!")

        # clipping
        nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_value)

        # loss / criterion
        if criterion == 'MSE':
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCELoss()
        criterion.to(self.device)

        # optimizer
        if optimizer == 'adam':
            optimizer = Adam(self.model.parameters(), lr=learn_rate)
        elif optimizer == 'adamW':
            optimizer = AdamW(self.model.parameters(), lr=learn_rate)
        else:
            optimizer = SGD(self.model.parameters(), lr=learn_rate)
        if self.from_load:
            optimizer.load_state_dict = self.optim
            del self.optim

        # scheduler
        if scheduler == 'step':
            teacher = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        elif scheduler == 'cycle':
            teacher = lr_scheduler.OneCycleLR(optimizer, max_lr=learn_rate, steps_per_epoch=len(trainLoader), epochs=epochs-self.tmp_epoch, verbose=self.verbose)
        else:
            teacher = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=self.verbose)

        if self.verbose:
            print(f"{ACTION} training for {BOLD}{epochs - self.tmp_epoch}{CLR} epochs with batch size of {BOLD}{batch_size}{CLR}")
            print(f"{ACTION} using {BOLD}{criterion}{CLR} loss criterion")
            optimizer_items = '\n'.join('\t{}={}{}{}'.format(k, BOLD, v, CLR) for k, v in (optimizer.__dict__['defaults'].items()))
            print(f"{ACTION} using {BOLD}{optimizer.__class__.__name__}{CLR} gradient optimizer with:\n{BOLD}{optimizer_items}{CLR}")
            scheduler_items = '\n'.join('\t{}={}{}{}'.format(k, BOLD, v, CLR) for k, v in (teacher.__dict__.items()) if k != 'optimizer')
            print(f"{ACTION} using {BOLD}{scheduler}{CLR} scheduler with:\n{scheduler_items}")
            print(f"{ACTION} learn rate scheduling starting at {BOLD}{learn_rate}{CLR}")
            print(f"{ACTION} gradient clipping value {BOLD}{gradient_clip_value}{CLR}")
            print(f"{ACTION} using early stopping with patience {BOLD}{early_stopping.patience}{CLR}")

        # plot computation graph for the first run
        _, _, files = (next(walk(f"{self.log_dir}")))
        the_net = None
        tmp_data = None
        if len(files) == 1:
           _, X, _ = next(iter(trainLoader))
           the_net = self.model
           tmp_data = X.to(self.device)
           del X

        file_count = len([x for x in next(walk(self.log_dir))[2] if '.txt' in x])

        # register parameters
        if WRITER:
            pre_callback(WRITER=WRITER,
                        EXP_NAME=self.name,
                        param_dict=self.param_dict,
                        loss=inf,
                        log_file_path=log_path,
                        net=the_net,
                        data=tmp_data)

        # initlize vars
        valid_loss_value_avg = inf
        loss_value_avg = valid_loss_value_avg
        loss_value = valid_loss_value_avg
        last_loss_value_avg = valid_loss_value_avg
        train_len = len(trainLoader)
        snapshot_interval = int(train_len*0.2)
        grid = None
        if self.device_type != 'cpu':
            scaler = cuda.amp.GradScaler()

        print(f"[>] {BOLD}fitting{CLR} data to compressor")
        max_epochs = epochs# + 1
        for epoch in range(self.tmp_epoch, max_epochs):
            ############
            # training #
            ############
            self.model.train()
            tqdm_train = tqdm(trainLoader,
                                total=train_len,
                                leave=True,
                                unit="batch",
                                dynamic_ncols=True,
                                postfix={"epoch": f"{BOLD}{epoch}{CLR}/{max_epochs}",
                                         "loss": f"{BOLD}{loss_value:.9f}{CLR}",
                                         "lr": f"{BOLD}{learn_rate:.9f}{CLR}",
                                         "avgV":f"{BOLD}{valid_loss_value_avg:.6f}{CLR}"},
                                disable=False)

            loss_value_avg = 0

            for idx, (targetVar, inputVar, _) in enumerate(tqdm_train):
                optimizer.zero_grad(set_to_none=True)
                if self.device_type != 'cpu':
                    # cast operations to mixed precision
                    with cuda.amp.autocast():
                        pred = self.model(inputVar.to(self.device, non_blocking=True))
                        loss = criterion(pred, targetVar.to(self.device, non_blocking=True))

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = self.model(inputVar)
                    loss = criterion(pred, targetVar)

                    loss.backward()
                    optimizer.step()

                tmp_lr = optimizer.param_groups[0]['lr']
                loss_value = loss.item() / batch_size
                tqdm_train.set_postfix({'epoch':f"{BOLD}{epoch}{CLR}/{max_epochs}",
                                       'loss':f"{BOLD}{loss_value:.9f}{CLR}",
                                       'lr':f"{BOLD}{tmp_lr:.9f}{CLR}",
                                       'avgV':f"{BOLD}{valid_loss_value_avg:.6f}{CLR}"})

                if WRITER:
                    if (idx % ((train_len-1 if snapshot_interval == 0 else snapshot_interval)+1)-1) == 0:
                        self.model.eval()
                        grid = plot_images(self.model, inputVar, pred, self.device, idx, loss_value, targetVar)
                        self.model.train()
                        early_stopping(loss_value,
                               {'epoch': epoch,
                                'state_dict': self.model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                                epoch,
                                step=idx+epoch*train_len)
                    in_epoch_callback(WRITER=WRITER,
                                        EXP_NAME=self.name,
                                        index=idx+epoch*train_len,
                                        learning_rate=tmp_lr,
                                        loss=loss_value,
                                        validation_loss=valid_loss_value_avg,
                                        images=grid)
                    grid = None

            loss_value_avg = loss_value
            if validLoader:
                tqdm_vali = tqdm(validLoader,
                                 leave=False,
                                 total=len(validLoader),
                                 dynamic_ncols=True,
                                 unit="batch",
                                 postfix={"validation_loss": f"{valid_loss_value_avg:.9f}"})
            ##############
            # validation #
            ##############
            with no_grad():
                self.model.eval()
                valid_loss_value_avg = []
                for targetVar, inputVar, _ in tqdm_vali:
                    valid_loss = criterion(self.model(inputVar.to(self.device, non_blocking=True)), targetVar.to(self.device, non_blocking=True))
                    valid_loss_value = valid_loss.item() / batch_size
                    valid_loss_value_avg.append(valid_loss_value)
                    tqdm_vali.set_postfix({'validation_loss':f"{BOLD}{valid_loss_value:.9f}{CLR}"})
            valid_loss_value_avg = array(valid_loss_value_avg).mean()
            last_loss_value_avg = loss_value_avg
            if WRITER:
                post_epoch_callback(WRITER=WRITER,
                                    EXP_NAME=self.name,
                                    index=epoch,
                                    loss=last_loss_value_avg,
                                    validation_loss=valid_loss_value_avg,
                                    param_dict=self.param_dict)

            early_stopping(valid_loss_value_avg,
                           {'epoch': epoch,
                            'state_dict': self.model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                            epoch)
            if early_stopping.early_stop:
                break

            if isinstance(teacher, lr_scheduler.ReduceLROnPlateau):
                teacher.step(valid_loss)
            else:
                teacher.step()

        print(f"\n{BOLD}[$] training completed!{CLR}")
        print(f"{ACTION} saving results -> {BOLD}file://{self.log_dir}/save_model/{self.name}{CLR}")
        if return_net:
            return self.model


    def transform(
        self,
        data,
        num_workers,
        save_path=None,
        return_data=False
    ):
        """Summary
        Args:
            print(f"{ACTION} generating dataset loaders using {BOLD}{fraction*100}%{CLR} of the data")
            data (TYPE): Description
        """
        print(f"[>] {BOLD}transforming{CLR} data with compressor")
        test_file = assert_file(path.abspath(data))
        tmp_shape = get_shape(test_file)
        testLoader = utils.data.DataLoader(H5Dataset(path=test_file,
                                            train=False,
                                            n_frames_input=self.n_frames_input,
                                            n_frames_output=self.n_frames_output,
                                            shape=tmp_shape,
                                            use_cache=False,
                                            do_preload=False,
                                            device=self.device,
                                            verbose=self.verbose),
                                            batch_size=1,
                                            pin_memory=False,
                                            drop_last=False,
                                            num_workers=num_workers,
                                            shuffle=False)
        self.model.eval()
        tqdm_test = tqdm(testLoader,
                        total=len(testLoader),
                        leave=True,
                        unit=" fragments",
                        dynamic_ncols=True,
                        desc=f"{ACTION} {data}",
                        disable=False)
        codes = []
        grounds = []
        with no_grad():
            for _, in_, ground in tqdm_test:
                if self.device_type == 'cpu':
                    codes.append(flatten(self.model.encode(in_)))
                else:
                    codes.append(flatten(self.model.encode(in_.to(self.device, non_blocking=True))))
                grounds.append(ground)

        if self.verbose:
            print(f"{ACTION} transformed shape: ({len(codes)}, {codes[0].shape[0]})")

        if save_path is not None:
            if not path.isdir(save_path):
                makedirs(save_path)
            dst = f"{save_path}/{data.replace('/', '_')}_cpd"
            print(f"{ACTION} saving '{data}' to: {BOLD}file://{save_path}{CLR} as '{data.replace('/', '_')}_cpd'")
            if not path.isdir(dst):
                makedirs(dst)
            t_save(codes, f"{dst}/compressed.pt")
            t_save(grounds, f"{dst}/ground.pt")
            with open(f"{dst}/info.txt", 'a') as f:
                print({'encoder':self.encoder_params[2],\
                       'decoder':self.decoder_params[2],\
                       'params':self.param_dict}, file=f)
        # deprecated
        if return_data:
            return codes, grounds


    def predict(
        self,
        datas,
        num_workers,
        NORMAL_LABEL=-1,
        ANOMALY_LABEL=1,
        threshold=None,
        WRITER=None,
        name='dummy',
        **kwargs
    ):
        name_lable = name
        if 'extra_lable' in kwargs.keys():
            name_lable = kwargs['extra_lable']
        if 'log_dir' in kwargs.keys():
            log_dir = kwargs['log_dir']
        # slack tolerance
        tol = 2.5
        self.model.eval()
        for counter, data in enumerate(datas):
            print(f"[>] {BOLD}predicting{CLR} {data} with compressor")
            start_time = time.now()
            work_file = assert_file(path.abspath(data))
            # remove debug
            frac=1
            if counter == 0:
                frac=.8
            testLoader = utils.data.DataLoader(H5Dataset(path=work_file,
                                                train=False,
                                                n_frames_input=self.n_frames_input,
                                                n_frames_output=self.n_frames_output,
                                                shape=get_shape(work_file),
                                                use_cache=True,
                                                do_preload=True,
                                                device=self.device,
                                                verbose=self.verbose,
                                                fraction=frac),
                                                batch_size=1,
                                                pin_memory=False,
                                                drop_last=False,
                                                num_workers=num_workers,
                                                shuffle=False)
            tqdm_predict = tqdm(testLoader,
                                total=len(testLoader),
                                leave=True,
                                unit=" fragments",
                                dynamic_ncols=True,
                                desc=f"{ACTION} {data}",
                                disable=False)
            preds = []
            grounds = []
            criterion = nn.MSELoss()
            criterion.to(self.device, non_blocking=True)
            with no_grad():
                for idx, (target_, input_, ground) in enumerate(tqdm_predict):
                    if self.device_type == 'cpu':
                        pred_loss = criterion(self.model(input_), target_)
                    else:
                        pred_loss = criterion(self.model(input_.to(self.device, non_blocking=True)),
                                              target_.to(self.device, non_blocking=True)).to('cpu')
                    if WRITER is not None:
                        WRITER.add_scalars(f"predict/loss", {f"{self.name.split('*')[0]}_{data.replace('/','_')}": pred_loss}, idx)
                        WRITER.add_scalars(f"predict/loss", {f"GT_{self.name}_{data.replace('/','_')}": (0 if ground[:,:,1].max().item() == -1 else 1)}, idx)
                    preds.append(pred_loss)
                    grounds.append(ground)

            print(f"{ACTION} predicting took: {time.now()-start_time}")
            if len(datas) == 1:
                avg_line, set_std = threshold
                offset = set_std*tol
                preds_predict_raw = None
                predict_preds = None
                max_norm_cutoff = avg_line + offset
                min_norm_cutoff = avg_line - offset
                print(f"[*] using threshold cutoff: {BOLD}{avg_line:.6f}(±{offset:.6f}){CLR}")
                predict_preds_d = plot_predict(array(preds), grounds,
                                        lines=(min_norm_cutoff, max_norm_cutoff),
                                        tag=name,
                                        tol=tol,
                                        finput=self.n_frames_input,
                                        file_type=work_file,
                                        ds_name=data,
                                        lable=name_lable,
                                        WRITER=WRITER,
                                        log_dir=log_dir)
                return (avg_line, offset), [(predict_preds_d, grounds)]
            else:
                # train
                if counter == 0:
                    train_lable = data
                    train_gt_tmp_path = f"runs/tmp_train_grounds.pt"
                    t_save(grounds, train_gt_tmp_path)
                    
                    preds_predict_raw = array(preds)
                    train_preds_tmp_path = f"runs/tmp_train_preds_raw.pt"
                    t_save(preds_predict_raw, train_preds_tmp_path)
                    avg_line = preds_predict_raw.mean()
                    offset = np_std(preds_predict_raw)*tol
                    max_norm_cutoff = avg_line + offset
                    min_norm_cutoff = avg_line - offset
                    del preds_predict_raw
                    print(f"[*] found threshold cutoff: {BOLD}{avg_line:.6f}(±{offset:.6f}){CLR}")
                # vali
                elif counter == 1:
                    vali_lable = data
                    vali_preds_tmp_path = f"runs/tmp_vali_preds_raw.pt"
                    t_save(array(preds), vali_preds_tmp_path)
                    
                    vali_gt_tmp_path = f"runs/tmp_vali_grounds.pt"
                    t_save(grounds, vali_gt_tmp_path)

        preds = array(preds)
        predict_preds, vali_preds, eval_preds = plot_baseline(datas=[(train_preds_tmp_path),(vali_preds_tmp_path),(preds)],
                                                     ground_truth=grounds,
                                                     lines=(avg_line, offset),
                                                     tag=name,
                                                     WRITER=WRITER,
                                                     log_dir=log_dir,
                                                     lables=[train_lable, vali_lable,data],
                                                     save_name=self.model_path.split('/save_model/')[1].split('/')[0],
                                                     finput=self.n_frames_input,
                                                     tol=tol,
                                                     file_type=work_file)
        del preds
        return (avg_line, offset), [(predict_preds, train_gt_tmp_path),\
                                    (vali_preds, vali_gt_tmp_path),\
                                    (eval_preds, grounds)]
