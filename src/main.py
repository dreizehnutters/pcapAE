#! /usr/bin/env python3
from lib.utils import *
from lib.pcapAE import PcapAE
from lib.earlystopping import EarlyStopping

from ad.ad import *

try:
    blob = prolog()
except Exception as error:
    exit(error)
else:
    ARGS = blob['ARGS']
    PARSER = blob['PARSER']
    EXP_NAME = blob['EXP_NAME']
    EXTRA = blob['EXTRA']
    DEVICE = blob['DEVICE']
    WRITER = blob['WRITER']
    CALL_STRING = blob['CALL_STRING']
    del blob

# pcapAE
if ARGS.train is not None:
    encoder_params, decoder_params = get_net(cell='GRU' if ARGS.cell.upper() == 'GRU' else 'LSTM',
                                             device=DEVICE,
                                             size=infer_size(ARGS.train),
                                             no_bn=ARGS.no_bn,
                                             dropout=ARGS.dropout,
                                             verbose=ARGS.verbose)
    if ARGS.retrain:
        compressor = PcapAE(model_path=ARGS.model,
                            device=DEVICE,
                            n_frames_input=ARGS.finput,
                            n_frames_output=ARGS.foutput,
                            log_dir=ARGS.dir,
                            verbose=ARGS.verbose,
                            name=EXP_NAME)
    else:
        compressor = PcapAE(encoder_params=encoder_params,
                            decoder_params=decoder_params,
                            n_frames_input=ARGS.finput,
                            n_frames_output=ARGS.foutput,
                            device=DEVICE,
                            log_dir=ARGS.dir,
                            verbose=ARGS.verbose,
                            name=EXP_NAME)
    savelog(ARGS.dir,
            EXP_NAME,
            PARSER.parse_args(),
            compressor.model.__str__(),
            encoder_params,
            decoder_params,
            ARGS.dropout,
            CALL_STRING,
            ARGS.loss,
            ARGS.optim,
            ARGS.scheduler,
            ARGS.clipping,
            show_print=ARGS.verbose)

    compressor.fit(train_set=ARGS.train,
                    vali_set=ARGS.vali,
                    criterion=ARGS.loss,
                    optimizer=ARGS.optim,
                    scheduler=ARGS.scheduler,
                    early_stopping=EarlyStopping(ARGS.dir,
                                                 patience=7,
                                                 model=compressor,
                                                 verbose=ARGS.verbose,
                                                 exp_tag=EXP_NAME),
                    log_dir=ARGS.dir,
                    epochs=ARGS.epochs,
                    batch_size=ARGS.batch_size,
                    learn_rate=ARGS.learn_rate,
                    gradient_clip_value=ARGS.clipping,
                    num_workers=ARGS.workers,
                    fraction=ARGS.fraction,
                    no_cache=ARGS.noCache,
                    writer=WRITER,
                    return_net=True)

if ARGS.fit != '' and 'redu_data' not in ARGS.fit and not ARGS.baseline:
    try:
        compressor
    except NameError:
        compressor = PcapAE(model_path=ARGS.model,
                            device=DEVICE,
                            log_dir=ARGS.dir,
                            verbose=ARGS.verbose,
                            name=EXP_NAME)

    return_data = compressor.transform(data=ARGS.fit,
                                       num_workers=ARGS.workers,
                                       save_path=f"{ARGS.dir}/redu_data/{EXP_NAME}/",
                                       return_data=ARGS.AD)
    if ARGS.AD:
        train_data = return_data[0]

    if ARGS.predict:
        compressor.transform(data=ARGS.predict,
                             num_workers=ARGS.workers,
                             save_path=f"{ARGS.dir}/redu_data/{EXP_NAME}/")

if ARGS.fit == '' and ARGS.predict != '' and ARGS.model and 'AD' not in ARGS.model and ARGS.baseline != 'pcapAE':
    compressor = PcapAE(model_path=ARGS.model,
                        device=DEVICE,
                        n_frames_input=ARGS.finput,
                        n_frames_output=ARGS.foutput,
                        log_dir=ARGS.dir,
                        verbose=ARGS.verbose,
                        name=EXP_NAME)
    compressor.transform(data=ARGS.predict,
                         num_workers=ARGS.workers,
                         save_path=f"{ARGS.dir}/redu_data/{EXP_NAME}/")

# AD
if ARGS.AD or all([x in ARGS.model for x in ['save_model', 'AD']]) or ARGS.baseline == 'pcapAE':
    classifier = AD(blueprint=None if ARGS.baseline == 'pcapAE' else ARGS.model,
                    EXP_NAME=EXP_NAME,
                    n_jobs=ARGS.n_jobs,
                    verbose=ARGS.verbose)
    start_time = time.now()
    if ARGS.baseline == 'noDL':
        train_data, _ = get_raw_data(ARGS.fit, ARGS)
    else:
        try:
            train_data
        except NameError:
            if ARGS.fit != '':
                train_data = load_compressed_data(ARGS.fit)
            else:
                train_data = None

    if train_data is not None or ARGS.baseline == 'pcapAE':
        if ARGS.grid_search:
            with console.status("[bold green]grid search...", spinner='dots') as status:
                classifier.grid_search(data=train_data,
                                       truth=ARGS.fit)
        if ARGS.baseline == 'pcapAE':
            compressor = PcapAE(model_path=ARGS.model,
                                device=DEVICE,
                                n_frames_input=ARGS.finput,
                                n_frames_output=ARGS.foutput,
                                log_dir=ARGS.dir,
                                verbose=ARGS.verbose,
                                name=EXP_NAME)
            threshold = ARGS.threshold
            # returns (threshold, [(preds_predict, ground_predict), (preds, grounds)])
            datas = [ARGS.predict, ARGS.vali, ARGS.eval] if ARGS.eval != '' else [ARGS.predict]
            threshold, pred_gt_list = compressor.predict(datas=datas,
                                                        num_workers=ARGS.workers,
                                                        NORMAL_LABEL=NORMAL,
                                                        ANOMALY_LABEL=ANOMALY,
                                                        threshold=None if (threshold == '') else sani(threshold),
                                                        WRITER=WRITER,
                                                        log_dir=ARGS.dir,
                                                        extra_lable=EXTRA,
                                                        name=EXP_NAME)
            for ds_name, (preds, truth) in zip(datas, pred_gt_list):
                classifier.calc_metrics(predicted=preds,
                                        truth=truth,
                                        show=True,
                                        save=ds_name,
                                        WRITER=WRITER,
                                        log_dir=ARGS.dir,
                                        threshold=(0, threshold),
                                        in_file=ARGS.predict,
                                        time=time.now()-start_time,
                                        save_ano_pids='y',)
            classifier.clean(ARGS.dir)
        else:
            classifier.fit_data(train_data, ARGS.dir)
            del train_data
            if ARGS.baseline == 'noDL':
                ## new begin data
                predict_data, truth = get_raw_data(ARGS.vali, ARGS)
                classifier.calc_metrics(predicted=classifier.predict_data(data=predict_data),
                                        truth=truth,
                                        show=True,
                                        save=ARGS.vali,
                                        WRITER=WRITER,
                                        log_dir=ARGS.dir,
                                        in_file=ARGS.vali,
                                        time=time.now()-start_time,
                                        save_ano_pids='y',)
                ## eval
                eval_data, truth = get_raw_data(ARGS.predict, ARGS)
                classifier.calc_metrics(predicted=classifier.predict_data(data=eval_data),
                                        truth=truth,
                                        show=True,
                                        save=ARGS.predict,
                                        WRITER=WRITER,
                                        log_dir=ARGS.dir,
                                        in_file=ARGS.predict,
                                        time=time.now()-start_time,
                                        save_ano_pids='y',)

    if ARGS.predict != '' and 'redu_data' in ARGS.predict:
        classifier.calc_metrics(predicted=classifier.predict_data(data=load_compressed_data(ARGS.predict)),
                                truth=ARGS.predict,
                                show=True,
                                save=ARGS.predict,
                                WRITER=WRITER,
                                log_dir=ARGS.dir,
                                in_file=ARGS.predict,
                                time=time.now()-start_time,
                                save_ano_pids='y',)
