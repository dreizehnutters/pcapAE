from os import path, makedirs, getcwd, unlink, scandir

from joblib import *
from numpy import where, ravel, array_equal, flatnonzero, savetxt, unique
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from yaml import safe_load, dump as ydump

from ad.utils import *

from sys import path as spath
spath.append(f"{spath[0]}/ad/thundersvm/python")
try:
    from thundersvm import *
except ModuleNotFoundError:
    svm_from_sk = False
    from sklearn.svm import OneClassSVM
    svm_from_sk = True


class AD(object):
    """docstring for AD"""
    def __init__(self, blueprint, EXP_NAME='debug', n_jobs=1, verbose=True):
        super(AD, self).__init__()
        self.n_jobs = n_jobs
        self.EXP_NAME = EXP_NAME
        self.verbose = verbose
        self.loaded = False
        self.em, self.mv = -1, -1
        if blueprint is None:
            self.pseudo = True
            self.model_type = f"pseudo"
        else:
            self.pseudo = False
            self.from_save_model = True if '.jlib' in blueprint else False
            if self.from_save_model:
                print(f"{ACTION} loading trained model....")
                if 'OneClassSVM' in blueprint and not svm_from_sk:
                    self.classifier = OneClassSVM()
                    self.classifier.load_from_file(blueprint)
                else:
                    self.classifier = load(blueprint)
                self.loaded = True
                self.model_path = blueprint
            else:
                self.classifier = self._get_classifier(blueprint)
                self.model_path = None
            print(f"{'loaded' if self.loaded else 'build'} {self.__str__()}")
            self.model_type = f"{self.classifier.__str__().split('(')[0]}"
            salt = hash(self.classifier.get_params())[:8]
            print(f"[*] model name: {BOLD}{salt}_{self.model_type}{CLR}")
    def __str__(self):
        return f"{self.classifier}@{self.classifier.get_params()}"

    def fit_data(self, data, LOG_DIR=''):
        start_time = time.now()
        with console.status(f"fitting data..", spinner='dots') as status:
            self.classifier.fit(data)
        print(f"{ACTION} fitting took: {time.now()-start_time}")
        if LOG_DIR != '':
            AD_DIR = f"{LOG_DIR}/save_model/{self.EXP_NAME}/AD"
            if not path.isdir(AD_DIR):
                makedirs(AD_DIR)
            params = self.classifier.get_params()
            salt = hash(params)[:8]
            print(f"{ACTION} saving {BOLD}{self.classifier}{CLR} classifier to -> {BOLD}file://{AD_DIR}{CLR}")
            
            if 'OneClassSVM' in self.model_type and not svm_from_sk:
                self.classifier.save_to_file(f"{AD_DIR}/{salt}_{self.model_type}_model.jlib")
            else:
                dump(self.classifier, f"{AD_DIR}/{salt}_{self.model_type}_model.jlib")
            self.model_path = f"{AD_DIR}/{salt}_{self.model_type}_model.jlib"
            with open(f"{AD_DIR}/{salt}_{self.model_type}_model.yaml", 'w') as outfile:
                ydump({self.model_type: params}, outfile, default_flow_style=False)

    def predict_data(self, data, em_mv=False):
        start_time = time.now()
        with console.status(f"predicting data..", spinner='dots') as status:
            preds = self.classifier.predict(data)
        print(f"{ACTION} predicting took: {time.now()-start_time}")
        return preds

    def grid_search(self, data, truth):
        if self.model_type == 'OneClassSVM':
            prefix = 'svm'
            param_dist = {
                            "kernel": ['polynomial'],
                            "degree": [3, 5, 8, 11, 13, 15, 18],
                            "gamma": ['auto'],
                            "coef0": [0, 0.1, 0.5],
                            "tol": [0.001, 0.00001],
                            "nu": [0.5, 0.2, 0.001]
                         }

        elif self.model_type == 'IsolationForest':
            prefix = 'if'
            param_dist = {
                            "n_estimators": [150, 100, 50],
                            "max_samples": [1, .5],
                            "max_features": [1, 1.5, 2],
                            "bootstrap": [True, False],
                            "contamination": [0, 0.25 ,0.5]
                         }

        elif self.model_type == 'LocalOutlierFactor':
            prefix = 'lof'
            param_dist = {
                            "n_neighbors": [40 ,20, 10],
                            "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                            "leaf_size": [10, 20, 40],
                            "p": [1, 1.5, 2],
                         }

        WRITER = spawn_tensorboard(prefix)    
        n_iter_search = 20
        random_search = GridSearchCV(self.classifier,
                                           param_dist,
                                           scoring=f1_scorer,)
                                           #n_iter=n_iter_search)
        start_time = time.now()
        random_search.fit(data, get_gt(truth))
        print(f"[*] grid search took: {time.now()-start_time} for {n_iter_search} candidates parameter settings.")

        results = random_search.cv_results_
        n_top = 5
        for i in range(1, n_top + 1):
            candidates = flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print(f"Model with rank: {i}")
                print(f"Mean validation score: {results['mean_test_score'][candidate]:.3f} (std: {results['std_test_score'][candidate]:.3f})")
                print(f"Parameters: {results['params'][candidate]}\n")
                WRITER.add_hparams(hparam_dict=results['params'][candidate],
                        metric_dict={f'hparam/f1':results['mean_test_score'][candidate]},
                        run_name=f"{self.EXP_NAME}_{i}")


    def clean(self, log_dir):
        buffer = [fd.path for fd in scandir(f"{log_dir}") if 'tmp' in fd.path]
        for file in buffer:
            unlink(file)
        print(f"[*] cleand {len(buffer)} files")


    def calc_metrics(self, predicted, truth, show=True, save='', WRITER=None, **kwargs):
        log_dir = kwargs['log_dir']
        truth_path = None
        if isinstance(truth, str):
            truth_path = truth
            truth = get_gt(truth)

        num_of_preds = len(predicted)
        assert num_of_preds == len(truth), exit(f"{ACTION} number of predicts {num_of_preds} does not match number of labels {num_of_ground}")
        threshold = kwargs['threshold'] if 'threshold' in kwargs.keys() else None
        time = f"{kwargs['time']}" if 'time' in kwargs.keys() else "N/A"
        # calculate metrics
        metric_dict = get_metrics(truth, predicted, show=False, NORMAL_LABEL=NORMAL, ANOMALY_LABEL=ANOMALY)
        if truth_path is not None:
            del truth
        TN = metric_dict['TN']
        FP = metric_dict['FP']
        FN = metric_dict['FN']
        TP = metric_dict['TP']
        pr = metric_dict['pr']
        re = metric_dict['re']
        f1 = metric_dict['f1']
        fpr = metric_dict['fpr']

        if show:
            plot_cm_to_console(TN=TN, FP=FP, FN=FN, TP=TP, EXP_NAME=f"{self.EXP_NAME}{save}")
            plot_metrics_to_console(pr=pr, re=re, f1=f1, fpr=fpr, em=self.em, mv=self.mv, EXP_NAME=self.EXP_NAME)

        if save != '':
            hash_ = hash(self.classifier.get_params()) if not self.pseudo else hash("dummy")
            blob = {'hash': hash_[:8],
                    'type': self.model_type,
                    'name': self.EXP_NAME,
                    'data path': save,
                    'threshold': threshold,
                    'TN': TN,
                    'FP': FP,
                    'FN': FN,
                    'TP': TP,
                    'FPR': fpr,
                    'PR': pr,
                    'RE': re,
                    'f1': f1,
                    'em': self.em,
                    'mv': self.mv,
                    'time':time,
                    'model path': self.model_path if not self.pseudo else "pcapAE"}
            save_results(save_path=f"{log_dir}/logs/results.csv", **blob)
            print(f"{ACTION} adding row to {BOLD}{log_dir}/logs/results.csv{CLR}")
            if 'save_ano_pids' in kwargs.keys():
                tmp_path = f"ano_pids_{kwargs['in_file'].replace('/','_')}_{self.EXP_NAME.split('*')[0].replace('/','_')}_{self.model_type.replace('/','_')}"
                if not path.isdir(f'{log_dir}/logs/{tmp_path}'):
                    makedirs(f'{log_dir}/logs/{tmp_path}')
                print(f"{ACTION} saving confusion matrix PIDs to {BOLD}runs/logs/{tmp_path}{CLR}")
                
                if truth_path is not None:
                    ano_pids = get_ano_pids(get_gt(truth_path, skip=True), predicted)
                else:
                    ano_pids = get_ano_pids(truth, predicted)
                    del truth
                for case, data in zip(['TP','FP','FN'], ano_pids):
                    save_file = f"{log_dir}/logs/{tmp_path}/{case}.txt"
                    savetxt(save_file,
                            data,
                            fmt='%s')
           

            if WRITER is not None:
                WRITER.add_text(f"AD_{self.model_type}_{self.EXP_NAME}", str(blob), global_step=None, walltime=None)


    def _get_classifier(self, blueprint):
        with open(blueprint, 'r') as stream:
            blueprint = safe_load(stream)

        model = list(blueprint.keys())[0]
        params = blueprint[model]

        if model == 'OneClassSVM':
            estimator = OneClassSVM()
            params['verbose'] = self.verbose

        elif model == 'IsolationForest':
            estimator = IsolationForest()
            params['verbose'] = self.verbose
            params['n_jobs'] = self.n_jobs
            params['random_state'] = 42

        elif model == 'LocalOutlierFactor':
            estimator = LocalOutlierFactor()
            params['n_jobs'] = self.n_jobs

        else:
            raise(f"NYI\n{blueprint}")

        estimator.set_params(**params)
        return estimator
