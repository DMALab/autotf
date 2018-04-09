import os
import sys
import warnings
from datetime import datetime
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils.validation import check_X_y, check_array


def model_params(model):
    """get the parameters of the model"""
    s = ''
    if hasattr(model, 'get_params'):
        params_dict = model.get_params()
        max_len = 0
        for key in params_dict:
            if len(key) > max_len:
                max_len = len(key)
        sorted_keys = sorted(params_dict.keys())
        for key in sorted_keys:
            s += '%-*s %s\n' % (max_len, key, params_dict[key])
    elif hasattr(model, '__repr__'):
        s = model.__repr__()
        print(s)
        s += '\n'
    else:
        s = 'Model has no ability to show parameters (has no <get_params> or <__repr__>)\n'
    s += '\n'
    return s


def model_action(model, X_train, y_train, X_test,
                 sample_weight=None, action=None,):
    """choose fit,predict or predict_proba(only avaliable for classification)"""
    if 'fit' == action:
        if sample_weight is not None:
            return model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            return model.fit(X_train, y_train)
    elif 'predict' == action:
        return model.predict(X_test)
    elif 'predict_proba' == action:
        return model.predict_proba(X_test)
    else:
        raise ValueError('The action must be fit,predict or predict_proba')

class subsemble():
    """
    sample_weight:numpy array of shape [n_train_samples]
        Individual weights for each sample
    regression : boolean, default True
        If True - perform stacking for regression task,
        if False - perform stacking for classification task
    bagged_pred : boolean, default True
        If True - bagged predictions for tests set(model is fitted on each fold's train data, then predicts test set)
        if False - predictions for tests set (model is fitted once on full train set, then predicts test set)
    needs_proba: boolean, default False
        Whether to predict probabilities (instead of class labels)
    save_dir: str, default None
        a valid directory (must exist) where log  will be saved
    metric:callable, default None
        Evaluation metric (score function) which is used to calculate results of cross-validation.
    MEAN/FULL interpretation:
            MEAN - mean (average) of scores for each fold.
            FULL - metric calculated using combined oof predictions
                for full train set and target.
    n_folds : int, default 4
        Number of folds in cross-validation
    stratified : boolean, default False, meaningful only for classification task
        If True - use stratified folds in cross-validation
        Ignored if regression=True
    shuffle : boolean, default False
        Whether to perform a shuffle before cross-validation split
    random_state : int, default 0
        Random seed
    verbose : int, default 0
        Level of verbosity.
        0 - show no messages
        1 - for each model show mean score
        2 - for each model show score for each fold and mean score
    """
    def __init__(self,X_train, y_train, X_test,
             sample_weight=None, regression=True,
             num_splits=3, needs_proba=False, save_dir=None,
             metric=None, n_folds=4, stratified=False,
             shuffle=False, random_state=0, verbose=0):
        self.X_train, self.y_train = check_X_y(X_train, y_train,
                                     accept_sparse=['csr'],  # allow csr and cast all other sparse types to csr
                                     force_all_finite=False,  # allow nan and inf
                                     multi_output=False)  # do not allow several columns in y_train
        self.X_test = check_array(X_test,
                             accept_sparse=['csr'],
                             force_all_finite=False)
        if sample_weight is not None:
            self.sample_weight = np.array(sample_weight)
        self.sample_weight = sample_weight
        self.regression = bool(regression)
        self.num_splits = num_splits
        self.needs_proba = bool(needs_proba)
        if save_dir is not None:
            save_dir = os.path.normpath(save_dir)
            if not os.path.isdir(save_dir):
                raise ValueError('Path does not exist or is not a directory. Check <save_dir> parameter')
            else:
                self.save_dir = save_dir
        if not isinstance(n_folds, int):
            raise ValueError('Parameter <n_folds> must be integer')
        elif not n_folds > 1:
            raise ValueError('Parameter <n_folds> must be not less than 2')
        else:
            self.n_folds = n_folds
        self.stratified = bool(stratified)
        self.shuffle = bool(shuffle)
        if verbose not in [0, 1, 2]:
            raise ValueError('Parameter <verbose> must be 0, 1, or 2')
        else:
            self.verbose = verbose
        if regression and (needs_proba or stratified):
            warn_str = 'Task is regression <regression=True> hence function ignored classification-specific parameters which were set as <True>:'
            if needs_proba:
                self.needs_proba = False
                warn_str += ' <needs_proba>'
            if stratified:
                self.stratified = False
                warn_str += ' <stratified>'
            warnings.warn(warn_str, UserWarning)
        if needs_proba and metric == 'accuracy':
            self.metric = log_loss
            warn_str = 'Task needs probability, so the metric is set to log_loss '
            warnings.warn(warn_str, UserWarning)
        if metric is None and regression:
            self.metric = mean_absolute_error
        elif metric is None and not regression:
            if needs_proba:
                self.metric = log_loss
            else:
                self.metric = accuracy_score
        self.save_dir = save_dir
        self.metric = metric
        self.random_state = random_state
        self.next_train = X_train
        self.next_test = X_test
        self.layer = 1

    def add(self, models, propagate_features=None, subset=None):
        """
        models:list
            List of  models
        propagate_features:list,default None
            List of column indexes to propagate from the input of
            the layer to the output of the layer.
        subset:list,default None
            List of column indexes to propagate from the original train/test set
            to the output of the  layer.
        """
        if self.save_dir is not None or self.verbose > 0:
            if self.regression:
                task_str = 'task:       [regression]'
            else:
                task_str = 'task:       [classification]'
                n_classes_str = 'n_classes:  [%d]' % len(np.unique(self.y_train))
            metric_str = 'metric:     [%s]' % self.metric.__name__
            num_splits_str = 'num_splits:     [%s]' % self.num_splits
            layer_str = 'layer:      [%d]' % self.layer
            n_models_str = 'n_models:   [%d]' % len(models)
            # Print the header of the report
        if self.verbose > 0:
            print(layer_str)
            print(task_str)
            if not self.regression:
                print(n_classes_str)
            print(metric_str)
            print(num_splits_str)
            print(n_models_str + '\n')

        ''' split the train data and its sample weights according to the num_splits parameter'''
        train_splits = []
        y_splits = []
        weight_splits = []
        for array in np.array_split(self.next_train, self.num_splits, axis=0):
            train_splits.append(array)
        for array in np.array_split(self.y_train, self.num_splits, axis=0):
            y_splits.append(array)
        if self.sample_weight:
            for weight in np.array_split(self.sample_weight, self.num_splits, axis=0):
                weight_splits.append(weight)

        if not self.regression and self.stratified:
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.random_state)
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.random_state)
        if not self.regression and self.needs_proba:
            n_classes = len(np.unique(self.y_train))
            action = 'predict_proba'
        else:
            n_classes = 1
            action = 'predict'

        '''Transform the test data into S_test'''
        S_test = np.zeros((self.next_test.shape[0], self.num_splits*len(models) * n_classes))

        models_folds_str = ''
        col_slice = 0
        fitted_models = []
        print('-' * 80)
        print('Fitting each model on each subset and predict on the test data:')
        print('-' * 80 + '\n')
        if weight_splits:
            '''1.save log'''
            for model_counter, model in enumerate(models):
                if self.save_dir is not None or self.verbose > 0:
                    model_str = 'model %d:    [%s] ' % (
                        model_counter, model.__class__.__name__)
                if self.save_dir is not None:
                    models_folds_str += '-' * 40 + '\n'
                    models_folds_str += model_str + '\n'
                    models_folds_str += '-' * 40 + '\n\n'
                    models_folds_str += model_params(model)
                if self.verbose > 0:
                    print(model_str)
            '''2.  get S_test'''
            for split_counter, (train_part, y_part, weight_part) in enumerate(zip(train_splits, y_splits, weight_splits)):
                for model_counter, model in enumerate(models):
                    if self.verbose > 0:
                        print('Fitting model [%d] on the train set [%d]\n'%(model_counter, split_counter))
                    _ = model_action(model, train_part, y_part, None, sample_weight=weight_part,
                                     action='fit')
                    # save all the fitted models
                    fitted_models.append(_)
                    if 'predict_proba' == action:
                        col_slice_pro = slice(col_slice * n_classes, col_slice * n_classes + n_classes)
                        S_test[:, col_slice_pro] = model_action(_, None, None, self.next_test, action=action)
                        col_slice += 1
                    else:
                        S_test[:, col_slice] = model_action(_, None, None, self.next_test, action=action)
                        col_slice += 1
                    print(S_test.shape)
        else:
            '''1. save log'''
            for model_counter, model in enumerate(models):
                if self.save_dir is not None or self.verbose > 0:
                    model_str = 'model %d:    [%s] ' % (
                        model_counter, model.__class__.__name__)
                if self.save_dir is not None:
                    models_folds_str += '-' * 40 + '\n'
                    models_folds_str += model_str + '\n'
                    models_folds_str += '-' * 40 + '\n\n'
                    models_folds_str += model_params(model)
                if self.verbose > 0:
                    print(model_str)

            '''2. get S_test'''
            for split_counter, (train_part, y_part) in enumerate(
                    zip(train_splits, y_splits)):
                for model_counter, model in enumerate(models):

                    if self.verbose > 0:
                        print('Fitting model [%d] on the train set [%d]\n' % (model_counter, split_counter))
                    _ = model_action(model, train_part, y_part, None,
                                     action='fit')
                    # save all fitted models
                    fitted_models.append(_)
                    if 'predict_proba' == action:
                        col_slice_pro = slice(col_slice * n_classes, col_slice * n_classes + n_classes)
                        S_test[:, col_slice_pro] = model_action(_, None, None, self.next_test, action=action)
                        col_slice += 1
                    else:
                        S_test[:, col_slice] = model_action(_, None, None, self.next_test, action=action)
                        col_slice += 1

        '''3. transform the train data into S_train'''
        print('-' * 80)
        print('using cross validation and the fitted models to transform  the subset data:')
        print('-' * 80 + '\n')

        if weight_splits:
            for split_counter, (train_part, y_part, weight_part) in enumerate(
                    zip(train_splits, y_splits, weight_splits)):
                # save the transformed train_part
                train_part_next = np.zeros((train_part.shape[0], self.num_splits*len(models) * n_classes))
                for model_counter, model in enumerate(models):
                    # save each fold's metric score
                    scores = np.array([])
                    # -----------------------------------------------------------------------
                    # Loop across folds
                    # -----------------------------------------------------------------------
                    S_train_temp = np.zeros((train_part.shape[0], len(models) * n_classes))
                    for fold_counter, (tr_index, te_index) in enumerate(kf.split(train_part, y_part)):
                        # Split data and target
                        X_tr = train_part[tr_index]
                        y_tr = y_part[tr_index]
                        X_te = train_part[te_index]
                        y_te = y_part[te_index]

                        # Split sample weights accordingly (if passed)
                        if self.sample_weight is not None:
                            sample_weight_tr = weight_part[tr_index]
                            sample_weight_te = weight_part[te_index]
                        else:
                            sample_weight_tr = None
                            sample_weight_te = None

                        # Train on each fold
                        _ = model_action(model, X_tr, y_tr, None, sample_weight=sample_weight_tr, action='fit')
                        # Predict out-of-fold part of train set
                        if 'predict_proba' == action:
                            col_slice_model = slice(model_counter * n_classes, model_counter * n_classes + n_classes)
                        else:
                            col_slice_model = model_counter
                        S_train_temp[te_index, col_slice_model] = model_action(_, None, None, X_te, action=action)

                        # Compute scores
                        if self.save_dir is not None or self.verbose > 0:
                            score = self.metric(y_te, S_train_temp[te_index, col_slice_model])
                            scores = np.append(scores, score)
                            fold_str = '    fold %d: [%.8f]' % (fold_counter, score)
                        if self.save_dir is not None:
                            models_folds_str += fold_str + '\n'
                        if self.verbose > 1:
                            print(fold_str)

                    # Compute scores: mean + std and full
                    if self.save_dir is not None or self.verbose > 0:
                        sep_str = '    ----'
                        mean_str = '    MEAN:   [%.8f] + [%.8f]' % (np.mean(scores), np.std(scores))
                        full_str = '    FULL:   [%.8f]\n' % (self.metric(y_part, S_train_temp[:, col_slice_model]))
                    if self.save_dir is not None:
                        models_folds_str += sep_str + '\n'
                        models_folds_str += mean_str + '\n'
                        models_folds_str += full_str + '\n'
                    if self.verbose > 0:
                        print(sep_str)
                        print(mean_str)
                        print(full_str)

                col_slice_split = slice(split_counter*len(models)*n_classes, (split_counter+1)*len(models)*n_classes)
                train_part_next[:, col_slice_split] = S_train_temp

                # find the models used in this train_part
                used_slice = slice(split_counter*len(models), (split_counter+1)*len(models))
                total_num = self.num_splits*len(models) * n_classes
                used_model = np.arange(total_num)[used_slice]
                # use the model which is not fitted on this train_part
                for i in np.arange(len(fitted_models)):
                    col_slice_i = slice(i * n_classes, (i + 1) * n_classes)
                    if i not in used_model:
                        if self.needs_proba:
                            train_part_next[:, col_slice_i] = fitted_models[i].predict_proba(train_part).reshape(-1, n_classes)
                        else:
                            train_part_next[:, col_slice_i] = fitted_models[i].predict(train_part).reshape(-1, n_classes)
                S_train.append(train_part_next)
                # get S_train
                data_0 = S_train[0]
                for data in S_train:
                    if data is not data_0:
                        data_0 = np.vstack((data_0, data))
                S_train = data_0
        else:
            S_train = []
            for split_counter, (train_part, y_part) in enumerate(
                    zip(train_splits, y_splits)):
                # save the transformed train_part
                train_part_next = np.zeros((train_part.shape[0], self.num_splits * len(models) * n_classes))
                for model_counter, model in enumerate(models):
                    # save each fold's metric score
                    scores = np.array([])
                    # -----------------------------------------------------------------------
                    # Loop across folds
                    # -----------------------------------------------------------------------
                    S_train_temp = np.zeros((train_part.shape[0], len(models) * n_classes))
                    cross_str = 'Cross validation of  model [%d] on the subset [%d]:' % (model_counter, split_counter)
                    print(cross_str)
                    models_folds_str += cross_str + '\n'
                    for fold_counter, (tr_index, te_index) in enumerate(kf.split(train_part, y_part)):
                        # Split data and target
                        X_tr = train_part[tr_index]
                        y_tr = y_part[tr_index]
                        X_te = train_part[te_index]
                        y_te = y_part[te_index]

                        # Train on each fold
                        _ = model_action(model, X_tr, y_tr, None, action='fit')
                        # Predict out-of-fold part of train set
                        if 'predict_proba' == action:
                            col_slice_model = slice(model_counter * n_classes, model_counter * n_classes + n_classes)
                        else:
                            col_slice_model = model_counter
                        S_train_temp[te_index, col_slice_model] = model_action(_, None, None, X_te, action=action)

                        # Compute scores
                        if self.save_dir is not None or self.verbose > 0:
                            score = self.metric(y_te, S_train_temp[te_index, col_slice_model])
                            scores = np.append(scores, score)
                            fold_str = '    fold %d: [%.8f]' % (fold_counter, score)
                        if self.save_dir is not None:
                            models_folds_str += fold_str + '\n'
                        if self.verbose > 1:
                            print(fold_str)

                    # Compute scores: mean + std and full
                    if self.save_dir is not None or self.verbose > 0:
                        sep_str = '    ----'
                        mean_str = '    MEAN:   [%.8f] + [%.8f]' % (np.mean(scores), np.std(scores))
                        full_str = '    FULL:   [%.8f]\n' % (self.metric(y_part, S_train_temp[:, col_slice_model]))
                    if self.save_dir is not None:
                        models_folds_str += sep_str + '\n'
                        models_folds_str += mean_str + '\n'
                        models_folds_str += full_str + '\n'
                    if self.verbose > 0:
                        print(sep_str)
                        print(mean_str)
                        print(full_str)
                '''将S_train_temp与其他几列合并'''
                col_slice_split = slice(split_counter * len(models) * n_classes,
                                        (split_counter + 1) * len(models) * n_classes)
                train_part_next[:, col_slice_split] = S_train_temp

                # find the models used in this train_part
                used_slice = slice(split_counter * len(models), (split_counter + 1) * len(models))
                total_num = self.num_splits * len(models) * n_classes
                used_model = np.arange(total_num)[used_slice]

                for i in np.arange(len(fitted_models)):
                    col_slice_i = slice(i * n_classes, (i + 1) * n_classes)
                    if i not in used_model:
                        if self.needs_proba:
                            train_part_next[:, col_slice_i] = fitted_models[i].predict_proba(train_part).reshape(-1, n_classes)
                        else:
                            train_part_next[:, col_slice_i] = fitted_models[i].predict(train_part).reshape(-1, n_classes)
                S_train.append(train_part_next)
                # get S_train
            data_0 = S_train[0]
            for data in S_train:
                if data is not data_0:
                    data_0 = np.vstack((data_0, data))
            S_train = data_0

        # ---------------------------------------------------------------------------
        # Cast class labels to int
        # ---------------------------------------------------------------------------
        if not self.regression and not self.needs_proba:
            if S_train is not None:
                S_train = S_train.astype(int)
            if S_test is not None:
                S_test = S_test.astype(int)
        # Save  log
        # ---------------------------------------------------------------------------
        if self.save_dir is not None:
            try:
                time_str = datetime.now().strftime('[%Y.%m.%d].[%H.%M.%S].%f')
                # Prepare paths for log files
                log_file_name = 'layer'+str(self.layer)+'.log.txt'
                log_full_path = os.path.join(self.save_dir, log_file_name)

                # Save log
                log_str = 'stack log '
                log_str += time_str + '\n\n'
                log_str += task_str + '\n'
                log_str += layer_str + '\n'
                if not self.regression:
                    log_str += n_classes_str + '\n'
                log_str += metric_str + '\n'
                log_str += num_splits_str + '\n'
                log_str += n_models_str + '\n\n'
                log_str += models_folds_str
                log_str += '-' * 40 + '\n'
                log_str += 'END\n'
                log_str += '-' * 40 + '\n'

                with open(log_full_path, 'w') as f:
                    _ = f.write(log_str)

                if self.verbose > 0:
                    print('log was saved to [%s]' % log_full_path)
            except:
                print('Error while saving files:\n%s' % sys.exc_info()[1])
        # Add 1 to layer as we call the add function again
        self.layer = self.layer + 1
        # The propagate_features and the subset can't be larger than the original features
        if propagate_features is not None and len(propagate_features) > self.next_train.shape[1]:
            propagate_features = slice(0, self.next_train.shape[1])
        if subset is not None and len(subset) > self.next_train.shape[1]:
            subset = slice(0,        self.X_train.shape[1])
        # Consider four kinds of situations
        if propagate_features is None and subset is None:
            self.next_train = S_train
            self.next_test = S_test
        elif propagate_features is None and subset is not None:
            self.next_train = np.hstack((S_train, self.X_train[:, subset]))
            self.next_test = np.hstack((S_test, self.X_test[:, subset]))
        elif propagate_features is not None and subset is None:
            self.next_train = np.hstack((S_train, self.next_train[:, propagate_features]))
            self.next_test = np.hstack((S_test, self.next_test[:, propagate_features]))
        else:
            self.next_train = np.hstack((S_train, self.X_train[:, subset], self.next_train[:, propagate_features]))
            self.next_test = np.hstack((S_test, self.X_test[:, subset], self.next_test[:, propagate_features]))
    # Compute the predictions
    def add_meta(self,meta_model):
        model = meta_model.fit(self.next_train, self.y_train)
        if self.needs_proba:
            y_pred = model.predict_proba(self.next_test)
        else:
            y_pred = model.predict(self.next_test)
        return y_pred
