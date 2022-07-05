import joblib

import lightgbm as lgb
import numpy as np
import json
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.tree import DecisionTreeClassifier

import config
from utils import set_logger
from models.dataset import Dataset
from models.tuner import grid_search, bayes_search_lgb
from models.metric import evaluate

logger = set_logger(config.log_path)


class Model:
    def __init__(self, model_path=None, train_mode=True, debug_mode=False):
        self.dataset = Dataset(debug_mode=debug_mode, train_mode=train_mode)
        self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.process_data()

        if train_mode:
            self.model = lgb.LGBMClassifier(objective='multiclass',
                                            n_jobs=10,
                                            num_class=33,
                                            num_leaves=30,
                                            reg_alpha=10,
                                            reg_lambda=200,
                                            max_depth=3,
                                            learning_rate=0.05,
                                            n_estimators=2000,
                                            bagging_freq=1,
                                            bagging_fraction=0.9,
                                            feature_fraction=0.8,
                                            seed=1440)
        else:
            self.load(model_path)
            labelNameToIndex = json.load(open(config.labels_path, encoding='utf-8'))
            self.ix2label = {v: k for k, v in labelNameToIndex.items()}

    def param_search(self, search_method='grid'):
        '''
        search_method: 'grid' or 'bayesian'
        '''
        if search_method == 'grid':
            logger.info("[Parameter Search] use grid search")
            parameters = {
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.05],
                'n_estimators': [1000, 2000],
                'subsample': [0.6, 0.75, 0.9],
                'colsample_bytree': [0.6, 0.75, 0.9],
                'reg_alpha': [5, 10],
                'reg_lambda': [10, 30, 50]
            }
            grid_result = grid_search(self.model, parameters, self.X_train, self.y_train)
            return grid_result.best_params_
        elif search_method == 'bayesian':
            logger.info("[Parameter Search] use bayesian optimization")
            trn_data = lgb.Dataset(data=self.X_train,
                                   label=self.y_train,
                                   free_raw_data=False)
            param = bayes_search_lgb(trn_data,
                                     init_round=2,
                                     opt_round=5,
                                     n_folds=5)
            logger.info("[Parameter Search] best param", param)
            return param

    def unbalance_helper(self,
                         imbalance_method='under_sampling',
                         search_method='grid'):
        '''
        imbalance_method: ClusterCentroids for under_sampling, SMOTE for over_sampling,
                          BalancedBaggingClassifier for ensemble
        search_method: 'grid' or 'bayesian'
        '''
        logger.info("get all freature")

        model_name = None
        if imbalance_method == 'over_sampling':
            logger.info("[Train] Use SMOTE dealing with unbalance data ")
            # x new = x ordi + lambda(x ordj - x ord i), 且 lambda -> (0,1)
            self.X_train, self.y_train = SMOTE().fit_resample(self.X_train, self.y_train)
            self.X_test, self.y_test = SMOTE().fit_resample(self.X_train, self.y_train)
            model_name = 'lgb_over_sampling'
        elif imbalance_method == 'under_sampling':
            logger.info("[Train] Use ClusterCentroids dealing with unbalance data ")
            self.X_train, self.y_train = ClusterCentroids(
                random_state=42).fit_resample(self.X_train, self.y_train)
            self.X_test, self.y_test = ClusterCentroids(
                random_state=42).fit_resample(self.X_test, self.y_test)
            model_name = 'lgb_under_sampling'
        elif imbalance_method == 'ensemble':
            logger.info("[Train] Use BalancedBaggingClassifier dealing with unbalance data ")
            self.model = BalancedBaggingClassifier(
                base_estimator=DecisionTreeClassifier(),
                sampling_strategy='auto',
                replacement=False,
                random_state=0)
            model_name = 'ensemble'

        logger.info('[Train] search best param')
        if imbalance_method != 'ensemble':
            param = self.param_search(search_method=search_method)
            self.model = self.model.set_params(**param)

        logger.info('[Train] fit model ')
        self.model.fit(self.X_train, self.y_train)

        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        per, acc, recall, f1 = evaluate(self.y_train, self.y_test, train_pred, test_pred)

        logger.info('train accuracy %s' % per)
        logger.info('test accuracy %s' % acc)
        logger.info('test recall %s' % recall)
        logger.info('test F1_score %s' % f1)
        self.save(model_name)

    def pred_process(self, data):
        # 处理预测所需要的特征
        data['text'] = data['title'] + data['desc']
        data["queryCut"] = data["text"].apply(self.dataset.cut)
        data["queryCutRMStopWord"] = data["queryCut"].apply(
            lambda x: [w for w in x if w not in self.dataset.stop_words])

        data = self.dataset.get_embedding_features(data)
        data = self.dataset.get_basic_feature(data)
        data = self.dataset.get_bert_feature(data)
        data = self.dataset.get_lda_feature(data)

        return data

    def predict(self, title_pd, desc_pd):
        data = pd.DataFrame([[title_pd, desc_pd]], columns=['title', 'desc'])
        inputs = self.process(data)
        label = self.ix2label[self.model.predict(inputs)[0]]
        proba = np.max(self.model.predict_proba(inputs))
        return label, proba

    def save(self, model_name):
        joblib.dump(self.model, config.lgb_path + model_name)

    def load(self, path):
        self.model = joblib.load(path)