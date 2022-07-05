import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from bayes_opt import BayesianOptimization

import config


def grid_search(model, parameters, x_train, y_train):
    gsearch = GridSearchCV(model,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=3,
                           verbose=True)
    gsearch.fit(x_train, y_train)
    print("Best parameters set found on development set:{}".format(gsearch.best_params_))
    return gsearch


# pip install bayes_opt
def bayes_search_lgb(trn_data,
                    init_round=3,
                    opt_round=5,
                    n_folds=5,
                    random_seed=6,
                    n_estimators=10000,
                    learning_rate=0.05):

    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth,
                 lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {
            'application': 'multiclass',
            'num_iterations': n_estimators,
            'learning_rate': learning_rate,
            'early_stopping_round': 100,
            'num_class': config.num_classes,
            'metric': 'multi_logloss'
        }
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params,
                           trn_data,
                           nfold=n_folds,
                           seed=random_seed,
                           stratified=True,
                           verbose_eval=200)
        return max(cv_result['multi_logloss-mean'])

    bayes_tuner = BayesianOptimization(lgb_eval,
                                {'num_leaves': (20, 50),
                                'feature_fraction': (0.1, 0.9),
                                'bagging_fraction': (0.8, 1),
                                'max_depth': (5, 10),
                                'lambda_l1': (0, 5),
                                'lambda_l2': (0, 3),
                                'min_split_gain': (0.001, 0.1),
                                'min_child_weight': (5, 50)
                                },
                                random_state=random_seed)

    bayes_tuner.maximize(init_points=init_round, n_iter=opt_round)
    # print(bayes_tuner.max)
    return bayes_tuner.max


# ====================================================================
# skopt 包：

from skopt import BayesSearchCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


bayes_cv_tuner = BayesSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 application='multiclass',
                                 n_jobs=-1,
                                 verbose=1),
    search_spaces={
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'num_leaves': (2, 500),
        'max_depth': (0, 500),
        'min_child_samples': (0, 200),
        'max_bin': (100, 100000),
        'subsample': (0.01, 1.0, 'uniform'),
        'subsample_freq': (0, 10),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'min_child_weight': (0, 10),
        'subsample_for_bin': (100000, 500000),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'scale_pos_weight': (1e-6, 500, 'log-uniform'),
        'n_estimators': (10, 10000),
    },
    scoring='f1_macro',
    cv=StratifiedKFold(n_splits=2),
    n_iter=30,
    verbose=1,
    refit=True)


# 递进特征筛选
def rfecv_opt(model, n_jobs, X, y, cv=StratifiedKFold(2)):
    rfecv = RFECV(estimator=model,
                  step=1,
                  cv=cv,
                  n_jobs=n_jobs,
                  scoring='f1_macro',
                  verbose=1)
    rfecv.fit(X.values, y.values.ravel())
    print('Optimal number of features : %d', rfecv.n_features_)
    print('Max score with current model :', round(np.max(rfecv.grid_scores_), 3))
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel('Number of features selected')
    plt.ylabel('Cross validation score (f1_macro)')
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    important_columns = []
    n = 0
    for i in rfecv.support_:
        if i:
            important_columns.append(X.columns[n])
        n += 1
    return important_columns, np.max(rfecv.grid_scores_), rfecv


def searching(X, y, n_iter_max, n_jobs, n_class):
    list_models = []
    list_scores_max = []
    list_features = []
    for i in range(n_iter_max):
        print('Currently on iteration', i + 1, 'of', n_iter_max, '.')
        if i == 0:
            model = lgb.LGBMClassifier(
                max_depth=-1,
                learning_rate=0.1,
                objective='multiclass',
                silent=False,
                metric='None',
                num_class=n_class,
                n_jobs=n_jobs,
                n_estimators=8000,
                class_weight='unbalanced')
        else:
            print('Adjusting model.')
            # Get current parameters and the best parameters
            result = bayes_cv_tuner.fit(X.values, y.values.ravel())
            best_params = pd.Series(result.best_params_)
            param_dict = pd.Series.to_dict(best_params)
            model = lgb.LGBMClassifier(
                colsample_bytree=param_dict['colsample_bytree'],
                learning_rate=param_dict['learning_rate'],
                max_bin=int(param_dict['max_bin']),
                max_depth=int(param_dict['max_depth']),
                min_child_samples=int(param_dict['min_child_samples']),
                min_child_weight=param_dict['min_child_weight'],
                n_estimators=int(param_dict['n_estimators']),
                num_leaves=int(param_dict['num_leaves']),
                reg_alpha=param_dict['reg_alpha'],
                reg_lambda=param_dict['reg_lambda'],
                scale_pos_weight=param_dict['scale_pos_weight'],
                subsample=param_dict['subsample'],
                subsample_for_bin=int(param_dict['subsample_for_bin']),
                subsample_freq=int(param_dict['subsample_freq']),
                n_jobs=n_jobs,
                class_weight='unbalanced',
                objective='multiclass')
        imp_columns, max_score, rfecv = rfecv_opt(model, n_jobs, X, y)
        list_models.append(model)
        list_scores_max.append(max_score)
        list_features.append(imp_columns)
    return list_models, list_scores_max, list_features