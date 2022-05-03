import os
import time
from sklearn.metrics import f1_score, precision_score
from sklearn import ensemble
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from statistics import mean

from load_data import init_directories, load_data_from_folder
from get_cluster_results import get_cluster_results

import warnings

warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})


def random_forest(X_train, X_test, y_train, y_test, df_X_test, optimization):
    X_test_diminue = X_test
    X_test_diminue = X_test_diminue.values
    X_test_diminue = X_test_diminue[:, 1:-1]

    if optimization == "yes":
        param_grid_rf = {
            'n_estimators': [10, 15, 20, 30, 50, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'min_samples_split': [2, 5, 8, 10, 12, 15],
            'min_samples_leaf': [1, 2, 3, 4, 5, 10],
            'bootstrap': [True, False]}

        rf_clf = ensemble.RandomForestClassifier(random_state=42)
        try:
            RandomGrid = RandomizedSearchCV(estimator=rf_clf, n_iter=250, param_distributions=param_grid_rf,
                                            cv=5, n_jobs=-1)
            RandomGrid.fit(X_train, y_train)
            rf_clf = ensemble.RandomForestClassifier(**RandomGrid.best_params_, random_state=42)
            print("optimized rf")
        except:
            rf_clf = ensemble.RandomForestClassifier(random_state=42)  # Build
            print("NON-optimized rf")
    else:
        rf_clf = ensemble.RandomForestClassifier(random_state=42)  # Build

    rf_clf.fit(X_train, y_train)  # Train

    y_pred = rf_clf.predict(X_test_diminue)

    df_X_test_RF = df_X_test.copy()
    nb_of_orders = df_X_test_RF['order_number']
    nb_of_orders = nb_of_orders[0]

    f1_rf = float("{0:.3f}".format(f1_score(y_test, y_pred, average='macro')))

    precision_rf = float("{0:.3f}".format(precision_score(y_test, y_pred, average='macro')))

    recall_rf = float("{0:.3f}".format(recall_score(y_test, y_pred, average='macro')))

    return rf_clf, f1_rf, precision_rf, recall_rf, nb_of_orders


def gbt(X_train, X_test, y_train, y_test, df_X_test, optimization):
    X_test_diminue = X_test
    X_test_diminue = X_test_diminue.values
    X_test_diminue = X_test_diminue[:, 1:-1]

    if optimization == "yes":
        param_grid_gb = {'n_estimators': [10, 50, 100, 250, 500, 1000],
                         'learning_rate': [0.001, 0.01, 0.1, 0.3, 0.6, 0.7, 0.85, 1, 10, 100],
                         'max_depth': [3, 7, 9, 11, 13, 15],
                         'min_samples_split': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
                         'min_samples_leaf': [30, 40, 50, 60, 70],
                         'subsample': [0.5, 0.7, 1.0]}

        gb_clf = ensemble.GradientBoostingClassifier(random_state=42)  # Build

        try:
            RandomGrid = RandomizedSearchCV(estimator=gb_clf, n_iter=250, param_distributions=param_grid_gb,
                                            cv=5, n_jobs=-1)
            RandomGrid.fit(X_train, y_train)
            gb_clf = ensemble.GradientBoostingClassifier(**RandomGrid.best_params_, random_state=42)
            print("optimized gb")
        except:
            gb_clf = ensemble.GradientBoostingClassifier(random_state=42)  # Build
            print("NON-optimized gb")
    else:
        gb_clf = ensemble.GradientBoostingClassifier(random_state=42)  # Build

    gb_clf.fit(X_train, y_train)  # Train

    y_pred = gb_clf.predict(X_test_diminue)

    f1_gb = float("{0:.3f}".format(f1_score(y_test, y_pred, average='macro')))

    precision_gb = float("{0:.3f}".format(precision_score(y_test, y_pred, average='macro')))

    recall_gb = float("{0:.3f}".format(recall_score(y_test, y_pred, average='macro')))

    return gb_clf, f1_gb, precision_gb, recall_gb


def xgb_model(X_train, X_test, y_train, y_test, df_X_test):
    X_test_diminue = X_test
    X_test_diminue = X_test_diminue.values
    X_test_diminue = X_test_diminue[:, 1:-1]

    try:
        xgb_clf = xgb.XGBClassifier(objective="multi:softprob", random_state=42, eval_metric='mlogloss',
                                    verbosity=0, silent=True)  # Build

        xgb_clf.fit(X_train, y_train)  # Train
    except:
        xgb_clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eval_metric='mlogloss',
                                    verbosity=0, silent=True)  # Build

    xgb_clf.fit(X_train, y_train)  # Train

    y_pred = xgb_clf.predict(X_test_diminue)

    f1_xgb = float("{0:.3f}".format(f1_score(y_test, y_pred, average='macro')))

    precision_xgb = float("{0:.3f}".format(precision_score(y_test, y_pred, average='macro')))

    recall_xgb = float("{0:.3f}".format(recall_score(y_test, y_pred, average='macro')))

    return xgb_clf, f1_xgb, precision_xgb, recall_xgb


def lr_model(X_train, X_test, y_train, y_test, df_X_test):
    X_test_diminue = X_test
    X_test_diminue = X_test_diminue.values
    X_test_diminue = X_test_diminue[:, 1:-1]

    lr_clf = LogisticRegression(random_state=0)  # Build

    lr_clf.fit(X_train, y_train)  # Train

    y_pred = lr_clf.predict(X_test_diminue)

    f1_lr = float("{0:.3f}".format(f1_score(y_test, y_pred, average='macro')))

    precision_lr = float("{0:.3f}".format(precision_score(y_test, y_pred, average='macro')))

    recall_lr = float("{0:.3f}".format(recall_score(y_test, y_pred, average='macro')))

    return lr_clf, f1_lr, precision_lr, recall_lr


def catboost_model(X_train, X_test, y_train, y_test, df_X_test, optimization):
    X_test_diminue = X_test
    X_test_diminue = X_test_diminue.values
    X_test_diminue = X_test_diminue[:, 1:-1]

    if optimization == "yes":
        param_grid_catboost = {'learning_rate': [0.03, 0.1, 0.5, 1],
                               'depth': [2, 4, 6, 10],
                               'l2_leaf_reg': [1, 3, 5, 7, 9]}

        try:
            catboost_clf = CatBoostClassifier(iterations=2, logging_level='Silent')  # Build

            try:
                RandomGrid = RandomizedSearchCV(estimator=catboost_clf, n_iter=250,
                                                param_distributions=param_grid_catboost,
                                                cv=5, n_jobs=8)
                RandomGrid.fit(X_train, y_train)
                catboost_clf = CatBoostClassifier(**RandomGrid.best_params_, iterations=2, logging_level='Silent')
                print("optimized catboost")
            except:
                catboost_clf = CatBoostClassifier(iterations=10, learning_rate=1, depth=2, logging_level='Silent')
                print("NON-optimized catboost")

            catboost_clf.fit(X_train, y_train)  # Train
        except:
            catboost_clf = CatBoostClassifier(iterations=10, loss_function='MultiClass',
                                              logging_level='Silent')  # Build
            try:
                RandomGrid = RandomizedSearchCV(estimator=catboost_clf, n_iter=250,
                                                param_distributions=param_grid_catboost,
                                                cv=5, n_jobs=8)
                RandomGrid.fit(X_train, y_train)
                catboost_clf = CatBoostClassifier(**RandomGrid.best_params_, iterations=10, loss_function='MultiClass',
                                                  logging_level='Silent')
                print("optimized catboost")
            except:
                catboost_clf = CatBoostClassifier(iterations=10, learning_rate=1, depth=2, loss_function='MultiClass',
                                                  logging_level='Silent')
                print("NON-optimized catboost")
            catboost_clf.fit(X_train, y_train)  # Train

    else:
        try:
            catboost_clf = CatBoostClassifier(iterations=2, learning_rate=1, depth=2, logging_level='Silent')  # Build

            catboost_clf.fit(X_train, y_train)  # Train
        except:
            catboost_clf = CatBoostClassifier(iterations=10, learning_rate=1, depth=2, loss_function='MultiClass',
                                              logging_level='Silent')  # Build

        catboost_clf.fit(X_train, y_train)  # Train

    y_pred = catboost_clf.predict(X_test_diminue)

    f1_catboost = float("{0:.3f}".format(f1_score(y_test, y_pred, average='macro')))

    precision_catboost = float("{0:.3f}".format(precision_score(y_test, y_pred, average='macro')))

    recall_catboost = float("{0:.3f}".format(recall_score(y_test, y_pred, average='macro')))

    return catboost_clf, f1_catboost, precision_catboost, recall_catboost


def decision_tree(X_train, X_test, y_train, y_test, df_X_test, optimization):
    X_test_diminue = X_test
    X_test_diminue = X_test_diminue.values
    X_test_diminue = X_test_diminue[:, 1:-1]

    if optimization == "yes":
        param_grid_dt = {'max_depth': [1, 2, 3, 4, 5, 6, 8, 10, 12],
                         'min_samples_leaf': [1, 2, 3, 4, 5],
                         'min_samples_split': [2, 3, 4, 5],
                         'criterion': ['gini', 'entropy']}

        dt_clf = tree.DecisionTreeClassifier(random_state=42)  # Build
        try:
            RandomGrid = RandomizedSearchCV(estimator=dt_clf, n_iter=250, param_distributions=param_grid_dt,
                                            cv=5, n_jobs=-1)
            RandomGrid.fit(X_train, y_train)
            dt_clf = tree.DecisionTreeClassifier(**RandomGrid.best_params_, random_state=42)
            print("optimized dt")
        except:
            dt_clf = tree.DecisionTreeClassifier(random_state=42)  # Build
            print("NON-optimized dt")
    else:
        dt_clf = tree.DecisionTreeClassifier(random_state=42)  # Build

    dt_clf.fit(X_train, y_train)  # Train

    y_pred = dt_clf.predict(X_test_diminue)

    f1_dt = float("{0:.3f}".format(f1_score(y_test, y_pred, average='macro')))

    precision_dt = float("{0:.3f}".format(precision_score(y_test, y_pred, average='macro')))

    recall_dt = float("{0:.3f}".format(recall_score(y_test, y_pred, average='macro')))

    return dt_clf, f1_dt, precision_dt, recall_dt


def naives_bayes(X_train, X_test, y_train, y_test, df_X_test):
    X_test_diminue = X_test
    X_test_diminue = X_test_diminue.values
    X_test_diminue = X_test_diminue[:, 1:-1]

    nb_clf = GaussianNB()  # Build
    nb_clf.fit(X_train, y_train)  # Train

    y_pred = nb_clf.predict(X_test_diminue)

    f1_nb = float("{0:.3f}".format(f1_score(y_test, y_pred, average='macro')))

    precision_nb = float("{0:.3f}".format(precision_score(y_test, y_pred, average='macro')))

    recall_nb = float("{0:.3f}".format(recall_score(y_test, y_pred, average='macro')))

    return nb_clf, f1_nb, precision_nb, recall_nb


def svc(X_train, X_test, y_train, y_test, df_X_test):
    X_test_diminue = X_test
    X_test_diminue = X_test_diminue.values
    X_test_diminue = X_test_diminue[:, 1:-1]

    sv_clf = SVC(probability=True, random_state=42)  # Build
    sv_clf.fit(X_train, y_train)  # Train

    y_pred = sv_clf.predict(X_test_diminue)

    f1_sv = float("{0:.3f}".format(f1_score(y_test, y_pred, average='macro')))

    precision_sv = float("{0:.3f}".format(precision_score(y_test, y_pred, average='macro')))

    recall_sv = float("{0:.3f}".format(recall_score(y_test, y_pred, average='macro')))

    return sv_clf, f1_sv, precision_sv, recall_sv


def mlp_model(X_train, X_test, y_train, y_test, df_X_test):
    X_test_diminue = X_test
    X_test_diminue = X_test_diminue.values
    X_test_diminue = X_test_diminue[:, 1:-1]

    mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2, 5), random_state=42)  # Build

    mlp_clf.fit(X_train, y_train)  # Train

    y_pred = mlp_clf.predict(X_test_diminue)

    f1_mlp = float("{0:.3f}".format(f1_score(y_test, y_pred, average='macro')))

    precision_mlp = float("{0:.3f}".format(precision_score(y_test, y_pred, average='macro')))

    recall_mlp = float("{0:.3f}".format(recall_score(y_test, y_pred, average='macro')))

    return mlp_clf, f1_mlp, precision_mlp, recall_mlp


def calculate_metrics_mean(rf_f1, rf_precision, rf_recall,
                           gb_f1, gb_precision, gb_recall,
                           xgb_f1, xgb_precision, xgb_recall,
                           lr_f1, lr_precision, lr_recall,
                           catboost_f1, catboost_precision, catboost_recall,
                           dt_f1, dt_precision, dt_recall,
                           nb_f1, nb_precision, nb_recall,
                           sv_f1, sv_precision, sv_recall,
                           mlp_f1, mlp_precision, mlp_recall,
                           dict_cluster_metrics, dir_count):

    # print(len(rf_f1))
    rf_cluster_f1_mean = float("{0:.3f}".format(mean(rf_f1)))
    rf_cluster_precision_mean = float("{0:.3f}".format(mean(rf_precision)))
    rf_cluster_recall_mean = float("{0:.3f}".format(mean(rf_recall)))

    # print(len(gb_f1))
    gb_cluster_f1_mean = float("{0:.3f}".format(mean(gb_f1)))
    gb_cluster_precision_mean = float("{0:.3f}".format(mean(gb_precision)))
    gb_cluster_recall_mean = float("{0:.3f}".format(mean(gb_recall)))

    # print(len(xgb_f1))
    xgb_cluster_f1_mean = float("{0:.3f}".format(mean(xgb_f1)))
    xgb_cluster_precision_mean = float("{0:.3f}".format(mean(xgb_precision)))
    xgb_cluster_recall_mean = float("{0:.3f}".format(mean(xgb_recall)))

    # print(len(lr_f1))
    lr_cluster_f1_mean = float("{0:.3f}".format(mean(lr_f1)))
    lr_cluster_precision_mean = float("{0:.3f}".format(mean(lr_precision)))
    lr_cluster_recall_mean = float("{0:.3f}".format(mean(lr_recall)))

    # print(len(catboost_f1))
    catboost_cluster_f1_mean = float("{0:.3f}".format(mean(catboost_f1)))
    catboost_cluster_precision_mean = float("{0:.3f}".format(mean(catboost_precision)))
    catboost_cluster_recall_mean = float("{0:.3f}".format(mean(catboost_recall)))

    # print(len(dt_f1))
    dt_cluster_f1_mean = float("{0:.3f}".format(mean(dt_f1)))
    dt_cluster_precision_mean = float("{0:.3f}".format(mean(dt_precision)))
    dt_cluster_recall_mean = float("{0:.3f}".format(mean(dt_recall)))

    # print(len(nb_f1))
    nb_cluster_f1_mean = float("{0:.3f}".format(mean(nb_f1)))
    nb_cluster_precision_mean = float("{0:.3f}".format(mean(nb_precision)))
    nb_cluster_recall_mean = float("{0:.3f}".format(mean(nb_recall)))

    sv_cluster_f1_mean = float("{0:.3f}".format(mean(sv_f1)))
    sv_cluster_precision_mean = float("{0:.3f}".format(mean(sv_precision)))
    sv_cluster_recall_mean = float("{0:.3f}".format(mean(sv_recall)))

    mlp_cluster_f1_mean = float("{0:.3f}".format(mean(mlp_f1)))
    mlp_cluster_precision_mean = float("{0:.3f}".format(mean(mlp_precision)))
    mlp_cluster_recall_mean = float("{0:.3f}".format(mean(mlp_recall)))

    cluster_f1 = {'rf': rf_cluster_f1_mean,
                  'gb': gb_cluster_f1_mean,
                  'xgb': xgb_cluster_f1_mean,
                  'lr': lr_cluster_f1_mean,
                  'catboost': catboost_cluster_f1_mean,
                  'dt': dt_cluster_f1_mean,
                  'nb': nb_cluster_f1_mean,
                  'sv': sv_cluster_f1_mean,
                  'mlp': mlp_cluster_f1_mean
                  }

    cluster_accuracy = {'rf': rf_cluster_precision_mean,
                        'gb': gb_cluster_precision_mean,
                        'xgb': xgb_cluster_precision_mean,
                        'lr': lr_cluster_precision_mean,
                        'catboost': catboost_cluster_precision_mean,
                        'dt': dt_cluster_precision_mean,
                        'nb': nb_cluster_precision_mean,
                        'sv': sv_cluster_precision_mean,
                        'mlp': mlp_cluster_precision_mean
                        }

    cluster_recall = {'rf': rf_cluster_recall_mean,
                      'gb': gb_cluster_recall_mean,
                      'xgb': xgb_cluster_recall_mean,
                      'lr': lr_cluster_recall_mean,
                      'catboost': catboost_cluster_recall_mean,
                      'dt': dt_cluster_recall_mean,
                      'nb': nb_cluster_recall_mean,
                      'sv': sv_cluster_recall_mean,
                      'mlp': mlp_cluster_recall_mean
                      }

    dict_cluster_metrics['cluster'].append('g' + str(int(dir_count)))

    dict_cluster_metrics['RF_F1_mean'].append(rf_cluster_f1_mean)
    dict_cluster_metrics['RF_precision_mean'].append(rf_cluster_precision_mean)
    dict_cluster_metrics['RF_recall_mean'].append(rf_cluster_recall_mean)

    dict_cluster_metrics['GB_F1_mean'].append(gb_cluster_f1_mean)
    dict_cluster_metrics['GB_precision_mean'].append(gb_cluster_precision_mean)
    dict_cluster_metrics['GB_recall_mean'].append(gb_cluster_recall_mean)

    dict_cluster_metrics['XGB_F1_mean'].append(xgb_cluster_f1_mean)
    dict_cluster_metrics['XGB_precision_mean'].append(xgb_cluster_precision_mean)
    dict_cluster_metrics['XGB_recall_mean'].append(xgb_cluster_recall_mean)

    dict_cluster_metrics['LR_F1_mean'].append(lr_cluster_f1_mean)
    dict_cluster_metrics['LR_precision_mean'].append(lr_cluster_precision_mean)
    dict_cluster_metrics['LR_recall_mean'].append(lr_cluster_recall_mean)

    dict_cluster_metrics['Catboost_F1_mean'].append(catboost_cluster_f1_mean)
    dict_cluster_metrics['Catboost_precision_mean'].append(catboost_cluster_precision_mean)
    dict_cluster_metrics['Catboost_recall_mean'].append(catboost_cluster_recall_mean)

    dict_cluster_metrics['DT_F1_mean'].append(dt_cluster_f1_mean)
    dict_cluster_metrics['DT_precision_mean'].append(dt_cluster_precision_mean)
    dict_cluster_metrics['DT_recall_mean'].append(dt_cluster_recall_mean)

    dict_cluster_metrics['NB_F1_mean'].append(nb_cluster_f1_mean)
    dict_cluster_metrics['NB_precision_mean'].append(nb_cluster_precision_mean)
    dict_cluster_metrics['NB_recall_mean'].append(nb_cluster_recall_mean)

    dict_cluster_metrics['SV_F1_mean'].append(sv_cluster_f1_mean)
    dict_cluster_metrics['SV_precision_mean'].append(sv_cluster_precision_mean)
    dict_cluster_metrics['SV_recall_mean'].append(sv_cluster_recall_mean)

    dict_cluster_metrics['MLP_F1_mean'].append(mlp_cluster_f1_mean)
    dict_cluster_metrics['MLP_precision_mean'].append(mlp_cluster_precision_mean)
    dict_cluster_metrics['MLP_recall_mean'].append(mlp_cluster_recall_mean)

    return cluster_f1, cluster_accuracy, cluster_recall


def main_ml(optimization):
    start_time = time.time()
    path_to_postprocess_all_products = "../postprocess_all_products/"
    path_to_results_all_users = "../results/ml_results/results_by_user/new_users_results_without_clustering.csv"
    dir_count = 0

    dict_user_metrics = {'user': [], 'number_of_orders': [],
                         'RF_F1': [], 'RF_precision': [], 'RF_recall': [],
                         'GB_F1': [], 'GB_precision': [], 'GB_recall': [],
                         'XGB_F1': [], 'XGB_precision': [], 'XGB_recall': [],
                         'LR_F1': [], 'LR_precision': [], 'LR_recall': [],
                         'Catboost_F1': [], 'Catboost_precision': [], 'Catboost_recall': [],
                         'DT_F1': [], 'DT_precision': [], 'DT_recall': [],
                         'NB_F1': [], 'NB_precision': [], 'NB_recall': [],
                         'SV_F1': [], 'SV_precision': [], 'SV_recall': [],
                         'MLP_F1': [], 'MLP_precision': [], 'MLP_recall': []
                         }

    dict_cluster_metrics = {'cluster': [],
                            'RF_F1_mean': [], 'RF_precision_mean': [], 'RF_recall_mean': [],
                            'GB_F1_mean': [], 'GB_precision_mean': [], 'GB_recall_mean': [],
                            'XGB_F1_mean': [], 'XGB_precision_mean': [], 'XGB_recall_mean': [],
                            'LR_F1_mean': [], 'LR_precision_mean': [], 'LR_recall_mean': [],
                            'Catboost_F1_mean': [], 'Catboost_precision_mean': [], 'Catboost_recall_mean': [],
                            'DT_F1_mean': [], 'DT_precision_mean': [], 'DT_recall_mean': [],
                            'NB_F1_mean': [], 'NB_precision_mean': [], 'NB_recall_mean': [],
                            'SV_F1_mean': [], 'SV_precision_mean': [], 'SV_recall_mean': [],
                            'MLP_F1_mean': [], 'MLP_precision_mean': [], 'MLP_recall_mean': []
                            }

    for directory in init_directories("all_products"):
        dir = directory

        dir_count += 1
        dir_count_max = len(init_directories("all_products"))
        # print(dir_count_max)
        print(f"Processing g{dir_count} ...")

        rf_cluster_F1_list, rf_cluster_precision_list, rf_cluster_recall_list = [], [], []

        gb_cluster_F1_list, gb_cluster_precision_list, gb_cluster_recall_list = [], [], []

        xgb_cluster_F1_list, xgb_cluster_precision_list, xgb_cluster_recall_list = [], [], []

        lr_cluster_F1_list, lr_cluster_precision_list, lr_cluster_recall_list = [], [], []

        catboost_cluster_F1_list, catboost_cluster_precision_list, catboost_cluster_recall_list = [], [], []

        dt_cluster_F1_list, dt_cluster_precision_list, dt_cluster_recall_list = [], [], []

        nb_cluster_F1_list, nb_cluster_precision_list, nb_cluster_recall_list = [], [], []

        sv_cluster_F1_list, sv_cluster_precision_list, sv_cluster_recall_list = [], [], []

        mlp_cluster_F1_list, mlp_cluster_precision_list, mlp_cluster_recall_list = [], [], []

        for filename in os.listdir(dir):
            filename_path = dir + "/" + filename + '/Recommended_products'
            print(filename)

            X_train, X_test, y_train, y_test, moyenne_taille_paniers, df_X_test, dataset_test_diminue, y_test2 = load_data_from_folder(
                dir, filename)

            rf = random_forest(X_train, X_test, y_train, y_test, df_X_test, optimization)
            gb = gbt(X_train, X_test, y_train, y_test, df_X_test, optimization)
            xgb = xgb_model(X_train, X_test, y_train, y_test, df_X_test)
            lr = lr_model(X_train, X_test, y_train, y_test, df_X_test)
            catboost = catboost_model(X_train, X_test, y_train, y_test, df_X_test, optimization)
            dt = decision_tree(X_train, X_test, y_train, y_test, df_X_test, optimization)
            nb = naives_bayes(X_train, X_test, y_train, y_test, df_X_test)
            sv = svc(X_train, X_test, y_train, y_test, df_X_test)
            mlp = mlp_model(X_train, X_test, y_train, y_test, df_X_test)

            rf_cluster_F1_list.append(rf[1])
            gb_cluster_F1_list.append(gb[1])
            xgb_cluster_F1_list.append(xgb[1])
            lr_cluster_F1_list.append(lr[1])
            catboost_cluster_F1_list.append(catboost[1])
            dt_cluster_F1_list.append(dt[1])
            nb_cluster_F1_list.append(nb[1])
            sv_cluster_F1_list.append(sv[1])
            mlp_cluster_F1_list.append(mlp[1])

            rf_cluster_precision_list.append(rf[2])
            gb_cluster_precision_list.append(gb[2])
            xgb_cluster_precision_list.append(xgb[2])
            lr_cluster_precision_list.append(lr[2])
            catboost_cluster_precision_list.append(catboost[2])
            dt_cluster_precision_list.append(dt[2])
            nb_cluster_precision_list.append(nb[2])
            sv_cluster_precision_list.append(sv[2])
            mlp_cluster_precision_list.append(mlp[2])

            rf_cluster_recall_list.append(rf[3])
            gb_cluster_recall_list.append(gb[3])
            xgb_cluster_recall_list.append(xgb[3])
            lr_cluster_recall_list.append(lr[3])
            catboost_cluster_recall_list.append(catboost[3])
            dt_cluster_recall_list.append(dt[3])
            nb_cluster_recall_list.append(nb[3])
            sv_cluster_recall_list.append(sv[3])
            mlp_cluster_recall_list.append(mlp[3])

            dict_user_metrics['user'].append(str(filename))
            dict_user_metrics['number_of_orders'].append(rf[4])
            # print(dict_user_metrics)

            dict_user_metrics['RF_F1'].append(rf[1])
            dict_user_metrics['RF_precision'].append(rf[2])
            dict_user_metrics['RF_recall'].append(rf[3])

            dict_user_metrics['GB_F1'].append(gb[1])
            dict_user_metrics['GB_precision'].append(gb[2])
            dict_user_metrics['GB_recall'].append(gb[3])

            dict_user_metrics['XGB_F1'].append(xgb[1])
            dict_user_metrics['XGB_precision'].append(xgb[2])
            dict_user_metrics['XGB_recall'].append(xgb[3])

            dict_user_metrics['LR_F1'].append(lr[1])
            dict_user_metrics['LR_precision'].append(lr[2])
            dict_user_metrics['LR_recall'].append(lr[3])

            dict_user_metrics['Catboost_F1'].append(catboost[1])
            dict_user_metrics['Catboost_precision'].append(catboost[2])
            dict_user_metrics['Catboost_recall'].append(catboost[3])

            dict_user_metrics['DT_F1'].append(dt[1])
            dict_user_metrics['DT_precision'].append(dt[2])
            dict_user_metrics['DT_recall'].append(dt[3])

            dict_user_metrics['NB_F1'].append(nb[1])
            dict_user_metrics['NB_precision'].append(nb[2])
            dict_user_metrics['NB_recall'].append(nb[3])

            dict_user_metrics['SV_F1'].append(sv[1])
            dict_user_metrics['SV_precision'].append(sv[2])
            dict_user_metrics['SV_recall'].append(sv[3])

            dict_user_metrics['MLP_F1'].append(mlp[1])
            dict_user_metrics['MLP_precision'].append(mlp[2])
            dict_user_metrics['MLP_recall'].append(mlp[3])

            df_results = pd.DataFrame(dict_user_metrics)

            savepath_all_users_results = '../results/ml_results/results_by_user'
            if not os.path.exists('../results/'):
                os.mkdir('../results/')
            if not os.path.exists('../results/ml_results/'):
                os.mkdir('../results/ml_results/')
            if not os.path.exists(savepath_all_users_results):
                os.mkdir(savepath_all_users_results)
            df_results.to_csv(
                r'../results/ml_results/results_by_user/results.csv',
                index=False)

    dict_user_metrics['user'].append("mean")
    dict_user_metrics['number_of_orders'].append("None")

    dict_user_metrics['RF_F1'].append(float("{0:.3f}".format(mean(dict_user_metrics['RF_F1']))))
    dict_user_metrics['RF_precision'].append(float("{0:.3f}".format(mean(dict_user_metrics['RF_precision']))))
    dict_user_metrics['RF_recall'].append(float("{0:.3f}".format(mean(dict_user_metrics['RF_recall']))))

    dict_user_metrics['GB_F1'].append(float("{0:.3f}".format(mean(dict_user_metrics['GB_F1']))))
    dict_user_metrics['GB_precision'].append(float("{0:.3f}".format(mean(dict_user_metrics['GB_precision']))))
    dict_user_metrics['GB_recall'].append(float("{0:.3f}".format(mean(dict_user_metrics['GB_recall']))))

    dict_user_metrics['XGB_F1'].append(float("{0:.3f}".format(mean(dict_user_metrics['XGB_F1']))))
    dict_user_metrics['XGB_precision'].append(float("{0:.3f}".format(mean(dict_user_metrics['XGB_precision']))))
    dict_user_metrics['XGB_recall'].append(float("{0:.3f}".format(mean(dict_user_metrics['XGB_recall']))))

    dict_user_metrics['LR_F1'].append(float("{0:.3f}".format(mean(dict_user_metrics['LR_F1']))))
    dict_user_metrics['LR_precision'].append(float("{0:.3f}".format(mean(dict_user_metrics['LR_precision']))))
    dict_user_metrics['LR_recall'].append(float("{0:.3f}".format(mean(dict_user_metrics['LR_recall']))))

    dict_user_metrics['Catboost_F1'].append(float("{0:.3f}".format(mean(dict_user_metrics['Catboost_F1']))))
    dict_user_metrics['Catboost_precision'].append(float("{0:.3f}".format(mean(dict_user_metrics['Catboost_precision']))))
    dict_user_metrics['Catboost_recall'].append(float("{0:.3f}".format(mean(dict_user_metrics['Catboost_recall']))))

    dict_user_metrics['DT_F1'].append(float("{0:.3f}".format(mean(dict_user_metrics['DT_F1']))))
    dict_user_metrics['DT_precision'].append(float("{0:.3f}".format(mean(dict_user_metrics['DT_precision']))))
    dict_user_metrics['DT_recall'].append(float("{0:.3f}".format(mean(dict_user_metrics['DT_recall']))))

    dict_user_metrics['NB_F1'].append(float("{0:.3f}".format(mean(dict_user_metrics['NB_F1']))))
    dict_user_metrics['NB_precision'].append(float("{0:.3f}".format(mean(dict_user_metrics['NB_precision']))))
    dict_user_metrics['NB_recall'].append(float("{0:.3f}".format(mean(dict_user_metrics['NB_recall']))))

    dict_user_metrics['SV_F1'].append(float("{0:.3f}".format(mean(dict_user_metrics['SV_F1']))))
    dict_user_metrics['SV_precision'].append(float("{0:.3f}".format(mean(dict_user_metrics['SV_precision']))))
    dict_user_metrics['SV_recall'].append(float("{0:.3f}".format(mean(dict_user_metrics['SV_recall']))))

    dict_user_metrics['MLP_F1'].append(float("{0:.3f}".format(mean(dict_user_metrics['MLP_F1']))))
    dict_user_metrics['MLP_precision'].append(float("{0:.3f}".format(mean(dict_user_metrics['MLP_precision']))))
    dict_user_metrics['MLP_recall'].append(float("{0:.3f}".format(mean(dict_user_metrics['MLP_recall']))))

    df_results = pd.DataFrame(dict_user_metrics)
    print(df_results)

    savepath_all_users_results = '../results/ml_results/results_by_user'
    if not os.path.exists(savepath_all_users_results):
        os.mkdir(savepath_all_users_results)
    df_results.to_csv(
        r'../results/ml_results/results_by_user/results.csv',
        index=False)
    savepath_all_users_results_full = savepath_all_users_results + '/results.csv'
    print("Processing completed.")

    get_cluster_results(savepath_all_users_results_full)

    print("--- It took " + str((time.time() - start_time)) + " seconds to run ML algorithms on dataset ---")

    return savepath_all_users_results_full


if __name__ == "__main__":
    main_ml("no")

