import pandas as pd
import os


def get_cluster_results(savepath_all_users_results_full):
    path_to_users_groups = "../postprocess_all_products/"
    path_to_results_all_users = savepath_all_users_results_full

    dict_cluster_metrics = {'cluster': [],
                            'RF_F1_mean': [],
                            'GB_F1_mean': [],
                            'XGB_F1_mean': [],
                            'LR_F1_mean': [],
                            'Catboost_F1_mean': [],
                            'DT_F1_mean': [],
                            'NB_F1_mean': [],
                            'SV_F1_mean': [],
                            'MLP_F1_mean': []
                            }

    users_g1, users_g2, users_g3, users_g4 = [], [], [], []

    for folder in os.listdir(path_to_users_groups):
        # print(folder)
        filepath = (path_to_users_groups + str(folder) + '/')
        for file in os.listdir(filepath):
            if folder == "g1":
                users_g1.append(str(file))
            elif folder == "g2":
                users_g2.append(str(file))
            elif folder == "g3":
                users_g3.append(str(file))
            elif folder == "g4":
                users_g4.append(str(file))

    """ ================================================  RF  ==============================================  """

    # Moyenne RF_F1 pour le groupe 1
    df_results_all_users = pd.read_csv(path_to_results_all_users)
    # print(df_results_all_users)
    df_results_g1 = df_results_all_users[df_results_all_users['user'].isin(users_g1)]
    rf_f1_list_g1 = df_results_g1.iloc[:, 2:3].values
    rf_f1_list_g1 = float("{0:.3f}".format(rf_f1_list_g1.mean()))
    # print('rf_f1_list_g1')
    # print(rf_f1_list_g1)
    # print()

    # Moyenne RF_F1 pour le groupe 2
    df_results_g2 = df_results_all_users[df_results_all_users['user'].isin(users_g2)]
    rf_f1_list_g2 = df_results_g2.iloc[:, 2:3].values
    rf_f1_list_g2 = float("{0:.3f}".format(rf_f1_list_g2.mean()))
    # print('rf_f1_list_g2')
    # print(rf_f1_list_g2)
    # print()

    # Moyenne RF_F1 pour le groupe 3
    df_results_g3 = df_results_all_users[df_results_all_users['user'].isin(users_g3)]
    rf_f1_list_g3 = df_results_g3.iloc[:, 2:3].values
    rf_f1_list_g3 = float("{0:.3f}".format(rf_f1_list_g3.mean()))
    # print('rf_f1_list_g3')
    # print(rf_f1_list_g3)
    # print()

    # Moyenne RF_F1 pour le groupe 4
    df_results_g4 = df_results_all_users[df_results_all_users['user'].isin(users_g4)]
    rf_f1_list_g4 = df_results_g4.iloc[:, 2:3].values
    rf_f1_list_g4 = float("{0:.3f}".format(rf_f1_list_g4.mean()))
    # print('rf_f1_list_g4')
    # print(rf_f1_list_g4)
    # print()

    # Moyenne pour tous les groupes
    rf_f1_mean = ((rf_f1_list_g1 * len(users_g1)) + (rf_f1_list_g2 * len(users_g2)) + (rf_f1_list_g3 * len(users_g3)) +
                  (rf_f1_list_g4 * len(users_g4))) / \
                 (len(users_g1) + len(users_g2) + len(users_g3) + len(users_g4))
    rf_f1_mean = float("{0:.3f}".format(rf_f1_mean))
    # print('rf_f1_mean_g1')
    # print(rf_f1_mean)

    """ ================================================  GB  ==============================================  """

    # Moyenne F1 pour le groupe 1
    df_results_all_users = pd.read_csv(path_to_results_all_users)
    df_results_g1 = df_results_all_users[df_results_all_users['user'].isin(users_g1)]
    gb_f1_list_g1 = df_results_g1.iloc[:, 5:6].values
    gb_f1_list_g1 = float("{0:.3f}".format(gb_f1_list_g1.mean()))
    # print('gb_f1_list_g1')
    # print(gb_f1_list_g1)
    # print()

    # Moyenne F1 pour le groupe 2
    df_results_g2 = df_results_all_users[df_results_all_users['user'].isin(users_g2)]
    gb_f1_list_g2 = df_results_g2.iloc[:, 5:6].values
    gb_f1_list_g2 = float("{0:.3f}".format(gb_f1_list_g2.mean()))
    # print('gb_f1_list_g2')
    # print(gb_f1_list_g2)
    # print()

    # Moyenne F1 pour le groupe 3
    df_results_g3 = df_results_all_users[df_results_all_users['user'].isin(users_g3)]
    gb_f1_list_g3 = df_results_g3.iloc[:, 5:6].values
    gb_f1_list_g3 = float("{0:.3f}".format(gb_f1_list_g3.mean()))
    # print('rf_f1_list_g3')
    # print(rf_f1_list_g3)
    # print()

    # Moyenne F1 pour le groupe 4
    df_results_g4 = df_results_all_users[df_results_all_users['user'].isin(users_g4)]
    gb_f1_list_g4 = df_results_g4.iloc[:, 5:6].values
    gb_f1_list_g4 = float("{0:.3f}".format(gb_f1_list_g4.mean()))
    # print('rf_f1_list_g4')
    # print(rf_f1_list_g4)
    # print()

    # Moyenne pour tous les groupes
    gb_f1_mean = ((gb_f1_list_g1 * len(users_g1)) + (gb_f1_list_g2 * len(users_g2)) + (
                gb_f1_list_g3 * len(users_g3)) + (gb_f1_list_g4 * len(users_g4))) / (
                             len(users_g1) + len(users_g2) + len(users_g3) + len(users_g4))
    gb_f1_mean = float("{0:.3f}".format(gb_f1_mean))
    # print('gb_f1_mean_g1')
    # print(gb_f1_mean)

    """ ================================================  XGB  ==============================================  """

    # Moyenne F1 pour le groupe 1
    df_results_all_users = pd.read_csv(path_to_results_all_users)
    df_results_g1 = df_results_all_users[df_results_all_users['user'].isin(users_g1)]
    xgb_f1_list_g1 = df_results_g1.iloc[:, 8:9].values
    xgb_f1_list_g1 = float("{0:.3f}".format(xgb_f1_list_g1.mean()))
    # print('gb_f1_list_g1')
    # print(gb_f1_list_g1)
    # print()

    # Moyenne F1 pour le groupe 2
    df_results_g2 = df_results_all_users[df_results_all_users['user'].isin(users_g2)]
    xgb_f1_list_g2 = df_results_g2.iloc[:, 8:9].values
    xgb_f1_list_g2 = float("{0:.3f}".format(xgb_f1_list_g2.mean()))
    # print('gb_f1_list_g2')
    # print(gb_f1_list_g2)
    # print()

    # Moyenne F1 pour le groupe 3
    df_results_g3 = df_results_all_users[df_results_all_users['user'].isin(users_g3)]
    xgb_f1_list_g3 = df_results_g3.iloc[:, 8:9].values
    xgb_f1_list_g3 = float("{0:.3f}".format(xgb_f1_list_g3.mean()))
    # print('rf_f1_list_g3')
    # print(rf_f1_list_g3)
    # print()

    # Moyenne F1 pour le groupe 4
    df_results_g4 = df_results_all_users[df_results_all_users['user'].isin(users_g4)]
    xgb_f1_list_g4 = df_results_g4.iloc[:, 8:9].values
    xgb_f1_list_g4 = float("{0:.3f}".format(xgb_f1_list_g4.mean()))
    # print('rf_f1_list_g4')
    # print(rf_f1_list_g4)
    # print()

    # Moyenne pour tous les groupes
    xgb_f1_mean = ((xgb_f1_list_g1 * len(users_g1)) + (xgb_f1_list_g2 * len(users_g2)) + (
                xgb_f1_list_g3 * len(users_g3)) + (xgb_f1_list_g4 * len(users_g4))) / (
                              len(users_g1) + len(users_g2) + len(users_g3) + len(users_g4))
    xgb_f1_mean = float("{0:.3f}".format(xgb_f1_mean))
    # print('gb_f1_mean_g1')
    # print(xgb_f1_mean)

    """ ================================================  LR  ==============================================  """

    # Moyenne F1 pour le groupe 1
    df_results_all_users = pd.read_csv(path_to_results_all_users)
    df_results_g1 = df_results_all_users[df_results_all_users['user'].isin(users_g1)]
    lr_f1_list_g1 = df_results_g1.iloc[:, 11:12].values
    lr_f1_list_g1 = float("{0:.3f}".format(lr_f1_list_g1.mean()))
    # print('gb_f1_list_g1')
    # print(gb_f1_list_g1)
    # print()

    # Moyenne F1 pour le groupe 2
    df_results_g2 = df_results_all_users[df_results_all_users['user'].isin(users_g2)]
    lr_f1_list_g2 = df_results_g2.iloc[:, 11:12].values
    lr_f1_list_g2 = float("{0:.3f}".format(lr_f1_list_g2.mean()))
    # print('gb_f1_list_g2')
    # print(gb_f1_list_g2)
    # print()

    # Moyenne F1 pour le groupe 3
    df_results_g3 = df_results_all_users[df_results_all_users['user'].isin(users_g3)]
    lr_f1_list_g3 = df_results_g3.iloc[:, 11:12].values
    lr_f1_list_g3 = float("{0:.3f}".format(lr_f1_list_g3.mean()))
    # print('rf_f1_list_g3')
    # print(rf_f1_list_g3)
    # print()

    # Moyenne F1 pour le groupe 4
    df_results_g4 = df_results_all_users[df_results_all_users['user'].isin(users_g4)]
    lr_f1_list_g4 = df_results_g4.iloc[:, 11:12].values
    lr_f1_list_g4 = float("{0:.3f}".format(lr_f1_list_g4.mean()))
    # print('rf_f1_list_g4')
    # print(rf_f1_list_g4)
    # print()

    # Moyenne tous les groupes
    lr_f1_mean = ((lr_f1_list_g1 * len(users_g1)) + (lr_f1_list_g2 * len(users_g2)) + (
                lr_f1_list_g3 * len(users_g3)) + (lr_f1_list_g4 * len(users_g4))) / (
                             len(users_g1) + len(users_g2) + len(users_g3) + len(users_g4))
    lr_f1_mean = float("{0:.3f}".format(lr_f1_mean))
    # print('lr_f1_mean_g1')
    # print(lr_f1_mean)

    """ ================================================  CATBOOST  ==============================================  """

    # Moyenne F1 pour le groupe 1
    df_results_all_users = pd.read_csv(path_to_results_all_users)
    df_results_g1 = df_results_all_users[df_results_all_users['user'].isin(users_g1)]
    cat_f1_list_g1 = df_results_g1.iloc[:, 14:15].values
    cat_f1_list_g1 = float("{0:.3f}".format(cat_f1_list_g1.mean()))
    # print('gb_f1_list_g1')
    # print(gb_f1_list_g1)
    # print()

    # Moyenne F1 pour le groupe 2
    df_results_g2 = df_results_all_users[df_results_all_users['user'].isin(users_g2)]
    cat_f1_list_g2 = df_results_g2.iloc[:, 14:15].values
    cat_f1_list_g2 = float("{0:.3f}".format(cat_f1_list_g2.mean()))
    # print('gb_f1_list_g2')
    # print(gb_f1_list_g2)
    # print()

    # Moyenne F1 pour le groupe 3
    df_results_g3 = df_results_all_users[df_results_all_users['user'].isin(users_g3)]
    cat_f1_list_g3 = df_results_g3.iloc[:, 14:15].values
    cat_f1_list_g3 = float("{0:.3f}".format(cat_f1_list_g3.mean()))
    # print('rf_f1_list_g3')
    # print(rf_f1_list_g3)
    # print()

    # Moyenne F1 pour le groupe 4
    df_results_g4 = df_results_all_users[df_results_all_users['user'].isin(users_g4)]
    cat_f1_list_g4 = df_results_g4.iloc[:, 14:15].values
    cat_f1_list_g4 = float("{0:.3f}".format(cat_f1_list_g4.mean()))
    # print('rf_f1_list_g4')
    # print(rf_f1_list_g4)
    # print()

    # Moyenne tous les groupes
    cat_f1_mean = ((cat_f1_list_g1 * len(users_g1)) + (cat_f1_list_g2 * len(users_g2)) + (
                cat_f1_list_g3 * len(users_g3)) + (cat_f1_list_g4 * len(users_g4))) / (
                              len(users_g1) + len(users_g2) + len(users_g3) + len(users_g4))
    cat_f1_mean = float("{0:.3f}".format(cat_f1_mean))
    # print('cat_f1_mean_g1')
    # print(cat_f1_mean)

    """ ================================================  DT  ==============================================  """

    # Moyenne F1 pour le groupe 1
    df_results_all_users = pd.read_csv(path_to_results_all_users)
    df_results_g1 = df_results_all_users[df_results_all_users['user'].isin(users_g1)]
    dt_f1_list_g1 = df_results_g1.iloc[:, 17:18].values
    dt_f1_list_g1 = float("{0:.3f}".format(dt_f1_list_g1.mean()))
    # print('gb_f1_list_g1')
    # print(gb_f1_list_g1)
    # print()

    # Moyenne F1 pour le groupe 2
    df_results_g2 = df_results_all_users[df_results_all_users['user'].isin(users_g2)]
    dt_f1_list_g2 = df_results_g2.iloc[:, 17:18].values
    dt_f1_list_g2 = float("{0:.3f}".format(dt_f1_list_g2.mean()))
    # print('gb_f1_list_g2')
    # print(gb_f1_list_g2)
    # print()

    # Moyenne F1 pour le groupe 3
    df_results_g3 = df_results_all_users[df_results_all_users['user'].isin(users_g3)]
    dt_f1_list_g3 = df_results_g3.iloc[:, 17:18].values
    dt_f1_list_g3 = float("{0:.3f}".format(dt_f1_list_g3.mean()))
    # print('rf_f1_list_g3')
    # print(rf_f1_list_g3)
    # print()

    # Moyenne F1 pour le groupe 4
    df_results_g4 = df_results_all_users[df_results_all_users['user'].isin(users_g4)]
    dt_f1_list_g4 = df_results_g4.iloc[:, 17:18].values
    dt_f1_list_g4 = float("{0:.3f}".format(dt_f1_list_g4.mean()))
    # print('rf_f1_list_g4')
    # print(rf_f1_list_g4)
    # print()

    # Moyenne tous les groupes
    dt_f1_mean = ((dt_f1_list_g1 * len(users_g1)) + (dt_f1_list_g2 * len(users_g2)) + (
                dt_f1_list_g3 * len(users_g3)) + (dt_f1_list_g4 * len(users_g4))) / (
                             len(users_g1) + len(users_g2) + len(users_g3) + len(users_g4))
    dt_f1_mean = float("{0:.3f}".format(dt_f1_mean))
    # print('dt_f1_mean_g1')
    # print(dt_f1_mean)

    """ ================================================  NB  ==============================================  """

    # Moyenne F1 pour le groupe 1
    df_results_all_users = pd.read_csv(path_to_results_all_users)
    df_results_g1 = df_results_all_users[df_results_all_users['user'].isin(users_g1)]
    nb_f1_list_g1 = df_results_g1.iloc[:, 20:21].values
    nb_f1_list_g1 = float("{0:.3f}".format(nb_f1_list_g1.mean()))
    # print('gb_f1_list_g1')
    # print(gb_f1_list_g1)
    # print()

    # Moyenne F1 pour le groupe 2
    df_results_g2 = df_results_all_users[df_results_all_users['user'].isin(users_g2)]
    nb_f1_list_g2 = df_results_g2.iloc[:, 20:21].values
    nb_f1_list_g2 = float("{0:.3f}".format(nb_f1_list_g2.mean()))
    # print('gb_f1_list_g2')
    # print(gb_f1_list_g2)
    # print()

    # Moyenne F1 pour le groupe 3
    df_results_g3 = df_results_all_users[df_results_all_users['user'].isin(users_g3)]
    nb_f1_list_g3 = df_results_g3.iloc[:, 20:21].values
    nb_f1_list_g3 = float("{0:.3f}".format(nb_f1_list_g3.mean()))
    # print('rf_f1_list_g3')
    # print(rf_f1_list_g3)
    # print()

    # Moyenne F1 pour le groupe 4
    df_results_g4 = df_results_all_users[df_results_all_users['user'].isin(users_g4)]
    nb_f1_list_g4 = df_results_g4.iloc[:, 20:21].values
    nb_f1_list_g4 = float("{0:.3f}".format(nb_f1_list_g4.mean()))
    # print('rf_f1_list_g4')
    # print(rf_f1_list_g4)
    # print()

    # Moyenne tous les groupes
    nb_f1_mean = ((nb_f1_list_g1 * len(users_g1)) + (nb_f1_list_g2 * len(users_g2)) + (
                nb_f1_list_g3 * len(users_g3)) + (nb_f1_list_g4 * len(users_g4))) / (
                             len(users_g1) + len(users_g2) + len(users_g3) + len(users_g4))
    nb_f1_mean = float("{0:.3f}".format(nb_f1_mean))
    # print('nb_f1_mean_g1')
    # print(nb_f1_mean)

    """ ================================================  SVM  ==============================================  """

    # Moyenne F1 pour le groupe 1
    df_results_all_users = pd.read_csv(path_to_results_all_users)
    df_results_g1 = df_results_all_users[df_results_all_users['user'].isin(users_g1)]
    svm_f1_list_g1 = df_results_g1.iloc[:, 23:24].values
    svm_f1_list_g1 = float("{0:.3f}".format(svm_f1_list_g1.mean()))
    # print('gb_f1_list_g1')
    # print(gb_f1_list_g1)
    # print()

    # Moyenne F1 pour le groupe 2
    df_results_g2 = df_results_all_users[df_results_all_users['user'].isin(users_g2)]
    svm_f1_list_g2 = df_results_g2.iloc[:, 23:24].values
    svm_f1_list_g2 = float("{0:.3f}".format(svm_f1_list_g2.mean()))
    # print('gb_f1_list_g2')
    # print(gb_f1_list_g2)
    # print()

    # Moyenne F1 pour le groupe 3
    df_results_g3 = df_results_all_users[df_results_all_users['user'].isin(users_g3)]
    svm_f1_list_g3 = df_results_g3.iloc[:, 23:24].values
    svm_f1_list_g3 = float("{0:.3f}".format(svm_f1_list_g3.mean()))
    # print('rf_f1_list_g3')
    # print(rf_f1_list_g3)
    # print()

    # Moyenne F1 pour le groupe 4
    df_results_g4 = df_results_all_users[df_results_all_users['user'].isin(users_g4)]
    svm_f1_list_g4 = df_results_g4.iloc[:, 23:24].values
    svm_f1_list_g4 = float("{0:.3f}".format(svm_f1_list_g4.mean()))
    # print('rf_f1_list_g4')
    # print(rf_f1_list_g4)
    # print()

    # Moyenne tous les groupes
    svm_f1_mean = ((svm_f1_list_g1 * len(users_g1)) + (svm_f1_list_g2 * len(users_g2)) + (
                svm_f1_list_g3 * len(users_g3)) + (svm_f1_list_g4 * len(users_g4))) / (
                              len(users_g1) + len(users_g2) + len(users_g3) + len(users_g4))
    svm_f1_mean = float("{0:.3f}".format(svm_f1_mean))
    # print('svm_f1_mean')
    # print(svm_f1_mean)

    """ ================================================  MLP  ==============================================  """

    # Moyenne F1 pour le groupe 1
    df_results_all_users = pd.read_csv(path_to_results_all_users)
    df_results_g1 = df_results_all_users[df_results_all_users['user'].isin(users_g1)]
    mlp_f1_list_g1 = df_results_g1.iloc[:, 26:27].values
    mlp_f1_list_g1 = float("{0:.3f}".format(mlp_f1_list_g1.mean()))
    # print('gb_f1_list_g1')
    # print(gb_f1_list_g1)
    # print()

    # Moyenne F1 pour le groupe 2
    df_results_g2 = df_results_all_users[df_results_all_users['user'].isin(users_g2)]
    mlp_f1_list_g2 = df_results_g2.iloc[:, 26:27].values
    mlp_f1_list_g2 = float("{0:.3f}".format(mlp_f1_list_g2.mean()))
    # print('gb_f1_list_g2')
    # print(gb_f1_list_g2)
    # print()

    # Moyenne F1 pour le groupe 3
    df_results_g3 = df_results_all_users[df_results_all_users['user'].isin(users_g3)]
    mlp_f1_list_g3 = df_results_g3.iloc[:, 26:27].values
    mlp_f1_list_g3 = float("{0:.3f}".format(mlp_f1_list_g3.mean()))
    # print('rf_f1_list_g3')
    # print(rf_f1_list_g3)
    # print()

    # Moyenne F1 pour le groupe 4
    df_results_g4 = df_results_all_users[df_results_all_users['user'].isin(users_g4)]
    mlp_f1_list_g4 = df_results_g4.iloc[:, 26:27].values
    mlp_f1_list_g4 = float("{0:.3f}".format(mlp_f1_list_g4.mean()))
    # print('rf_f1_list_g4')
    # print(rf_f1_list_g4)
    # print()

    # Moyenne tous les groupes
    mlp_f1_mean = ((mlp_f1_list_g1 * len(users_g1)) + (mlp_f1_list_g2 * len(users_g2)) + (
            mlp_f1_list_g3 * len(users_g3)) + (mlp_f1_list_g4 * len(users_g4))) / (
                          len(users_g1) + len(users_g2) + len(users_g3) + len(users_g4))
    mlp_f1_mean = float("{0:.3f}".format(mlp_f1_mean))
    # print('mlp_f1_mean')
    # print(mlp_f1_mean)

    """ ===================================================================================================  """
    """ ================================================  G1  ==============================================  """
    dict_cluster_metrics['cluster'].append('g1')

    dict_cluster_metrics['RF_F1_mean'].append(rf_f1_list_g1)
    dict_cluster_metrics['GB_F1_mean'].append(gb_f1_list_g1)
    dict_cluster_metrics['XGB_F1_mean'].append(xgb_f1_list_g1)
    dict_cluster_metrics['LR_F1_mean'].append(lr_f1_list_g1)
    dict_cluster_metrics['Catboost_F1_mean'].append(cat_f1_list_g1)
    dict_cluster_metrics['DT_F1_mean'].append(dt_f1_list_g1)
    dict_cluster_metrics['NB_F1_mean'].append(nb_f1_list_g1)
    dict_cluster_metrics['SV_F1_mean'].append(svm_f1_list_g1)
    dict_cluster_metrics['MLP_F1_mean'].append(mlp_f1_list_g1)

    """ ================================================  G2  ==============================================  """

    dict_cluster_metrics['cluster'].append('g2')

    dict_cluster_metrics['RF_F1_mean'].append(rf_f1_list_g2)
    dict_cluster_metrics['GB_F1_mean'].append(gb_f1_list_g2)
    dict_cluster_metrics['XGB_F1_mean'].append(xgb_f1_list_g2)
    dict_cluster_metrics['LR_F1_mean'].append(lr_f1_list_g2)
    dict_cluster_metrics['Catboost_F1_mean'].append(cat_f1_list_g2)
    dict_cluster_metrics['DT_F1_mean'].append(dt_f1_list_g2)
    dict_cluster_metrics['NB_F1_mean'].append(nb_f1_list_g2)
    dict_cluster_metrics['SV_F1_mean'].append(svm_f1_list_g2)
    dict_cluster_metrics['MLP_F1_mean'].append(mlp_f1_list_g2)

    """ ================================================  G3  ==============================================  """

    dict_cluster_metrics['cluster'].append('g3')

    dict_cluster_metrics['RF_F1_mean'].append(rf_f1_list_g3)
    dict_cluster_metrics['GB_F1_mean'].append(gb_f1_list_g3)
    dict_cluster_metrics['XGB_F1_mean'].append(xgb_f1_list_g3)
    dict_cluster_metrics['LR_F1_mean'].append(lr_f1_list_g3)
    dict_cluster_metrics['Catboost_F1_mean'].append(cat_f1_list_g3)
    dict_cluster_metrics['DT_F1_mean'].append(dt_f1_list_g3)
    dict_cluster_metrics['NB_F1_mean'].append(nb_f1_list_g3)
    dict_cluster_metrics['SV_F1_mean'].append(svm_f1_list_g3)
    dict_cluster_metrics['MLP_F1_mean'].append(mlp_f1_list_g3)

    """ ================================================  G4  ==============================================  """

    dict_cluster_metrics['cluster'].append('g4')

    dict_cluster_metrics['RF_F1_mean'].append(rf_f1_list_g4)
    dict_cluster_metrics['GB_F1_mean'].append(gb_f1_list_g4)
    dict_cluster_metrics['XGB_F1_mean'].append(xgb_f1_list_g4)
    dict_cluster_metrics['LR_F1_mean'].append(lr_f1_list_g4)
    dict_cluster_metrics['Catboost_F1_mean'].append(cat_f1_list_g4)
    dict_cluster_metrics['DT_F1_mean'].append(dt_f1_list_g4)
    dict_cluster_metrics['NB_F1_mean'].append(nb_f1_list_g4)
    dict_cluster_metrics['SV_F1_mean'].append(svm_f1_list_g4)
    dict_cluster_metrics['MLP_F1_mean'].append(mlp_f1_list_g4)

    """ ================================================  Global  ==============================================  """

    dict_cluster_metrics['cluster'].append('Global')

    dict_cluster_metrics['RF_F1_mean'].append(rf_f1_mean)
    dict_cluster_metrics['GB_F1_mean'].append(gb_f1_mean)
    dict_cluster_metrics['XGB_F1_mean'].append(xgb_f1_mean)
    dict_cluster_metrics['LR_F1_mean'].append(lr_f1_mean)
    dict_cluster_metrics['Catboost_F1_mean'].append(cat_f1_mean)
    dict_cluster_metrics['DT_F1_mean'].append(dt_f1_mean)
    dict_cluster_metrics['NB_F1_mean'].append(nb_f1_mean)
    dict_cluster_metrics['SV_F1_mean'].append(svm_f1_mean)
    dict_cluster_metrics['MLP_F1_mean'].append(mlp_f1_mean)

    df_cluster_metrics = pd.DataFrame(dict_cluster_metrics)
    print("The global metrics means are :")
    print(df_cluster_metrics)

    if not os.path.exists('../results/ml_results/results_by_cluster'):
        os.mkdir('../results/ml_results/results_by_cluster')
    df_cluster_metrics.to_csv(r'../results/ml_results/results_by_cluster/cluster_results.csv', index=False)


if __name__ == "__main__":
    get_cluster_results("../results/ml_results/results_by_user/results.csv")
