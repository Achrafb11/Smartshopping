import gc
import time

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer


def calculate_max_group(path):
    print("Calculating max group")
    all_prods = np.zeros((1, 5))
    for folder in os.listdir(path):
        filepath = (path + str(folder) + '/')
        for file in os.listdir(filepath):
            print(file)
            for train_file in os.listdir(filepath + file):
                if train_file == "train.csv":
                    # print(train_file)
                    df = pd.read_csv(filepath + file + "/" + train_file)
                    df = df.loc[df['store_id'] != 0]
                    df = df.iloc[:, 0:]
                    df = df.drop(['user_id', 'order_id', 'order_number', 'name', 'reordered', 'category', 'price',
                                  'special', 'brand', 'distance_moyenne', 'store_avlb_1',
                                  'store_avlb_2',
                                  'store_avlb_3',
                                  'store_avlb_4', 'store_avlb_5', 'store_avlb_6', 'store_avlb_7', 'store_avlb_8',
                                  'store_avlb_9',
                                  'store_avlb_10', 'store_id', 'group'], axis=1)

                    if folder == 'g1':
                        df['g1'], df['g2'], df['g3'], df['g4'] = 1, 0, 0, 0
                    elif folder == 'g2':
                        df['g1'], df['g2'], df['g3'], df['g4'] = 0, 1, 0, 0
                    elif folder == 'g3':
                        df['g1'], df['g2'], df['g3'], df['g4'] = 0, 0, 1, 0
                    elif folder == 'g4':
                        df['g1'], df['g2'], df['g3'], df['g4'] = 0, 0, 0, 1

                    all_prods = np.concatenate((all_prods, df.values), axis=0)

    all_prods = pd.DataFrame(all_prods).astype(int)
    grouped = all_prods.groupby([0]).sum().reset_index()
    del all_prods
    gc.collect()

    grouped.rename(
        columns={0: 'product_id', 1: 'g1', 2: 'g2', 3: 'g3', 4: 'g4'},
        inplace=True)
    grouped['max_group'] = grouped.iloc[:, 1:].idxmax(axis=1)
    grouped['total_bought'] = grouped.iloc[:, 1:].sum(axis=1)

    lb = LabelBinarizer()
    binarized = lb.fit_transform(grouped['max_group'])
    binarized = pd.DataFrame(binarized, columns=['mg1', 'mg2', 'mg3', 'mg4'])
    df_mg = pd.concat([grouped, binarized], axis=1)
    del grouped
    gc.collect()
    df_mg.drop('max_group', inplace=True, axis=1)

    df_mg.to_csv("../feature_engineering/features_to_merge.csv", index=False)
    del df_mg
    gc.collect()


def add_features(path):
    start_time = time.time()
    print("Adding features :")
    print("total_bought for users")

    all_prods = pd.read_csv("../feature_engineering/features_to_merge.csv")
    for folder in os.listdir(path):
        print(folder)
        filepath_all_products = (path + str(folder) + '/')
        for file in os.listdir(filepath_all_products):
            # print(file)
            df_train = pd.read_csv(filepath_all_products + file + "/train.csv")
            # print(df_train)
            df_train = df_train.iloc[:, :]
            df_test = pd.read_csv(filepath_all_products + file + "/test.csv")

            df_train = pd.merge(df_train, all_prods, on=["product_id"], how="left")
            df_train.dropna(axis=0, how='any', inplace=True)

            df_test = pd.merge(df_test, all_prods, on=["product_id"], how="left")
            df_test.dropna(axis=0, how='any', inplace=True)

            cols_to_normalize_train = ['category', 'price', 'special', 'distance_moyenne', 'total_bought']
            cols_to_normalize_test = ['category', 'price', 'special', 'distance_moyenne', 'total_bought']

            df_train[cols_to_normalize_train] = (df_train[cols_to_normalize_train] - df_train[cols_to_normalize_train].mean()) / df_train[
                cols_to_normalize_train].std(ddof=0)
            df_test[cols_to_normalize_test] = (df_test[cols_to_normalize_test] - df_test[cols_to_normalize_test].mean()) / df_test[
                cols_to_normalize_test].std(ddof=0)

            df_train = df_train[
                ['user_id', 'order_id', 'order_number', 'product_id', 'name', 'reordered', 'category',  # 'cat_1', 'cat_2',
                 'price', 'special', 'brand', 'total_bought',
                 'distance_moyenne', 'store_avlb_1', 'store_avlb_2', 'store_avlb_3',
                 'store_avlb_4', 'store_avlb_5', 'store_avlb_6', 'store_avlb_7', 'store_avlb_8', 'store_avlb_9',
                 'store_avlb_10', 'store_id']].copy()

            df_test = df_test[
                ['user_id', 'order_id', 'order_number', 'product_id', 'name', 'reordered', 'category',  # 'cat_1', 'cat_2',
                 'price', 'special', 'brand', 'total_bought',
                 'distance_moyenne', 'store_avlb_1', 'store_avlb_2', 'store_avlb_3',
                 'store_avlb_4', 'store_avlb_5', 'store_avlb_6', 'store_avlb_7', 'store_avlb_8', 'store_avlb_9',
                 'store_avlb_10', 'store_id']].copy()

            df_train.to_csv(filepath_all_products + file + "/" + "train.csv", index=False)
            df_test.to_csv(filepath_all_products + file + "/" + "test.csv", index=False)
    del df_train
    del df_test
    gc.collect()
    print("Features added successfully.")

    for folder in os.listdir(path):
        filepath_user_n = (path + str(folder) + '/')
        for file in os.listdir(filepath_user_n):
            print(file)
            df_train = pd.read_csv(filepath_user_n + file + '/' + "train.csv")
            panier_custom_to_save = df_train.loc[df_train['order_number'] == 99999999].copy()
            panier_custom_to_save.to_csv(filepath_user_n + file + '/' + "order_99999999.csv", index=False)

    print("--- It took " + str((time.time() - start_time)) + " seconds to add features to users ---")
    return


def start_new_features():
    path = "../postprocess_all_products/"
    calculate_max_group(path)
    add_features(path)


if __name__ == "__main__":
    start_new_features()
