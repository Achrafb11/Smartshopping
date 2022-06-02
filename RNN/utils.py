import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import FeatureHasher
import math
import torch

def load_data_from_folder(directory_ldff, filename):

    dataset_train = pd.read_csv(directory_ldff + '/' + filename + '/' + 'train.csv')

    panier_99999999 = pd.read_csv(directory_ldff + '/' + filename + '/' + 'order_99999999.csv')

    y_train = dataset_train.values
    y_train = y_train[:, -1]
    y_train = y_train.astype('int')

    dataset_test = pd.read_csv(directory_ldff + '/' + filename + '/' + 'test.csv')
    dataset_test_to_add = dataset_test.loc[dataset_test['store_id'] == 0].reset_index(drop=True)

    n_products = min(len(dataset_test_to_add), 150)
    dataset_test_to_add = dataset_test_to_add.sample(n=n_products)
    dataset_test = dataset_test.loc[dataset_test['store_id'] != 0].reset_index(drop=True)

    dataset_test = dataset_test.append(panier_99999999).reset_index(drop=True)
    dataset_test = dataset_test.append(dataset_test_to_add).reset_index(drop=True)
    # Hashage de colonne product_id
    product_id_hashed_test = dataset_test["product_id"]
    product_id_hashed_train = dataset_train["product_id"]

    len_product_id_train = product_id_hashed_train.append(product_id_hashed_test)
    len_product_id_train = len_product_id_train.nunique()
    n_feature_train = math.ceil((math.log(len_product_id_train)/math.log(2)))

    if math.pow(2, n_feature_train) < len_product_id_train:
        n_feature_train += 1

    h_train = FeatureHasher(n_features=n_feature_train, input_type='string')
    f_train = h_train.transform(product_id_hashed_train.astype('str'))
    hashed_categories_train = pd.DataFrame(f_train.toarray())

    product_id_hashed_train = pd.concat([product_id_hashed_train.astype('str'), hashed_categories_train], axis=1)
    product_id_hashed_train = product_id_hashed_train.iloc[:, 1:]
    order_ids_train = dataset_train["order_id"].values
    product_ids_train = dataset_train['product_id'].values
    dataset_train = dataset_train.iloc[:, 5:-1]
    dataset_train = pd.concat([product_id_hashed_train.astype('str'), dataset_train], axis=1)
    dataset_train = dataset_train.values

    X_train = dataset_train

    """Test part starts here"""

    dataset_test = pd.read_csv(directory_ldff + '/' + filename + '/' + 'test.csv')
    dataset_test_to_add = dataset_test.loc[dataset_test['store_id'] == 0].reset_index(drop=True)

    n_products = min(len(dataset_test_to_add), 150)
    dataset_test_to_add = dataset_test_to_add.sample(n=n_products)
    dataset_test = dataset_test.loc[dataset_test['store_id'] != 0].reset_index(drop=True)

    dataset_test = dataset_test.append(panier_99999999).reset_index(drop=True)

    dataset_test = dataset_test.append(dataset_test_to_add).reset_index(drop=True)

    y_test = dataset_test.values
    y_test = y_test[:, -1]
    y_test = y_test.astype('int')

    product_id_hashed_test = dataset_test["product_id"]
    order_ids_test = dataset_test["order_id"].values
    product_ids_test = dataset_test['product_id'].values

    h_test = FeatureHasher(n_features=n_feature_train, input_type='string')
    f_test = h_test.transform(product_id_hashed_test.astype('str'))
    hashed_categories_test = pd.DataFrame(f_test.toarray())
    product_id_hashed_test = pd.concat([product_id_hashed_test.astype('str'), hashed_categories_test], axis=1)
    dataset_test = dataset_test.iloc[:, 5:]
    dataset_test2 = pd.concat([product_id_hashed_test.astype('str'), dataset_test], axis=1)

    df_X_test = pd.read_csv(directory_ldff + '/' + filename + '/' + 'test.csv')

    df_X_test_to_add = df_X_test.loc[df_X_test['store_id'] == 0].reset_index(drop=True)
    n_products = min(len(df_X_test_to_add), 150)
    df_X_test_to_add = df_X_test_to_add.sample(n=n_products)
    df_X_test = df_X_test.loc[df_X_test['store_id'] != 0].reset_index(drop=True)

    df_X_test = df_X_test.append(panier_99999999).reset_index(drop=True)

    df_X_test = df_X_test.append(df_X_test_to_add).reset_index(drop=True)

    df_X_test = pd.concat([product_id_hashed_test.astype('str'), df_X_test], axis=1)
    df_X_test = df_X_test.iloc[:, :-1]
    dataset_test = pd.concat([product_id_hashed_test.astype('str'), dataset_test], axis=1)
    dataset_test = dataset_test.values

    X_test = dataset_test

    dataset_test_diminue = pd.read_csv(directory_ldff + '/' + filename + '/' + 'test.csv')

    dataset_test_diminue_to_add = dataset_test_diminue.loc[dataset_test_diminue['store_id'] == 0].reset_index(drop=True)

    n_products = min(len(dataset_test_diminue_to_add), 150)
    dataset_test_diminue_to_add = dataset_test_diminue_to_add.sample(n=n_products)
    dataset_test_diminue = dataset_test_diminue.loc[dataset_test_diminue['store_id'] != 0].reset_index(drop=True)

    dataset_test_diminue = dataset_test_diminue.append(panier_99999999).reset_index(drop=True)

    dataset_test_diminue = dataset_test_diminue.append(dataset_test_diminue_to_add).reset_index(drop=True)

    dataset_test_diminue = dataset_test_diminue.values

    df_for_panier_count = pd.read_csv(directory_ldff + '/' + filename + '/' + 'train.csv')
    number_of_rows_in_train_file = df_for_panier_count.loc[df_for_panier_count['store_id'] != 0]
    number_of_rows_in_train_file = number_of_rows_in_train_file.shape[0]

    panier_count = df_for_panier_count["order_id"].nunique()

    moyenne_taille_paniers = number_of_rows_in_train_file // panier_count

    y_test2 = dataset_test_diminue[:, -1]
    y_test2 = y_test2.astype('int')

    dataset_test_diminue = dataset_test_diminue[:, 5:-1]

    return X_train, dataset_test2, y_train, y_test, moyenne_taille_paniers, df_X_test, dataset_test_diminue, y_test2, order_ids_train, order_ids_test, product_ids_train, product_ids_test

def pool_max(tensor, dim):
    return torch.max(tensor, dim)[0]

def pool_avg(tensor, dim):
    return torch.mean(tensor, dim)
