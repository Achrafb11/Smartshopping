import pandas as pd
import os
import matplotlib.pyplot as plt


def new_results_file(path_to_users_groups, path_to_results):
    df_results_all_users = pd.read_csv(path_to_results)

    user_id = []
    number_of_orders = []
    moyenne_taille_panier = []
    for folder in os.listdir(path_to_users_groups):
        filepath = (path_to_users_groups + folder + '/')
        for file in os.listdir(filepath):
            for train_file in os.listdir(filepath + file):
                if train_file == "train.csv":
                    df_for_panier_count = pd.read_csv(filepath + file + "/" + train_file)
                    number_of_rows_in_train_file = df_for_panier_count.loc[df_for_panier_count['store_id'] != 0]
                    number_of_rows_in_train_file = number_of_rows_in_train_file.shape[0]
                    panier_count = df_for_panier_count["order_id"].nunique()
                    panier_count += 1
                    moyenne_taille_paniers = number_of_rows_in_train_file // panier_count

                    user_id.append(file)
                    number_of_orders.append(panier_count)
                    moyenne_taille_panier.append(moyenne_taille_paniers)

    dataframe_temp = pd.DataFrame(
        {'user_id': user_id,
         'number_of_orders': number_of_orders,
         'avg_basket_size': moyenne_taille_panier
         })

    df_rf_results = pd.DataFrame(columns=["user_id", "RF_F1"])

    df_rf_results["user_id"] = df_results_all_users.user
    df_rf_results["RF_F1"] = df_results_all_users.RF_F1
    print(df_rf_results)

    df_final = pd.merge(dataframe_temp, df_rf_results, how="outer", on=["user_id"])

    print(df_final)

    savepath_all_users_results = '../results/ml_results/'
    if not os.path.exists(savepath_all_users_results):
        os.mkdir(savepath_all_users_results)
    df_final.to_csv(
        r'../results/ml_results/users_results_to_plot.csv',
        index=False)


if __name__ == "__main__":
    path_to_postprocess_all_products = "../postprocess_all_products/"

    path_to_results_all_users = "../results/ml_results/results_by_user/results.csv"

    new_results_file(path_to_postprocess_all_products, path_to_results_all_users)
