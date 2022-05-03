from matplotlib import pyplot
import pandas as pd
from matplotlib.pyplot import figure

def generate_boxplot():
    path_to_results_to_plot = "../results/ml_results/users_results_to_plot.csv"

    df_results_to_plot = pd.read_csv(path_to_results_to_plot)
    df_results_to_plot = df_results_to_plot.reset_index(drop=True)

    number_of_orders = df_results_to_plot[["number_of_orders", "RF_F1"]]

    avg_basket_size = df_results_to_plot[["avg_basket_size", "RF_F1"]]

    """
    ============================================ Setting values for number of baskets
    """

    number_of_orders_itv_3_10 = number_of_orders[number_of_orders["number_of_orders"] < 10].reset_index(drop=True)
    number_of_orders_itv_3_10_nb_orders = list(number_of_orders_itv_3_10.iloc[:, 0])
    number_of_orders_itv_3_10_f1 = list(number_of_orders_itv_3_10.iloc[:, 1])

    number_of_orders_itv_11_25 = number_of_orders[number_of_orders["number_of_orders"] >= 10].reset_index(drop=True)
    number_of_orders_itv_11_25 = number_of_orders_itv_11_25[number_of_orders_itv_11_25["number_of_orders"] < 25].reset_index(drop=True)
    number_of_orders_itv_11_25_nb_orders = list(number_of_orders_itv_11_25.iloc[:, 0])
    number_of_orders_itv_11_25_f1 = list(number_of_orders_itv_11_25.iloc[:, 1])

    number_of_orders_itv_26_40 = number_of_orders[number_of_orders["number_of_orders"] >= 26].reset_index(drop=True)
    number_of_orders_itv_26_40 = number_of_orders_itv_26_40[number_of_orders_itv_26_40["number_of_orders"] < 40].reset_index(drop=True)
    number_of_orders_itv_26_40_nb_orders = list(number_of_orders_itv_26_40.iloc[:, 0])
    number_of_orders_itv_26_40_f1 = list(number_of_orders_itv_26_40.iloc[:, 1])

    number_of_orders_itv_41_75 = number_of_orders[number_of_orders["number_of_orders"] >= 40].reset_index(drop=True)
    number_of_orders_itv_41_75 = number_of_orders_itv_41_75[number_of_orders_itv_41_75["number_of_orders"] < 75].reset_index(drop=True)
    number_of_orders_itv_41_75_nb_orders = list(number_of_orders_itv_26_40.iloc[:, 0])
    number_of_orders_itv_41_75_f1 = list(number_of_orders_itv_26_40.iloc[:, 1])

    number_of_orders_itv_76_100 = number_of_orders[number_of_orders["number_of_orders"] > 76].reset_index(drop=True)
    number_of_orders_itv_76_100_nb_orders = list(number_of_orders_itv_76_100.iloc[:, 0])
    number_of_orders_itv_76_100_f1 = list(number_of_orders_itv_76_100.iloc[:, 1])

    data2_nb_order = [number_of_orders_itv_3_10_nb_orders, number_of_orders_itv_11_25_nb_orders, number_of_orders_itv_26_40_nb_orders,
                    number_of_orders_itv_41_75_nb_orders, number_of_orders_itv_76_100_nb_orders]

    data1_nb_order = [number_of_orders_itv_3_10_f1, number_of_orders_itv_11_25_f1, number_of_orders_itv_26_40_f1,
             number_of_orders_itv_41_75_f1, number_of_orders_itv_76_100_f1]

    """
    ============================================ Same for average basket size
    """

    avg_basket_size_itv_1_5 = avg_basket_size[avg_basket_size["avg_basket_size"] <= 5].reset_index(drop=True)
    avg_basket_size_itv_1_5_avg_b_s = list(avg_basket_size_itv_1_5.iloc[:, 0])
    avg_basket_size_itv_1_5_f1 = list(number_of_orders_itv_3_10.iloc[:, 1])

    avg_basket_size_itv_6_10 = avg_basket_size[avg_basket_size["avg_basket_size"] > 5].reset_index(drop=True)
    avg_basket_size_itv_6_10 = avg_basket_size_itv_6_10[avg_basket_size_itv_6_10["avg_basket_size"] <= 10].reset_index(drop=True)
    navg_basket_size_itv_6_10_avg_b_s = list(avg_basket_size_itv_6_10.iloc[:, 0])
    avg_basket_size_itv_6_10_f1 = list(avg_basket_size_itv_6_10.iloc[:, 1])

    avg_basket_size_itv_11_15 = avg_basket_size[avg_basket_size["avg_basket_size"] > 10].reset_index(drop=True)
    avg_basket_size_itv_11_15 = avg_basket_size_itv_11_15[avg_basket_size_itv_11_15["avg_basket_size"] <= 15].reset_index(drop=True)
    navg_basket_size_itv_11_15_avg_b_s = list(avg_basket_size_itv_11_15.iloc[:, 0])
    avg_basket_size_itv_11_15_f1 = list(avg_basket_size_itv_11_15.iloc[:, 1])

    avg_basket_size_itv_16_20 = avg_basket_size[avg_basket_size["avg_basket_size"] > 15].reset_index(drop=True)
    avg_basket_size_itv_16_20 = avg_basket_size_itv_16_20[avg_basket_size_itv_16_20["avg_basket_size"] <= 20].reset_index(drop=True)
    navg_basket_size_itv_16_20_avg_b_s = list(avg_basket_size_itv_16_20.iloc[:, 0])
    avg_basket_size_itv_16_20_f1 = list(avg_basket_size_itv_16_20.iloc[:, 1])

    avg_basket_size_itv_21_31 = avg_basket_size[avg_basket_size["avg_basket_size"] > 21].reset_index(drop=True)
    navg_basket_size_itv_21_31_avg_b_s = list(avg_basket_size_itv_21_31.iloc[:, 0])
    avg_basket_size_itv_21_31_f1 = list(avg_basket_size_itv_21_31.iloc[:, 1])

    data2_avg_basket_size = [avg_basket_size_itv_1_5_avg_b_s, navg_basket_size_itv_6_10_avg_b_s, navg_basket_size_itv_11_15_avg_b_s,
                    navg_basket_size_itv_16_20_avg_b_s, navg_basket_size_itv_21_31_avg_b_s]

    data1_avg_basket_size = [avg_basket_size_itv_1_5_f1, avg_basket_size_itv_6_10_f1, avg_basket_size_itv_11_15_f1,
             avg_basket_size_itv_16_20_f1, avg_basket_size_itv_21_31_f1]


    """
    ============================================ Plot
    """

    medianpropsdict = dict(linewidth=3.5, color='firebrick')
    whiskerpropsdict = dict(linewidth=2, color='black')

    pyplot.rcParams.update({'font.size': 26})
    figure(figsize=(16, 10), dpi=100)

    pyplot.boxplot(x=data1_nb_order, medianprops=medianpropsdict, notch=False, widths=(0.75, 0.75, 0.75, 0.75, 0.75),
                   whiskerprops=whiskerpropsdict, boxprops=whiskerpropsdict, capprops=whiskerpropsdict)
    pyplot.title('(a) F-score distribution vs number of baskets')
    pyplot.xlabel("Number of baskets")
    pyplot.ylabel("F-score")
    pyplot.xticks([1, 2, 3, 4, 5], ['3 to 10', '11 to 25', '26 to 40', '41 to 75', '76 to 100'])
    pyplot.savefig('../results/Boxplot_Number_of_baskets.png', bbox_inches='tight')
    pyplot.show()

    pyplot.rcParams.update({'font.size': 26})
    figure(figsize=(16, 10), dpi=100)

    pyplot.boxplot(x=data1_avg_basket_size, medianprops=medianpropsdict, notch=False, widths=(0.75, 0.75, 0.75, 0.75, 0.75),
                   whiskerprops=whiskerpropsdict, boxprops=whiskerpropsdict, capprops=whiskerpropsdict)
    pyplot.title('(b) F-score distribution vs average basket size')
    pyplot.xlabel("Average basket size (number of products)")
    pyplot.ylabel("F-score")
    pyplot.xticks([1, 2, 3, 4, 5], ['1 to 5', '6 to 10', '11 to 15', '16 to 20', '21 to 31'])
    pyplot.savefig('../results/Boxplot_Average_basket_size.png', bbox_inches='tight')
    pyplot.show()



if __name__ == "__main__":
    generate_boxplot()
