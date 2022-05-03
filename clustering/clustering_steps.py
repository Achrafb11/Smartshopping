import pandas as pd
import numpy as np
import os
from shutil import copytree
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import figure
plt.rcParams.update(plt.rcParamsDefault)

pd.options.mode.chained_assignment = None  # default='warn'
plt.rcParams.update({'font.size': 11})
figure(figsize=(12, 10), dpi=100)


def load_data(usr_path):
    train = pd.read_csv(usr_path + 'train.csv')
    test = pd.read_csv(usr_path + 'test.csv')
    df = train.append(test)
    df = df.loc[df['store_id'] != 0]
    return df.reset_index(drop=True)


def calculate_visit_fidelity_ratio(df, n_stores):
    stores = df.store_id.to_list()
    occurences = [stores.count(x) for x in set(stores)]
    xmax = max(occurences)
    xtotal = (len(df))
    sumx = (len(df) - xmax)
    if n_stores != 1:
        fidelity_ratio = ((xmax - (1 / (n_stores - 1)) * sumx) / xtotal)
    elif n_stores == 1:
        fidelity_ratio = xmax / xtotal
    return fidelity_ratio


def calculate_price_fidelity_ratio(df, n_stores):
    store_prices = df.groupby('store_id').sum()['price'].to_list()
    pmax = max(store_prices)
    ptotal = (sum(store_prices))
    sump = ptotal - pmax
    if n_stores != 1:
        fidelity_ratio = ((pmax - (1 / (n_stores - 1)) * sump) / ptotal)
    elif n_stores == 1:
        fidelity_ratio = pmax / ptotal
    return fidelity_ratio


def vectorize_category(df):
    category_vector = 24 * [0]
    category_count = pd.value_counts(df.category)
    for idx in category_count.index:
        category_vector[int(idx) - 1] = category_count[idx]
    category_vector.insert(0, df.user_id.unique()[0])
    return category_vector


def make_cluster_data(path, cluster_data, category_matrix):
    print("make_cluster_data")
    for user in os.listdir(path):
        usr_path = path + '/' + user + '/'
        df = load_data(usr_path)
        user_id = df.user_id.unique()[0]
        x_special = np.mean(df.special)
        x_price = np.mean(df.price)
        n_stores = len(df.store_id.unique())
        x_basket_size = len(df) / len(df.order_id.unique())
        group = df.group.unique()[0]
        visit_fidelity_ratio = calculate_visit_fidelity_ratio(df, n_stores)
        price_fidelity_ratio = calculate_price_fidelity_ratio(df, n_stores)
        x_fidelity_ratio = (visit_fidelity_ratio + price_fidelity_ratio) / 2
        category_vector = vectorize_category(df)
        row = [user_id, group, x_price, x_special, x_fidelity_ratio, x_basket_size]
        row = pd.Series(row, index=cluster_data.columns)
        cluster_data = cluster_data.append(row, ignore_index=True)

        category_vector = pd.Series(category_vector, index=category_matrix.columns)
        category_matrix = category_matrix.append(category_vector, ignore_index=True)
        cluster_data['user_id'] = cluster_data['user_id'].astype('int')
    # reduce dimension of category matix
    reduced_dimension = PCA(n_components=1, random_state=1).fit_transform(category_matrix.iloc[:, -24:])
    embedded_category = pd.DataFrame(reduced_dimension, columns=['category_embbed'])
    cluster_data = cluster_data.join(embedded_category)
    return cluster_data


def prep_data(path):
    print("prep_data")
    path = path
    cluster_data = pd.DataFrame(columns=['user_id', 'group', 'x_price',
                                         'x_special', 'x_fidelity_ratio', 'x_basket_size'])
    columns = [*range(1, 25)]
    columns.insert(0, 'user_id')
    category_matrix = pd.DataFrame(columns=columns)
    cluster_data = make_cluster_data(path, cluster_data, category_matrix)
    # cluster_data = cluster_data.sample(frac=1) #random shuffle df
    return cluster_data


def make_clusters(cluster_data):
    print("make_clusters")
    norm = MinMaxScaler()
    cluster_data.iloc[:, 2:] = norm.fit_transform(cluster_data.iloc[:, 2:])
    X = cluster_data.iloc[:, 2:].values
    clt_ = AgglomerativeClustering(n_clusters=4)
    labels = clt_.fit_predict(X)
    cluster_data['label'] = labels
    labeled_cluster_data = cluster_data[['user_id', 'label']]
    labeled_cluster_data.to_csv('labeled_cluster_data.csv')
    plt.clf()
    plt.close('all')
    dim = TSNE(n_components=2, random_state=1, perplexity=30, learning_rate=925, init='pca', early_exaggeration=1)
    reduced_dimension = dim.fit_transform(X)
    df = pd.DataFrame(reduced_dimension, columns=['pca1', 'pca2'])
    df['labels'] = labels
    colors = ['red', 'limegreen', 'dodgerblue', 'yellow']

    plt.scatter(df.values[:, 0], df.values[:, 1], c=df.labels, s=75,
                cmap=matplotlib.colors.ListedColormap(colors),
                alpha=0.95, edgecolors='black')

    for i, txt in enumerate(labels):
        plt.annotate(int(txt)+1, (reduced_dimension[:, 0][i], reduced_dimension[:, 1][i]), fontsize=7)

    plt.axis([-50, 62, -30, 50])
    plt.savefig('clustering.png')
    plt.show()

    return labeled_cluster_data


def redistribute_users(labeled_cluster_data):
    print("redistribute_users")
    df = labeled_cluster_data
    n_clusters = len(df.label.unique())

    if not os.path.exists('../postprocess_all_products/'):
        os.mkdir('../postprocess_all_products/')

    #print(df.label.unique())
    for i in range(n_clusters):
        try:
            os.mkdir('../postprocess/g' + str(i + 1))
        except:
            print('File already exists.')
        cluster = df.loc[df.label == i]
        cluster_users = cluster.user_id.to_list()
        for file in os.listdir('../data'):
            if int(file) in cluster_users:
                copytree('../data/' + str(file),
                         '../postprocess_all_products/g' + str(i + 1) + '/' + str(file))


def start_clustering_steps():
    path = '../data'
    cluster_data = prep_data(path)
    clusterized_data = make_clusters(cluster_data)
    redistribute_users(clusterized_data)


if __name__ == "__main__":
    start_clustering_steps()
