import torch
from utils import load_data_from_folder, pool_max
import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score

import argparse

parser = argparse.ArgumentParser(description='Process some args.')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--seed', type=int, help='Seed')
args = parser.parse_args()

# torch.manual_seed(args.seed)
np.random.seed(args.seed)

class MLP(torch.nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.layers = []
        for k in range(len(dims)-1):
            self.layers.append(torch.nn.Linear(dims[k], dims[k+1]))
            self.layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

class RNN_recommender(torch.nn.Module):
    def __init__(self, n_input, n_products, n_embedding, n_rnn_layers, n_classes, dropout, device):
        super(RNN_recommender, self).__init__()
        # Layer definitons
        self.n_products = n_products
        self.n_embedding = n_embedding
        self.n_rnn_layers = n_rnn_layers
        self.dropout = dropout
        self.device = device
        self.encode = torch.nn.Embedding(n_products, 
                                         n_embedding,
                                         padding_idx = 0) # Item embedding layer, 商品编码
        self.pool = pool_max
        # RNN type specify
        self.embed_mlp = MLP(dims=[n_input, n_embedding])
        self.rnn1 = torch.nn.GRU(n_embedding, 
                                n_embedding, 
                                n_rnn_layers, 
                                batch_first=True, 
                                dropout=dropout)
        self.rnn2 = torch.nn.GRU(n_embedding, 
                                n_embedding, 
                                n_rnn_layers, 
                                batch_first=True, 
                                dropout=dropout)
        self.concat_mlp = MLP(dims=[2*n_embedding, n_embedding])
        # Class weight matrix
        self.class_weights = torch.nn.Parameter(torch.randn(size=(n_embedding, n_classes)), requires_grad=True)
    
    def forward(self, seqs, neg_labels=None):
        embed_seqs = []
        lens = []
        if neg_labels is not None:
            neg_labels_order = neg_labels['product_ids'].values.astype(np.int32)
        for order_id, seq in seqs:
            pos_labels = seq.iloc[:,1].values.astype(np.int32)
            if neg_labels is not None:
                neg_labels_order = np.setdiff1d(neg_labels_order, pos_labels)
                product_ids = torch.LongTensor(np.concatenate([pos_labels, neg_labels_order])).to(self.device) # First feature is product_id
                features = np.concatenate([seq.iloc[:,2:].values.astype(np.float32),
            neg_labels[neg_labels["product_ids"].isin(neg_labels_order)].iloc[:, 2:].values.astype(
                np.float32
            )],0)
            else:
                product_ids = torch.LongTensor(pos_labels).to(self.device)
                features = seq.iloc[:,2:].values.astype(np.float32)
            
            features = self.embed_mlp(torch.FloatTensor(features).to(self.device))
            embedding = self.encode(product_ids) # shape: 1, len(basket), embedding_dim
            features = self.concat_mlp(torch.cat([features, embedding], -1))            

            embed_seqs.append(features)
            lens.append(features.shape[0])
        # Input for rnn
        # embed_seqs =  torch.stack(embed_seqs, 0) # shape: batch_size, max_len, embedding_dim
        padded_embed_seqs = torch.nn.utils.rnn.pad_sequence(embed_seqs, batch_first=True)
        packed_embed_seqs = torch.nn.utils.rnn.pack_padded_sequence(padded_embed_seqs, lens, enforce_sorted=False, batch_first=True) # packed sequence as required by pytorch

        hidden1 = self.init_hidden(len(padded_embed_seqs))
        # RNN
        output, h_u = self.rnn1(packed_embed_seqs, hidden1)
        output, h_u = self.rnn2(output, h_u)
        dynamic_user, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # shape: batch_size, max_len, embedding_dim
        return dynamic_user, h_u

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return weight.new(self.n_rnn_layers, batch_size, self.n_embedding).zero_()


    def bpr_loss(self, seqs, dynamic_user):
        nll = 0.
        # n_batch x n_T x n_products
        embed_product = torch.einsum("ijk,kl->ijl", dynamic_user, self.encode.weight.t())
        for t, (order_id, seq) in enumerate(seqs):
            
            pos_idx = torch.LongTensor(seq.iloc[:,1].values.astype(np.int32)).to(self.device) # First feature is product_id
            
            neg_idx = torch.randint(low=1, high=self.n_products, size=(len(pos_idx),)).to(self.device)
            if pos_idx[0].item() != 0 and t != 0:
                # features = seq.iloc[:,2:].values.astype(np.float32) # Take 26 features
                # features = self.embed_mlp(torch.FloatTensor(features).to(self.device))
                pos_scores = torch.einsum("ijk,kl->ijl", dynamic_user, self.encode(pos_idx).t())[t-1]
                neg_scores = torch.einsum("ijk,kl->ijl", dynamic_user, self.encode(neg_idx).t())[t-1]
                
                # Score p(u, t, v > v')
                score = pos_scores - neg_scores
                # Average Negative log likelihood for basket_t
                nll += - torch.mean(torch.nn.LogSigmoid()(score))
        return nll

    def cce_loss(self, seqs, dynamic_user, neg_labels, ys, neg_ys):
        loss = 0.
        criterion = torch.nn.CrossEntropyLoss()
        # (n_batch x n_T x n_products) @ (n_products x n_classes) -> n_batch x n_T x n_classes
        embed_product = torch.einsum("ijk,kl->ijl", dynamic_user, self.class_weights)
        for t, ((order_id, seq), (_, y)) in enumerate(zip(seqs,ys)):
            pos_labels = y.iloc[:,2].values.astype(np.float32).astype(np.int32)
            # neg_labels_order = neg_labels['product_ids'].values.astype(np.int32)
            # neg_labels_order = np.setdiff1d(neg_labels_order, seq.iloc[:, 1].values.astype(np.float32).astype(np.int32))
            # neg_labels_order = np.zeros_like(neg_labels_order)
            neg_labels_order = np.zeros(shape=(len(embed_product[t])-pos_labels.shape[0])).astype(np.int32)
            
            y_true = torch.LongTensor(np.concatenate([pos_labels, neg_labels_order])).to(self.device)
            y_pred = embed_product[t]
            loss += criterion(y_pred, y_true)
        return loss / float(len(seqs))

    def predict(self, seqs, dynamic_user, ys):
        loss = 0.
        # (n_batch x n_T x n_products) @ (n_products x n_classes) -> n_batch x n_T x n_classes
        embed_product = torch.einsum("ijk,kl->ijl", dynamic_user, self.class_weights)
        y_true_acc = []
        y_pred_acc = []
        for t, ((order_id, seq), (_, y)) in enumerate(zip(seqs,ys)):
            n_T = len(seq)
            y_true = y.iloc[:,-1].values.astype(np.int32)
            y_pred = embed_product[t,:n_T].detach().cpu().numpy().argmax(1)
            y_true_acc.append(y_true)
            y_pred_acc.append(y_pred)
        return y_true_acc, y_pred_acc

def wrap_X(X, order_ids, product_ids, enc, split=True):
    product_ids = enc.transform(product_ids.reshape(-1, 1))
    X = pd.DataFrame(
        np.concatenate([order_ids.reshape(-1, 1), product_ids, X], 1)
    )
    cols = list(X.columns)
    cols[:2] = ['order_ids','product_ids']
    X.columns = cols
    if split:
        negative_X = X[X['order_ids']==99999999]
        positive_X = X[X['order_ids']!=99999999]
        return positive_X, negative_X
    else:
        return X

def wrap_y(y, order_ids, product_ids, enc, split=True):
    product_ids = enc.transform(product_ids.reshape(-1, 1))
    y = pd.DataFrame(
        np.concatenate([order_ids.reshape(-1, 1), product_ids, y.reshape(-1, 1)], 1)
    )
    cols = list(y.columns)
    cols[:2] = ['order_ids','product_ids']
    y.columns = cols
    if split:
        negative_y = y[y['order_ids']==99999999]
        positive_y = y[y['order_ids']!=99999999]
        return positive_y, negative_y
    else:
        return y

scores = []
for directory in glob.glob("data/*"):
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            # if os.path.isfile(directory+'/'+str(filename)+'/f1_score.csv'):
            #     continue
            if (directory == 'data/g1' and filename == '1'):
                continue
            X_train, X_test, y_train, y_test, moyenne_taille_paniers, df_X_test, dataset_test_diminue, y_test2, order_ids_train, order_ids_test, product_ids_train, product_ids_test = load_data_from_folder(
                directory, filename)
            n_products = np.unique(np.concatenate([product_ids_train,product_ids_test])).shape[0]
            n_classes = np.max(np.concatenate([y_train,y_test])).item()+1
            enc = OrdinalEncoder()
            enc.fit(np.concatenate([product_ids_train,product_ids_test]).reshape(-1, 1))
            
            del X_test['product_id'], X_test['store_id']
            X_train_pos, X_train_neg = wrap_X(X_train, order_ids_train, product_ids_train, enc, split=True)
            X_test = wrap_X(X_test, order_ids_test, product_ids_test, enc, split=False)
            y_train_pos, y_train_neg = wrap_y(y_train, order_ids_train, product_ids_train, enc, split=True)
            y_test = wrap_y(y_test, order_ids_test, product_ids_test, enc, split=False)

            # n_classes = 10 # number of stores
            n_input = X_train_pos.shape[-1]-2
            # n_batch = 8
            n_embedding = 64
            n_rnn_layers = 1
            dropout = 0.
            n_epochs = 100
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            model = RNN_recommender(n_input=n_input, n_products=n_products, n_embedding=n_embedding, n_rnn_layers=n_rnn_layers, n_classes=n_classes, dropout=dropout, device=device).to(device)
            model.train()
            # opt = torch.optim.AdamW(model.parameters(), lr = 1e-4)
            opt = torch.optim.RMSprop(model.parameters(), lr = args.lr)

            seqs_train = X_train_pos.groupby(["order_ids"])

            for i in range(n_epochs):
                features, hidden = model(seqs_train, X_train_neg)
                # nll_loss = model.bpr_loss(seqs_train, features)
                cce_loss = model.cce_loss(seqs_train, features, X_train_neg, y_train_pos.groupby(["order_ids"]), y_train_neg)
                cce_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                opt.step()
                print('Epoch %d: %.3f'%(i,cce_loss.detach().item()))
            seqs_test = X_test.groupby(["order_ids"])
            features, hidden = model(seqs_test, None)
            y_true_acc, y_pred_acc =  model.predict(seqs_test, features, y_test.groupby(["order_ids"]))
            f1_score_avg = 0.
            for y_true, y_pred in zip(y_true_acc, y_pred_acc):
                f1_score_avg += f1_score(y_true, y_pred, average='macro')
            f1_score_avg = f1_score_avg / len(y_true_acc)
            
            # Logistic regression code
            # from sklearn.linear_model import LogisticRegression
            # clf = LogisticRegression(random_state=0).fit(X_train_pos.iloc[:, 2:].values.astype(np.float32), y_train_pos.iloc[:, 2:].values.astype(np.int32))
            # clf.score(X_train_pos.iloc[:, 2:].values.astype(np.float32), y_train_pos.iloc[:, 2:].values.astype(np.int32))

            df_f1 = pd.DataFrame({'user_id': [filename], 'cluster_id': [directory.split('/')[1]], 'f1_score': [f1_score_avg]})
            # df_f1.to_csv(directory+'/'+str(filename)+'/f1_score.csv', index=False)

            scores.append(df_f1)
scores = pd.concat(scores,0)
scores.to_csv('data/f1_score_%f.csv'%args.lr, index=False)
# print('Learning rate: %f'%args.lr)
# print(scores['f1_score'].mean())
# print('Mean')
# print(scores.groupby("cluster_id").apply(lambda x: x["f1_score"].mean()))
# print('Std')
# print(scores.groupby("cluster_id").apply(lambda x: x["f1_score"].std()))