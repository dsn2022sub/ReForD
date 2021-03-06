import json
import re

from tqdm import tqdm
import networkx as nx
from node2vec import Node2Vec
from sklearn import datasets
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import easygraph as eg
import sklearn.metrics
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest

x = []
x1 = []
x_train = []
y_train = []
x_test = []
y_test = []


# load data, a weighted undirected graph
def load_data(path, index):
    normal_stream_path = path + '/normal/filtered-stream/'
    normal_base_path = path + '/normal/filtered-base/'
    attack_stram_path = path + '/attack/filtered-stream/'
    attack_base_path = path + '/attack/filtered-base/'
    y_true = []
    G = nx.Graph()
    G1 = nx.Graph()

    path1 = normal_stream_path + 'stream-wget-normal-' + str(index) + '.txt'
    data1 = np.loadtxt(path1, dtype=str).astype('str')
    tmpset1 = set()

    for ite in data1:
        G.add_edge(int(ite[0]), int(ite[1]))

    path2 = normal_base_path + 'base-wget-normal-' + str(index) + '.txt'
    data2 = np.loadtxt(path2, dtype=str).astype('str')

    for ite in data2:
        G.add_edge(int(ite[0]), int(ite[1]))

    if index <= 24:
        path3 = attack_base_path + 'base-wget-attack-' + str(index) + '.txt'
        data3 = np.loadtxt(path3, dtype=str).astype('str')

        for ite in data3:
            G1.add_edge(int(ite[0]), int(ite[1]))

        path4 = attack_stram_path + 'stream-wget-attack-' + str(index) + '.txt'
        data4 = np.loadtxt(path4, dtype=str).astype('str')

        for ite in data4:
            G1.add_edge(int(ite[0]), int(ite[1]))

    return G, G1


for i in range(125):
    print('epoch:' + str(i))
    G, G1 = load_data('data1', i)
    print("load finished")
    '''
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec1 = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs
    node2vec2 = Node2Vec(G1, dimensions=64, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs
    # Embed nodes
    model1 = node2vec1.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
    for word in model1.wv.index_to_key:
        x.append(list(model1.wv[word]))
    print(x)
    model2 = node2vec2.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
    for word in model2.wv.index_to_key:
        x1.append(list(model2.wv[word]))
    print(x1)
    '''
    model1 = eg.functions.graph_embedding.line.LINE(G,
                                                    embedding_size=16,
                                                    order='all')  # The order of model LINE. 'first'???'second' or 'all'.
    model1.train(batch_size=1024, epochs=1, verbose=2)
    y = model1.get_embeddings()  # Returns the graph embedding results.
    # print(y)
    x = []
    for key in y.keys():
        x.append(y[key])
    # print(x)

    print("emdedding ok")
    if i <= 24:
        model2 = eg.functions.graph_embedding.line.LINE(G1,
                                                        embedding_size=16,
                                                        order='all')  # The order of model LINE. 'first'???'second' or 'all'.
        model2.train(batch_size=1024, epochs=1, verbose=2)
        y1 = model2.get_embeddings()  # Returns the graph embedding results.
        # print(y1)
        x1 = []
        for key in y1.keys():
            x1.append(y1[key])
        x1 = np.array(x1)
    # print(x1)

    x = np.array(x)
    print('emdedding ok')
    model_anomaly = IsolationForest(n_estimators=80, contamination=0.1)
    normal_list = model_anomaly.fit_predict(x)
    print("fit normal")
    if i <= 24:
        model_anomaly1 = IsolationForest(n_estimators=80, contamination=0.1)
        attlist = model_anomaly1.fit_predict(x1)
        print("fit attack")
        # y1 = model_anomaly1._predict(x1)

    res1 = model_anomaly.score_samples(x)
    cnt = 0

    # print(cnt)
    if i <= 24:
        res2 = model_anomaly1.score_samples(x)
        # print(res2)
        sum2 = np.sum(res2)
        cnt1 = 0
        # print(res2)
        # for i in res2:
        #    print(i)
        attres = np.sum(res2)  # sum(1 if i ==-1 else 0 for i in attlist)

        print("attack score :" + str(attres))
    # sum1 = np.sum(res1)
    norres = np.sum(res1)  # sum(1 if i ==-1 else 0 for i in normal_list)

    print("normal score :" + str(norres))

    if i <= 24:
        x_train.append([attres])
        y_train.append(1)

x_train.append([norres])
y_train.append(0)

if (i == 124):
    model = XGBClassifier()
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    # pretest = model.predict(x_test)
    print(cross_validate(model, x_train, y_train, scoring=scoring, cv=5, n_jobs=4))
