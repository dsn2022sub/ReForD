import os
import re

# import matplotlib.pyplot as plt
import numpy as np

# from openTSNE import TSNE
import networkx as nx
# from examples import utils

import easygraph
# from matplotlib import cm
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate

from sklearn.manifold import TSNE
import random
from pyecharts import options as opts
from pyecharts.charts import Scatter3D
from pyecharts.faker import Faker
from sklearn.decomposition import PCA


def visualize(dataset, mode, i):
    filtered_base_graph_file = f'sword01-data/{mode}/filtered-base/base-{dataset}-{mode}-{i}.txt'
    # filtered_base_graph_file = f'data/{mode}/all-{mode}-{i}.txt'
    filtered_stream_graph_file = f'sword01-data/{mode}/filtered-stream/stream-{dataset}-{mode}-{i}.txt'
    G = nx.MultiDiGraph()

    with open(filtered_base_graph_file, 'r', encoding='utf-8', errors='ignore') as all_map_fh:
        for edge in all_map_fh:
            parts = edge.strip().split(':')
            src, dst, src_type = parts[0].split()
            dst_type = parts[1]
            edge_type = parts[2]
            logical_ts = int(parts[-3])
            actual_ts = parts[-2]
            isnetwork = parts[-1]
            # if isnetwork == '1':
            #    print(str(src)+':'+str(dst))

            G.add_edge(src, dst, **{
                'src': src,
                'dst': dst,
                'src_type': src_type,
                'dst_type': dst_type,
                'edge_type': edge_type,
                'logical_ts': logical_ts,
                'actual_ts': actual_ts,
                'isnetwork': isnetwork
            })

    with open(filtered_stream_graph_file, 'r', encoding='utf-8', errors='ignore') as all_map_fh:
        for edge in all_map_fh:
            parts = edge.strip().split(':')
            src, dst, src_type = parts[0].split()
            dst_type = parts[1]
            edge_type = parts[2]
            logical_ts = int(parts[-3])
            actual_ts = parts[-2]
            isnetwork = parts[-1]

            G.add_edge(src, dst, **{
                'src': src,
                'dst': dst,
                'src_type': src_type,
                'dst_type': dst_type,
                'edge_type': edge_type,
                'logical_ts': logical_ts,
                'actual_ts': actual_ts,
                'isnetwork': isnetwork
            })

    model = easygraph.functions.graph_embedding.line.LINE(G, embedding_size=16,
                                                          order='all')  # The order of model LINE. 'first'，'second' or 'all'.
    model.train(batch_size=1024, epochs=10, verbose=2)
    # model = easygraph.functions.graph_embedding.sdne.SDNE(G, hidden_size=[256, 128]) # The hidden size in SDNE.
    # model.train(batch_size=3000, epochs=10, verbose=2)
    embeddings = model.get_embeddings()  # Returns the graph embedding results.
    emb_matrix = []
    for k, v in embeddings.items():
        emb_matrix.append(v)
    # print(embeddings)
    emb_matrix = np.array(emb_matrix)

    # A=np.array(nx.adjacency_ma{mode}-trix(G).todense())

    print('embedding done')
    print('transform begin')
    # tsne = TSNE(n_components=2,perplexity=30,metric="euclidean",n_jobs=32,  exaggeration=26, random_state=42,  verbose=True,)

    tsne = TSNE(perplexity=50, n_components=3, init='pca', n_iter=250)
    # init = openTSNE.initialization.rescale(emb_matrix[:, :2])
    '''
    aff500 = openTSNE.affinity.PerplexityBasedNN(
        np.array(emb_matrix) ,

        perplexity=500,
        n_jobs=32,
        random_state=0,
    )
    '''
    '''
    low_dim_embs = openTSNE.TSNE(
        n_components=3,
        #initialization='pca',
        #perplexity= 500,
        exaggeration=12,
        n_jobs=32,
        verbose=True,
        metric="cosine",
        random_state=3
    ).fit(np.array(emb_matrix))
    '''

    pca = PCA(n_components=3)
    low_dim_embs = pca.fit_transform(np.array(emb_matrix))

    # low_dim_embs = tsne.fit_transform(np.array(emb_matrix))
    # low_dim_embs = np.array(tsne.embedding_)
    low_dim_embs = np.array(low_dim_embs)
    '''
    with open('params.txt','a') as a:
        a.write(mode + '-'+str(i))
        a.write(str(tsne.embedding_))
        a.write('\n')
        a.write(str(tsne.kl_divergence_))
        a.write('\n')
        a.write(str(tsne.n_features_in_))
        a.write('\n')
        #a.write(str(tsne.feature_name_in_))

        #a.write('\n')
        a.write(str(tsne.n_iter_))
        a.write('\n')
    '''

    print(low_dim_embs)
    print('transform done')
    if mode == 'attack':
        tag = 'attack'
    else:
        tag = 'normal'
    y_label = [tag for i in range(len(low_dim_embs))]
    print('tags prepared')
    return low_dim_embs, y_label

    # utils.plot(embedding_train, y_train, colors=utils.MACOSKO_COLORS)


if __name__ == '__main__':
    all_embs = []
    all_labels = []
    for i in range(25):
        emb, label = visualize('wget', 'attack', i)
        all_embs.extend(emb)
        all_labels.extend(label)
    for i in range(125):
        emb, label = visualize('wget', 'normal', i)

        # label = label[0:100]
        all_embs.extend(emb)

        all_labels.extend(label)
    all_embs = np.array(all_embs)

    with open('coordinatespca.txt', 'w') as co_fh:
        for emb in all_embs:
            co_fh.write(str(emb[0]) + ' ' + str(emb[1]) + ' ' + str(emb[2]) + '\n')
    with open('labelspca.txt', 'w') as lb_fh:
        for label in all_labels:
            lb_fh.write(label + '\n')
    # print(all_embs)
    # print(all_labels)
