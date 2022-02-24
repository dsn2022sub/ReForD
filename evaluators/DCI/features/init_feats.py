import numpy as np
import sklearn.preprocessing as preprocessing
from scipy.sparse import linalg
import scipy.sparse as sp
import sys

np.random.seed(0)

def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
           
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = x.astype("float64")
    return x

def intial_embedding(n, adj, in_degree,hidden_size, retry=10):
    in_degree = in_degree.clip(1) ** -0.5
    norm = sp.diags(in_degree, 0, dtype=float)
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    
    return x

def process_adj(dataSetName):
    normal_stream_path = '/normal/filtered-stream/'
    normal_base_path = '/normal/filtered-base/'
    attack_stream_path = '/attack/filtered-stream/'
    attack_base_path = '/attack/filtered-base/'
    node_num = 0
    row = []
    col = []
    mx = 0
    nodeset = set()
    #上面这些先预定义，后面再加
    for i in range(1):
        path1 = '../'+datasetName+normal_stream_path+'stream-wget-normal-'+str(i)+'.txt'
        path2 = '../'+datasetName+normal_base_path+'base-wget-normal-'+str(i)+'.txt'
        path3 = '../'+datasetName+attack_stream_path+'stream-wget-attack-'+str(i)+'.txt'
        path4 = '../'+datasetName+attack_base_path+'base-wget-attack-'+str(i)+'.txt'
        edges = np.loadtxt(path1, dtype='str').astype('str')
        for edge in edges:
            mx=max(mx, int(edge[0]))

            mx=max(mx, int(edge[1]))

        #node_num = node_num + len(set(edges[:, 0])) + len(set(edges[:, 1]))

        row = row+list(edges[:, 0].astype('int').T) + list(edges[:, 1].astype('int').T)
        col = col+list(edges[:, 1].astype('int').T) + list(edges[:, 0].astype('int').T)

        edges = np.loadtxt(path2, dtype='str').astype('str')
        for edge in edges:
            mx=max(mx,int(edge[0]))
            mx=max(mx,int(edge[1]))
        # node_num = node_num + len(set(edges[:, 0])) + len(set(edges[:, 1]))

        row = row+list(edges[:, 0].astype('int').T) + list(edges[:, 1].astype('int').T)
        col = col+list(edges[:, 1].astype('int').T) + list(edges[:, 0].astype('int').T)

        edges = np.loadtxt(path3, dtype='str').astype('str')
        for edge in edges:
            mx=max(mx,int(edge[0]))
            mx=max(mx,int(edge[1]))
        # node_num = node_num + len(set(edges[:, 0])) + len(set(edges[:, 1]))

        row = row+list(edges[:, 0].astype('int').T) + list(edges[:, 1].astype('int').T)
        col = col+list(edges[:, 1].astype('int').T) + list(edges[:, 0].astype('int').T)

        edges = np.loadtxt(path4, dtype='str').astype('str')
        for edge in edges:
            print(int(edge[0]))
            mx = max(mx,int(edge[0]))
            mx = max(mx,int(edge[1]))
        # node_num = node_num + len(set(edges[:, 0])) + len(set(edges[:, 1]))

        row = row+list(edges[:, 0].astype('int').T) + list(edges[:, 1].astype('int').T)
        col = col+list(edges[:, 1].astype('int').T) + list(edges[:, 0].astype('int').T)

    node_num = mx+1
    print(node_num)
    data = [1.0 for _ in range(len(row))]
    adj = sp.csr_matrix((data, (row, col)), shape=(node_num, node_num))
    return adj, node_num
print(sys.path)
datasetName = sys.argv[1]

adj, n = process_adj(datasetName)
'''
adj, n = process_adj('../data/'+datasetName+'.txt')
'''
hidden_size = 64
in_degree = [np.sum(adj.data[adj.indptr[i]: adj.indptr[i+1]]) for i in range(n)]
in_degree = np.array(in_degree)
x = intial_embedding(n, adj, in_degree, hidden_size, retry=10)
np.save(datasetName+'_feature64.npy', x)
