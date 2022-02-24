import numpy as np
from sklearn.model_selection import StratifiedKFold

def load_data(datasets, num_folds):
    # load the adjacency
    normal_stream_path = '/normal/filtered-stream/'
    normal_base_path = '/normal/filtered-base/'
    attack_stream_path = '/attack/filtered-stream/'
    attack_base_path = '/attack/filtered-base/'
    adj = []
    cnt = 0
    for i in range(1):
        # adj = np.loadtxt('./data/'+datasets+'.txt')
        data1 = np.loadtxt('./'+datasets+normal_stream_path+'stream-wget-normal-'+str(i)+'.txt', dtype='str').astype('str')
        data2 = np.loadtxt('./'+datasets+normal_base_path+'base-wget-normal-'+str(i)+'.txt', dtype='str').astype('str')
        data3 = np.loadtxt('./'+datasets+attack_stream_path+'stream-wget-attack-'+str(i)+'.txt', dtype='str').astype('str')
        data4 = np.loadtxt('./'+datasets+attack_base_path+'base-wget-attack-'+str(i)+'.txt', dtype='str').astype('str')
        adj.append(data1[:, 0:2])
        adj.append(data2[:, 0:2])
        adj.append(data3[:, 0:2])
        adj.append(data4[:, 0:2])

    adj = np.concatenate(adj)
    num_user = len(set(adj[:, 0]))
    num_object = len(set(adj[:, 1]))
    adj = adj.astype('int')
    nb_nodes = np.max(adj) + 1
    edge_index = adj.T
    print(adj)
    print('Load the edge_index done!')
    
    # load the user label
    # label = np.loadtxt('./data/'+datasets+'_label.txt')
    label = []
    for i in range(1):
        # adj = np.loadtxt('./data/'+datasets+'.txt')
        data1 = np.loadtxt('./'+datasets+normal_stream_path+'stream-wget-normal-'+str(i)+'.txt', dtype='str').astype('str')
        tmpdata1 = data1[:, 0:2]
        tmpset1 = set()
        for num_line in tmpdata1:
            tmpset1.add(num_line[0])
            tmpset1.add(num_line[1])
            cnt=cnt+1
        tmp = list(tmpset1)

        label1=[]
        for j in range(len(tmp)):

            label1.append(np.array([float(tmp[j]), float(0.0)]))
            # label1.append(t)
        label.append(label1)
        tmp.clear()


        data2 = np.loadtxt('./'+datasets+normal_base_path+'base-wget-normal-'+str(i)+'.txt', dtype='str').astype('str')
        tmpdata2 = data2[:, 0:2]
        tmpset2 = set()
        for num_line in tmpdata2:
            tmpset2.add(num_line[0])
            tmpset2.add(num_line[1])
            cnt=cnt+1
        tmp = list(tmpset2)
        label2=[]
        for j in range(len(tmp)):
            t = []
            label2.append(np.array([float(tmp[j]), 0.0]))

            # label2.append(t)
        label.append(label2)
        tmp.clear()
        data3 = np.loadtxt('./'+datasets+attack_stream_path+'stream-wget-attack-'+str(i)+'.txt', dtype='str').astype('str')
        tmpdata3 = data3[:, 0:2]
        tmpset3 = set()
        for num_line in tmpdata3:
            tmpset3.add(num_line[0])
            tmpset3.add(num_line[1])
            cnt=cnt+1
        tmp = list(tmpset3)
        label3=[]
        for j in range(len(tmp)):
            t = []
            label3.append(np.array([float(tmp[j]), 1.0]))


            # label3.append(t)
        label.append(label3)
        tmp.clear()

        data4 = np.loadtxt('./'+datasets+attack_base_path+'base-wget-attack-'+str(i)+'.txt', dtype='str').astype('str')
        tmpdata4 = data4[:, 0:2]
        tmpset4 = set()
        for num_line in tmpdata4:
            tmpset4.add(num_line[0])
            tmpset4.add(num_line[1])
            cnt=cnt+1
        tmp = list(tmpset4)
        label4=[]
        for j in range(len(tmp)):
            t = []
            label4.append(np.array([float(tmp[j]), float(1.0)]))

            # label4.append(t)
        label.append(label4)
        tmp.clear()


    label = np.concatenate(label)
    print(label)
    y = label[:, 1]
    print(np.sum(y))
    print(cnt)

    print('Ratio of fraudsters: ', np.sum(y) / len(y))
    print('Number of edges: ', edge_index.shape[1])
    print('Number of users: ', num_user)
    print('Number of objects: ', num_object)
    print('Number of nodes: ', nb_nodes)

    # split the train_set and validation_set
    split_idx = []
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
    for (train_idx, test_idx) in skf.split(y, y):
        split_idx.append((train_idx, test_idx))
   
    # load initial features

    feats = np.load('./features/'+'data'+'_feature64.npy')
    print(feats)
    # return edge_index, feats, split_idx, label, nb_nodes
    return edge_index, feats, split_idx, label, nb_nodes

