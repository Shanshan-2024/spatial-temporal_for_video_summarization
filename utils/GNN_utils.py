import torch
import torch.nn as nn
import numpy as np

def Generate_Adjacency(feature_dic, args):
    Adj_dic = {}

    for video_index in feature_dic.keys():
        feature = feature_dic[video_index] #N * 1024
        #计算节点个数
        N = feature.shape[0]
        adjacency = torch.zeros((N, N))

        if args.Gene_ways == 'cosine_similarity':
                    smi = nn.functional.cosine_similarity(feature[i], feature[j], dim=0)
                    adjacency[i, j] = smi
        elif args.Gene_ways == 'dot':
                    smi = np.dot(feature[i], feature[j])
                    adjacency[i, j] = smi
        elif args.Gene_ways == 'Gauuss':
            for i in range(N):
                for j in range(N):
                    # 计算欧氏距离的平方
                    distance_squared = np.sum((feature[i] - feature[j]) ** 2)

                    # 设置高斯核函数的方差参数
                    sigma_squared = 1.0  # 可以根据需求调整这个值

                    # 计算高斯核函数
                    gaussian_kernel = np.exp(-distance_squared / (2 * sigma_squared))
                    adjacency[i, j] = gaussian_kernel

        Adj_dic[video_index] = adjacency

    return Adj_dic


def Generate_Adjacency_PE(feature_dic, args):
    Laplas_dic = {}
    Adj_dic = {}

    for video_index in feature_dic.keys():
        feature = torch.from_numpy(feature_dic[video_index])
        # feature = feature_dic[video_index]  # N * 1024
        # 计算节点个数
        N = feature.shape[0]
        adjacency = torch.zeros((N, N))

        if args.Gene_ways == 'cosine_similarity':

            for i in range(N):
                for j in range(N):
                    smi = nn.functional.cosine_similarity(feature[i], feature[j], dim=0)
                    adjacency[i, j] = smi
        # 计算方形矩阵的特征值和特征向量
        EigVal, EigVec = np.linalg.eig(adjacency.numpy())
        idx = EigVal.argsort()  # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
        gPE = torch.from_numpy(EigVec[:, 0: 100]).float()
        Adj_dic[video_index] = adjacency
        if gPE.shape[1] < 100:
            patch = torch.ones(gPE.shape[0], (100-gPE.shape[1]))
            # patch = torch.zeros(gPE.shape[0], (10-gPE.shape[1]))
            gPE = torch.cat((gPE, patch), dim=1)
        Laplas_dic[video_index] = gPE

    return Adj_dic, Laplas_dic

def Generate_fragment_Adjacency_PE(fragment_feature, args):
    Adj_list = []
    Laplas_list = []

    for fragment in fragment_feature:
        # fragment  #10*1024
        N = fragment.shape[0]
        adjacency = torch.zeros((N, N))

        if args.Gene_ways == 'cosine_similarity':
            for i in range(N):
                for j in range(N):
                    smi = nn.functional.cosine_similarity(fragment[i], fragment[j], dim=0)
                    adjacency[i, j] = smi
        Adj_list.append(adjacency)
        # 计算方形矩阵的特征值和特征向量
        EigVal, EigVec = np.linalg.eig(adjacency.numpy())
        idx = EigVal.argsort()  # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
        gPE = torch.from_numpy(EigVec[:, 0: 11]).float()
        Laplas_list.append(gPE)

    return Adj_list, Laplas_list











