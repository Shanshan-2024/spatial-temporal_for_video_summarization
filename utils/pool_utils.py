import torch


def adj_pool(adj, k):
    '''
    调用方法：adj_pool(adj=adj, k=0.5)
    k表示池化率，1代表全选择，0代表全不选择，adj为邻接矩阵
    :param adj:
    :param k:
    :return:
    '''
    num_node = adj.shape[0]
    filter_value = torch.zeros(1).to(adj.device)
    if isinstance(k, float):
        indices_to_remove = adj < torch.topk(adj, pool, sorted=False)[0][..., -1, None]
    else:
        pool = torch.ceil(k * num_node)
        indices_to_remove = adj < torch.topk(adj, int(pool.detach().cpu().numpy()), sorted=False)[0][..., -1, None]
    adj[indices_to_remove] = filter_value

    return adj