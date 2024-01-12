
import argparse

def Get_args():
    parser = argparse.ArgumentParser()

    #数据集设置
    parser.add_argument('--Train_dataset', default='canonical',
                        help='有canonical, augment, transfer三种模式，选择以哪种模式读取数据集')
    parser.add_argument('--Dataset_root', default='data/', help='存放数据集的路径')
    parser.add_argument('--dataset', default='summe', help='选用的数据集{tvsum, summe}')
    parser.add_argument('--generate_mode', default='max', help='在生成f1的时候选择评价指标，tvsum->arg, summe->max')
    parser.add_argument('--split_seed', default='1226', help='随机划分的随机数种子')
    parser.add_argument('--shot_mode', type=bool, default=False, help='是否使用镜头模式')

    #参数设置
    parser.add_argument('--lr', default=1e-3, help='学习率')
    parser.add_argument('--lr_step', type=int, default=5, help='调整学习率的轮数')
    parser.add_argument('--lr_gamma', default=0.7, help='学习率衰减的频率')
    parser.add_argument('--cuda', default='cuda:1', help='使用哪个GPU学习')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--weight_decay', default=0., help='l2正则化权重')
    parser.add_argument('--model_seed', default=1226, help='模型初始化的随机数种子')
    parser.add_argument('--save_dir', default='save_result/save_model', help='模型存储位置')
    parser.add_argument('--logist_classes', default='logist', help='回归问题还是分类问题， classes, logist')
    parser.add_argument('--act_fun', default=False, help='最后一层是否用激活函数')
    parser.add_argument('--dropout', default=0.0, help='')

    #图设置
    parser.add_argument('--GNN_Use', type=bool, default=True, help='使用图神经网络')
    parser.add_argument('--Generate_adj', type=bool, default=False, help='计算邻接矩阵')
    parser.add_argument('--Gene_ways', default='cosine_similarity', help='生成邻接矩阵的计算方法，余弦距离：cosine_similarity')
    parser.add_argument('--lim_adj', default=False, help='计算邻接矩阵的极限稳态值')
    parser.add_argument('--pool', default=True, help='图池化')
    parser.add_argument('--pool_rate', default='0.8', help='图池化率')

    #TimeBlock
    parser.add_argument('--inchannel', type=int, default=1024, help='输入通道数')
    parser.add_argument('--outchannel', type=int, default=256, help='输出通道数')
    parser.add_argument('--spatialchannel', type=int, default=128, help='空间卷积通道数')

    '''d_input = args.d_input, d_model = args.hidden_dim,
    d_output = args.output_dim,
    q = args.q,
    v = args.v,
    h = args.h,
    N = args.N,
    attention_size = None,
    dropout = args.transformer_dropout,'''
    #temporal_transformer
    parser.add_argument('--d_input', type=int, default=1024, help='输入维度')
    parser.add_argument('--d_model', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--d_output', type=int, default=1024, help='输出维度')
    parser.add_argument('--q', type=int, default=128, help='查询向量的大小')
    parser.add_argument('--v', type=int, default=128, help='值向量的大小')
    parser.add_argument('--h', type=int, default=8, help='头数')
    parser.add_argument('--N', type=int, default=1, help='编码器和解码器堆叠的层数')
    parser.add_argument('--transformer_dropout', default=0.2)

    #GraphTransformer参数设置
    parser.add_argument('--g_hidden_dim', type=int, default=1024, help='GraphTransformer的隐藏层特征维度,与输入特征维度同')
    parser.add_argument('--g_n_heads', type=int, default=8, help='GraphTransformer中多头注意力的头数')
    parser.add_argument('--g_out_dim', type=int, default=128, help='GraphTransformer的输出特征维度')
    parser.add_argument('--g_dropout', default=0.2, help='')
    parser.add_argument('--g_n_layers', type=int, default=6, help='GraphTransformer的层数')

    #结果保存
    parser.add_argument('--save_embedding', default='save_result', help='二维可视化存储位置')

    return parser.parse_args()
# print(Get_args())