'''
    预训练transformer
'''

import copy
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools
from utils.data_utils import load_data
from utils.shot_utils import *

import random
import tqdm
import math
from config import Get_args
from models.LTT import temporal_transformer
args = Get_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Object():
    def __init__(self, args, model=None, k=None):
        self.args = args
        self.k = k
        self.device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

        self.split_key, self.feature_dic, self.labels_score_dic, self.labels_classes_dic, \
        self.cps_dic, self.num_frames_dic, self.nfps_dic, self.positions_dic, \
        self.user_summary_dic, self.shot_score_dic, self.shot_classes_dic = load_data(self.args)

        #k从1取到5，将k-1的数据集作为验证集
        self.test_keys = self.split_key[k - 1]
        num = [1, 2, 3, 4, 0]
        num.remove(k-1)
        self.train_keys = []
        for i in num:
            self.train_keys += self.split_key[i]
        #定义模型
        if model == None:
            # self.model = mlptomlp().to(self.device)
            # self.model0 = Transf().to(self.device)
            # self.model = STGCN(num_features=args.inchannel, args=args).to(self.device)
            self.model = temporal_transformer(d_input=args.d_input, d_model=args.d_model,
                 d_output=args.d_output,
                 q=args.q,
                 v=args.v,
                 h=args.h,
                 N=args.N,
                 attention_size = None,
                 dropout = args.transformer_dropout,
                 chunk_mode = 'window',
                 pe = "regular",
                 pe_period = 6).to(self.device)
            # self.model = Transformer_STGCN()
            # self.model1 = GCN(args).to(self.device)
        else:
            pass

        #定义优化器
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr)
        if args.lr_step > 0:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size=args.lr_step,
                                                          gamma=float(args.lr_gamma))
        #定义损失函数
        if args.logist_classes == 'logist':
            self.criterion = nn.MSELoss().to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.train_loss = []
        self.max_f1 = -1
        self.val_loss = []
        self.pre_f1 = []
        self.real_f1 = []
        self.great_model = None
        self.video_all_logist = dict([(k, []) for k in self.test_keys])

    def train(self, args):
        print("---------------开始训练-------------")

        #将模型设置为训练模式
        self.model.train()

        #计算模型的参数数量
        tot_params = sum([np.prod(p.size()) for p in self.model.parameters()])
        print("\n Total number of parameters: {}".format(tot_params))

        #设置视频摘要的任务为回归问题还是分类问题
        if args.logist_classes == 'logist':
            feature_dic = self.feature_dic
            cps_dic = self.cps_dic
            positions_dic = self.positions_dic
            colors = self.shot_score_dic
            if args.shot_mode == False:
                labels_dic = self.labels_score_dic
            else:
                feature_dic = torch.load('data/{}_represent_frame.pt'.format(args.dataset))
                labels_dic = self.shot_score_dic
        else:
            feature_dic = self.feature_dic
            cps_dic = self.cps_dic
            positions_dic = self.positions_dic
            colors = self.shot_classes_dic
            if args.shot_mode == False:
                labels_dic = self.labels_classes_dic
            else:
                feature_dic = torch.load('data/{}_represent_frame.pt'.format(args.dataset))
                labels_dic = self.shot_score_dic

        if args.shot_mode:
            name = 'shot'
        else:
            name = 'frame'

        #进行多轮的训练
        for epoch in range(args.epochs):
            sum_loss = []
            for video_index in self.train_keys:
                fragment_feature = []
                fragment_label = []
                feature = torch.Tensor(feature_dic[video_index]).to(self.device)  #[332, 1024]
                labels = torch.Tensor(labels_dic[video_index]).to(self.device)  #[332]
                if (labels.shape[0] % 10):
                    patch = torch.zeros(10 - (labels.shape[0] % 10)).to(self.device)
                    labels = torch.cat((labels, patch), dim=0)
                n_feature = math.ceil(feature.shape[0]/10)
                n_labels = math.ceil(labels.shape[0]/10)
                M = torch.chunk(feature, n_feature, dim=0)
                Y = torch.chunk(labels, n_labels, dim=0)
                for index, fragment in enumerate(M):
                    if fragment.shape[0] != 10:
                        patch = torch.zeros((10-fragment.shape[0]), 1024).to(self.device)
                        fragment = torch.cat((fragment, patch), 0).to(self.device)
                    fragment_feature.append(fragment)
                for index, y in enumerate(Y):
                    if y.shape[0] != 10:
                        patch = torch.zeros((10-y.shape[0])).to(self.device)
                        y = torch.cat((y, patch), 0).to(self.device)
                    fragment_label.append(y)
                video_feature = torch.stack(fragment_feature).to(self.device)
                video_feature = video_feature.unsqueeze(0).to(self.device)
                # X = X.permute(0, 3, 1, 2)
                video_feature = video_feature.permute(0, 2, 1, 3)
                video_labels = torch.stack(fragment_label).to(self.device)
                video_labels = video_labels.unsqueeze(0)
                # 是否设置生成邻接矩阵
                if args.Generate_adj == True:
                    from utils.GNN_utils import Generate_fragment_Adjacency
                    Adj_list = Generate_fragment_Adjacency(fragment_feature=fragment_feature, args=args)
                    torch.save(Adj_list, 'data/adj/{}_{}_Adjacency_fragment_cosine_similarity'.format(args.dataset, video_index))
                else:
                    Adj_list = torch.load('data/adj/{}_{}_Adjacency_fragment_cosine_similarity'.format(args.dataset, video_index))
                for index, adj in enumerate(Adj_list):
                    # output, h = self.model(video_feature, adj.to(self.device), 10)
                    y_pred,x1,x2,x_bar = self.model(video_feature)
                # output, h = self.model(feature)
                if args.logist_classes == 'logist':
                    loss = self.criterion(y_pred.view(-1), labels) + 0.01 * self.criterion(x_bar, video_feature)
                else:
                    # loss = self.criterion(h, labels.long())
                    pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if args.lr_step > 0:
                    self.lr_scheduler.step()
                sum_loss.append(loss.item())
            print("\n-----The K{}_epoch {} train_loss is {}".format(self.k, epoch, np.mean(sum_loss)))

            self.val(args)
            self.train_loss.append(np.mean(sum_loss))

    def val(self, args):
        print("====================> Test")
        with torch.no_grad():
            self.model.eval()

            if args.logist_classes == 'logist':
                feature_dic = self.feature_dic
                cps_dic = self.cps_dic
                n_frames_dic = self.num_frames_dic
                nfps_dic = self.nfps_dic
                user_summary_dic = self.user_summary_dic
                positions_dic = self.positions_dic
                labels_dic = self.labels_score_dic
            else:
                feature_dic = self.feature_dic
                cps_dic = self.cps_dic
                n_frames_dic = self.num_frames_dic
                nfps_dic = self.nfps_dic
                user_summary_dic = self.user_summary_dic
                positions_dic = self.positions_dic
                labels_dic = self.labels_classes_dic

            sum_loss = []
            real_f1 = []
            pre_f1 = []
            pre_score = {}
            real_score = {}

            for video_index in self.test_keys:
                fragment_feature = []
                fragment_label = []
                feature = torch.Tensor(feature_dic[video_index]).to(self.device)  # [332, 1024]
                labels = torch.Tensor(labels_dic[video_index]).to(self.device)  # [332]
                if (labels.shape[0] % 10):
                    patch = torch.zeros(10 - (labels.shape[0] % 10)).to(self.device)
                    labels = torch.cat((labels, patch), dim=0)
                n_feature = math.ceil(feature.shape[0] / 10)
                n_labels = math.ceil(labels.shape[0] / 10)
                M = torch.chunk(feature, n_feature, dim=0)
                Y = torch.chunk(labels, n_labels, dim=0)
                for index, fragment in enumerate(M):
                    if fragment.shape[0] != 10:
                        patch = torch.zeros((10 - fragment.shape[0]), 1024).to(self.device)
                        fragment = torch.cat((fragment, patch), 0).to(self.device)
                    fragment_feature.append(fragment)
                for index, y in enumerate(Y):
                    if y.shape[0] != 10:
                        patch = torch.zeros((10 - y.shape[0])).to(self.device)
                        y = torch.cat((y, patch), 0).to(self.device)
                    fragment_label.append(y)
                video_feature = torch.stack(fragment_feature).to(self.device)
                video_feature = video_feature.unsqueeze(0)
                # X = X.permute(0, 3, 1, 2)
                video_feature = video_feature.permute(0, 2, 1, 3)
                video_labels = torch.stack(fragment_label).to(self.device)
                video_labels = video_labels.unsqueeze(0)
                # 是否设置生成邻接矩阵
                if args.Generate_adj == True:
                    from utils.GNN_utils import Generate_fragment_Adjacency
                    Adj_list = Generate_fragment_Adjacency(fragment_feature=fragment_feature, args=args)
                    torch.save(Adj_list,
                               'data/adj/{}_{}_Adjacency_fragment_cosine_similarity'.format(args.dataset, video_index))
                else:
                    Adj_list = torch.load(
                        'data/adj/{}_{}_Adjacency_fragment_cosine_similarity'.format(args.dataset, video_index))
                for index, adj in enumerate(Adj_list):
                    y_pred,x1,x2,x_bar = self.model(video_feature)
                # y_pred = y_pred.view(len(video_feature), -1)
                pre_spot_score = y_pred.view(-1)

                if args.logist_classes == 'logist':
                    loss = self.criterion(pre_spot_score, labels) + 0.01 * self.criterion(x_bar, video_feature)
                else:
                    # loss = self.criterion(h, labels.long())
                    pass
                sum_loss.append(loss.item())
                if args.shot_mode == False:
                    machine_summary = generate_summary(pre_spot_score.cpu().detach().numpy(), cps_dic[video_index], n_frames_dic[video_index], nfps_dic[video_index], positions_dic[video_index])
                    pre_fm, pre_prec, pre_rec = evaluate_summary(machine_summary, user_summary_dic[video_index], eval_metric=args.generate_mode)
                    pre_f1.append(pre_fm)
                    pre_score[video_index] = pre_spot_score.cpu().detach().numpy().tolist()

                    real_user_score = generate_summary(labels_dic[video_index], cps_dic[video_index], n_frames_dic[video_index], nfps_dic[video_index], positions_dic[video_index])
                    real_fm, real_prec, real_rec = evaluate_summary(real_user_score, user_summary_dic[video_index], eval_metric=args.generate_mode)
                    real_f1.append(real_fm)
                    real_score[video_index] = labels_dic[video_index]
                else:
                    pre_user_score1 = SpotScore_FrameScore(pre_spot_score.cpu().detach().numpy().tolist(), cps_dic[video_index], n_frames_dic[video_index], nfps_dic[video_index])
                    pre_fm, pre_prec, pre_rec = evaluate_summary(pre_user_score1, user_summary_dic[video_index], eval_metric=args.generate_mode)
                    pre_f1.append(pre_fm)
                    pre_score[video_index] = pre_spot_score.cpu().detach().numpy().tolist()

                    real_user_score = SpotScore_FrameScore(labels_dic[video_index], cps_dic[video_index], n_frames_dic[video_index], nfps_dic[video_index])
                    real_fm, real_prec, real_rec = evaluate_summary(real_user_score, user_summary_dic[video_index], eval_metric=args.generate_mode)
                    real_f1.append(real_fm)
                    real_score[video_index] = labels_dic[video_index]
            if np.mean(pre_f1) >= self.max_f1:
                self.max_f1 = np.mean(pre_f1)
                self.real_spot_score = real_score
                self.pre_spot_score = pre_score
                self.great_model = copy.deepcopy(self.model)
            self.val_loss.append(np.mean(sum_loss))
            print("the real_f1 is {}, the pre_f1 is {}".format(np.mean(real_f1), np.mean(pre_f1)))
            print("最好的f1为{}".format(self.max_f1))

            self.pre_f1.append(np.mean(pre_f1))
            self.real_f1.append(np.mean(real_f1))
        return

if __name__ == "__main__":
    print("-----------------------------------------------------------------------------------")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    t1 = time.time()
    args = Get_args()
    data_sets = ['summe', 'tvsum']
    for data_set in data_sets:
        print(f"dataset is {data_set}")
        args.dataset = data_set
        if args.dataset == 'tvsum':
            args.generate_mode = 'avg'
        else:
            args.generate_mode = 'max'
        setup_seed(args.model_seed)

        pre_f1 = [0, 0, 0, 0, 0]
        real_f1 = []
        spearman_rou = []
        kendall_tao = []
        if args.logist_classes != 'logist':
            label_f1 = []
            real_label_f1 = []
        for k in range(1, 6):
            print('\n')
            print("==============================This is k{}=======================".format(k))
            object = Object(args=args, k=k)

            object.train(args=args)
            object.val(args=args)


            if object.max_f1 >= pre_f1[k - 1]:
                pre_f1[k - 1] = object.max_f1

            real_f1.append(object.real_f1[0])

            if args.shot_mode == True:
                name = 'shot'
            else:
                name = 'frame'
            # torch.save(object.great_model,
            #            '{}/{}_{}_model_{}_k{}.pth'.format(args.save_dir, args.dataset,
            #                                               object.model.__class__.__name__,
            #                                               name, k))
            torch.save(object.great_model.state_dict(), args.save_dir)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            del object
        print("-----------------------------------------------")
        print("-----------------------------------------------")
        for i in range(len(real_f1)):
            print("This is K{}".format(i + 1))
            print("the real_f1 is {}, the pre_f1 is {}".format(real_f1[i], pre_f1[i]))
        print("------------------------------------------------")
        print("------------------------------------------------")
        print("the avg real_f1 is {}, the avg pre_f1 is {}".format(np.mean(real_f1), np.mean(pre_f1)))
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))












