import numpy as np
import math
import torch

'''
用动态规划解决0/1背包问题
时间复杂度o(nW)， n是项的个数，W是背包容量
knapsack_dp(values, weights, n_items, capacity, return_all=False)
输入参数：
    1. values: int 或 float类型的数字列表， 指定的项的值
    2. weights: 指定项目权重的整型数字列表
    3. n_items: 表示项数
    4. capacity: 一个表示背包容量的整型数字
    5. return_all: 是否返回所有信息，defaulty为False(可选)
返回值：
    1. picks: 存储所选项目位置的数字列表
    2. max_val: 最大值
'''

def check_inputs(values, weights, n_items, capacity):
    #check variable type
    # assert(isinstance(values, list))
    assert(isinstance(weights, list))
    assert(isinstance(n_items, int))
    assert(isinstance(capacity, int))

    #check value type
    assert(all(isinstance(val, int) or isinstance(val, float) for val in values))
    assert(all(isinstance(val, int) for val in weights))

    #check validity of value
    assert(all(val >= 0 for val in weights))
    assert(n_items > 0)
    assert(capacity > 0)

def knapsack_dp(values, weights, n_items, capacity, return_all=False):
    check_inputs(values, weights, n_items, capacity)

    table = np.zeros((n_items+1, capacity+1), dtype=np.float32)
    keep = np.zeros((n_items+1, capacity+1), dtype=np.float32)

    for i in range(1, n_items+1):
        for w in range(0, capacity+1):
            wi = weights[i-1]  #当前项的权重
            vi = values[i-1]  #当前项的值
            if (wi <= w) and (vi + table[i-1, w-wi] > table[i-1, w]):
                table[i, w] = vi + table[i-1, w-wi]
                keep[i, w] = 1
            else:
                table[i, w] = table[i-1, w]

    picks = []
    K = capacity

    for i in range(n_items, 0, -1):
        if keep[i, K] == 1:
            picks.append(i)
            K -= weights[i-1]

#镜头排序
    picks.sort()
    picks = [x-1 for x in picks] #改变，让下标从0开始

    if return_all:
        max_val = table[n_items, capacity]
        return picks, max_val
    return picks


def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    '''
    对比机器生成的摘要和用户注释的摘要，基于关键镜头的
    :param machine_summary:
    :param user_summary:
    :param eval_metric: {'avg', 'max'}
    :return:
    '''
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    n_users, n_frames = user_summary.shape

    #binarization 二值化
    machine_summary[machine_summary > 0] = 1
    user_summary[user_summary > 0] = 1

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])

    f_scores = []
    prec_arr = []
    rec_arr = []

    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx, :]
        #计算有多少个真的正样本
        overlap_duration = (machine_summary * gt_summary).sum()
        #精准率和召回率
        precision = overlap_duration / (machine_summary.sum() + 1e-8)
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        f_scores.append(f_score)
        prec_arr.append(precision)
        rec_arr.append(recall)
    if eval_metric == 'avg':
        final_f_score = np.mean(f_scores)
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)
    elif eval_metric == 'max':
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]
    return final_f_score

def Get_shot_score(ypred, cps, n_frames, positions):
    ypred = ypred
    cps = cps
    n_frames = n_frames
    positions = positions
    n_segs = cps.shape[0] #镜头个数

    frame_scores = np.zeros((n_frames), dtype=np.float32)

    if eval_metric == 'avg':
        final_f_score = np.mean(f_scores)
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)
    elif eval_metric == 'max':
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]
    seg_score = []

    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx, 0]), int(cps[seg_idx, 1] + 1)

        scores = frame_scores[start:end]

        seg_score.append(float(scores.mean()))
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i + 1]
        frame_scores[pos_left:pos_right] = y_pred[i]

    return seg_score

def Get_shot_classes(y_pred, cps, n_frames, nfps, positions):
    n_segs = cps.shape[0] #镜头个数
    proportion = 0.15

    frame_scores = np.zeros((n_frames), dtype=np.float32)
    shot_label = []
    for seg_idx in range(n_segs):
        if seg_idx in picks:
            shot_label.append(1)
        else:
            shot_label.append(0)
    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx, 0]), int(cps[seg_idx, 1] + 1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))
    #0-1背包选择问题
    picks = knapsack_dp(seg_score, nfps, n_segs, limits)


    return shot_label

def SpotScore_FrameScore(seg_score, cps, n_frames, nfps, proportion=0.15):
    n_segs = cps.shape[0]  #镜头个数

    limits = int(math.floor(n_frames * proportion))
    #0-1背包选择问题
    picks = knapsack_dp(seg_score, nfps, n_segs, limits)

    summary = np.zeros((1), dtype=np.float32)  #这个元素应该被删除
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    summary = np.delete(summary, 0)
    return summary, picks

def Represent_Frame(feature_dic, cps_dic, positions_dic):
    represent_frame = {}

    for video_index in feature_dic.keys():
        feature = feature_dic[video_index]
        cps = cps_dic[video_index]
        positions = positions_dic[video_index]



        while i <= len(positions_dic):
            if i == len(positions):
                last_spot = True
            if last_spot == False and positions[i] <= num_cps:
                shot.append(feature[i, :])
                i += 1
            else:
                num_shot = len(cps) #镜头数量
                num_cps = cps[len(cps) - num_shot].max() #取当前镜头内的最大帧数
                shot = []
                i = 0
                first_spot = True
                last_spot = False
                if shot == []:
                    pass
                else:
                    frames = torch.Tensor(np.array(shot))

                if first_spot:
                    pre_frame = present_frame(frames)
                    first_spot = False
                else:
                    pre_frame = torch.cat([pre_frame, pre_frame(frames)], dim=0)

                num_shot -= 1
                if num_shot == 0:
                    break
                num_cps = cps[len(cps) - num_shot].max()
                shot = []
        represent_frame[video_index] = pre_frame

    return represent_frame

def present_frame(frames):
    smi = []

    for i in range(frames.shape[0]):
        temp_smi = 0
        values, index = smi.topk(k=1, largest=False)
        pre_frame = frames[index, :]
        for j in range(frames.shape[0]):
            temp_smi += torch.norm(frames[i, :] - frames[j, :], p=2, dim=0)
        temp_smi = temp_smi / frames.shape[0]
        smi.append(torch.exp(torch.mul(temp_smi, -1)))

    smi = torch.Tensor(smi)


    return pre_frame

def split_shot(feature_dic, cps_dic, positions_dic):
    shot_feature = {}
    for video_index in feature_dic.keys():
        feature = feature_dic[video_index]
        cps = cps_dic[video_index]
        positions = positions_dic[video_index]

        num_shot = len(cps) #镜头数量
        num_cps = cps[len(cps) - num_shot].max() #取当前镜头内的最大帧数
        shot = []
        i = 0
        first_spot = True
        last_spot = False

        while i <= len(positions_dic):
            if i == len(positions):
                last_spot = True
            if last_spot == False and positions[i] <= num_cps:
                shot.append(feature[i, :])
                i += 1
            else:

                shot = []
    if shot == []:
        pass
    else:
    frames = torch.Tensor(np.array(shot))

    if first_spot:
        pre_frame = frames
        first_spot = False
    else:
                    pre_frame = torch.cat([pre_frame, frames], dim=0)

                num_shot -= 1
                if num_shot == 0:
                    break
                num_cps = cps[len(cps) - num_shot].max()
        shot_feature[video_index] = pre_frame
    return shot_feature