import random
from utils.shot_utils import Get_shot_score, Get_shot_classes
import h5py


def load_data(args):
    print("‘\n------------>读取数据<--------------")
    keys, feature_dic, labels_score_dic, labels_classes_dic, cps_dic,shot_score_dic, shot_classes_dic = Read_data(args.Dataset_root, args.dataset)

        split_key = data_split(keys, args.split_seed)
        return split_key, feature_dic, labels_score_dic,  positions_dic, user_summary_dic, shot_score_dic, shot_classes_dic

def Read_data(root, dataset):
    root = root + 'eccv16_dataset_' + dataset + '_google_pool5.h5'
    f = h5py.File(root, 'r')
    '''
    change_points  每个镜头的起始位置
    features  特征（帧数（抽帧）*1024）
    gtscore  分数（帧数（抽帧）*1）
    gtsummary  0-1标签（帧数（抽帧）*1）
    n_frame_per_seg  每段视频的帧数
    n_frames  帧数
    picks  抽帧的位置
    user_summary  用户评价
    video_name  视频编号
    '''
    feature_dic = {}
    labels_score_dic = {}
    labels_classes_dic = {}
    cps_dic = {}
    shot_score_dic = {}
    shot_classes_dic = {}
    keys = []

    for key in f.keys(): #key为视频编号
        keys.append(key)
        feature_dic[key] = f[key]['features'][...] #特征
        labels_score_dic[key] = f[key]['gtscore'][...] #分数标签 #用户评价
        shot_score_dic[key] = Get_shot_score(labels_score_dic[key], cps_dic[key], num_frames_dic[key], positions_dic[key])
        shot_classes_dic[key] = Get_shot_classes(labels_score_dic[key], cps_dic[key], num_frames_dic[key], nfps_dic[key], positions_dic[key])

    return keys, feature_dic, labels_score_dic, labels_classes_dic, cps_dic, num_frames_dic, nfps_dic, positions_dic, user_summary_dic, shot_score_dic, shot_classes_dic

def Read_OVP_data(root, dataset):
    root = root + 'eccv16_dataset_' + dataset + '_google_pool5.h5'
    f = h5py.File(root, 'r')
    feature_dic = {}
    shot_score_dic = {}
    shot_classes_dic = {}
    for key in f.keys():
        feature_dic[key] = f[key]['features'][...]  # 特征
        shot_score_dic[key] = f[key]['gtscore'][...]  # 分数标签
        shot_classes_dic[key] = f[key]['gtsummary'][...]  # 0-1标签
    keys = []
    return root

def data_split(key, seed):
    random.seed(seed)
    random.shuffle(key)
    long = int(len(key) / 5)
    split_key = []

    split_1 = key[:long]
    split_2 = key[long:8*long]


    names = locals()
    for i in range(1, 6):
        split_key.append(names['split_' + str(i)])
    return split_key
