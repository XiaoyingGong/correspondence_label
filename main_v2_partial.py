# author: 龚潇颖(Xiaoying Gong)
# date： 2019/9/18 20:43  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
from preprocessing import pre_matching
from label.label_show_points_pass import Label
from utils import constants
from utils.utils import utils
# 图：1-11和23
import numpy as np
# 图像路径
img_path_r = "./img/train_set_9_20/1_r.jpg"
img_path_s = "./img/train_set_9_20/1_s.jpg"

# 这个类用于部分的标记 即一张图提了点 只标记300个点
def get_partial_points(pre_matches):
    pre_matches_len = len(pre_matches)
    split_len = int(pre_matches_len / 35)
    index = np.array([], dtype=np.int)
    split_index = np.linspace(0, 9, 10, dtype=np.int)
    for i in range(35):
        index = np.append(index, split_index)
        split_index += split_len
    return pre_matches[index], index

# test
def test():
    constants.SIFT_THRESHOLD = 1.0
    pre_matches_s_strict, pre_matches_r_strict, des_s_strict, des_r_strict, img_s, img_r, resize_h, resize_w =\
        pre_matching.get_pre_matches(img_path_s, img_path_r, False)

    pre_matches_s_strict = np.transpose(pre_matches_s_strict)
    pre_matches_r_strict = np.transpose(pre_matches_r_strict)

    # 从头开始
    label = Label(img_path_s, img_path_r, img_s, img_r, pre_matches_s_strict, pre_matches_r_strict)
    label.start_label()

    # 以下为测试2 读取保存的文件来进行标注
    # label = Label(img_path_s, img_path_r, img_s, img_r, pre_matches_s_strict, pre_matches_r_strict, './label_result/1_s.png_1_r.png_0.3.npz')
    # label.start_label()

# 用于筛选内点
def inlier_choose():
    constants.SIFT_THRESHOLD = 0.3
    pre_matches_s_strict, pre_matches_r_strict, des_s_strict, des_r_strict, img_s, img_r, resize_h, resize_w =\
        pre_matching.get_pre_matches(img_path_s, img_path_r, False)

    pre_matches_s_strict = np.transpose(pre_matches_s_strict)
    pre_matches_r_strict = np.transpose(pre_matches_r_strict)

    # 从头开始
    label = Label(img_path_s, img_path_r, img_s, img_r, pre_matches_s_strict, pre_matches_r_strict)
    label.start_label()

    # 以下为测试2 读取保存的文件来进行标注
    # label = Label(img_path_s, img_path_r, img_s, img_r, pre_matches_s_strict, pre_matches_r_strict, './label_result/1_s.png_1_r.png_0.3.npz')
    # label.start_label()


# 用于筛选内点
def potential_inlier_choose():
    constants.SIFT_THRESHOLD = 0.3
    pre_matches_s_strict, pre_matches_r_strict, des_s_strict, des_r_strict, img_s, img_r, resize_h, resize_w =\
        pre_matching.get_pre_matches(img_path_s, img_path_r, False)

    constants.SIFT_THRESHOLD = 1.0
    pre_matches_s_loose, pre_matches_r_loose, des_s_loose, des_r_loose, img_s, img_r, resize_h, resize_w = \
        pre_matching.get_pre_matches(img_path_s, img_path_r, True)

    _, repeated_index= utils.set_difference_2d(pre_matches_s_loose, pre_matches_s_strict)
    non_repeated_index = np.setdiff1d(np.arange(0, len(pre_matches_s_loose)), repeated_index)

    pre_matches1 = pre_matches_r_loose[non_repeated_index]
    pre_matches2 = pre_matches_s_loose[non_repeated_index]

    pre_matches1, _ = get_partial_points(pre_matches1)
    pre_matches2, _ = get_partial_points(pre_matches2)

    # 从头开始
    label = Label(img_path_s, img_path_r, img_s, img_r, np.transpose(pre_matches2), np.transpose(pre_matches1))
    label.start_label()

    # 以下为测试2 读取保存的文件来进行标注
    # label = Label(img_path_s, img_path_r, img_s, img_r, np.transpose(pre_matches2), np.transpose(pre_matches1), './label_result/1_s.jpg_1_r.jpg_1.0.npz')
    # label.start_label()

# inlier_choose()
potential_inlier_choose()
# test()



