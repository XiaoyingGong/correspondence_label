import matplotlib.pyplot as plt
import cv2
import numpy as np

from feature_matching import sift_matching
from label.label import Label
from utils import constants


# 这个类用于部分的标记 即一张图提了点 只标记300个点
def get_partial_points(pre_matches):
    pre_matches_len = len(pre_matches)
    split_len = int(pre_matches_len / 30)
    index = np.array([], dtype=np.int)
    split_index = np.linspace(0, 9, 10, dtype=np.int)
    for i in range(30):
        index = np.append(index, split_index)
        split_index += split_len
    return pre_matches[index], index


img_1_path = "./img/4_r.png"
img_2_path = "./img/4_s.png"

img_1 = cv2.imread(img_1_path)
img_2 = cv2.imread(img_2_path)
# resize
img1 = cv2.resize(img_1, (800, 600))
img2 = cv2.resize(img_2, (800, 600))
h_img = np.hstack((img1, img2))
#sift的阈值, 推荐设置高一点以增加负样
sift_threshold = constants.SIFT_THRESHOLD
# 通过sift进行预匹配
pre_matches1, pre_matches2, des1, des2, good_match = sift_matching.get_matches(img1, img2, sift_threshold)

# 因为匹配里面也有可能存在一对多的情况所以，这里进行一次将一对多的情况剔除
pre_matches1, index1 = np.unique(pre_matches1, return_index=True, axis=0)
pre_matches2 = pre_matches2[index1]
pre_matches2, index2 = np.unique(pre_matches2, return_index=True, axis=0)
pre_matches1 = pre_matches1[index2]

pre_matches1_partial, _ = get_partial_points(pre_matches1)
pre_matches2_partial, _ = get_partial_points(pre_matches2)

pre_matches1_partial = np.transpose(pre_matches1_partial)
pre_matches2_partial = np.transpose(pre_matches2_partial)

# 以下为测试1：从头开始的标注
label = Label(img1, img2, pre_matches1_partial, pre_matches2_partial)
label.start_label()

# 以下为测试2 读取保存的文件来进行标注
# label = Label(img1, img2, pre_matches1_partial, pre_matches2_partial, './label_result/3_r.png_3_s.png_0.9.npz')
# label.start_label()










