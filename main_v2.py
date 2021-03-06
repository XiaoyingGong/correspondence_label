# author: 龚潇颖(Xiaoying Gong)
# date： 2019/9/10 10:21  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：

import matplotlib.pyplot as plt
import cv2
import numpy as np

from preprocessing import pre_matching
from label.label import Label
from utils import constants
from utils.utils import utils

# 取点的策略 每份10个 取30份 每张图300个点

# 图像路径
img_path_r = "./img/train_image_set/1_r.png"
img_path_s = "./img/train_image_set/1_s.png"

# 以下提出的点集都是sensed points 1对多 reference points
constants.IMG_RESIZE_RATIO = 1.5

# strict threshold
constants.SIFT_THRESHOLD = 0.4
pre_matches_s_strict, pre_matches_r_strict, des_s_strict, des_r_strict, img_s, img_r, resize_h, resize_w =\
    pre_matching.get_pre_matches(img_path_s, img_path_r, False)
print(len(pre_matches_s_strict))

# loose threshold
constants.SIFT_THRESHOLD = 0.8
pre_matches_s_loose, pre_matches_r_loose, des_s_loose, des_r_loose, img_s, img_r, resize_h, resize_w =\
    pre_matching.get_pre_matches(img_path_s, img_path_r, False)

_, repeated_index = utils.set_difference_2d(pre_matches_s_loose, pre_matches_s_strict)
non_repeated_index = np.setdiff1d(np.arange(0, len(pre_matches_s_loose)), repeated_index)

pre_matches1 = np.transpose(pre_matches_r_loose[non_repeated_index])
pre_matches2 = np.transpose(pre_matches_s_loose[non_repeated_index])

# 以下为测试1：从头开始的标注
label = Label(img_path_r, img_path_s, img_r, img_s, pre_matches1, pre_matches2)
label.start_label()

# 以下为测试2 读取保存的文件来进行标注
# label = Label(img_path_r, img_path_s, img_r, img_s, pre_matches1, pre_matches2, './label_result/1_r.png_1_s.png_1.0.npz')
# label.start_label()
