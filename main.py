import matplotlib.pyplot as plt
import cv2
import numpy as np

from feature_matching import sift_matching
from label.label import Label
from utils import constants

# 取点的策略 每份10个 取30份 每张图300个点

# 图像路径
img1_path = "./img/9_r.jpg"
img2_path = "./img/9_s.jpg"
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
# resize
img1 = cv2.resize(img1, (800, 600))
img2 = cv2.resize(img2, (800, 600))
h_img = np.hstack((img1, img2))
#sift的阈值, 推荐设置高一点以增加负样
sift_threshold = constants.SIFT_THRESHOLD

# 通过sift进行预匹配
pre_matches1, pre_matches2, des1, des2, good_match = sift_matching.get_matches(img1, img2, sift_threshold)
len_1 = len(pre_matches1)

# 因为匹配里面也有可能存在一对多的情况所以，这里进行一次将一对多的情况剔除
pre_matches1, index1 = np.unique(pre_matches1, return_index=True, axis=0)
pre_matches2 = pre_matches2[index1]
pre_matches2, index2 = np.unique(pre_matches2, return_index=True, axis=0)
pre_matches1 = pre_matches1[index2]


pre_matches1 = np.transpose(pre_matches1)
pre_matches2 = np.transpose(pre_matches2)
# 以下为测试1：从头开始的标注
label = Label(img1, img2, pre_matches1, pre_matches2)
label.start_label()

# 以下为测试2 读取保存的文件来进行标注
# label = Label(img1, img2, pre_matches1, pre_matches2, './label_result/3_r.png_3_s.png_0.9.npz')
# label.start_label()



