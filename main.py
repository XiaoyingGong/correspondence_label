import matplotlib.pyplot as plt
import cv2
import numpy as np

from feature_matching import sift_matching
from label.label import  Label
# 主类，汇总各个类的功能

# 图像路径
img1_path = "./img/1.png"
img2_path = "./img/2.png"
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
h_img = np.hstack((img1, img2))
#sift的阈值
sift_threshold = 0.8

# 通过sift进行预匹配
pre_matches1, pre_matches2, kp1, kp2, good_match = sift_matching.get_matches(img1_path, img2_path, sift_threshold)
index1 = good_match[0].queryIdx
index2 = good_match[0].trainIdx

# 以下为测试1：从头开始的标注
pre_matches1 = pre_matches1[:, :10]#只取前10个，方便测试
pre_matches2 = pre_matches2[:, :10]
label = Label(img1_path, img2_path, pre_matches1, pre_matches2)
label.start_label()

# 以下为测试2 读取保存的文件来进行标注
# pre_matches1 = pre_matches1[:, :10]#只取前10个，方便测试
# pre_matches2 = pre_matches2[:, :10]
# label = Label(img1_path, img2_path, pre_matches1, pre_matches2, './label_result/1.png_2.png.npz')
# label.start_label()