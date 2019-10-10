# author: 龚潇颖(Xiaoying Gong)
# date： 2019/9/23 10:26  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
from utils import constants
from preprocessing import pre_matching

import numpy as np
# 图像路径
img_path_r = "./img/exp_img/1_r.jpg"
img_path_s = "./img/exp_img/1_s.jpg"

constants.SIFT_THRESHOLD = 0.4
pre_matches_s_strict, pre_matches_r_strict, des_s_strict, des_r_strict, img_s, img_r, resize_h, resize_w = \
    pre_matching.get_pre_matches(img_path_s, img_path_r, False)

# 生成正样结果
pre_matches1 = np.transpose(pre_matches_r_strict)
pre_matches2 = np.transpose(pre_matches_s_strict)
matching_result = np.ones(len(pre_matches1[0]))
result = np.vstack((pre_matches2, pre_matches1, matching_result))

print(len(result[0]))
filename = img_path_s.split("/")[len(img_path_s.split("/")) - 1] + \
           img_path_r.split("/")[len(img_path_r.split("/")) - 1] + "_"

np.savez('./label_result/' + filename + "_0.3_i", correspondence_label=result)