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
img_path_r = "./img/train_set_9_20/9_r.jpg"
img_path_s = "./img/train_set_9_20/9_s.jpg"

constants.SIFT_THRESHOLD = 0.3
pre_matches_s_strict, pre_matches_r_strict, des_s_strict, des_r_strict, img_s, img_r, resize_h, resize_w = \
    pre_matching.get_pre_matches(img_path_s, img_path_r, False)

# 生成正样结果
pre_matches1 = np.transpose(pre_matches_r_strict)
pre_matches2 = np.transpose(pre_matches_s_strict)
matching_result = np.ones(len(pre_matches1[0]))
result = np.vstack((pre_matches2, pre_matches1, matching_result))



# filename = img_path_s.split("/")[len(img_path_s.split("/")) - 1] + \
#            img_path_r.split("/")[len(img_path_r.split("/")) - 1] + "_"
#
# np.savez('./label_result/' + filename + "_0.3_i", correspondence_label=result)