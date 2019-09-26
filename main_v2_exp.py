# author: 龚潇颖(Xiaoying Gong)
# date： 2019/9/25 16:04  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
from utils import constants
from utils.utils import utils
from preprocessing import pre_matching
from label.label_show_points_pass import Label

import numpy as np
# 图像路径
img_path_r = "./img/exp_img/5_l.jpg"
img_path_s = "./img/exp_img/5_r.jpg"

constants.SIFT_THRESHOLD = 1.0
pre_matches_s_loose, pre_matches_r_loose, des_s_loose, des_r_loose, img_s, img_r, resize_h, resize_w = \
    pre_matching.get_pre_matches(img_path_s, img_path_r, False)


pre_matches1 = pre_matches_r_loose
pre_matches2 = pre_matches_s_loose

# 从头开始
label = Label(img_path_s, img_path_r, img_s, img_r,
              np.transpose(pre_matches2), np.transpose(pre_matches1), des_s_loose, des_r_loose)
label.start_label()