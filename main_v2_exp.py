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
img_path_r = "./img/exp_img/4_r.jpg"
img_path_s = "./img/exp_img/4_s.jpg"

constants.SIFT_THRESHOLD = 1.0
pre_matches_s_loose, pre_matches_r_loose, des_s_loose, des_r_loose, img_s, img_r, resize_h, resize_w = \
    pre_matching.get_pre_matches(img_path_s, img_path_r, False)

print(len(pre_matches_s_loose))

constants.SIFT_THRESHOLD = 0.4
pre_matches_s_strict, pre_matches_r_strict, des_s_strict, des_r_strict, _, _, _, _ = \
    pre_matching.get_pre_matches(img_path_s, img_path_r, False)

_, index = utils.set_difference_2d(pre_matches_s_loose, pre_matches_s_strict)
print(index)

pre_matches1 = pre_matches_r_loose
pre_matches2 = pre_matches_s_loose

# 从头开始
label = Label(img_path_s, img_path_r, img_s, img_r,
              np.transpose(pre_matches2), np.transpose(pre_matches1), des_s_loose, des_r_loose, load_path=None, is_all_points_show=True)
label.start_label()

# 以下为测试2 读取保存的文件来进行标注
# label = Label(img_path_s, img_path_r, img_s, img_r, np.transpose(pre_matches2), np.transpose(pre_matches1), des_s_loose, des_r_loose, './label_result/5_r.jpg_5_l.jpg_1.0.npz')
# label.start_label()

