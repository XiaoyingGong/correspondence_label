# author: 龚潇颖(Xiaoying Gong)
# date： 2019/9/18 20:43  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
from preprocessing import pre_matching
from label.label import Label
from utils import constants
from utils.utils import utils
# 图：1-11和23
import numpy as np
# 图像路径
img_path_r = "./img/train_image_set/1_r.png"
img_path_s = "./img/train_image_set/1_s.png"

constants.SIFT_THRESHOLD = 0.3

# 用于筛选内点
def inlier_choose():
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

inlier_choose()


