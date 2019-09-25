'''
author：龚潇颖
date：2019_8_27
des: 给定图像对，提取出相应的预匹配特征
     constant.IMG_RESIZE_RATIO与constant.SIFT_THRESHOLD需要改变时，可以在外部改变
inputs: 两幅图像的路径
outputs: 预匹配的点对，特征描述，resize后的图像，以及resize后的图像的h与w
'''
import cv2
import numpy as np

from utils.feature_extraction import sift_matching
from utils import constants


# 图像大小的调整
def img_resize(img_r, img_s):
    shape = img_r.shape
    resize_h = int(shape[0] / constants.IMG_RESIZE_RATIO)
    resize_w = int(shape[1] / constants.IMG_RESIZE_RATIO)
    img_r = cv2.resize(img_r, (resize_w, resize_h))
    img_s = cv2.resize(img_s, (resize_w, resize_h))
    return img_r, img_s, resize_h, resize_w

def img_resize_fixed(img_1, img_2):
    shape = img_1.shape
    resize_h = 300
    resize_w = 400
    img_1 = cv2.resize(img_1, (resize_w, resize_h))
    img_2 = cv2.resize(img_2, (resize_w, resize_h))
    return img_1, img_2, resize_h, resize_w

# 用于一对多的去除，强行变成一对一
def become_one_to_one(pre_matches_1, pre_matches_2):
    pre_matches_1, index_1 = np.unique(pre_matches_1, return_index=True, axis=0)
    pre_matches_2 = pre_matches_2[index_1]
    pre_matches_2, index_2 = np.unique(pre_matches_2, return_index=True, axis=0)
    pre_matches_1 = pre_matches_1[index_2]
    return pre_matches_1, pre_matches_2

def get_pre_matches(img_path_1, img_path_2, is_unique=None):
    """
    :param img_path_1:
    :param img_path_2:
    :param is_unique:
    :return:
    """
    # 图像路径
    img_1 = cv2.imread(img_path_1)[:, :, [2, 1, 0]]
    img_2 = cv2.imread(img_path_2)[:, :, [2, 1, 0]]
    # resize
    img_1, img_2, resize_h, resize_w = img_resize_fixed(img_1, img_2)
    # 通过sift进行预匹配
    pre_matches_1, pre_matches_2, des_1, des_2, match_index = \
        sift_matching.get_matches(img_1, img_2, constants.SIFT_THRESHOLD)
    # 是否强行变成一一对应
    if is_unique:
        pre_matches_1, pre_matches_2 = become_one_to_one(pre_matches_1, pre_matches_2)
    return pre_matches_1, pre_matches_2, des_1, des_2, img_1, img_2, resize_h, resize_w

