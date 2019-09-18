# author: 龚潇颖(Xiaoying Gong)
# date： 2019/9/10 21:30  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：

# result = np.vstack((self.pre_matches1, self.pre_matches2, self.is_right_match))
# # 存储结果，并保存遍历到的图像的图像特征点的序号，便于中间暂停后下次操作
# filename = self.img_path1.split("/")[len(self.img_path1.split("/")) - 1] + "_" \
#            + self.img_path2.split("/")[len(self.img_path2.split("/")) - 1] + "_" + str(constants.SIFT_THRESHOLD)
# np.savez('./label_result/' + filename, correspondence_label=result, index=self.index)

import numpy as np

from preprocessing import pre_matching
from utils import constants
from utils.utils import utils

# 图像路径
img_path_r = "./img/train_image_set/23_r.png"
img_path_s = "./img/train_image_set/23_s.png"

# 以下提出的点集都是sensed points 1对多 reference points
constants.IMG_RESIZE_RATIO = 1.5

# strict threshold
constants.SIFT_THRESHOLD = 0.3
pre_matches_s_strict, pre_matches_r_strict, des_s_strict, des_r_strict, img_s, img_r, resize_h, resize_w =\
    pre_matching.get_pre_matches(img_path_s, img_path_r, False)

# loose threshold
constants.SIFT_THRESHOLD = 0.4
pre_matches_s_loose, pre_matches_r_loose, des_s_loose, des_r_loose, img_s, img_r, resize_h, resize_w =\
    pre_matching.get_pre_matches(img_path_s, img_path_r, False)

_, repeated_index = utils.set_difference_2d(pre_matches_s_loose, pre_matches_s_strict)
non_repeated_index = np.setdiff1d(np.arange(0, len(pre_matches_s_loose)), repeated_index)

# 生成正样结果
pre_matches1 = np.transpose(pre_matches_r_loose[non_repeated_index])
pre_matches2 = np.transpose(pre_matches_s_loose[non_repeated_index])
matching_result = np.ones(len(pre_matches1[0]))
result = np.vstack((pre_matches1, pre_matches2, matching_result))

# 生成负样的结果
print(len(pre_matches1[0]))
matching_result_negative = np.zeros(len(pre_matches1[0]))
index_negative = np.arange(len(pre_matches1[0]))
for i in range(len(index_negative)):
    while index_negative[i] == i:
        index_negative[i] = np.random.randint(0, len(index_negative))
pre_matches1_negative = pre_matches1[:, index_negative]
result_negative = np.vstack((pre_matches1_negative, pre_matches2, matching_result_negative))

final_result = np.hstack((result, result_negative))

filename = img_path_r.split("/")[len(img_path_r.split("/")) - 1] + "_" \
           + img_path_s.split("/")[len(img_path_s.split("/")) - 1]
np.savez('./label_result/' + filename, correspondence_label=final_result)
print(len(final_result[0]))



