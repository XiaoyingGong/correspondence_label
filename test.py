import numpy as np
import cv2

from utils.feature_extraction import sift_matching

# a = np.load("./label_result/5_r.jpg_5_l.jpg_1.0.npz")
# b = a["correspondence_label"]
# c = a["des"]
#
# kp1 = b[:2, :]
# kp2 = b[2:4, :]
# des_len = len(c)
# half_des_len = int(des_len / 2)
# des1 = c[:half_des_len, :]
# des2 = c[half_des_len: des_len, :]
label = np.load("./label_result/test_label.npz")
des1 = label["test_label_1"]
des2 = label["test_label_2"]
#
# sift_matching.get_matches_form_dataset(None, None, des1,
#                                        des2, 0.3)
# print(len(b[0]))
#
# bf = cv2.BFMatcher()
#
# print("des1_shape:", des1.shape)
# print("des1_type:", type(des1))
# print("des1:", des1)
#
# print("des2_shape:", des2.shape)
# print("des2_type:", type(des2))
# print("des2:", des2)
#
#
# matches = bf.knnMatch(des1, des2, k=2)
# good = []
# for m, n in matches:
#     if m.distance < 0.3 * n.distance:
#         good.append(m)
