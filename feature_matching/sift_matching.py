import cv2
import numpy as np
'''
 输入为两张图像的地址，返回值为指定阈值下的匹配
'''


def get_matches(img1_path, img2_path, sift_threshold):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    _, kp1, des1 = sift_kp(img1)
    _, kp2, des2 = sift_kp(img2)
    index1 = repeat_removal(kp1)
    index2 = repeat_removal(kp2)
    kp1 = np.array(kp1)[index1]
    kp2 = np.array(kp2)[index2]
    des1 = np.array(des1)[index1]
    des2 = np.array(des2)[index2]
    good_match = get_good_match(des1, des2, sift_threshold)
    matching_points_1, matching_points_2 = get_matching_points(kp1, kp2, good_match)
    return matching_points_1, matching_points_2, kp1, kp2, good_match


# 去重返回值为不重复的值的下标
def repeat_removal(kp):
    temp = np.zeros([len(kp), 2])
    for i in range(len(kp)):
        temp[i] = kp[i].pt
    _, index = np.unique(temp, return_index=True, axis=0)
    return index

# 得到在预匹配过后筛选的点,matching_points是一个2乘以n的二维矩阵，第一行为x坐标，第二行为y坐标
def get_matching_points(kp1, kp2, good_match):
    matching_points_1 = np.zeros((2, len(good_match)))
    matching_points_2 = np.zeros((2, len(good_match)))
    for i in range(len(good_match)):
        index1 = good_match[i].queryIdx
        index2 = good_match[i].trainIdx
        matching_points_1[0][i] = kp1[index1].pt[0]
        matching_points_1[1][i] = kp1[index1].pt[1]
        matching_points_2[0][i] = kp2[index2].pt[0]
        matching_points_2[1][i] = kp2[index2].pt[1]
    return matching_points_1, matching_points_2


# 得到关键点
def sift_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SIFT.create()
    kp,des = sift.detectAndCompute(image,None)
    kp_image = cv2.drawKeypoints(gray_image,kp,None)
    return kp_image, kp, des


# 做匹配
def get_good_match(des1, des2, sift_threshold):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < sift_threshold * n.distance:
            good.append(m)
    return good


