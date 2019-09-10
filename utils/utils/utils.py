# author: 龚潇颖(Xiaoying Gong)
# date： 2019/8/28 14:57  
# IDE：PyCharm 
# des: 一些工具方法，例如集合求差等
# input(s)：
# output(s)：

import numpy as np


def set_difference_2d(larger_set, less_set):
    """
    用于二维集合的差集操作
    :param larger_set: 一个较大的集合
    :param less_set: 一个较小的集合
    :return: 较大的集合去掉较小的集合
    """
    less_set_len = less_set.shape[0]
    delete_index = np.array([], dtype=np.int)
    for i in range(less_set_len):
       mask = larger_set == less_set[i]
       index = np.where(np.all(mask, axis=1))[0]
       delete_index = np.append(delete_index, index)
    result = np.delete(larger_set, delete_index, axis=0)
    return result, delete_index

# 计算两个点的欧式距离
def cal_Euclidean_distance(point_1, point_2):
    Euclidean_distance = np.sqrt(np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1]))
    return Euclidean_distance

if __name__ == "__main__":
    a = np.array([[1, 2, 3, 4], [3, 3, 3, 5], [4, 5, 3, 6], [3, 4.1, 3, 7], [4, 4.1, 3, 7]])
    b = np.array([[1, 2, 3, 4], [4, 4.1, 3.1, 7]])
    print(set_difference_2d(a, b))


