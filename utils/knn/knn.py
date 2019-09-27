'''
author:龚潇颖
des:找到相领的内点
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors


class K_NearestNeighbors:
    def __init__(self, train_data):
        self.train_data = train_data
        self.nn = NearestNeighbors(algorithm='kd_tree').fit(self.train_data)

    # 寻找的点不会在kd树当中
    def get_k_neighbors_v0(self, aim_point, k):
        k_nearest_neighbors_dist, k_nearest_neighbors_index = self.nn.kneighbors(X=aim_point, n_neighbors=k)
        return k_nearest_neighbors_dist, k_nearest_neighbors_index

    # 寻找的点在kd树种
    def get_k_neighbors_v1(self, aim_point, k):
        aim_point = np.array([aim_point])
        # 没找到足量的点 就一直循环
        k_new = k
        while True:
            k_nearest_neighbors_dist, k_nearest_neighbors_index = self.nn.kneighbors(X=aim_point, n_neighbors=k_new)
            #去除掉自己
            zero_index = np.array(np.where(k_nearest_neighbors_dist[0] == 0.))
            k_nearest_neighbors_dist = np.delete(k_nearest_neighbors_dist, zero_index)
            k_nearest_neighbors_index = np.delete(k_nearest_neighbors_index, zero_index)
            if len(k_nearest_neighbors_dist) != k or len(k_nearest_neighbors_index) != k:
                k_new = k_new + 1
            else:
                break
        return k_nearest_neighbors_dist, k_nearest_neighbors_index

    def get_k_neighbors(self, aim_point, k):
        aim_point = np.array([aim_point])
        k_nearest_neighbors_dist, k_nearest_neighbors_index = self.nn.kneighbors(X=aim_point, n_neighbors=k)
        return k_nearest_neighbors_dist[0], k_nearest_neighbors_index[0]

    def get_k_neighbors_boardcast(self, aim_point, k):
        k_nearest_neighbors_dist, k_nearest_neighbors_index = self.nn.kneighbors(X=aim_point, n_neighbors=k)
        return k_nearest_neighbors_dist, k_nearest_neighbors_index
