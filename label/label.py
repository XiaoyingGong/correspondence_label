import matplotlib.pyplot as plt
import numpy as np
import cv2
'''
input:大小相同的两幅图像的路径，以及预匹配后的点集
output:人工标注的点集,其为5行，n列的矩阵,其中
第一行，与第二行代表：第一幅图像中特征点的x和y坐标
第三行，与第四行代表：第二幅图像中特征点的x和y坐标
第五行代表：两特征点的匹配关系是否正确，1代表正确‘即匹配’，0代表不正确‘即不匹配’

键盘事件：
q:保存当前标注，并退出
r:该匹配关系正确
f:该匹配关系错误
b:返回上一 步

蓝线：错误匹配
黄线：正确匹配
注意：如果出现蓝线，且为错误匹配，则点击f，说明该匹配是错误的
      如果出现黄线，且为错误匹配，则点击f，说明该匹配是错误的
      如果出现蓝线，且为正确匹配，则点击r，说明该匹配是正确的
      如果出现黄线，且为正确匹配，则点击r，说明该匹配是正确的
      综上，错误的匹配都点击f，正确的匹配都点击r
'''


class Label:
    def __init__(self, img_path1, img_path2, pre_matches1, pre_matches2, load_path=None):
        self.img_path1 = img_path1
        self.img_path2 = img_path2
        self.img1 = cv2.imread(img_path1)[:, :, [2, 1, 0]]
        self.img2 = cv2.imread(img_path2)[:, :, [2, 1, 0]]
        self.img_width = self.img1.shape[1]
        self.h_img = np.hstack((self.img1, self.img2))
        self.pre_matches1 = pre_matches1
        self.pre_matches2 = pre_matches2
        # 用于记录是否这个对应关系是真的匹配 0代表不是，1代表是,初始时都默认为1
        self.is_right_match = np.ones(len(pre_matches1[0]))
        self.fig = None
        self.index = 0
        # 如果不是None，即以读文件的方式读入，之前的结果
        if load_path is not None:
            self.index = np.load(load_path)['index']
            self.is_right_match = np.load(load_path)['correspondence_label'][4, :]

    # 主方法,开始标注
    def start_label(self):
        fig, ax = plt.subplots(num='label')
        self.fig = fig
        self.draw_fig(self.pre_matches1[:, self.index], self.pre_matches2[:, self.index], self.is_right_match[self.index])
        self.print_index()
        # 取消默认快捷键的注册
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show()

    # 定义键盘点击事件
    def on_key_press(self, event):
        if event.key == 'r':
            self.right_match()
        if event.key == 'f':
            self.false_match()
        if event.key == 'b':
            self.back()
        if event.key == 'q':
            self.save_and_quit()
        self.fig.canvas.draw_idle()  # 重新绘制整个图表，

    # 定义绘制事件,每次绘制前都要清空
    # 最后一个参数即两个点是否是正确的匹配，如果是则为1，不是则为0
    def draw_fig(self, point1, point2, is_inlier):
        plt.clf()
        plt.axis('off')
        plt.imshow(self.h_img)
        plt.scatter(point1[0], point1[1], c='red', s=2)
        plt.scatter(point2[0]+self.img_width, point2[1], c='red', s=2)
        color = 'yellow'
        if is_inlier == 1:
            color = 'yellow'
        else:
            color = 'blue'
        plt.plot([point1[0], point2[0]+self.img_width], [point1[1], point2[1]], linewidth=1, c=color)

    # 操作
    # 正确的匹配进行的操作，将is_right_match[self.index]设置为1
    def right_match(self):
        flag = self.set_index(1)
        if flag == 1: # 已经遍历完了
            self.print_index()
            self.set_is_right_match(self.index, 1)
            self.draw_fig(self.pre_matches1[:, self.index], self.pre_matches2[:, self.index],
                          self.is_right_match[self.index])
        else:# 还没遍历完了
            self.print_index()
            self.set_is_right_match(self.index - 1, 1)
            self.draw_fig(self.pre_matches1[:, self.index], self.pre_matches2[:, self.index], self.is_right_match[self.index])

    # 错误的匹配进行的操作，将is_right_match[self.index]设置为0
    def false_match(self):
        flag = self.set_index(1)
        if flag == 1: # 已经遍历完了
            self.print_index()
            self.set_is_right_match(self.index, 0)
            self.draw_fig(self.pre_matches1[:, self.index], self.pre_matches2[:, self.index],
                          self.is_right_match[self.index])
        else:# 还没遍历完了
            self.print_index()
            self.set_is_right_match(self.index - 1, 0)
            self.draw_fig(self.pre_matches1[:, self.index], self.pre_matches2[:, self.index], self.is_right_match[self.index])

    # 返回上一步
    def back(self):
        flag = self.set_index(-1)
        self.print_index()
        self.draw_fig(self.pre_matches1[:, self.index], self.pre_matches2[:, self.index], self.is_right_match[self.index])

    # 保存当前结果，并退出
    def save_and_quit(self):
        #拼接结果
        result = np.vstack((self.pre_matches1, self.pre_matches2, self.is_right_match))
        #存储结果，并保存遍历到的图像的图像特征点的序号，便于中间暂停后下次操作
        filename = self.img_path1.split("/")[len(self.img_path1.split("/")) - 1] + "_" \
            + self.img_path2.split("/")[len(self.img_path2.split("/")) - 1]
        np.savez('./label_result/'+filename, correspondence_label=result, index=self.index)
        plt.close()

    # 对is_right_match的操作
    def set_is_right_match(self, index, is_match):
        self.is_right_match[index] = is_match

    # 对index的操作 addition为+则为增加相应的数量，addition为负为减少相应的数量
    def set_index(self, addition):
        # 越界就直接不变,直接返回，返回值为0代表越下界（即不能再回退），返回值为1为越上界(即以及遍历完所有的预匹配)
        # 返回值为2代表正常
        if self.index + addition < 0:
            return 0
        elif self.index + addition > len(self.pre_matches1[0]) - 1:
            return 1
        else:
            self.index = self.index + addition
            return 2

    # 打印出当前的序号
    def print_index(self):
        print(self.index + 1, ' / ',  len(self.pre_matches1[0]))
