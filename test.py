import numpy as np

a = np.load("./label_result/5_r.jpg_5_l.jpg_1.0.npz")
b = a["correspondence_label"]
c = a["des"]
print(len(b[0]))
print(c)
