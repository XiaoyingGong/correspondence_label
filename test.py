import numpy as np

c = np.load("./label_result/1_s.jpg_1_r.jpg_1.0.npz")
a = c["correspondence_label"]
print(a[:, :10])
