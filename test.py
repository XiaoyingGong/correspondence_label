import numpy as np

c = np.load("./label_result/15_r.jpg_15_s.jpg.npz")
a = c["correspondence_label"]
print(a[:, :10])
