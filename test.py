import numpy as np

c = np.load("./label_result/1_r.png_1_s.png_1.0.npz")
a = c["correspondence_label"]
print(a[4])
