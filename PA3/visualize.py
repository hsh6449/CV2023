import numpy as np
import matplotlib.pyplot as plt

segmentation = np.load("coarse_seg/train/12.npy")

print(segmentation)

plt.imshow(segmentation[0], cmap="gray")
plt.show()