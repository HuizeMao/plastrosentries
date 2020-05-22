from matplotlib import pyplot as plt
import numpy as np
#set up V object
inp = np.load("Feed_Images.npy")
V = inp[1]
V = (V).astype(np.uint8)

plt.imshow(V)
plt.show()
