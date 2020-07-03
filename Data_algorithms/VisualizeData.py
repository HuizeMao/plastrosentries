from matplotlib import pyplot as plt
import numpy as np

inp = np.load("testImgX.npy")
labels=np.load("testImgY.npy")
print(inp.shape)
print(labels.shape)
#set up V object
for i in range(0,5):
    V = inp[i]
    print(V.shape)
    V = (V).astype(np.uint8)

    plt.imshow(V)
    plt.show()
