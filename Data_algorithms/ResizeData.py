import numpy as np

X_train = np.load("C:/Users/guestml/PlastroSentries/feedsmol.npy")
Y_train = np.load("C:/Users/guestml/PlastroSentries/hotencodedsmol.npy")

newx=X_train[:5][:][:]
newy=Y_train[:5][:][:]
np.save("smallImg.npy",newx)
np.save("smalllabel.npy",newy)
