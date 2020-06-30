import numpy as np 
from keras.models import Model,load_model
import tensorflow as tf
from keras.backend import argmax

#prepare model and data 
M = load_model('DraftModel.h5')
im=np.load("ISS.npy")
kerasresult=M.predict(im,verbose=1)

num=kerasresult.shape[0]
N=num-1
x=kerasresult.shape[1]
y=kerasresult.shape[2]
c=6
iouarr=np.zeros((num,x,y,c))#change to tensor
for cnt in range(num):
    for i in range(x):
        for j in range(y):
            maxp=argmax(kerasresult[cnt][i][j][:])
            iouarr[cnt][i][j][maxp]=1
    print(str(cnt)+"out of"+str(N)+"completed")

np.save("PredForIoU.npy",iouarr)
