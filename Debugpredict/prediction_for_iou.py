import numpy as np 
from keras.models import Model,load_model
import tensorflow as tf
from keras.backend import argmax

kerasresult=#..
num=kerasresult.shape[0]
x=kerasresult.shape[1]
y=kerasresult.shape[2]
c=6
iouarr=np.zeros((num,x,y,c))#change to tensor
for cnt in range(num):
    for i in range(x):
        for j in range(y):
            maxp=argmax(kerasresult[cnt][i][j][:])
            iouarr[cnt][i][j][maxp]=1

np.save("PredForIoU.npy",)
