import numpy as np 
from keras.models import Model,load_model
import tensorflow as tf
from keras.backend import argmax

#prepare model and data 
M = load_model('DraftModel.h5')
im=np.load("ISS.npy")
kerasresult=M.predict(im,verbose=1)

num=kerasresult.shape[0]
x=kerasresult.shape[1]
y=kerasresult.shape[2]

water=np.array([0,0,255])
cloud=np.array([255,255,255])
land=np.array([0,255,0])

colorcode=np.zeros((num,x,y,3))#change to tensor

#color
def color(cnt,b,b,c):
    if(c==1) colorcode[cnt][a][b][:]=water
    if(c==2) colorcode[cnt][a][b][:]=cloud
    if(c==3) colorcode[cnt][a][b][:]=land

#work
for cnt in range(num):
    for i in range(x):
        for j in range(y):
            maxp=argmax(kerasresult[cnt][i][j][:])
            if(maxp==4):
                if(kerasresult[cnt][i][j][2]>kerasresult[cnt][i][j][3]):maxp=2
                else: maxp=3
                    
            if(maxp==5):
                if(kerasresult[cnt][i][j][1]>kerasresult[cnt][i][j][3]):maxp=1
                else: maxp=3
            if(maxp):color(cnt,i,j,maxp)


    print(str(cnt)+"out of"+str(num-1)+"completed")
np.save("ColorCode.npy",colorcode)
