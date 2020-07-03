import numpy as np 
from keras.models import Model,load_model
import tensorflow as tf
im=np.load("smallImg.npy")
s=im.shape
a=np.zeros((1,s[1],s[2],s[3]))
im=im[1][:][:][:]
print("im shape: "+str(im.shape))

M = load_model('DraftModel.h5')

print("type: "+str(type(M)))
#print(max(O[0][1][1][:]))
