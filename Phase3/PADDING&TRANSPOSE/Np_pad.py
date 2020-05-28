from matplotlib import pyplot as plt
import numpy as np
from TransposeArr import Transp
"""This file pads the np array 'indir' to size (972,1296)
It will also transpose the image from (W,H) to (H,W)
[delete Transping process if the image is already (H,W)]"""

####init####
#input dir output dir
indir = 'Feed_Images.npy' #Change as needed
cnt=0 #counter for the current image "id"

#init arrays bef/aft
old = np.load(indir)

oldW=old.shape[1] #dimension before
oldH=old.shape[2]
newW=1296 #dimension after
newH=972
difH=(newH-oldH)//2 #padding len
difW=(newW-oldW)//2
NUM=old.shape[0]
c=old.shape[-1]#channel: image or lebel

new = np.zeros([NUM,972,1296,c],dtype=np.uint8) #obj array

def pad():
    for cnt in range(0,NUM):
        cur=old[cnt]#extract
        cur=Transp(cur) #switch W*H to H*W or vice versa
        cur=np.pad(cur,((difH,difH), (difW,difW), (0, 0)), 'constant')#pad
        new[cnt]=cur #save
        print(str(cnt+1)+"/"+str(NUM),"padded")
        
    if(c==1): np.save("Padded_Labels",new)
    if(c==3): np.save("Padded_Images",new)
    return 0
pad()
