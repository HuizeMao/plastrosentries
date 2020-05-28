from matplotlib import pyplot as plt
import numpy as np
"""This file changes an array with size N*W*H*C to N*H*W*C, W:width H:Height"""
def Transp(old): #Transpose a single image from size (W,H) to (H,W) or vice versa
    ####init####
    W=old.shape[0] #dimension before
    H=old.shape[1]
    c=old.shape[-1]#channel: image or lebel

    new=np.zeros([H,W,c],dtype=np.uint8) #dimension after

    for i in range(0,H): #Transpose
        for j in range(0,W):
            new[i][j][:]=old[j][i][:]  

    return new

def main():
    indir="Feed_images.npy" #change name
    old = np.load(indir)
    
    NUM=old.shape[0]
    W=old.shape[1] #dimension before
    H=old.shape[2]
    c=old.shape[-1]#channel: image or lebel

    new = np.zeros([NUM,H,W,c],dtype=np.uint8)
    for cnt in range(0,NUM):
        cur=old[cnt]
        cur=Transp(cur)
        new[cnt]=cur
        print(str(cnt+1)+"/"+str(NUM),"transposed")
    np.save("Transposed",new)
    return 0

