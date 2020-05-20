import os
import PIL
import numpy as np
from PIL import Image

####init####
#input dir output dir
dir = 'Training_Data'
outdir = 'Training_Labels'
cnt=1 #counter for the current image "id"
NUM=len([f for f in os.listdir(dir)if os.path.isfile(os.path.join(dir, f))]) #total # of images
#groundtruth array
label = np.zeros([NUM, 1296, 972, 1], dtype=np.uint8)#init output array


####edit in "convert" function####
def convert(rgb,i,j):
    #get separate r, g, and b value of the pixel
    R=px[i,j][0]
    G=px[i,j][1]
    B=px[i,j][2]

    ##edit here
    #if r,g,b values in the range of label X: 
    #np[cnt][i][j][0]= X
    """
    classes: clouds, land, sea or night,ice.
    Apply special processing to the ISS window later
    """
    
    return 0

for file in os.listdir(dir):
    #init var
    ind=os.path.join(dir, file)
    im=Image.open(ind)
    px=im.load()
    w,h=im.size
    #print(str(w),str(h))
   
    #iter every pixel and classify
    for i in range(0,w):
        for j in range(1,h):
            convert(px[i,j],i,j) #label this pixel
        
    #update progress
    print(str(cnt)+"/"+str(NUM),"completed")
    cnt+=1

np.save(outdir,label) #saves np array with size(BATCH_SIZE,H,W,1)
