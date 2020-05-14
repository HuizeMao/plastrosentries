import os
import numpy as np
from PIL import Image

#input dir output dir
dir = 'Training_Data'
outdir = 'Training_Labels'

#counters for logging progress info
cnt=1
NUM=len([f for f in os.listdir(dir)if os.path.isfile(os.path.join(dir, f))]) #total # of files

for file in os.listdir(dir):
    #init var
    ind=os.path.join(dir, file)
    im=Image.open(ind)
    w,h=im.size

    label_i = np.zeros([h, w, 3], dtype=np.uint8)#init output array

    #classify
    """for i in range(1,h)
        for j in range(1,w)"""

    #save
    ind2=os.path.join(outdir,file)#+str(cnt) if needed
    np.save(ind2,label_i)

    #update info
    print(str(cnt)+"/"+str(NUM),"completed")
    cnt+=1
