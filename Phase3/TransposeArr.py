from matplotlib import pyplot as plt
import numpy as np



def Transp(indir):
    ####init####
    #input dir output dir
    cnt=0 #counter for the current image "

    #init arrays bef/aft
    old = np.load(indir)

    W=old.shape[1] #dimension before
    H=old.shape[2]
    NUM=old.shape[0]
    c=old.shape[-1]#channel: image or lebel

    new = np.zeros(NUM,H,W,c)

    for k in range(0,NUM):
        for(i in range(0,H)):
            for(j in range(0,W)):
                new[cnt][i][j][:]=old[j][i][:]    
        cnt+=1

    return new

