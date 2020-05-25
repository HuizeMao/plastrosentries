from matplotlib import pyplot as plt
import numpy as np

####init####
#input dir output dir
indir = 'Training_Data.np'
cnt=0 #counter for the current image "id"


#init arrays bef/aft
old = np.load(indir)

oldW=old.shape[1] #dimension before
oldH=old.shape[2]
newW=1296 #dimension after
newH=972
NUM=old.shape[0]
c=old.shape[-1]#channel: image or lebel

new = np.zeros(NUM,972,1296,c)



def pad(p):
    cur=old[cnt]#extract
    #pad equally in the two parallel sides
    difH=(newH-oldH)/2
    difW=(newW-oldW)/2
    cur=np.pad(old,((difH,difH), (difW,difW), (0, 0)), 'constant')
    new[cnt]=cur#save
    
for i in range(0,NUM)
    
    pad()
    cnt+=1

if(c==1) np.save("Padded_"+"labels")
if(c==3) np.save("Padded_"+"images")
