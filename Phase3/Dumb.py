import os
import PIL
import numpy as np
from PIL import Image
import random

####init####
#input dir output dir
dir = 'C:/Users/cocef/Desktop/PlastroSentries/200x200Crop'
outdir = 'C:/Users/cocef/Desktop/PlastroSentries/200x200Labels/'
cnt=0 #counter for the current image "id"
NUM=len([f for f in os.listdir(dir)if os.path.isfile(os.path.join(dir, f))]) #total # of images
#groundtruth array
letters = ['a','b','c','d','e','f','g']
arrays = []
for i in letters:
    arrays.append(np.zeros([NUM//7, 200, 200, 6], dtype=np.uint8))#init output array

####edit in "convert" function####
def convert(rgb,i,j, array): ##i,j is the pixel location
    #get separate r, g, and b value of the pixel
    R = rgb[0]
    G = rgb[1]
    B = rgb[2]
    ##edit here
    #if r,g,b values in the range of label X: 
    if B > G and B > R and B >= R+30 and 25 < B < 150:
        array[i][j][1] = 1 #1 = water

    elif R > 165 and G > 165 and B > 165:
        array[i][j][2] = 1 #2 = cloud/ice

    elif R - 20 < 110 and G - 20 < 110 and B - 20 < 110 and R > 40 and G > 40 and B > 40:
        array[i][j][5] = 1 #5 = uncertain land or water

    elif abs(R - G) <= 20 and abs(R - B) <= 20 and abs(B - G) <= 20 and 80 < B < 200:
        array[i][j][4] = 1 #4 = uncertain cloud/ice or land

    elif R - 30 < 165 and G - 30 < 165 and B - 30 < 165 and R + 20 > 110 and G + 20 > 110 and B + 20 > 110:
        array[i][j][3] = 1 #3 = land

    else:
        array[i][j][0] = 1 #0 = null



for cnt in range(NUM):
    im = Image.open(dir + '/' + str(cnt).zfill(5)+ ".jpg")
    px = im.load()
    w, h = im.size
    array = np.zeros([h, w, 6], dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            convert(px[i,j], j, i, array)
    arrays[(cnt)//1166][cnt%1166] = array
    if cnt % 10 == 0:
        print(cnt)
    if (cnt+1) % 1166 == 0:
        print(cnt)
        np.save(outdir + 'labels ' + str(letters[(cnt)//1166]), arrays[(cnt)//1166])




