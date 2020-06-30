###install keras-segmentation which contain all util required
###pip install keras-segmentation
import keras
from keras.utils import plot_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import concatenate,UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.metrics import MeanIoU
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import numpy as np
"""input result array and labels"""
##P=np.load() #prediction
##L=npload() #label

ClassIoU=np.zeros(6)

def Class_iou(labels, predictions):
    """
    labels,prediction with shape of [batch,height,width,class_number=6]
    """
    """mean_iou = K.variable(0.0)
    seen_classes = K.variable(0.0)"""

    for c in range(6): #iter every class
        #returns two bool array (1 for predict 1 for label)in which 
        #every pixel that is == to class c is labeled as 1 else 0 
        labels_c = K.cast(K.equal(labels, c), K.floatx()) 
        pred_c = K.cast(K.equal(predictions, c), K.floatx())

        #Total number of pixels that is == to class c in the two arrays
        labels_c_sum = K.sum(labels_c)
        pred_c_sum = K.sum(pred_c)

        #Calculate # intersection: Note that labels_c*pred_c is element wise multiplication, 
        #so it will first make an array of 0 and 1, where 1s is intersection(0*1=0,1*1=1)
        #Then it sums up the total number of pixels that are both 1
        intersect = K.sum(labels_c*pred_c)

        #self explanatory
        union = labels_c_sum + pred_c_sum - intersect

        #class iou 
        iou = intersect / union
        ClassIoU[c]=iou
        """
        condition = K.equal(union, 0) #if the image present class c condition=1  else 0
        mean_iou = K.switch(condition,#if 1
                            mean_iou, #then
                            mean_iou+iou) #else
        seen_classes = K.switch(condition,
                                seen_classes,
                                seen_classes+1)

    mean_iou = K.switch(K.equal(seen_classes, 0),
                        mean_iou,
                        mean_iou/seen_classes)
                        """
    return mean_iou
Class_iou(L,P)
outfile="classIoU.npy"
np.save(outfile,ClassIoU)
