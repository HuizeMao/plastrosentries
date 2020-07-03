###install keras-segmentation which contain all util required
###pip install keras-segmentation
import tensorflow as tf
import keras
from keras.utils import plot_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import concatenate,UpSampling2D, BatchNormalization,Activation
from keras.models import Model, load_model
from keras.metrics import MeanIoU
from keras.callbacks import ModelCheckpoint
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import numpy as np
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])
  except RuntimeError as e:
    print(e)

X_train, Y_train = [], []
letters = ['a','b','c','d','e','f','g']

def conv(chanels,shape,pad,inp):
    layer=Conv2D(chanels, shape, padding=pad)(inp)
    layer=BatchNormalization(axis = 3)(layer)
    layer = Activation('relu')(layer)
    return layer

def FCN(input_height,input_width):
    n_classes=6
    img_input = Input(shape=(input_height,input_width,3))

    ##Downsampling (Encoder)
    conv1=conv(32,(3,3),"same",img_input)
    #conv1 = Dropout(0.2)(conv1)
    conv1=conv(32,(3,3),"same",conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
     
    conv2=conv(64,(3, 3),'same',pool1)
    #conv2 = Dropout(0.2)(conv2)
    conv2=conv(64,(3, 3),'same',conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = conv(128,(3, 3),'same',pool2)
    #conv3 = Dropout(0.2)(conv3)
    conv3 = conv(128,(3, 3),'same',conv3)
    
##deeper layer begins 
    #add conv4 with size(25) do  not downsample again
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv33 = conv(256,(3,3),'same',pool3)
    #conv33 = Dropout(0.2)(conv33)
    conv33 = conv(256,(3,3),'same',conv33)

    
    #update up1 to concat (upsample conv4) with conv3 
    up0 = concatenate([UpSampling2D((2, 2))(conv33), conv3], axis=-1)
    conv33D = conv(128,(3, 3),'same',up0)
    #conv33D = Dropout(0.2)(conv33D)
    conv33D = conv(128,(3, 3),'same',conv33D)
##deeper layer ends

    ##Upsampling (Decoder)
    up1 = concatenate([UpSampling2D((2, 2))(conv33D), conv2], axis=-1)
    conv4 = conv(64,(3, 3),'same',up1)
    #conv4 = Dropout(0.2)(conv4)
    conv4 = conv(64,(3, 3),'same',conv4)

    up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = conv(32,(3, 3),'same',up2)
    #conv5 = Dropout(0.2)(conv5)
    conv(32,(3, 3),'same',conv5)
     
    """
    Here conv1 is concatenated with conv4, and conv2 is concatenated with conv3. 
    """
    out = Conv2D( n_classes, (1, 1), padding='same')(conv5)
    out = BatchNormalization(axis = 3)(out)
    out = Activation('softmax')(out)
    #model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model
    model = Model(inputs = img_input, outputs = out, name='FCN')
    
    return model


model= FCN(200,200)
 ##Config    
model.compile(loss='categorical_crossentropy',
            optimizer = keras.optimizers.Adam(lr=0.01),
            metrics=['acc',MeanIoU(num_classes=6)])
##save best
mc = ModelCheckpoint('DraftModel3.h5', monitor='val_MeanIoU', mode='max', verbose=1, save_best_only=True)

##Training
for _ in range(4):
    for i in range(7):
        r = random.randint(0,6)
        X_train = np.load('C:/Users/GuestML/PlastroSentries/Feed/feed ' + letters[r] + '.npy')
        Y_train = np.load('C:/Users/GuestML/PlastroSentries/Labels/labels ' + letters[r] + '.npy')
        X_train=X_train/255.0
        history = model.fit(X_train, Y_train, batch_size = 32, epochs = 10,verbose = 1, validation_split=0.15, shuffle=True,callbacks=[mc])
                
        #model history
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("LCaccuracy.png") #LC=Learning Curve
        plt.clf()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig("LCloss.png")
        plt.clf()

        del X_train, Y_train
    model.save('fakeFCN3.h5') #callbacks automatically saves
##save
model.save('fakeFCN3.h5') #callbacks automatically saves


#summary/Learning curve
model.summary()

#model history
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("LCaccuracy.png") #LC=Learning Curve

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("LCloss.png")