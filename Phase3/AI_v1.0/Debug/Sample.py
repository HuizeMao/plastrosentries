###install keras-segmentation which contain all util required
###pip install keras-segmentation
import tensorflow as tf
import keras
from keras.utils import plot_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import concatenate,UpSampling2D
from keras.models import Model
from keras.metrics import MeanIoU
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint

"""
Load numpy array here
X_train = #images
Y_train = #labels
X_CV = #validation images(20%)
Y_CV = #validation labels
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#theSession= tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

X_train = np.load("smallImg.npy")
Y_train = np.load("smalllabel.npy")
print(X_train.shape)
print(Y_train.shape)
X_train=X_train/255.0
Y_train=Y_train/255.0

def FCN(input_height,input_width):
    n_classes=6
    img_input = Input(shape=(input_height,input_width,3))

    ##Downsampling (Encoder)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    ##Upsampling (Decoder)
    up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    """
    Here conv1 is concatenated with conv4, and conv2 is concatenated with conv3. 
    """
    out = Conv2D( n_classes, (1, 1), activation='sigmoid', padding='same')(conv5)
    #model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model
    model = Model(inputs = img_input, outputs = out, name='FCN')
    
    return model


model=FCN(296, 296)
 ##Config    
model.compile(loss='categorical_crossentropy',
            optimizer = keras.optimizers.Adam(lr=0.01),
            metrics=['acc',MeanIoU(num_classes=6)])

##save best
mc = ModelCheckpoint('DraftModel.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

##Training
history = model.fit(X_train, Y_train, batch_size = 5,epochs = 15,verbose = 2, validation_split=0.2, shuffle=True,callbacks=[mc])
##save
model.save('fakeFCN.h5')


#summary/Learning curve
model.summary()

#model history
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("LCaccuracy.png") #LC=Learning Curve
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("LCloss.png")
