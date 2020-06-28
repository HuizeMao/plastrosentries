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
"""
Load numpy array here
X_train = #images
Y_train = #labels
X_CV = #validation images(20%)
Y_CV = #validation labels
"""

def FCN(input_height,input_width):
    n_classes=6
    img_input = Input(shape=(input_height,input_width , 3 ))
    
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
    pool3 = MaxPooling2D((3, 3))(conv3)
    
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

    ##Upsampling (Decoder)
    up1 = concatenate([UpSampling2D((3, 3))(conv4), conv3], axis=-1)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    
    up2 = concatenate([UpSampling2D((2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up3 = concatenate([UpSampling2D((2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    """
    Here conv1 is concatenated with conv4, and conv2 is concatenated with conv3. 
    """
    out = Conv2D( n_classes, (1, 1) , padding='same')(conv7)
    #model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model
    model = Model(inputs = img_input, outputs = out, name='PlastroSentries')

    return model


model=FCN(972,1296)

"""Preview Model"""
plot_model(model, show_shapes=True,to_file='DraftModel.png')

##Config
model.compile(loss='categorical_crossentropy',
            optimizer = keras.optimizers.SGD(lr=0.01, decay=0, momentum=0, nesterov=False),
            metrics=['acc',MeanIoU(num_classes=6)])
##Training
history = model.fit(X_train, Y_train, batch_size = 1000,epochs = 1,verbose = 1, validation_data = (X_CV,Y_CV),shuffle=True)
##safe
model.save('fakeFCN.h5')

#evaluate
preds = model.evaluate(X_CV, Y_CV)

print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

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
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
