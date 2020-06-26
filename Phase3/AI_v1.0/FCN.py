###install keras-segmentation which contain all util required
###pip install keras-segmentation
from keras_segmentation.models.model_utils import get_segmentation_model
from keras_segmentation.models.unet import vgg_unet ##change later


def FCN(input_height,input_width):  
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
    out = Conv2D( n_classes, (1, 1) , padding='same')(conv5)
    model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model
    return model


model=FCN(972,1296)
##Training
#dataset is the directory of the training images and checkpoints
#is the directory where all the model weights would be saved
model.train( 
    train_images =  "dataset_path/images_prepped_train/",
    train_annotations = "dataset_path/annotations_prepped_train/",
    checkpoints_path = "checkpoints/vgg_unet_1" , epochs=5
)
"""
#make prediction
out = model.predict_segmentation(
    inp="dataset_path/images_prepped_test/0016E5_07965.png",
    out_fname="output.png"
)
"""
