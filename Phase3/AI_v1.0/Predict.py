from keras_segmentation.predict import predict_multiple


predict_multiple( 
	checkpoints_path="checkpoints/vgg_unet_1", 
	inp_dir="dataset_path/images_prepped_test/", 
	out_dir="outputs/" 
)


