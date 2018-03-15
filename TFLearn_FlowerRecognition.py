from __future__ import division, print_function, absolute_import
import tflearn
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import build_hdf5_image_dataset
from tflearn.layers.normalization import local_response_normalization

#Restarting karnel
print('Kernel Restarting..')
tf.reset_default_graph()
print('Kernel Restarted..')

img_height = 100
img_width = 100
nb_classes = 5

def dataLoad(dataset_file):
    X,Y = tflearn.data_utils.image_preloader(dataset_file,image_shape=(img_width,img_height),mode='folder',categorical_labels=True,normalize=True)
    X,Y = tflearn.data_utils.shuffle(X,Y)
    imgList = []
    labelList = []

    for i in range(0,len(X)):
        imgList.append(X[i])
        labelList.append(Y[i])
        print('Serial - ' + str(i))

    imgList = np.array(imgList)
    labelList = np.array(labelList)
    #print('Size of ImageList = ' + str(imgList.shape))
    #print('Size of LabelList = ' + str(labelList.shape))
    return imgList,labelList

def create_model():
	# Building 'AlexNet'
	network = input_data(shape=[None, img_width, img_height, 3])

	network = conv_2d(network, 96, 11, strides=4, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)

	network = conv_2d(network, 256, 5, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)

	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)

	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)

	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)

	network = fully_connected(network, 60, activation='sigmoid')

	model = regression(network, optimizer='adam',
						 loss='categorical_crossentropy',
						 learning_rate=0.001)

	return model

def create_own_model():
    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    # Convolutional network building
    network = input_data(shape=[None, img_width, img_height, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, nb_classes, activation='softmax')

    model = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return model

def create_own_model_two_conv2():
    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    network = input_data(shape=[None, img_width, img_height,3])

    network = conv_2d(network, 64, 3, strides=2, activation='relu')
    network = max_pool_2d(network, 3, strides=2)

    network = conv_2d(network, 64, 3, strides=2, activation='relu')
    network = max_pool_2d(network, 3, strides=2)

    network = fully_connected(network, 128, activation='relu')

    network = fully_connected(network, nb_classes, activation='softmax')
    model = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    return model

def create_own_model_one_conv2():
    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)

    network = input_data(shape=[None, img_width, img_height,3])

    network = conv_2d(network, 64, 3, strides=2, activation='relu')
    network = max_pool_2d(network, 3, strides=2)


    network = fully_connected(network, 128, activation='relu')
    network = fully_connected(network, nb_classes, activation='softmax')

    model = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    return model

def train_model(model,x_train,y_train):
	model = tflearn.DNN(model, tensorboard_verbose=3)

	model.fit(x_train, y_train, n_epoch=40, shuffle=True, validation_set=0.15,
			show_metric=True, batch_size=100, snapshot_step=50,
			snapshot_epoch=False, run_id='tflean_flower_run03_2conv2')

    #print('Model Trained...')
	#save Model
	model.save('models/tflearn_flower_2Conv_model.model')
    #print('Model Saved...')

def load_model():
	model.load('models/tflearn_flower_2Conv_model.model')

def main():
    dataset_file_TRAIN = 'C:/Users/Nahid/Documents/MachineLearningProjects/zDataset/flowers/'
    #dataset_file_VAL = 'data/Validation'
    x_train,y_train = dataLoad(dataset_file_TRAIN)
    #x_val,y_val = dataLoad(dataset_file_VAL)
    print('Data Ready For Training...')

    #model = create_model()
    model = create_own_model_two_conv2()
    print('Model Created...')

    print('Training Started...')
    train_model(model,x_train,y_train)
    print('Model Trained & Saved...')
    print('Done...')

if __name__== "__main__":
  main()
