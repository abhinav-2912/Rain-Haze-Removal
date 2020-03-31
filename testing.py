import os

import cv2
import imageio
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models import ResidualModel2 as Network
from utils import guided_filter

tf.reset_default_graph()


file = '/home/two/final_code/synthetic_images/med2.jpg'
ori = plt.imread(file)
ori = ori / 255.0

#ori = ori * 0.5 + 0.5 * 1

input_tensor = np.expand_dims(ori[:,:,:], axis = 0)
detail_layer = input_tensor - guided_filter(input_tensor, input_tensor.shape[1], input_tensor.shape[2])
base_layer = guided_filter(input_tensor, input_tensor.shape[1], input_tensor.shape[2])

num_channels = 3
image = tf.placeholder(tf.float32, shape=(1, input_tensor.shape[1], input_tensor.shape[2], num_channels))
detail = tf.placeholder(tf.float32, shape=(1, input_tensor.shape[1], input_tensor.shape[2], num_channels))

output = Network.inference(image, detail, is_training = False)

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    saver.restore(sess, "/home/two/final_code/main_model/test-model/model") # try a pre-trained model 
    print ("load pre-trained model")


    final_output  = sess.run(output, feed_dict={image: input_tensor, detail: detail_layer})

    final_output[np.where(final_output < 0. )] = 0.
    final_output[np.where(final_output > 1. )] = 1.
    derained = final_output[0,:,:,:]
    
    detail_layer = detail_layer[0,:,:,:]
    base_layer = base_layer[0,:,:,:]
    
    imageio.imwrite('/home/two/final_code/derained.jpg', derained)
    imageio.imwrite('/home/two/final_code/ori.jpg', ori)
