import os
import random
import shutil
import sys
import time
import warnings
from time import time

import cv2
import imageio
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.python.client import device_lib

import albumentations as A
import segmentation_models as sm
from models import ResidualModel1, ResidualModel2
from utils import guided_filter, visualize

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
with tf.device("/device:GPU:0"):
    image_size = 64
    num_channels = 3
    x_train_path = "/home/two/final_code/data_generation/h5data/train/x/"
    y_train_path = "/home/two/final_code/data_generation/h5data/train/y/"
    x_valid_path = "/home/two/final_code/data_generation/h5data/validation/x/"
    y_valid_path = "/home/two/final_code/data_generation/h5data/validation/y/"
    x_test_path = "/home/two/final_code/data_generation/h5data/test/x/"
    y_test_path = "/home/two/final_code/data_generation/h5data/test/y/"

    image = Input((image_size, image_size, num_channels))
    detail = Input((image_size, image_size, num_channels))
    label = Input((image_size, image_size, num_channels))

    model = ResidualModel2(image, detail, True)

    #     model.summary()

    def round_clip_0_1(x, **kwargs):
        return x.round().clip(0, 1)

    # define heavy augmentations
    def get_training_augmentation():
        train_transform = [
            A.RandomCrop(height=image_size, width=image_size, always_apply=True),
        ]
        return A.Compose(train_transform)


    def get_validation_augmentation():
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            A.RandomCrop(height=image_size, width=image_size, always_apply=True)
        ]
        return A.Compose(test_transform)

    BATCH_SIZE = 64
    LR = 0.001
    EPOCHS = 15
    IMAGE_ORDERING = 'channels_last'

    optim = Adam(LR)
    loss = tf.keras.losses.MeanSquaredError()

    model.compile(optim, loss)

    random.seed(7)
    train_dataset = Dataset(
        x_train_path,
        y_train_path,
        augmentation=get_training_augmentation())

    # Dataset for validation images
    valid_dataset = Dataset(
        x_valid_path,
        y_valid_path,
        augmentation=get_validation_augmentation())

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    tensorboard_callback = TensorBoard(log_dir='/home/two/final_code/logs/{}'.format(time()), update_freq='batch')
    reduce_lr = ReduceLROnPlateau()
    callbacks = [tensorboard_callback, reduce_lr]

    train_samples = 300000
    valid_samples = train_samples * 0.2
    train_steps = int(train_samples / BATCH_SIZE)
    valid_steps = int(valid_samples / BATCH_SIZE)

    history = model.fit_generator(
        train_dataloader,
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=valid_steps)
