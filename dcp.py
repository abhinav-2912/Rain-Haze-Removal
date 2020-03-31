%matplotlib inline

import os

import cv2
import imageio
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def darkChannel(image):
    size = 15
    b,g,r = cv2.split(image)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(dc, kernel)
    return dark

def atmLight(image, dark):
    [height, width] = image.shape[:2]
    image_size = height * width
    numpx = int(max(math.floor(image_size / 1000), 1))
    darkvec = dark.reshape(image_size, 1)
    imvec = image.reshape(image_size, 3)
    indices = darkvec.argsort()
    indices = indices[image_size - numpx : :]
    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]
    A = atmsum / numpx
    return A

def transmissionEstimate(image, atm_light):
    omega = 0.15
    image3 = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    for i in range(3):
        image3[:,:,i] = image[:,:,i] / atm_light[0, i]
    transmission = 1 - omega * darkChannel(image3)
    return transmission

def getRadiance(atm_light, image, t_map):
    t0 = 0.1
    J = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    t_map[np.where(t_map < t0)] = t0
    for i in range(3):
        J[:,:,i] = (image[:,:,i] - atm_light[0, i]) / t_map
    
#     J = J / max(max(max(J)))
    return J

def deHaze(image):
    dark_channel = darkChannel(image)
    atm_light = atmLight(image, dark_channel)
    transmission = transmissionEstimate(image, atm_light)
    J = getRadiance(atm_light, image, transmission)
    return J
