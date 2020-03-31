import os
import pickle
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np

clean_path = "./image/label_clean/"
rainy_path = "./image/input_rainy/"

files = os.listdir(clean_path)

size_input = 64  # size of the training patch
num_channel = 3  # number of the input's channels.
num_files = len(files)  # total ( num_files - 1 ) training h5 files, the last one is used for validation
num_files_each_dir = 14
num_patches = 300  # number of patches (i.e. images) in each h5 file.
train_save = "./h5data/train/"
valid_save = "./h5data/validation/"
test_save = "./h5data/test/"
file_cnt = 0

def save_pickle_file(arr, path):
    with open(path, 'wb') as f:
        pickle.dump(arr, f)

for dir_num in range(1, num_files + 1):
    print("{} dir started.".format(dir_num))
    for file_num in range(1, num_files_each_dir + 1):
        print("{}-{} started.".format(dir_num, file_num))
        rainy_file_name = "{}_{}.jpg".format(dir_num, file_num)
        
        rainy_image = os.path.join(os.path.join(rainy_path, str(dir_num)), rainy_file_name)
        label_image = os.path.join(clean_path, "{}.jpg".format(str(dir_num)))
        
        for _ in range(num_patches):
            file_cnt += 1
            rainy = plt.imread(rainy_image)
            rainy = rainy/255.0

            label = plt.imread(label_image)
            label = label/255.0

            x = random.randint(0,rainy.shape[0] - size_input)
            y = random.randint(0,rainy.shape[1] - size_input)

            subim_input = rainy[x : x + size_input, y : y + size_input, :]
            subim_label = label[x : x + size_input, y : y + size_input, :]
            
            inp_lab = [subim_input, subim_label]
            save_file_name = "{}.pkl".format(str(file_cnt).zfill(7))
            
            if dir_num <= 715:
                path = os.path.join(train_save, save_file_name)
                save_pickle_file(inp_lab, path)
            elif dir_num <= 999:
                path = os.path.join(valid_save, save_file_name)
                save_pickle_file(inp_lab, path)
            else:
                path = os.path.join(test_save, save_file_name)
                save_pickle_file(inp_lab, path)
