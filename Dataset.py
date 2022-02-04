import torch
import numpy as np
import math
from torch.utils.data import Dataset
import os
import pickle
from PIL import Image


class MyDataset(Dataset):

    def __init__(self, dataset_obj, transform=None):
        self.img_path = dataset_obj['img_path']
        self.gaze_vector = dataset_obj['gaze_vector']
        self.index_candidate = dataset_obj['index_candidate']
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_path[index]
        gaze_vector = self.gaze_vector[index].reshape(-1)
        gaze_vector = torch.tensor(gaze_vector).float()
        index_candidate = self.index_candidate[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, gaze_vector, index_candidate

    def __len__(self):
        return len(self.index_candidate)


def transform(img):
    img = torch.as_tensor(np.array(img, copy=True))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).to(dtype=torch.float32).div(255)
    return img


def dataset_division(root, dataset_file, test_set_index, val_set_index=0):
    val_set_index = (test_set_index + 1) % 15  # default
    train_set = {
        'img_path': [],
        'gaze_vector': [],
        'index_candidate': []
    }
    val_set = {
        'img_path': [],
        'gaze_vector': [],
        'index_candidate': []
    }
    test_set = {
        'img_path': [],
        'gaze_vector': [],
        'index_candidate': []
    }

    if not os.path.exists(dataset_file):
        FileNotFoundError("File doesn't exist.")

    with open(dataset_file, 'rb') as f:
        data = pickle.load(f)
        img_path = data['img_path']
        normalized_gaze_vector = data['normalized_gaze_vector']
        index_candidate = data['index_candidate']

    for i in range(len(index_candidate)):
        if index_candidate[i] == test_set_index:
            test_set['img_path'].append(root + img_path[i])
            test_set['gaze_vector'].append(normalized_gaze_vector[i])
            test_set['index_candidate'].append(index_candidate[i])
        elif index_candidate[i] == val_set_index:
            val_set['img_path'].append(root + img_path[i])
            val_set['gaze_vector'].append(normalized_gaze_vector[i])
            val_set['index_candidate'].append(index_candidate[i])
        else:
            train_set['img_path'].append(root + img_path[i])
            train_set['gaze_vector'].append(normalized_gaze_vector[i])
            train_set['index_candidate'].append(index_candidate[i])

    return train_set, val_set, test_set


def val_set_division(val_set, num_cali_samp):
    val_cali_set = {
        'img_path': [],
        'gaze_vector': [],
        'index_candidate': []
    }
    val_val_set = {
        'img_path': [],
        'gaze_vector': [],
        'index_candidate': []
    }
    val_test_set = {
        'img_path': [],
        'gaze_vector': [],
        'index_candidate': []
    }
    num = len(val_set['index_candidate'])
    num_part = math.ceil(num/num_cali_samp)
    for i in range(num):
        if i % num_part == 0:
            val_cali_set['img_path'].append(val_set['img_path'][i])
            val_cali_set['gaze_vector'].append(val_set['gaze_vector'][i])
            val_cali_set['index_candidate'].append(val_set['index_candidate'][i])
        elif (i - 1) % 4 == 0:
            val_val_set['img_path'].append(val_set['img_path'][i])
            val_val_set['gaze_vector'].append(val_set['gaze_vector'][i])
            val_val_set['index_candidate'].append(val_set['index_candidate'][i])
        else:
            val_test_set['img_path'].append(val_set['img_path'][i])
            val_test_set['gaze_vector'].append(val_set['gaze_vector'][i])
            val_test_set['index_candidate'].append(val_set['index_candidate'][i])
    return val_cali_set, val_val_set, val_test_set


def test_set_division(test_set, num_cali_samp):
    test_cali_set = {
        'img_path': [],
        'gaze_vector': [],
        'index_candidate': []
    }
    test_test_set = {
        'img_path': [],
        'gaze_vector': [],
        'index_candidate': []
    }
    num = len(test_set['index_candidate'])
    num_part = math.ceil(num / num_cali_samp)
    for i in range(num):
        if i % num_part == 0:
            test_cali_set['img_path'].append(test_set['img_path'][i])
            test_cali_set['gaze_vector'].append(test_set['gaze_vector'][i])
            test_cali_set['index_candidate'].append(test_set['index_candidate'][i])
        else:
            test_test_set['img_path'].append(test_set['img_path'][i])
            test_test_set['gaze_vector'].append(test_set['gaze_vector'][i])
            test_test_set['index_candidate'].append(test_set['index_candidate'][i])
    return test_cali_set, test_test_set
