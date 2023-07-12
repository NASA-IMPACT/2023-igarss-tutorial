import csv
import rasterio
import os
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
import logging
from PIL import Image
from torchvision import transforms

# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.addHandler(console_handler)
@DATASETS.register_module()
class Sen1Floods11(CustomDataset):
    """Sen1Floods11 dataset.
    """
    CLASSES = (0, 1)

    PALETTE = None

    def __init__(self, **kwargs):
        super(Sen1Floods11, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            #ignore_index=2,
            **kwargs)
#check for ignore_index


class InMemoryDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])

    def __len__(self):
        return len(self.data_list)


def process_and_augment_s1(data):
#try to balance for masks and burn
    (x, y) = data
    im, label = x.copy(), y.copy()
    label = label.astype(np.float64)

    im1 = Image.fromarray(im[0])
    im2 = Image.fromarray(im[1])
    label = Image.fromarray(label.squeeze())
    dim = 224
    i, j, h, w = transforms.RandomCrop.get_params(im1, (dim, dim))

    im1 = F.crop(im1, i, j, h, w)
    im2 = F.crop(im2, i, j, h, w)
    label = F.crop(label, i, j, h, w)
    if random.random() > 0.5:
        im1 = F.hflip(im1)
        im2 = F.hflip(im2)
        label = F.hflip(label)
    if random.random() > 0.5:
        im1 = F.vflip(im1)
        im2 = F.vflip(im2)
        label = F.vflip(label)

    norm = transforms.Normalize([442.9805145263672, 699.5281867980957, 679.5248565673828, 2093.850761413574], [232.17283718767763, 236.6401508696276, 369.91184775358425, 850.2287590280677])
    ims = [torch.stack((transforms.ToTensor()(im1).squeeze(),
                        transforms.ToTensor()(im2).squeeze(),
                        transforms.ToTensor()(im1).squeeze() * transforms.ToTensor()(im2).squeeze(),
                        transforms.ToTensor()(im1).squeeze() * transforms.ToTensor()(im2).squeeze()))]
    ims = [norm(im) for im in ims]
    im = torch.stack(ims).reshape(4, 1, dim, dim)

    label = transforms.ToTensor()(label).squeeze()
    return {"img": im,
            "img_metas": dict(),
            "gt_semantic_seg": label}


def process_and_augment_s2(data):
    (x, y) = data
    im, label = x.copy(), y.copy()
    label = label.astype(np.float64)
#change
    im1 = Image.fromarray(im[0])  # red
    im2 = Image.fromarray(im[1])  # green
    im3 = Image.fromarray(im[2])  # blue
    im4 = Image.fromarray(im[3])  # NIR narrow
    label = Image.fromarray(label.squeeze())
    dim = 224
    i, j, h, w = transforms.RandomCrop.get_params(im1, (dim, dim))

    im1 = F.crop(im1, i, j, h, w)
    im2 = F.crop(im2, i, j, h, w)
    im3 = F.crop(im3, i, j, h, w)
    im4 = F.crop(im4, i, j, h, w)
    label = F.crop(label, i, j, h, w)
    if random.random() > 0.5:
        im1 = F.hflip(im1)
        im2 = F.hflip(im2)
        im3 = F.hflip(im3)
        im4 = F.hflip(im4)
        label = F.hflip(label)
    if random.random() > 0.5:
        im1 = F.vflip(im1)
        im2 = F.vflip(im2)
        im3 = F.vflip(im3)
        im4 = F.vflip(im4)
        label = F.vflip(label)

    #norm = transforms.Normalize([0.21531178, 0.20978154, 0.18528642, 0.48253757], [0.10392396, 0.10210076, 0.11696766, 0.19680527])
    norm = transforms.Normalize([442.9805145263672, 699.5281867980957, 679.5248565673828, 2093.850761413574],[232.17283718767763, 236.6401508696276, 369.91184775358425, 850.2287590280677])
    ims = [torch.stack((transforms.ToTensor()(im1).squeeze(),
                        transforms.ToTensor()(im2).squeeze(),
                        transforms.ToTensor()(im3).squeeze(),
                        transforms.ToTensor()(im4).squeeze()))]
    ims = [norm(im) for im in ims]
    im = torch.stack(ims).reshape(4, 1, dim, dim)

    label = transforms.ToTensor()(label).squeeze()

    _img_metas = {
        'ori_shape': (dim, dim),
        'img_shape': (dim, dim),
        'pad_shape': (dim, dim),
        'scale_factor': [1., 1., 1., 1.],
        'flip': True,
        # 'flip_direction': 'horizontal',
        # 'img_norm_cfg': {
        #     'mean': [123.675, 116.28, 103.53],
        #     'std': [58.395, 57.12, 57.375],
        #     'to_rgb': True
        # },
        # 'batch_input_shape': (480, 640)
    }

    img_metas = [_img_metas] * 1
    return {"img": im,
            "img_metas": img_metas,
            "gt_semantic_seg": label}


def process_test_im(data):
    (x, y) = data
    im, label = x.copy(), y.copy()
    label = label.astype(np.float64)

    im1 = Image.fromarray(im[0])  # red
    im2 = Image.fromarray(im[1])  # green
    im3 = Image.fromarray(im[2])  # blue
    im4 = Image.fromarray(im[3])  # NIR narrow
    dim = 512
    label = Image.fromarray(label.squeeze())

    norm = transforms.Normalize([442.9805145263672, 699.5281867980957, 679.5248565673828, 2093.850761413574],[232.17283718767763, 236.6401508696276, 369.91184775358425, 850.2287590280677])
    ims = [torch.stack((transforms.ToTensor()(im1).squeeze(),
                        transforms.ToTensor()(im2).squeeze(),
                        transforms.ToTensor()(im3).squeeze(),
                        transforms.ToTensor()(im4).squeeze()))]
    ims = [norm(im) for im in ims]
    ims = torch.stack(ims).reshape(4, dim, dim)

    label = transforms.ToTensor()(label).squeeze()

    _img_metas = {
        'ori_shape': (dim, dim),
        'img_shape': (dim, dim),
        'pad_shape': (dim, dim),
        'scale_factor': [1., 1., 1., 1.],
        'flip': True,
        # 'flip_direction': 'horizontal',
        # 'img_norm_cfg': {
        #     'mean': [123.675, 116.28, 103.53],
        #     'std': [58.395, 57.12, 57.375],
        #     'to_rgb': True
        # },
        # 'batch_input_shape': (480, 640)
    }

    img_metas = [_img_metas] * 1
    return {"img": [ims],
            "img_metas": img_metas,
            "gt_semantic_seg": label}


def get_arr_flood(fname):
    return rasterio.open(fname).read()


def download_flood_water_data_from_list_s1(l):
#change
    i = 0
    flood_data = list()
    for (im_fname, mask_fname) in l:
	#change
        if "S2Hand" in im_fname:
            im_fname = im_fname.replace("S2Hand", "S1Hand")
        if not os.path.exists(os.path.join(im_fname)):
            continue
        # Converts the missing values (clouds etc.)
        arr_x = np.nan_to_num(get_arr_flood(os.path.join(im_fname)))
        arr_y = get_arr_flood(os.path.join(mask_fname))
        arr_y[arr_y == -1] = 2

        arr_x = np.clip(arr_x, -50, 1)
        arr_x = (arr_x + 50) / 51

        if i % 100 == 0:
            print(im_fname, mask_fname)
        i += 1
        flood_data.append((arr_x, arr_y))

    return flood_data


def download_flood_water_data_from_list_s2(l):
    i = 0
    flood_data = list()
    for (im_fname, mask_fname) in l:
        if not os.path.exists(os.path.join(im_fname)):
            continue
        # Converts the missing values (clouds etc.)
	# change 
        arr_x = get_arr_flood(os.path.join(im_fname))
        arr_y = get_arr_flood(os.path.join(mask_fname))
        arr_x = arr_x.astype(np.float32)
        arr_y = arr_y.astype(np.float32)
        arr_x = arr_x / np.max(arr_x)
        #arr_y = arr_y / np.max(arr_y)
        #arr_y[arr_y == -1] = 2

        if i % 100 == 0:
            print(im_fname, mask_fname)
        i += 1
        flood_data.append((arr_x, arr_y))

    return flood_data


def load_flood_train_data(input_root, label_root, fname):
    #logger.info(f("input root: {input_root}, {label_root}, {fname}"))
    #print(f'input root: {input_root}, {label_root}, {fname}')
    training_files = list()
    with open(fname) as f:
        for line in csv.reader(f):
            #logger.info(f'xxxxxx----{tuple((input_root + line[0], label_root + line[1])}')
            x= tuple((input_root + line[0], label_root + line[1]))
           # print(f'xxxxxx----{x}')
            training_files.append(tuple((input_root + line[0], label_root + line[1])))
    if "S1" in fname:
        return download_flood_water_data_from_list_s1(training_files)
    else:
        return download_flood_water_data_from_list_s2(training_files)


def load_flood_val_data(input_root, label_root, fname):
    #print("this is file name:",fname)
    #print(f'input root: {input_root}, {label_root}, {fname}')
    fname=os.path.join(input_root, fname)
    #print("this is file name:",fname)
    val_files = list()
    with open(fname) as f:
        for line in csv.reader(f):
            #print(line)
            val_files.append(tuple((input_root + line[0], label_root + line[1])))

    if "S1" in fname:
        return download_flood_water_data_from_list_s1(val_files)
    else:
        return download_flood_water_data_from_list_s2(val_files)


def load_flood_test_data(input_root, label_root, fname):
    testing_files = list()
    with open(fname) as f:
        for line in csv.reader(f):
            testing_files.append(tuple((input_root + line[0], label_root + line[1])))

    if "S1" in fname:
        return download_flood_water_data_from_list_s1(testing_files)
    else:
        return download_flood_water_data_from_list_s2(testing_files)
