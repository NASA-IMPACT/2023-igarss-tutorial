import argparse
from mmcv import Config
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,wrap_fp16_model)
from mmseg.models import build_segmentor

import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor

from mmseg.datasets import build_dataloader, build_dataset, load_flood_test_data
import rasterio
import torch

from torchvision import transforms
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmseg.apis import multi_gpu_test, single_gpu_test, init_segmentor
import pdb

import numpy as np
import glob
import os

import time

def parse_args():
    
    parser = argparse.ArgumentParser(description="Inference on flood detection fine-tuned model")
    parser.add_argument('-config', help='path to model configuration file')
    parser.add_argument('-ckpt', help='path to model checkpoint')
    parser.add_argument('-input', help='path to input images folder for inference')
    parser.add_argument('-output', help='path to save output image')
    parser.add_argument('-input_type', help='file type of input images',default="tif")
    
    args = parser.parse_args()
    
    return args

def open_tiff(fname):
    
    with rasterio.open(fname, "r") as src:
        
        data = src.read()
        
    return data

def write_tiff(img_wrt, filename, metadata):

    """
    It writes a raster image to file.

    :param img_wrt: numpy array containing the data (can be 2D for single band or 3D for multiple bands)
    :param filename: file path to the output file
    :param metadata: metadata to use to write the raster to disk
    :return:
    """

    with rasterio.open(filename, "w", **metadata) as dest:

        if len(img_wrt.shape) == 2:
            
            img_wrt = img_wrt[None]

        for i in range(img_wrt.shape[0]):
            dest.write(img_wrt[i, :, :], i + 1)
            

def get_meta(fname):
    
    with rasterio.open(fname, "r") as src:
        
        meta = src.meta
        
    return meta

def preprocess_image(data, nodata=-9999):
    
    data=np.where(data == nodata, 0, data)
    data = data.astype(np.float32)*0.0001
    
    if len(data)==2:
        (x, y) = data
    else:
        x=data
        y=np.full((x.shape[-2], x.shape[-1]), -1)
        
    im, label = x.copy(), y.copy()
    label = label.astype(np.float64)

    im1 = im[0]  # red
    im2 = im[1]  # green
    im3 = im[2]  # blue
    im4 = im[3]  # NIR narrow

    dim = x.shape[-1]
    label = label.squeeze()

    norm = transforms.Normalize([0.21531178, 0.20978154, 0.18528642, 0.48253757], [0.10392396, 0.10210076, 0.11696766, 0.19680527])
    ims = [torch.stack((transforms.ToTensor()(im1).squeeze(),
                        transforms.ToTensor()(im2).squeeze(),
                        transforms.ToTensor()(im3).squeeze(),
                        transforms.ToTensor()(im4).squeeze()))]
    ims = [norm(im) for im in ims]
    ims = torch.stack(ims)
    
    label = transforms.ToTensor()(label).squeeze()

    _img_metas = {
        'ori_shape': (dim, dim),
        'img_shape': (dim, dim),
        'pad_shape': (dim, dim),
        'scale_factor': [1., 1., 1., 1.],
        'flip': True,
    }

    img_metas = [_img_metas] * 1
    return {"img": ims,
            "img_metas": img_metas,
            "gt_semantic_seg": label}
    

def load_model(config, ckpt):
    
    print('Loading configuration...')
    cfg = Config.fromfile(config)
    print('Building model...')
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    print('Loading checkpoint...')
    checkpoint = load_checkpoint(model,ckpt, map_location='cpu')
    print('Evaluating model...')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    return model


def inference_on_file(model, target_image, output_image):
    
    try:
        st = time.time()
        data_orig = open_tiff(target_image)
        meta = get_meta(target_image)
        nodata = meta['nodata'] if meta['nodata'] is not None else -9999

        data = preprocess_image(data_orig, nodata)

        print('Running inference...')
        with torch.no_grad():
            result = model([data['img']], [data['img_metas']], return_loss=False, rescale=False)

        print("Output has shape: " + str(result[0].shape))
        #### TO DO: Post process (e.g. morphological operations)


        ##### Save file to disk
        meta["count"] = 1
        meta["dtype"] = "int16"
        meta["compress"] = "lzw"
        meta["nodata"] = -1
        meta["nodata"] = nodata
        print('Saving output...')
        # pdb.set_trace()
        result = np.where(data_orig[0] == nodata, nodata, result[0])
        
        write_tiff(result, output_image, meta)
        et = time.time()
        print(f'Inference completed in {str(np.round(et - st, 1))} seconds. Output available at: ' + output_image)
      
    except:
        print(f'Error on image {image} \nContinue to next input')
    
    

def main():
    
    args = parse_args()
    
    model = load_model(args.config, args.ckpt)
    
    target_images = glob.glob(args.input+"*."+args.input_type)
    
    print('Identified images to predict on: ' + str(len(target_images)))    
    
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
        
    for i, target_image in enumerate(target_images):
        
        print(f'Working on Image {i}')
        output_image = args.output+target_image.split("/")[-1].split(".")[0]+'_pred.'+args.input_type
        
        inference_on_file(model, target_image, output_image)
        
    
if __name__ == "__main__":
    main()