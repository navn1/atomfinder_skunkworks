# -*- coding: utf-8 -*-
"""
Scripts to publish generalized model followed with LG filter. 
Resize the input image
(set as 1 to remain the same, set greater than 1 to upsampling and smaller than 1 to downsampling)
"""
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import ndimage
from typing import List, Union
from fcn import SegResNet
from predictor import SegPredictor


def map8bit(data):
    return ((data - data.min()) / (data.max() - data.min()) * 255).astype('int8')


def run(inputdict, *args):
    imarray_original = inputdict['image']
    resize = inputdict['change_size']
    glfilter_sigma = inputdict['glfilter_sigma']
    w = torch.load('model_metadict_final.tar', map_location=torch.device('cpu'))

    model = SegResNet()
    model.load_state_dict(list(w.items())[8][1])
    modelpredictor = SegPredictor(model, resize=(
    int(imarray_original.shape[1] * resize), int(imarray_original.shape[0] * resize)),
                                  use_gpu=False, logits=True, nnfilter='gaussian_laplace', filter_thresh=0.02,
                                  glfilter_sigma=glfilter_sigma, )
    nn_output, coor = modelpredictor.run(imarray_original, compute_coords=True)
    predictpos_y, predictpos_x, _ = coor[0].T
    return predictpos_x, predictpos_y


def inference(dicts: Union[List[dict], dict]):
    """
    Run a model to predict atom column coordinates in the input STEM images
    change_size: set as 1 to remain the same, set greater than 1 to upsampling and smaller than 1 to downsampling
    filter_sigma: the sigma value for the Laplasian Gaussian filter. Default value is 3. 
    Example:

    >>> # Make a dict of your inputs
    >>> # use a list of dicts if you want to run it though multiple images at one time
    >>> dict = {'image': array,'change_size': 1, 'filter_sigma': 3} 

    """
    results = []
    if isinstance(dicts, list):
        for i in range(len(dicts)):
            results.append(run(dicts[i]))
        return results
    elif isinstance(dicts, dict):
        results.append(run(dicts))
        return results
    else:
        raise AssertionError("Input should be dict or list of dict")
