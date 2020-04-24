import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras import backend as K
from scipy import ndimage

import code

import numpy

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = np.ones((16, 16)) if labels[idx] == 1 else np.zeros((16, 16))
            idx = idx + 1
    return im


def concatenate_images(img, gt_img):
    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def balance_data(data, labels):
    c0 = 0
    c1 = 0
    for val in labels:
        if val[0] == 0:
            c0 += 1
        else:
            c1 += 1
    min_ = np.min([c0, c1])
    ind_datac0 = [i for i,j in enumerate(labels) if j[0] == 0]
    ind_datac1 = [i for i,j in enumerate(labels) if j[0] == 1]
    balanced_data = np.asarray([y for x in [data[ind_datac0[:min_]], data[ind_datac1[:min_]]] for y in x])
    balanced_labels = np.asarray([y for x in [labels[ind_datac0[:min_]], labels[ind_datac1[:min_]]] for y in x])
    return balanced_data, balanced_labels


def print_img_and_gt_img(imgs, gt_imgs, indexes):
    for index in indexes:
        cimg = concatenate_images(imgs[index], gt_imgs[index])
        fig1 = plt.figure(figsize=(10, 10))
        plt.imshow(cimg, cmap='Greys_r')
        
        
def extract_patches(imgs, gt_imgs, patch_size):
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(len(imgs))]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(len(gt_imgs))]

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    
    return img_patches, gt_patches

def extract_patches_testing(imgs, patch_size):
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(len(imgs))]

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    
    return img_patches

def standardize_color_features(img_patches):
    mean_r = np.mean(img_patches[:,:,:,0])
    mean_g = np.mean(img_patches[:,:,:,1])
    mean_b = np.mean(img_patches[:,:,:,2])

    imgs_r = img_patches[:,:,:,0]
    imgs_g = img_patches[:,:,:,1]
    imgs_b = img_patches[:,:,:,2]

    std_imgs_r = imgs_r - mean_r 
    std_imgs_g = imgs_g - mean_g 
    std_imgs_b = imgs_b - mean_b

    std_r = np.std(imgs_r)
    std_g = np.std(imgs_g)
    std_b = np.std(imgs_b)

    if std_r > 0: 
        std_imgs_r /= std_r
    if std_g > 0: 
        std_imgs_g /= std_g
    if std_b > 0: 
        std_imgs_b /= std_b

    return np.stack((std_imgs_r, std_imgs_g, std_imgs_b), axis=3)


def split_data(X, y, train_ratio, seed):
    np.random.seed(seed)
    
    N = len(y)
    
    indices = np.random.permutation(N)
    index_split = int(train_ratio*N)
    index_train = indices[: index_split]
    index_test = indices[index_split :]
    
    X_train = X[index_train]
    y_train = y[index_train]
    X_test = X[index_test]
    y_test = y[index_test]
    return X_train, y_train, X_test, y_test

def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# rotate input images with angles submitted
def get_rotated_images(images, angles):
    rotated_images = [None]*(len(images)*len(angles))
    i = 0
    for angle in angles:
        for image in images:
            rotated_images[i] = ndimage.rotate(image, angle, mode='reflect', order=0, reshape=False)
            i += 1
    return rotated_images