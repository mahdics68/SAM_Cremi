import numpy as np
import os
import glob
import imageio.v2 as imageio
import pandas as pd
from skimage.segmentation import find_boundaries
import sys
import torch_em
from skimage.io import imread
from skimage.transform import resize

#(if your ground truth is not binarised, make sure to put the parameter threshold_gt = 0.)


def dice_score(segmentation, groundtruth, threshold_seg=None, threshold_gt=None):
    """ Compute the dice score between binarized segmentation and ground-truth.
    Arguments:
        segmentation [np.ndarray] - candidate segmentation to evaluate
        groundtruth [np.ndarray] - groundtruth
        threshold_seg [float] - the threshold applied to the segmentation.
            If None the segmentation is not thresholded.
        threshold_gt [float] - the threshold applied to the ground-truth.
            If None the ground-truth is not thresholded.
    Returns:
        float - the dice score
    """
    assert segmentation.shape == groundtruth.shape, f"{segmentation.shape}, {groundtruth.shape}"
    if threshold_seg is None:
        seg = segmentation
    else:
        seg = segmentation > threshold_seg
    if threshold_gt is None:
        gt = groundtruth
    else:
        gt = groundtruth > threshold_gt

    nom = 2 * np.sum(gt * seg)
    denom = np.sum(gt) + np.sum(seg)

    eps = 1e-7
    score = float(nom) / float(denom + eps)
    return score

cell_types = ["sampleA-", "sampleB-","sampleC-"]
fg_eval_scores = np.zeros(3)
fg_final_list = []

bd_eval_scores = np.zeros(3)
bd_final_list = []



for ind,i in enumerate(cell_types):

    fg_dir = os.path.join("/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/cremi_75persample_vit_b/boundaries/", i+"*")
    n = 0 
    fg_dsl = []   
    fg_ds_dict = {}
    for fg_pred_seg in glob.glob(fg_dir):
        filename = os.path.split(fg_pred_seg)[-1]
        gt_path = os.path.join("/scratch-grete/usr/nimmahen/data/Cremi/test/test_label/", filename)
        

        fg_pred_y = imread(fg_pred_seg)
        fg_pred_y = np.squeeze(fg_pred_y, axis=0)
        breakpoint()
        fg_pred_y = np.where(fg_pred_y>.5,1,0) #changed to binarize
        gt_y = imread(gt_path)
        fg_ds = dice_score(fg_pred_y, gt_y, threshold_gt= 0,threshold_seg=None)
        fg_dsl.append(fg_ds)
        fg_ds_dict.update({filename: fg_ds})
    fg_dsl = sum(fg_dsl)/len(fg_dsl)
    fg_eval_scores[ind] = fg_dsl
    fg_final_list.append(fg_ds_dict)

    bd_dir = os.path.join("/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/cremi_75persample_vit_b/boundaries/", i+"*")
    n = 0 
    bd_dsl = []   
    bd_ds_dict = {}
    for bd_pred_seg in glob.glob(bd_dir):
        filename = os.path.split(bd_pred_seg)[-1]
        gt_path = os.path.join("/scratch-grete/usr/nimmahen/data/Cremi/test/test_label/", filename)
        

        bd_pred_y = imread(bd_pred_seg)
        bd_pred_y = np.squeeze(bd_pred_y, axis=0)
        bd_pred_y = np.where(bd_pred_y>.5,1,0) #changed to binarize
        gt_y = imread(gt_path)
        bd_gt = find_boundaries(gt_y,mode='thick').astype(np.float32)
        bd_ds = dice_score(bd_pred_y, bd_gt, threshold_gt=None, threshold_seg=None)
        bd_dsl.append(bd_ds)
        bd_ds_dict.update({filename: bd_ds})
    bd_dsl = sum(bd_dsl)/len(bd_dsl)
    bd_eval_scores[ind] = bd_dsl
    bd_final_list.append(bd_ds_dict)


#top_5_keys = sorted(final_list[0], key=lambda x: final_list[0][x], reverse=True)[:5]
fg_all_values = [value for dictionary in fg_final_list for value in dictionary.values()]

# Sort the values and get the top 5
fg_top_5_values = sorted(fg_all_values, reverse=True)[:5]

# Get the keys corresponding to the top 5 values
fg_top_5_keys = [key for dictionary in fg_final_list for key, value in dictionary.items() if value in fg_top_5_values]

with open(r'/home/nimmahen/code/results/UNETR_sam_cremi_75persample_vit_b_samples_BIN.txt', 'w') as fp:
    for item in fg_top_5_keys:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'/home/nimmahen/code/results/UNETR_sam_cremi_75persample_vit_b_BIN.txt', 'w') as fp:
    for item in fg_eval_scores:
        # write each item on a new line
        fp.write("%s\n" % item)

print('UNETR_sam_cremi_75persample_vit_b_BIN',fg_eval_scores)
#print('dict of images', final_list)



#top_5_keys = sorted(final_list[0], key=lambda x: final_list[0][x], reverse=True)[:5]
bd_all_values = [value for dictionary in bd_final_list for value in dictionary.values()]

# Sort the values and get the top 5
bd_top_5_values = sorted(bd_all_values, reverse=True)[:5]

# Get the keys corresponding to the top 5 values
bd_top_5_keys = [key for dictionary in bd_final_list for key, value in dictionary.items() if value in bd_top_5_values]

with open(r'/home/nimmahen/code/results/UNETR_sam_cremi_75persample_vit_b_samples_BIN.txt', 'w') as fp:
    for item in bd_top_5_keys:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'/home/nimmahen/code/results/UNETR_sam_cremi_75persample_vit_b_BIN.txt', 'w') as fp:
    for item in bd_eval_scores:
        # write each item on a new line
        fp.write("%s\n" % item)

print('UNETR_sam_cremi_75persample_vit_b_BIN',bd_eval_scores)
