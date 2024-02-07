import torch
import glob
import torch_em
from torch_em.model import UNETR
from torch_em.data.datasets import get_cremi_loader
import imageio.v2 as imageio
from torch_em.util.prediction import predict_with_halo
import os
import numpy as np
import sys
import h5py
from skimage.io import imsave
import matplotlib.pyplot as plt
import argparse

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def save_data(slice_number, data,sample_n, output_dir):
    filename = f"{sample_n}-{slice_number:03d}.tif"
    filepath = os.path.join(output_dir, filename)
    imsave(filepath, data)

def identity(raw):
        return raw

cremi_test_rois = {"A": np.s_[100:125, :, :], "B": np.s_[100:125, :, :], "C": np.s_[100:125, :, :]}

test_loader = get_cremi_loader(
        path="/scratch-grete/usr/nimmahen/data/Cremi/",
        patch_shape=(1, 512, 512),
        download=True,
        rois=cremi_test_rois,
        ndim=2,
        defect_augmentation_kwargs=None,
        boundaries=True,
        batch_size=2,
        raw_transform = identity,
        num_workers=16,
        shuffle=False,
        samples=("A", "B", "C")
    )







def test_unetr(args, model_weights: str, pred_dir:str):



       
    # Path to the directory containing the H5 files
    test_path = "/scratch-grete/usr/nimmahen/data/Cremi/" 


    # List all the H5 files in the train_path directory
    h5_files = [file for file in os.listdir(test_path) if file.endswith('.h5')]


    model = UNETR(
        backbone=args.backbone, encoder=args.encoder, out_channels=1, use_sam_stats=True, 
        final_activation="Sigmoid",
    )

    model.out_channels=1


    # NOTE:
    # here's the implementation for the custom unpickling to ignore the preproc. fns.
    from micro_sam.util import _CustomUnpickler
    import pickle

    # over-ride the unpickler with our custom one
    custom_pickle = pickle
    custom_pickle.Unpickler = _CustomUnpickler

    state = torch.load(model_weights, map_location="cpu", pickle_module=custom_pickle)
    model_state = state["model_state"]
    model.load_state_dict(model_state)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    model.eval()

    with torch.no_grad():

        for h5_file in h5_files:
            h5_file_path = os.path.join(test_path, h5_file)
            sample_n =h5_file.split(".")[0]
            


            with h5py.File(h5_file_path, 'r') as h5_file:
                # Assuming 'volumes' is the subgroup containing 'raw' dataset
                volumes_group = h5_file['volumes']
                raw_group = volumes_group['raw']
                label_group = volumes_group['labels']
                annotation_group = label_group['neuron_ids']


                
                # Get the volume data (assuming it's a 3D volume in the 'raw' dataset)
                volume_data = np.array(raw_group)
                gt_data = np.array(annotation_group)
                
                
                
                # Assuming you want to access slices 100 to 125 for testing
                start_slice = 100
                end_slice = 125

                # Directory to save the output images, labels, and predictions
                output_dir = os.path.join(args.base_dir, pred_dir, "boundaries")
                os.makedirs(output_dir, exist_ok=True)
                        

                test_image_dir = "/scratch-grete/usr/nimmahen/data/Cremi/test_image/"
                os.makedirs(test_image_dir, exist_ok=True)

                test_label_dir = "/scratch-grete/usr/nimmahen/data/Cremi/test_label/"
                os.makedirs(test_label_dir, exist_ok=True)
                
                # Loop through the slices and save data with generic nomenclature
                for slice_number in range(start_slice, end_slice):
                    # Access the slice data
                    slice_data_org = volume_data[slice_number]
                    slice_gt = gt_data[slice_number]
                    slice_data = slice_data_org.astype(np.float32)
                
                
                  
    
                    # Assuming you generate prediction_data for each slice
                    predictions = predict_with_halo(slice_data, model, gpu_ids=[device], block_shape=(256,256), halo = (128,128),preprocess=None)
                    
                    
                    
                    # Save images, labels, and predictions
                    # save_data(slice_number, slice_data_org, sample_n, test_image_dir)
                    # save_data(slice_number, slice_gt, sample_n, test_label_dir)  # Replace with your ground truth data
                    save_data(slice_number, predictions, sample_n ,  output_dir)
                    
                
                
                    

        print("Data saved successfully with a generic naming convention.")

        


def main(args):
    
    test_unetr(
        args,
        model_weights=args.model_weights,
        pred_dir=args.pred_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="vit_b")
    parser.add_argument("--backbone")
    parser.add_argument("--model_weights")
    parser.add_argument("--base_dir", default="/scratch-grete/usr/nimmahen/models/UNETR/sam/prediction/")
    parser.add_argument("--pred_dir", required=True)
    args = parser.parse_args()
    main(args)
                        
###--encoder "vit_b" --backbone "sam" --model_weights /scratch-grete/usr/nimmahen/models/UNETR/sam/checkpoints/cremi_10_vit_b/checkpoints/unetr-cremi/best.pt --pred_dir cremi_10_vit_b
#/scratch-grete/usr/nimmahen/models/UNETR/sam/checkpoints/new_cremi_10persample_vit_b/checkpoints/unetr-cremi/best.pt
