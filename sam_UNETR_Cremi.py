import os
import argparse
import numpy as np

import torch
import torch_em
from torch_em.model import UNETR
from torch_em.data.datasets import get_cremi_loader, get_cremi_dataset
from torch_em.data import ConcatDataset




def identity(raw):
        return raw


def do_unetr_training(args, data_path: str, save_root: str, iterations: int, device, patch_shape=(1, 512, 512)):
    os.makedirs(data_path, exist_ok=True)

    

    cremi_train_rois_A = {"A": np.s_[0:75, :, :]}
    cremi_train_rois_B = {"B": np.s_[0:75, :, :]}
    cremi_train_rois_C = {"C": np.s_[0:75, :, :]}



    cremi_train_dataset_A= get_cremi_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            rois=cremi_train_rois_A,
            ndim=2,
            defect_augmentation_kwargs=None,
            boundaries=True,
            raw_transform = identity,
            n_samples=args.n_samples,
            samples=("A")

    )

    cremi_train_dataset_B= get_cremi_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            rois=cremi_train_rois_B,
            ndim=2,
            defect_augmentation_kwargs=None,
            boundaries=True,
            raw_transform = identity,
            n_samples=args.n_samples,
            samples=("B")

    )

    cremi_train_dataset_C= get_cremi_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            rois=cremi_train_rois_C,
            ndim=2,
            defect_augmentation_kwargs=None,
            boundaries=True,
            raw_transform = identity,
            n_samples=args.n_samples,
            samples=("C")

    )

    cremi_train_dataset = ConcatDataset(cremi_train_dataset_A, cremi_train_dataset_B, cremi_train_dataset_C)
    train_loader = torch_em.get_data_loader(cremi_train_dataset, batch_size = 2, num_workers=16,shuffle=True)
    


    cremi_val_rois_A = {"A": np.s_[75:100, :, :]}
    cremi_val_rois_B = {"B": np.s_[75:100, :, :]}
    cremi_val_rois_C = {"C": np.s_[75:100, :, :]}



    cremi_val_dataset_A= get_cremi_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            rois=cremi_val_rois_A,
            ndim=2,
            defect_augmentation_kwargs=None,
            boundaries=True,
            raw_transform = identity,
            samples=("A")

    )

    cremi_val_dataset_B= get_cremi_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            rois=cremi_val_rois_B,
            ndim=2,
            defect_augmentation_kwargs=None,
            boundaries=True,
            raw_transform = identity,
            samples=("B")

    )

    cremi_val_dataset_C= get_cremi_dataset(
            path=data_path,
            patch_shape=patch_shape,
            download=True,
            rois=cremi_val_rois_C,
            ndim=2,
            defect_augmentation_kwargs=None,
            boundaries=True,
            raw_transform = identity,
            samples=("C")

    )

    cremi_val_dataset = ConcatDataset(cremi_val_dataset_A, cremi_val_dataset_B, cremi_val_dataset_C)
    val_loader = torch_em.get_data_loader(cremi_val_dataset, batch_size = 1, num_workers=16,shuffle=True)


    

    
    model = UNETR(
        backbone=args.backbone, encoder=args.encoder, out_channels=1, use_sam_stats=True,
        encoder_checkpoint=args.checkpoint, final_activation="Sigmoid",
    )
    print("UNETR Model successfully created and encoder initialized from", args.checkpoint)
    model.to(device)

    trainer = torch_em.default_segmentation_trainer(
        name="unetr-cremi",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-5,
        log_image_interval=10,
        save_root=save_root,
        compile_model=False,
        mixed_precision=True
    )

    trainer.fit(iterations)


def main(args):
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "GPU not available, hence running on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        print("Training a 2D UNETR on Cremi dataset")
        do_unetr_training(
            args,
            data_path=args.inputs,
            save_root=args.save_root,
            iterations=args.iterations,
            device=device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Enables UNETR training on Cremi dataset")
    parser.add_argument("-i", "--inputs", type=str, default="./cremi/",
                        help="Path where the dataset already exists/will be downloaded by the dataloader")
    parser.add_argument("-s", "--save_root", type=str, default=None,
                        help="Path where checkpoints and logs will be saved")
    parser.add_argument("--iterations", type=int, default=100000, help="No. of iterations to run the training for")
    parser.add_argument("--checkpoint")
    parser.add_argument("--encoder", default="vit_b")
    parser.add_argument("--backbone")
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()
    main(args)

#--train --inputs /scratch-grete/usr/nimmahen/data/Cremi --save_root /scratch-grete/usr/nimmahen/models/UNETR/sam/checkpoints/new_cremi_10persample_vit_b --checkpoint /scratch-grete/usr/nimmahen/models/SAM/checkpoints/sam_vit_b_01ec64.pth --iterations 10000 --encoder "vit_b" --backbone "sam" --n_samples 10
