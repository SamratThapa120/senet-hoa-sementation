import pandas as pd
import torch
import os 
from .base import Base
from senet_hoa.model.unet import TimmUnet_v2m
from senet_hoa.dataset.segment_3d_dataset import Slices3DDataset,LargeImageInferenceCollator
import albumentations as A
from senet_hoa.dataset.transforms import MaskBinarize,FilterSmallComponents,Normalize
import cv2

class Configs(Base):
    OUTPUTDIR=f"../workdir/{__name__}"

    TRAIN_DATA_PATHS=[
        ('/app/segment/dataset/train_3d/images/kidney_1_dense.npy','/app/segment/dataset/train_3d/masks/kidney_1_dense.npy',[0,1,2]),
        ('/app/segment/dataset/train_3d/images/kidney_1_voi.npy','/app/segment/dataset/train_3d/masks/kidney_1_voi.npy',[0,1,2]),
        ('/app/segment/dataset/train_3d/images/kidney_2.npy','/app/segment/dataset/train_3d/masks/kidney_2.npy',[0,1,2]),
    ]
    VALID_DATA_PATHS=[
        ('/app/segment/dataset/train_3d/images/kidney_3_dense.npy','/app/segment/dataset/train_3d/masks/kidney_3_dense.npy',[0]),
        ('/app/segment/dataset/train_3d/images/kidney_3_sparse.npy','/app/segment/dataset/train_3d/masks/kidney_3_sparse.npy',[0])
    ]

    CROP_SIZE=512
    OPTUNA_TUNING_TRAILS= 1000

    USE_DATASET_LEN=None   #Set to small number while debugging
    SAMPLES_PER_GPU=16
    N_GPU=1
    VALIDATION_BS=1
    PIN_MEMORY=True
    NUM_WORKERS=8
    NUM_WORKERS_VAL=6
    DISTRIBUTED=True

    LR=0.0001

    EPOCHS=50
    MIN_CONFIGS=2

    GRADIENT_STEPS=1
    VALIDATION_FREQUENCY=2  # Number of epochs

    CLIP_NORM=1e-2

    PRUNING_TOLERANCE=10
    PRED_THRESHOLD= 0.5
    def __init__(self,is_inference=False,sample_valid=True):
        self.device = "cuda"

        self.model = TimmUnet_v2m(encoder="tf_efficientnetv2_m_in21k", in_chans=1, num_class=1, pretrained=not is_inference)

        test_albums = [
            A.PadIfNeeded(min_height=self.CROP_SIZE,min_width=self.CROP_SIZE,value=80,mask_value=0,border_mode=cv2.BORDER_CONSTANT),
            MaskBinarize(),
            Normalize(),
        ]
        self.test_transform = A.Compose(test_albums)
        self.inference_collator = LargeImageInferenceCollator(self.CROP_SIZE,strides=7*self.CROP_SIZE//8)

        if is_inference:
            return
        
        train_albums = [
            A.PadIfNeeded(min_height=self.CROP_SIZE,min_width=self.CROP_SIZE,value=80,mask_value=0,border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=90,value=80,mask_value=0,border_mode=cv2.BORDER_CONSTANT,p=0.5),
            A.RandomCrop(self.CROP_SIZE,self.CROP_SIZE,always_apply=True),
            MaskBinarize(),
            Normalize(),
        ]
        self.train_transform = A.Compose(train_albums)
        self.train_dataset = Slices3DDataset(self.TRAIN_DATA_PATHS,transforms=self.train_transform)
        self.valid_dataset = Slices3DDataset(self.VALID_DATA_PATHS,transforms=self.test_transform)
        if sample_valid:
            self.valid_dataset.slices = [self.valid_dataset.slices[x] for x in range(0,len(self.valid_dataset.slices),len(self.valid_dataset.slices)//500)]
        print(f"length of train: {len(self.train_dataset)}, length of valid: {len(self.valid_dataset)}")
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.LR)
        self.steps_per_epoch = len(self.train_dataset)//(self.SAMPLES_PER_GPU*self.N_GPU)+1
        self.VALIDATION_FREQUENCY  = self.VALIDATION_FREQUENCY * self.steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.LR,steps_per_epoch=self.steps_per_epoch,epochs=self.EPOCHS,pct_start=0.1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
