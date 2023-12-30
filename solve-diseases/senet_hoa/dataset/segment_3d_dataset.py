import glob
from typing import Any
import numpy as np
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import random
import glob
from tqdm import tqdm
import cv2

import numpy as np

def sample_plane(images, masks, axis, index, slice=0):
    # Determine the shape of the images and masks
    shape = images.shape

    # Initialize start and end indices for slicing
    start_index = max(index - slice, 0)
    end_index = min(index + slice + 1, shape[axis])

    # Perform slicing based on the axis
    if axis == 0:  # Sampling along x-axis
        plane_images = images[start_index:end_index, :, :]
        plane_masks = masks[index, :, :]
    elif axis == 1:  # Sampling along y-axis
        plane_images = images[:, start_index:end_index, :]
        plane_masks = masks[:, index, :]
    else:  # Sampling along z-axis
        plane_images = images[:, :, start_index:end_index]
        plane_masks = masks[:, :, index]

    # Determine the padding required
    padding = [(0, 0), (0, 0), (0, 0)]
    padding[axis] = (slice - (index - start_index), slice - (end_index - index - 1))

    # Apply zero padding
    plane_images = np.pad(plane_images, padding, mode='constant')

    if axis==0:
        plane_images = plane_images.transpose(1,2,0)
    elif axis==1:
        plane_images = plane_images.transpose(0,2,1)


    return plane_images, plane_masks


class Slices3DDataset(Dataset):
    def __init__(self,files,transforms=None,slice=0):
        """
            files: lise of list of format ["folder_path",["slice_dim1","slice_dim2"]]
        """
        self.three_d = [] 
        self.slices = []
        self.files = files
        for idx,(fname_image,fname_mask,dims) in enumerate(files):
            # masks = []
            # images = []
            # for image_path in tqdm(glob.glob(os.path.join(fname,"labels/*.tif"))):
            #     images.append(cv2.imread(image_path.replace("labels","images"),cv2.IMREAD_GRAYSCALE))
            #     masks.append(cv2.imread(image_path,cv2.IMREAD_GRAYSCALE))
            masks = np.load(fname_mask.replace("images","masks"))
            images = np.load(fname_image)
            assert masks.shape==images.shape, "shape of images and masks must be same."
            self.three_d.append((images,masks))
            for dim in dims:
                self.slices.extend([(idx,dim,slice_idx) for slice_idx in range(masks.shape[dim])])
        self.transforms = transforms
        self.slice = slice
    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        idx,dim,slice_idx = self.slices[index]
        image,mask = sample_plane(self.three_d[idx][0],self.three_d[idx][1],dim,slice_idx,self.slice)
        if self.transforms:
            image,mask = self.transforms(image=image, mask=mask).values()
        if self.slice==0:
            return torch.tensor(image).unsqueeze(0).float(),torch.tensor(mask).unsqueeze(0).float()
        else:
            return torch.tensor(image.transpose(2,0,1)).float(),torch.tensor(mask).unsqueeze(0).float()

    

class LargeImageInferenceCollator:
    def __init__(self,size,strides):
        self.size=size
        self.strides=strides
    
    def crop_image_into_chunks(self,image_tensor):
        _, height, width = image_tensor.shape
        crops = []
        coordinates = []

        # Adjusted iteration to ensure full coverage of the image
        for y in range(0, height, self.strides):
            for x in range(0, width, self.strides):
                # Adjust x and y if they would result in a crop smaller than self.size
                y_start = min(y, height - self.size)
                x_start = min(x, width - self.size)

                # Crop the image
                crop = image_tensor[:, y_start:y_start + self.size, x_start:x_start + self.size]
                crops.append(crop)

                # Store the coordinates
                coordinate = torch.tensor([y_start, y_start + self.size, x_start, x_start + self.size])
                coordinates.append(coordinate)
        return crops, coordinates
    
    @staticmethod
    def combine_masks_into_image(full_image_shape, crops_coordinates, prediction_masks):
        _, height, width = full_image_shape
        combined_mask = torch.zeros((1, height, width))
        count = torch.zeros((1, height, width))

        # Iterate over each crop
        for crop_coord, pred_mask in zip(crops_coordinates, prediction_masks):
            y1, y2, x1, x2 = crop_coord

            # Add the predicted mask to the combined mask
            combined_mask[:, y1:y2, x1:x2] += pred_mask

            # Increment the count
            count[:, y1:y2, x1:x2] += 1

        # Averaging the overlapping areas
        combined_mask /= count.clamp(min=1)  # Avoid division by zero

        return combined_mask

    def __call__(self, batch):
        assert len(batch)==1,"only batch size 1 supported currently"
        image_crops,coords = self.crop_image_into_chunks(batch[0][0])
        return torch.stack(image_crops),coords,batch[0][0].shape,batch[0][1]