
import torch
from tqdm import tqdm
import os
import numpy as np
from scipy import stats
from collections import defaultdict
from senet_hoa.utils.surface_dice_metric import compute_surface_dice_at_tolerance,compute_surface_distances
from senet_hoa.dataset.segment_3d_dataset import LargeImageInferenceCollator

def compute_2d_dice(gt,pred):
    return compute_surface_dice_at_tolerance(compute_surface_distances(gt,pred,(1,1)),0)

class ModelValidationCallback:
    def __init__(self,model,metrics,valid_loader,threshold=0.5,device="cpu",output_dir="./"):
        self.model = model
        self.metrics = metrics
        self.valid_loader = valid_loader
        self.threshold = threshold
        self.device = device
        self.output_dir = output_dir
        self.score =-1
    def _savemodel(self,current_step,path):
        torch.save({
            'current_step': current_step,
            'model_state_dict': self.model.state_dict(),
        }, path)

    def __call__(self, current_step):
        self._savemodel(current_step, os.path.join(self.output_dir, "latest_model.pkl"))
        
        # Initializing storage for overall truths and predictions
        ground_truth = []
        predictions = []
        categories = []
        all_scores=[]
        idx=0
        for images,coors,shape,mask_gt in tqdm(self.valid_loader):
            with torch.no_grad():
                prediction = self.model(images.to(self.device)).detach().cpu()
                prediction = torch.sigmoid(LargeImageInferenceCollator.combine_masks_into_image(shape,coors,prediction))
            x= mask_gt.squeeze(0).numpy()
            y = prediction.squeeze(0).squeeze(1).numpy()
            all_scores.append(compute_2d_dice(x>0,y>self.threshold))


            f,dim,_ = self.valid_loader.dataset.slices[idx]
            fname = os.path.split(self.valid_loader.dataset.files[f][0])[-1].split(".")[0]
            categories.append(f"{fname}_{dim}")
            idx+=1
        # all_scores = [compute_2d_dice(x>0,y>self.threshold) for x,y in zip(ground_truth,predictions)] 
        total_scores = [x for x in all_scores if not np.isnan(x)]
        score = np.mean(total_scores) if len(total_scores)>0 else 0
        self.metrics(current_step, f"surfacedice", score)

        for val in np.unique(categories):
            subset_scores = [x for i,x in enumerate(all_scores) if categories[i]==val]
            subset_scores = [x for x in subset_scores if not np.isnan(x)]
            subset_scores = np.mean(subset_scores) if len(subset_scores)>0 else 0
            self.metrics(current_step, f"surfacedice_{val}", subset_scores)

        if score>=self.score:
            print(f"saving best model.surfacedice improved from {self.score} to {score}")
            self._savemodel(current_step,os.path.join(self.output_dir,"bestmodel_opa.pkl"))
            self.score = score