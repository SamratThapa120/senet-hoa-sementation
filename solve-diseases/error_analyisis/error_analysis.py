import torch
from tqdm import tqdm
import numpy as np
import sys
import os
from torch.utils.data import DataLoader
sys.path.append("../")
import importlib

from configs.baseline_effnetb7_3planes_v0_noinfoloss import Configs

import argparse
import torch
from tqdm import tqdm
import os
import numpy as np
from senet_hoa.utils.surface_dice_metric import compute_surface_dice_at_tolerance,compute_surface_distances
from senet_hoa.dataset.segment_3d_dataset import LargeImageInferenceCollator

import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
import copy

NAMES = {1:"FN",2:"FP",3:"TP"}

def compute_2d_dice(gt,pred):
    return compute_surface_dice_at_tolerance(compute_surface_distances(gt,pred,(1,1)),0)

class ModelValidationCallback:
    def __init__(self,model,valid_loader,threshold=0.5,device="cpu",output_dir="./"):
        self.model = model
        self.valid_loader = valid_loader
        self.threshold = threshold
        self.device = device
        self.output_dir = output_dir
        self.score =-1

    def __call__(self, current_step):
        # Initializing storage for overall truths and predictions
        ground_truth = []
        predictions = []
        
        for images,coors,shape,mask_gt in tqdm(self.valid_loader):
            with torch.no_grad():
                prediction = self.model(images.to(self.device)).detach().cpu()
                prediction = torch.sigmoid(LargeImageInferenceCollator.combine_masks_into_image(shape,coors,prediction))
            ground_truth.append(mask_gt.squeeze(0).numpy())
            predictions.append(prediction.squeeze(0).squeeze(1).numpy())
        return ground_truth,predictions

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='error analysis with 3d plot')

    parser.add_argument('--module_name', type=str, default="baseline_effnetb7_3planes_v0_noinfoloss",help='Name of the config to process')
    parser.add_argument('--downsample',type=str, default=8, help='downsampling rate')

    args = parser.parse_args()

    module = importlib.import_module(f"configs.{args.module_name}")

    cfg : Configs = module.Configs(sample_valid=False) 
    cfg.load_state_dict(os.path.join(cfg.OUTPUTDIR,"bestmodel_opa.pkl"))
    cfg.model.to(cfg.device)
    cfg.model.eval()
    all_slices = copy.deepcopy(cfg.valid_dataset.slices)
    for slices in np.unique([f"{x[0]}__{x[1]}" for x in cfg.valid_dataset.slices]):
        print("processing:",slices)
        IDX,DIM = [int(x) for x in slices.split("__")]
        cfg.valid_dataset.slices = [x for x in all_slices if x[0]==IDX and x[1]==DIM]

        valid_loader = DataLoader(cfg.valid_dataset, batch_size=1, pin_memory=cfg.PIN_MEMORY, num_workers=cfg.NUM_WORKERS_VAL,shuffle=False,collate_fn=cfg.inference_collator)
        evaluation_callback = ModelValidationCallback(cfg.model,valid_loader,cfg.PRED_THRESHOLD,cfg.device)
        gt,preds = evaluation_callback(0)

        gt = np.stack(gt)
        preds = np.stack(preds)

        preds = (preds>cfg.PRED_THRESHOLD).astype(np.uint8)
        gt = (gt>0).astype(np.uint8)

        TP = np.where((gt==1)&(preds==1),3,0)
        FP = np.where((gt==0)&(preds==1),2,0)
        FN = np.where((gt==1)&(preds==0),1,0)
        array_3d = TP+FP+FN
        length,breadth,height = array_3d.shape

        n = args.downsample  # Sampling rate, adjust as needed
        sampled_array = array_3d[::n, ::n, ::n]

        # Extract the points where the value needs to be plotted

        data = []
        for category in [1,2,3]:
            x, y, z = np.where(sampled_array==category)  # Adjust condition as per your data
            data.append(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=2,
                    opacity=0.8
                ),
                name=NAMES[category],  # This will create a legend entry for each category
            ))
        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(nticks=4, range=[0, length//n]),
                yaxis=dict(nticks=4, range=[0, breadth//n]),
                zaxis=dict(nticks=4, range=[0, height//n])
            ),
            legend=dict(
                x=1, 
                y=0,
                xanchor='right',
                yanchor='bottom',
                font=dict(
                    size=12  # Adjusting font size for legend items
                )
            )
        )

        fig = go.Figure(data=data, layout=layout)

        # Export to HTML
        html_filename = os.path.join(cfg.OUTPUTDIR,f'3d_plot_{IDX}_{DIM}.html')
        pyo.plot(fig, filename=html_filename, auto_open=False)

        print(f"Interactive 3D plot saved as {html_filename}")
