import torch
import os
from torch.utils.data import DataLoader
from utils import setup_logger,MetricsStore
import os
from tqdm import tqdm
from senet_hoa.callbacks.evaluation import ModelValidationCallback

from contextlib import nullcontext
import json

class Trainer:
    def __init__(self, base_obj):

        self.__dict__.update(base_obj.get_all_attributes())

        #Create a dictionary to save the configuration files
        all_configs = self.__dict__.copy()
        all_configs = {k:v for k,v in all_configs.items() if type(v) in {int,float,str,bool}}

        self.train_sampler = None
        self.model = self.model.to(self.device)

        self.start_epoch = 0
        self.current_step=0
        if os.path.exists(os.path.join(self.OUTPUTDIR,"latest_model.pkl")):
            statedict = torch.load(os.path.join(self.OUTPUTDIR,"latest_model.pkl"))
            self.current_step = statedict["current_step"]
            self.start_epoch = self.current_step//self.steps_per_epoch
            self.model.load_state_dict(statedict["model_state_dict"])
            print("loaded model state from step: ",self.current_step)
            if self.scheduler is not None:
                for _ in range(self.current_step):
                    self.scheduler.step()

        collate_func=None
        if hasattr(self,"dataloder_collate"):
            print("Using collate function..")
            collate_func= self.dataloder_collate
        self.train_loader = DataLoader(self.train_dataset,collate_fn=collate_func ,batch_size=self.SAMPLES_PER_GPU, sampler=self.train_sampler, pin_memory=self.PIN_MEMORY, num_workers=self.NUM_WORKERS)
        
        os.makedirs(self.OUTPUTDIR,exist_ok=True)
        self.logger = setup_logger(os.path.join(self.OUTPUTDIR,"logs.txt"))
        self.metrics = MetricsStore()
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=1, pin_memory=self.PIN_MEMORY, num_workers=self.NUM_WORKERS_VAL,shuffle=False,collate_fn=self.inference_collator)
        self.evaluation_callback = ModelValidationCallback(self.model,self.metrics,self.valid_loader,self.PRED_THRESHOLD,self.device,self.OUTPUTDIR)

        with open(os.path.join(self.OUTPUTDIR,'configs.json'), 'w') as file:
            json.dump(all_configs, file, indent=4)
        
        self.train_context = nullcontext()
        self.accum_steps = self.GRADIENT_STEPS
    
    def continue_training(self,tolerance=5):
        score = self.metrics.get_metric_all("surfacedice")
        if max(score) not in score[-tolerance:]:
            return False
        if len(score)>tolerance and max(score)<0.1:
            return False
        return True
    #@profile
    def train_one_epoch(self,epoch,early_stop=False,tolerance=5):
        continue_training=True
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        tqdm_loader = tqdm(self.train_loader,desc=f"Train epoch: {epoch}")
        updatefreq=1

        self.optimizer.zero_grad()
        for i,batch in enumerate(tqdm_loader):
            outputs = self.model(batch[0].to(self.device), )
            loss = self.criterion(outputs, batch[1].to(self.device))/self.accum_steps
            loss.backward()
            if ((i + 1) % self.accum_steps == 0):
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.CLIP_NORM)
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            if i%updatefreq==0:
                if torch.isnan(loss):
                    print("Found nan loss")
                    continue_training=False
                tqdm_loader.set_description(f"loss: {loss.item():.8f} ")
            num_batches += 1
            self.current_step+=1
            if self.current_step%self.VALIDATION_FREQUENCY==0:
                self.optimizer.zero_grad()
                self.validate(self.current_step)
                self.logger.info(f"###Iter: {self.current_step}  ::  {self.metrics.get_metrics_by_epoch(self.current_step)}")
                if early_stop==True:
                    continue_training = self.continue_training(tolerance=tolerance)

        avg_loss = total_loss / num_batches                
        # all_losses = self.metrics.get_metric_all("training_loss")
        # all_losses.append(10000)
        # if avg_loss<=min(all_losses):
        #     self.validate(self.current_step)
        self.metrics(self.current_step,"training_loss",avg_loss)
        self.logger.info(f"###Iter: {self.current_step}  ::  {self.metrics.get_metrics_by_epoch(self.current_step)}")
        return continue_training
    def validate(self,current_step):
        self.model.eval()
        if self.evaluation_callback:
            self.evaluation_callback(current_step)
        self.model.train()

    def get_state_dict(self):
        return  self.model.state_dict()
    
    def _savemodel(self,current_step,path):
        torch.save({
            'current_step': current_step,
            'model_state_dict': self.get_state_dict(),
        }, path)

    def train(self,prune_epochs=10):
        print("Starting training....")
        for epoch in range(self.start_epoch,self.EPOCHS):
            continue_train = self.train_one_epoch(epoch,early_stop=True,tolerance=prune_epochs)
 
            self._savemodel(self.current_step,os.path.join(self.OUTPUTDIR,"latest_model.pkl"))

            if not continue_train:
                print(f"Early stopping ..... ")
                return -1

        self.metrics.to_dataframe().to_csv(os.path.join(self.OUTPUTDIR,"metrics.csv"))
        
        return -1
    