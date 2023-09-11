#   python ./src/train.py /root/repo/help-repo/kaggle-pneumothorax/configs/resnet34_768_unet.yaml --fold=0

from collections import defaultdict
from typing import Dict
from typing import List
import sys 
sys.path.append ("/root/repo/siim-help/kaggle-pneumothorax/src")
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from .utils import batch2device
from utils import batch2device
tqdm.monitor_interval = 0
import matplotlib.pyplot as plt
import numpy as np
import shutil 
import os
import random
class Runner:
    def __init__(self, factory, device, callbacks=None, stages: Dict[str, dict] = None):
        self.factory = factory
        self.device = device
        self.stages = stages

        self._model = None
        self._loss = None
        self._metrics = None

        self.current_stage = None
        self.current_stage_name = None
        self.global_epoch = 0
        self.optimizer = None
        self.scheduler = None

        self.callbacks = callbacks
        if callbacks is not None:
            self.callbacks.set_runner(self)
        self.writer = SummaryWriter()


        self.sample_path="/root/repo/siim-help/kaggle-pneumothorax/figures/samples"
        shutil.rmtree(self.sample_path)
        os.makedirs(self.sample_path,exist_ok=True)

    @property
    def model(self):
        if self._model is None:
            self._model = self.factory.make_model(device=self.device)
        return self._model

    @property
    def loss(self):
        if self._loss is None:
            self._loss = self.factory.make_loss(device=self.device)
        return self._loss

    @property
    def metrics(self):
        if self._metrics is None:
            self._metrics = self.factory.make_metrics()
        return self._metrics

    def fit(self, data_factory):
        self.callbacks.on_train_begin()

        for stage_name, stage in self.stages.items():
            self.current_stage = stage
            self.current_stage_name = stage_name

            train_loader = data_factory.make_train_loader()
            val_loader = data_factory.make_val_loader()
            self.optimizer = self.factory.make_optimizer(self.model, stage)
            self.scheduler = self.factory.make_scheduler(self.optimizer, stage)

            self.callbacks.on_stage_begin()
            self._run_one_stage(train_loader, val_loader)
            self.callbacks.on_stage_end()
            torch.cuda.empty_cache()
        self.callbacks.on_train_end()

    def _run_one_stage(self, train_loader, val_loader):

        # print ("self epoch", self.current_stage['epochs'])
        print ("len", len(train_loader), len (val_loader))
        grad_dict={}   
        loss_dict={}
        dice_dict={}
        loss_dict["train"]=[]
        loss_dict["val"]=[]
        dice_dict["val"]=[]
        train_samples=[]
        val_samples=[]
        for epoch in range(self.current_stage['epochs']):
            train_loader.sampler.set_epoch(epoch)
            print(f'positive ratio: {train_loader.sampler.positive_ratio}')
            self.callbacks.on_epoch_begin(self.global_epoch)

            self.model.train()
            self.metrics.train_metrics,grad = self._run_one_epoch(epoch, train_loader, is_train=True)

            grad_dict[f'{epoch}']=grad
            loss_dict["train"].append(self.metrics.train_metrics["loss"])

            

            self.model.eval()
            self.metrics.val_metrics, _ = self._run_one_epoch(epoch, val_loader, is_train=False)

            loss_dict["val"].append(self.metrics.val_metrics["loss"] )
            dice_dict["val"].append (self.metrics.val_metrics["DiceMetric"])
           

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(self.metrics.val_metrics['dice'], epoch)
            else:
                self.scheduler.step(epoch)
            self.callbacks.on_epoch_end(self.global_epoch)
            self.global_epoch += 1
        

        gradient_path="/root/repo/siim-help/kaggle-pneumothorax/figures/grad.png"
        metric_path='/root/repo/siim-help/kaggle-pneumothorax/figures/metric.png'
        train_visual='/root/repo/siim-help/kaggle-pneumothorax/figures/trainsample.png'
        val_visual='/root/repo/siim-help/kaggle-pneumothorax/figures/valsample.png'
        plot_epoch_gradient(gradient_path,grad_dict)
        plot_metric(metric_path,dice_dict,loss_dict)
        # plot_sample(train_visual,train_samples)
        # plot_sample(val_visual,val_samples)


    def _run_one_epoch(self, epoch: int, loader, is_train: bool = True) -> Dict[str, float]:
        grads=[]
        image_sample=True
        n_samples=10
        samples=[]
        count=n_samples


        epoch_report = defaultdict(float)
        progress_bar = tqdm(
            iterable=enumerate(loader),
            total=len(loader),
            desc=f"Epoch {epoch} {['validation', 'train'][is_train]}ing...",
            ncols=0
        )
        metrics = {}
        with torch.set_grad_enabled(is_train):
            for i, data in progress_bar:
                self.callbacks.on_batch_begin(i)
                step_report , sample= self._make_step(data, is_train,count)
                if sample is not None and count >0:
                    samples.append(sample)
                    count-=1

                for key, value in step_report.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    epoch_report[key] += value

                if is_train:
                    grads.append(step_report["grad"].item())

                metrics = {k: v / (i + 1) for k, v in epoch_report.items()}
                
                progress_bar.set_postfix(**{k: f'{v:.5f}' for k, v in metrics.items()})
                self.callbacks.on_batch_end(i, step_report=step_report, is_train=is_train)
        
        print (f" {['validation', 'train'][is_train]} {metrics}")

        epoch_sample_path=os.path.join(self.sample_path,f" {['validation', 'train'][is_train]}_epoch_{epoch}.png")
        plot_sample(epoch_sample_path,samples)
        return metrics,grads

    def _make_step(self, data: Dict[str, torch.Tensor], is_train: bool,count) -> Dict[str, float]:
        
        sample=None
        prob=0.1

        report = {}
        data = self.batch2device(data)
        images = data['image'].float()

        labels = data['mask'].float()


        labels=labels.unsqueeze(1)
        non_empty_labels = data['non_empty']
        predictions, empty_predictions = self.model(images)
        if np.random.rand()<prob and count>0:

            sample=[images.detach().cpu().numpy(),predictions.detach().cpu().numpy(),labels.detach().cpu().numpy()]

        loss = self.loss(predictions, labels, empty_predictions, non_empty_labels)
        report['loss'] = loss.data

        if is_train:
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            report['grad'] = grad_norm
            self.optimizer.step()
            self.optimizer.zero_grad()

            inp=torch.sigmoid(predictions.detach()).cpu().numpy()
            target=labels.detach().cpu().numpy()
            for metric, f in self.metrics.functions.items():

                val=f(inp, target)

                report[metric] = val
            
        else:
            for metric, f in self.metrics.functions.items():
                
                report[metric] = f(torch.sigmoid(predictions).cpu().numpy(), labels.cpu().numpy())

        return report,sample

    def batch2device(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return batch2device(data, device=self.device)
    
def plot_epoch_gradient(save_path,gradient: Dict):
    # print ("len ls",len(gradient))
    
    assert len(gradient)>1 , "must >1 epoch"

    fig,ax= plt.subplots(len(gradient),1)
    plt.tight_layout()
    for i , (epoch,grad )in enumerate(gradient.items()):

        ax[i].plot (range (len(grad)),grad, label =f" grad epoch {epoch}")
        # ax[i].set_title(f" grad norm of epoch {epoch}")
        ax[i].legend()
    plt.show()
    plt.savefig(save_path)
    plt.clf()
def plot_metric(save_path,dice:Dict, loss:Dict):

    fig,ax= plt.subplots(2,1)
    plt.tight_layout()
    for i , (mode,metric ) in enumerate (dice.items()):
        ax[0].plot ( range (1, len(metric)+1),metric, label=f"dice {mode} ")
        ax[0].legend()
    for j , (mode,metric ) in enumerate (loss.items()):
        ax[1]. plot ( range (1, len(metric)+1), metric,label=f"loss {mode} ")
        ax[1].legend()
    plt.show()
    plt.savefig(save_path)
    plt.clf()

def plot_sample (save_path,samples):
    assert (len(samples)>1) , "only plot for >1 samples"

   
    fig,ax=plt.subplots(len(samples),5)


    for i, sample in enumerate (samples):

        img=sample[0]
        pred=sample[1]
        label=sample[2]
        # get a sample in a batch
        idx=random.choice(range(img.shape[0]))
        img=img[idx]
        pred=np.squeeze(pred[idx])
        pred[pred<0.5]=0
        pred[pred>0.5]=1
        label=np.squeeze(label[idx])
        ax[i,0].imshow(np.transpose(img, [1,2,0]) )
        ax[i,1].imshow(pred,cmap="Reds")
        ax[i,2].imshow(np.transpose(img, [1,2,0]))
        ax[i,2].imshow(pred,cmap="Reds", alpha=0.5)
        ax[i,3].imshow(label,cmap="Reds")
        ax[i,4].imshow(np.transpose(img, [1,2,0]))
        ax[i,4].imshow(label,cmap="Reds",alpha=0.5)
    plt.show()
    plt.savefig(save_path)
    plt.clf()





if __name__=="__main__":
    path="/root/repo/siim-help/kaggle-pneumothorax/figures/grad.png"
    grads={"1":[1,1.3,1.9],"2":[2,3,1]}
    path2="/root/repo/siim-help/kaggle-pneumothorax/figures/metric.png"
    dice={"val":[1,1,2]}
    loss={"train": [1,2,3],"val":[1,1,2]}
    plot_metric(path2, dice, loss)
    plot_epoch_gradient(path,grads)

    samples=[(np.random.rand(3,256,256),np.random.rand(256,256),np.random.rand(256,256)),(np.random.rand(3,256,256),np.random.rand(256,256),np.random.rand(256,256))]
    save="/root/repo/siim-help/kaggle-pneumothorax/figures/compar.png"
    plot_sample(save,samples)



