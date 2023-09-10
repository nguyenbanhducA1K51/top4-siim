import os

import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2


class PneumothoraxDataset(Dataset):
    def __init__(self, data_folder, mode, transform=None,
                 fold_index=None, folds_distr_path=None):
        
        self.transform = transform
        self.mode = mode
        
        # change to your path
        self.train_image_path = '{}/img/'.format(data_folder)
        self.train_mask_path = '{}/msk/'.format(data_folder)
        self.test_image_path = '{}/test/'.format(data_folder)
        
        self.fold_index = None
        self.folds_distr_path = folds_distr_path

        folds = pd.read_csv(self.folds_distr_path)
        folds=folds[:20]
        folds.Fold = folds.Fold.astype(str)
        if self.mode == 'train':
            
            folds = folds[folds.Fold != fold_index]        
            self.train_list = folds.ImageId.values.tolist()
            self.exist_labels = folds.has_pneumo.values.tolist()

        self.set_mode(mode, fold_index)
        print ("len", mode,self.num_data)
        self.to_tensor = ToTensorV2()

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index

        if self.mode == 'train':
            folds = pd.read_csv(self.folds_distr_path)
            print (folds.columns)
            folds=folds[:20]
            folds.Fold = folds.Fold.astype(str)
            folds = folds[folds.Fold != fold_index]
            
            # self.train_list = folds.fname.values.tolist()
            # self.exist_labels = folds.exist_labels.values.tolist()
            self.train_list = folds.ImageId.values.tolist()
            # print (self.train_list)
            self.exist_labels = folds.has_pneumo.values.tolist()
            # self.num_data = len(self.train_list)
            self.num_data=len(folds)

        elif self.mode == 'val':
            folds = pd.read_csv(self.folds_distr_path)
            folds=folds[:20]
            folds.Fold = folds.Fold.astype(str)
            folds = folds[folds.Fold == fold_index]
            
            self.val_list = folds.ImageId.values.tolist()
            self.num_data = len(self.val_list)
            self.num_data=len(folds)

        elif self.mode == 'test':
            self.test_list = sorted(os.listdir(self.test_image_path))
            self.num_data = len(self.test_list)
            self.num_data=len(folds)

    def __getitem__(self, index):
        if self.fold_index is None and self.mode != 'test':
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return
        
        if self.mode == 'test':
            image = cv2.imread(os.path.join(self.test_image_path, self.test_list[index]), 1)
            
            if self.transform:
                sample = {"image": image}
                sample = self.transform(**sample)
                sample = self.to_tensor(**sample)
                image = sample['image']
            image_id = self.test_list[index].replace('.png', '')
            return image_id, image
        
        elif self.mode == 'train':

            # assert index<len(self.train_list) ,f"out bound {(self.__len__())},{index} ,{len(self.train_list)}"
            image = cv2.imread(os.path.join(self.train_image_path, f'{self.train_list[index]}.png'), 1)
            assert image is not None, "none image"
            if self.exist_labels[index] == 0:
                label = np.zeros((1024, 1024))
            else:
                label = cv2.imread(os.path.join(self.train_mask_path, f'{self.train_list[index]}.png'), 0)           

        elif self.mode == 'val':
            image = cv2.imread(os.path.join(self.train_image_path, self.val_list[index]), 1)
            label = cv2.imread(os.path.join(self.train_mask_path, self.val_list[index]), 0)

        if self.transform:
            sample = {"image": image, "mask": label}
            sample = self.transform(**sample)
            sample = self.to_tensor(**sample)
            image, label = sample['image'], sample['mask']
        return image, label
         
    def __len__(self):
        return self.num_data


from torch.utils.data.sampler import Sampler
class PneumoSampler(Sampler):
    def __init__(self, folds_distr_path, fold_index, demand_non_empty_proba):
        assert demand_non_empty_proba > 0, 'frequensy of non-empty images must be greater then zero'
        self.fold_index = fold_index
        self.positive_proba = demand_non_empty_proba
        
        self.folds = pd.read_csv(folds_distr_path)
        self.folds.Fold = self.folds.Fold.astype(str)
        self.folds = self.folds[self.folds.Fold != fold_index].reset_index(drop=True)

        self.positive_idxs = self.folds[self.folds.has_pneumo == 1].index.values
        self.negative_idxs = self.folds[self.folds.has_pneumo == 0].index.values

        self.n_positive = self.positive_idxs.shape[0]
        self.n_negative = int(self.n_positive * (1 - self.positive_proba) / self.positive_proba)
        
    def __iter__(self):
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_idxs)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative
if __name__=="__main__":
    import albumentations as albu
    import matplotlib.pyplot as plt
    import random
    data_folder="/root/data/siim_png_convert/"
    csv="/root/data/siim_png_convert/k_fold.csv"

    # print (d.index.tolist())
    df=pd.read_csv(csv)
    print (df.head())

    transform=albu.load("/root/repo/help-repo/pneumothorax-segmentation/unet_pipeline/transforms/train_transforms_complex_1024.json")
    dataset= PneumothoraxDataset(data_folder,mode='train',transform=transform,fold_index=1,folds_distr_path=csv)

    n=5
    repeat=4
    fig, ax = plt.subplots(n, repeat, figsize=(30, 30))
    
    print (len(dataset))
    for i in range (n):
        s=random.choice(range(len(dataset)))
        for  j in range (repeat):
            # ax[i,j].imshow(d.__getitem__(s)["image"][0],cmap="bone")
            dataset.__getitem__(s)
            ax[i,j].imshow( np.transpose(dataset.__getitem__(s)[0] ,(1,2,0)) ) 
    plt.show()
    plt.savefig("/root/repo/help-repo/pneumothorax-segmentation/unet_pipeline/transforms/samples.png")