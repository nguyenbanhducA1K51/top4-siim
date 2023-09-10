import numpy as np

EPS = 1e-10


class DiceMetric:
    def __init__(self, score_threshold=0.5):
        self.score_threshold = score_threshold

    def __call__(self, predictions, gt):

        mask = predictions > self.score_threshold
        batch_size = mask.shape[0]

        mask = mask.reshape(batch_size, -1).astype(int)
        gt = gt.reshape(batch_size, -1).astype(int)

        intersection =2*(mask*gt).sum(1)+EPS

        union =(mask+gt).sum(1)+EPS

        loss=intersection/union    
        
     

        return loss.mean()

# EPS = 1e-4


class DiceMetric2:
    def __init__(self, score_threshold=0.5):
        self.score_threshold = score_threshold

    def __call__(self, predictions, gt):

        mask = predictions > self.score_threshold
        batch_size = mask.shape[0]

        mask = mask.reshape(batch_size, -1).astype(bool)
        gt = gt.reshape(batch_size, -1).astype(bool)

        intersection = np.logical_and(mask, gt).sum(axis=1)
        union = mask.sum(axis=1) + gt.sum(axis=1) + EPS
        # print ("2", intersection,union)
        loss = (2.0 * intersection + EPS) / union
        return loss.mean()

if __name__=="__main__":
    import torch
    d1=DiceMetric1()
    d2=DiceMetric2()
    x=torch.load("/root/repo/help-repo/kaggle-pneumothorax/figures/img.pt")
    y=torch.load("/root/repo/help-repo/kaggle-pneumothorax/figures/msk.pt")


    x= np.random.rand(8,1,512,512)
    y=np.random.rand(8,1,512,512)
    y=(y>0.5).astype(int)

    while d1(x,y)==d2(x,y):
        x= np.random.rand(8,1,512,512)
        y=np.random.rand(8,1,512,512)
        y=(y>0.5).astype(int)
    print (d1(x,y),d2(x,y))
        
   
   

