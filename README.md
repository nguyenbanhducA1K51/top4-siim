the repo should be focused is kaggle-pneumothorax
original repo :https://github.com/amirassov/kaggle-pneumothorax
the dataset can be found at this https://www.kaggle.com/datasets/jesperdramsch/siim-acr-pneumothorax-segmentation-data

to prepare, run file ./kaggle-pneumothorax/srrc/prepare.py 
I only modify the datset.py (__getitem__) and add file transform to switch between 2 type of load data: read image in gray channel and stack 3 times, or use cv2.imread(img,1). For read image in gray channel , it achive maximum around 0.76 dice at epoch 40 but seem diverge and drop to 20~30 when finish. As for second loading method with much heavier augmentation, it max dice score is around 0.4, which is very weird.
