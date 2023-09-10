import albumentations as albu 

path='/root/repo/help-repo/pneumothorax-segmentation/unet_pipeline/transforms/valid_transforms_1024_old.json'


path2='/root/repo/help-repo/pneumothorax-segmentation/unet_pipeline/transforms/train_transforms_complex_1024_old.json'
trans=albu.load(path2)
print (trans)

