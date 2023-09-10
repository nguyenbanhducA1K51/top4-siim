import albumentations as A
import numpy as np
from typing import Union, Tuple, List,Literal




def load_transform(mode:Literal["train","val"]="train") :     
	factor=0.05

	img_size=512

	def norm (mask,*args,**kargs):
		mask=mask/255
		mask.astype(float) 
		return mask
	def norm_and_stackChannel(image,*args,**kargs):
		# print (" at transform", np.max(image))
		image=image/255
		image=np.expand_dims(image,axis=0)
		return np.repeat(image,3,axis=0)

	# train_transform=A.Compose([
	# A.HorizontalFlip(),
	# # A.OneOf([
	# # 	A.RandomContrast(),
	# # 	A.RandomGamma(),
	# # 	A.RandomBrightness(),
	# # 	], p=0.3),
	# # A.OneOf([
	# # 	A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
	# # 	A.GridDistortion(),
	# # 	A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
	# # 	], p=0.3),
	# A.ShiftScaleRotate( shift_limit_x=(-0.1, 0.1), shift_limit_y=(-0.1, 0.1), scale_limit=(-0.1, 0.1), rotate_limit=(-10, 10)),
	# A.Resize(img_size,img_size,always_apply=True),
	# A.Lambda(image=norm_and_stackChannel, mask=norm),       

	# ])  
	# test_transform=A.Compose([
	# A.Resize( height=img_size, width=img_size),
	# A.Lambda(image=norm_and_stackChannel, mask=norm),       

	# ])
	train_transform = A.Compose([                     
					A.ShiftScaleRotate( scale_limit =(-0.2, 0.2) ,rotate_limit=(-10,10)),
					A.RandomResizedCrop(height=img_size,width=img_size,scale=(0.9, 1.0),ratio=(0.75, 1.3333333333333333)),

					A.HorizontalFlip(),
					A.Lambda(image=norm_and_stackChannel, mask=norm),              
								])

	test_transform=A.Compose([  
					A.Resize(height=img_size,width=img_size),
					A.Lambda(image=norm_and_stackChannel, mask=norm),
									]) 

	if mode=="train":
		return train_transform
	elif mode=="test" or mode=="val":
		return test_transform
	else:
		raise RuntimeError("invalid mode in transform file ")

