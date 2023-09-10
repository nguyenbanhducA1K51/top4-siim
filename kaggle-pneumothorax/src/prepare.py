
import os
from tqdm import tqdm
import shutil

from joblib import Parallel, delayed

import argparse
# from mask_functions import rle2mask, mask2rle


import numpy as np 
import pandas as pd 

from glob import glob
import pydicom

import cv2
from skimage.transform import resize
def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    if rle == ' -1' or rle == '-1':
        return mask.reshape(width,height)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)
def argparser():
	parser = argparse.ArgumentParser(description='Prepare png dataset for pneumatorax')
	parser.add_argument('-train_path', default='../input/train/', type=str, nargs='?', help='directory with train')
	parser.add_argument('-test_path', default='../input/test/', type=str, nargs='?', help='directory with test')
	parser.add_argument('-rle_path', default='../input/train-rle.csv', type=str, nargs='?', help='path for rle csv file')
	parser.add_argument('-out_path', default='../dataset', type=str, nargs='?', help='path for saving dataset')
	parser.add_argument('-n_train', default=-1, type=int, nargs='?', help='size of train dataset')
	parser.add_argument('-img_size', default=1024, type=int, nargs='?', help='image size')
	parser.add_argument('-n_threads', default=4, type=int, nargs='?', help='number of using threads')
	return parser.parse_args()


def get_mask(encode, width, height):
	if encode == [] or encode == [' -1']:
		return rle2mask(' -1',width,height)
	mask = rle2mask(encode[0],width,height)
	for e in encode[1:]:
		mask += rle2mask(e,width,height)
	return mask.T

def save_train_file(f, encode_df, out_path, img_size):
	img = pydicom.read_file(f).pixel_array
	name = f.split('/')[-1][:-4]
	encode = list(encode_df.loc[encode_df['ImageId'] == name, ' EncodedPixels'].values)
	encode = get_mask(encode,img.shape[1],img.shape[0])
	encode = resize(encode,(img_size,img_size))
	img = resize(img,(img_size,img_size))
	
	cv2.imwrite('{}/train/{}.png'.format(out_path, name), img * 255)
	cv2.imwrite('{}/mask/{}.png'.format(out_path, name), encode)  


def save_test_file(f, out_path, img_size):
	img = pydicom.read_file(f).pixel_array
	name = f.split('/')[-1][:-4]
	img = resize(img,(img_size,img_size)) * 255
	cv2.imwrite('{}/test/{}.png'.format(out_path, name), img)


def save_train(train_images_names, encode_df, out_path='../dataset128', 
	img_size=128, n_train=-1, n_threads=1):
	if os.path.isdir(out_path):
		shutil.rmtree(out_path)
	os.makedirs(out_path + '/train', exist_ok=True)
	os.makedirs(out_path + '/mask', exist_ok=True)
	if n_train < 0:
		n_train = len(train_images_names)
	try:
		Parallel(n_jobs=n_threads, backend='threading')(delayed(save_train_file)(
			f, encode_df, out_path, img_size) for f in tqdm(train_images_names[:n_train]))
	except pydicom.errors.InvalidDicomError:
		print('InvalidDicomError')


def save_test(test_images_names, out_path='../dataset128', img_size=128, n_threads=1):
	os.makedirs(out_path + '/test', exist_ok=True)
	try:
		Parallel(n_jobs=n_threads, backend='threading')(delayed(save_test_file)(
			f, out_path, img_size) for f in tqdm(test_images_names))
	except pydicom.errors.InvalidDicomError:
		print('InvalidDicomError')


def main():
	args = argparser()
	train_fns = sorted(glob('{}/*/*/*.dcm'.format(args.train_path)))
	test_fns = sorted(glob('{}/*/*/*.dcm'.format(args.test_path))) 

	rle = pd.read_csv(args.rle_path)
	out_path = args.out_path
	img_size = args.img_size
	n_train = args.n_train
	n_threads = args.n_threads

	save_train(train_fns, rle, out_path, img_size, n_train, n_threads)
	# save_test(test_fns, out_path, img_size, n_threads)

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	path="/root/data/siim-other" 

	img_glob=f'{path}/dicom-images-train/**/**/*.dcm'

	train_paths=glob(img_glob)
	rle = pd.read_csv(f'{path}/train-rle.csv')
	out_path=f'/root/data/siim-sample'
	n_train=-1
	img_size=1024
	n_threads=2
	msk_dir1="/root/data/siim_png_convert/msk"
	msk_dir2="/root/data/siim-sample/mask"
	img_dir1="/root/data/siim_png_convert/img"
	img_dir2="/root/data/siim-sample/train"
	msk1_glob=glob(f"{msk_dir1}/*.png")
	print (len(msk1_glob))
	save_path="/root/repo/help-repo/kaggle-pneumothorax/figures/compare.png"
	tuples=[]
	n_visualize=10
	for msk1 in msk1_glob:
		print (f"process {msk1}")
			
		base=msk1.split("/")[-1]
		img1=f"{img_dir1}/{base}"
		img2=f"{img_dir2}/{base}"
		msk2=f"{msk_dir2}/{base}"
		
		msk1_arr=cv2.imread(msk1,0)
		msk2_arr=cv2.imread(msk2,0)
		img1_arr=cv2.imread(img1,1)
		img2_arr=cv2.imread(img2,1)
		assert msk1_arr.shape==msk2_arr.shape
		if not np.array_equal(msk1_arr,msk2_arr):
			print (f"msk not equal{base}")
			if len(tuples)>n_visualize:
				break
			else:
				tuples.append([img1_arr,msk1_arr,img2_arr,msk2_arr])
	fig,ax=plt.subplots(n_visualize,6 ,figsize=(50,50))
	for i,ls in enumerate(tuples):
		ax[i,0].imshow(ls[0])
		ax[i,1].imshow(ls[1])
		ax[i,2].imshow(ls[0])
		ax[i,2].imshow(ls[1], cmap="red",alpha=0.3)

		ax[i,3].imshow(ls[2])
		ax[i,4].imshow(ls[3])
		ax[i,5].imshow(ls[2])
		ax[i,5].imshow(ls[3], cmap="red",alpha=0.3)
	plt.show()
	plt.savefig(save_path)


		

			
		
		

	# save_train(train_paths, rle, out_path, img_size, n_train, n_threads)
    