'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160625
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
'''

import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    #img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    #img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    #img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

train_lmdb = 'C:/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/input/train_lmdb'
validation_lmdb = 'C:/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/input/validation_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)


train_data = [img for img in glob.glob("C:/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/input/train/*tiff")]
test_data = [img for img in glob.glob("C:/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/input/test/*tiff")]

#Shuffle train_data
random.shuffle(train_data)

print 'Creating train_lmdb'
count_trainlmdb=0
count_valmdb=0

in_db = lmdb.open(train_lmdb, map_size=int(1e11))
with in_db.begin(write=True) as in_txn:
	for in_idx, img_path in enumerate(train_data):
		if in_idx %  6 == 0:
			continue
		count_trainlmdb=count_trainlmdb+1
		img = cv2.imread(img_path,cv2.IMREAD_COLOR )#cv2.IMREAD_COLOR
		#print img.shape
		px=img[100,100]
		#print img.dtype
		#print px
		img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
		#print img.shape
		if 'benign' in img_path:
			label = 0
		else:
			label = 1
		datum = make_datum(img, label)
		in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
		print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()

print 'count_trainlmdb:'+str(count_trainlmdb)+'\n'
print '\nCreating validation_lmdb'

in_db = lmdb.open(validation_lmdb, map_size=int(1e11))
with in_db.begin(write=True) as in_txn:
	for in_idx, img_path in enumerate(train_data):
		if in_idx % 6!= 0:
			continue
		count_valmdb=count_valmdb+1
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
		if 'benign' in img_path:
			label = 0
		else:
			label = 1
		datum = make_datum(img, label)
		in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
		print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()

print 'count_valmdb:'+str(count_valmdb)+'\n'
print '\nCreating validation_lmdb'
print '\nFinished processing all images'
