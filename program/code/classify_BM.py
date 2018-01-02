

import dicom 
import os
from PIL import Image
import cv2
import numpy as np
import random


dataPath = "$藍偉任/test_data/EBUSimages_converttotiff/"






def main():
	DirList = []
	getDirList(dataPath, DirList)
	mFilePathList = []
	indexM=0
	indexB=0
	for dir in DirList:
		imagelist = []
		getFileList(dir, imagelist)		
		
		for imagepath in imagelist:
			if imagepath.split('.')[-1] == 'tiff':
				(indexM, indexB) =RENAME(imagepath,indexM,indexB)

			
def RENAME(imagepath,indexM,indexB):
	#outTrainingpath = 'C:/Users/repon/Desktop/EBUS_B_M_rewrite_denoise/'
	outTrainingpath = '$藍偉任/test_data/EBUSimages_BM/'
	#outTrainingpath = 'C:/Users/repon/Desktop/EBUS_B_M_rewrite_otsu/'
	layersize=227.0 #model input layer size
	image = cv2.imread(imagepath,0)
	r = 227.0 / image.shape[0]
	dim = (227, 227)
	
	# perform the actual resizing of the image and show it
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	#resized = cv2.fastNlMeansDenoising(resized,None,10,7,21)
	#resized = cv2.equalizeHist(resized)
	#esized = cv2.adaptiveThreshold(resized,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #       cv2.THRESH_BINARY,11,2)
	#ret2,resized = cv2.threshold(resized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
	cv2.imwrite(os.path.join('resized'+'.tiff'), resized)
	im=Image.open( "resized.tiff" )
	print im.size
	if imagepath.split('/')[-2] in ('Adenocarcinoma_M','largecell_M','SCLC_M','SqCC_M'):
		im.save( os.path.join(outTrainingpath,'M','malignant'+'.'+str(indexM)+'.'+imagepath.split('/')[-2]+'.tiff'))
		#im.save( os.path.join(outTrainingpath,'M',imagepath.split('/')[-2]+'.'+str(indexM)+'.tiff'))
		indexM +=1
	else:
		im.save( os.path.join(outTrainingpath,'B','benign'+'.'+str(indexB)+'.'+imagepath.split('/')[-2]+'.tiff'))
		indexB +=1
	return indexM,indexB
			
	
	
	
def getFileList(path, FileList):
    for item in os.listdir(path):
        if not item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            FileList.append(path + item)
    return	
	
def getDirList(path, DirList):
    for item in os.listdir(path):
        if not item.startswith('.') and os.path.isdir(os.path.join(path, item)):
            getDirList(path+item+'/', DirList)
            DirList.append(path+item+'/')
    return

# main()  
if __name__ == '__main__':
    main()