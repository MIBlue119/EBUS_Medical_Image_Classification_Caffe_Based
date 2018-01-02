import dicom 
import os
from PIL import Image
import cv2
import numpy as np


dataPath = "$藍偉任/test_data/EBUSimagestraining_test/"


def main():
	DirList = []
	getDirList(dataPath, DirList)
	mFilePathList = []
	count=0
	for dir in DirList:
		print dir
		imagelist = []
		getFileList(dir, imagelist)	
		imagelist.sort(key= lambda x:int(x.split('.')[1]))	
		indexB = 0
		indexM = 0
		for imagepath in imagelist:
			if imagepath.split('/')[-2] in ('train0','train1','train2','train3','train4'):
				if imagepath.split('.')[-1] == 'tiff':
					if 'benign' in imagepath:
						indexB=BENIGN_AUG(imagepath,indexB)
					if 'malignant' in imagepath:
						indexM=MAN_AUG(imagepath,indexM)
				
				print "count:"+str(count)
				count += 1
	
			
def BENIGN_AUG(imagepath,index):
	outTrainingpath = '$藍偉任/test_data/EBUSimagestraining_test/'
	im=Image.open(imagepath)
	
	state=[0,1,2]
	
	for st in state:	
	
		if st==1:
			rotatestate=Image.FLIP_LEFT_RIGHT		
		elif st==2:
			rotatestate=Image.FLIP_TOP_BOTTOM
		
		if st!=0:
			nim4 = im.transpose( rotatestate )
		else:
			nim4=im
		if not os.path.exists( os.path.join(outTrainingpath,imagepath.split('/')[-2]+'_flip')):    
			os.makedirs( os.path.join(outTrainingpath,imagepath.split('/')[-2]+'_flip'))
		nim4.save( os.path.join(outTrainingpath,imagepath.split('/')[-2]+'_flip','benign'+'.'+str(index)+'.'+imagepath.split('.')[-2]+'.tiff'))
		print "indexB:"+str(index)
		
		index += 1			
	
	return index
	
def MAN_AUG(imagepath,index):
	outTrainingpath = '$藍偉任/test_data/EBUSimagestraining_test/'
	im=Image.open(imagepath)
	
	state=[0,1,2]
	
	for st in state:	
	
		if st==1:
			rotatestate=Image.FLIP_LEFT_RIGHT		
		elif st==2:
			rotatestate=Image.FLIP_TOP_BOTTOM
		
		if st!=0:
			nim4 = im.transpose( rotatestate )
		else:
			nim4=im
		if not os.path.exists( os.path.join(outTrainingpath,imagepath.split('/')[-2]+'_flip')):   
			os.makedirs( os.path.join(outTrainingpath,imagepath.split('/')[-2]+'_flip'))
		nim4.save( os.path.join(outTrainingpath,imagepath.split('/')[-2]+'_flip','malignant'+'.'+str(index)+'.'+imagepath.split('.')[-2]+'.tiff'))
		print "indexM:"+str(index)
		index += 1			
	
	return index
	
	
	
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
