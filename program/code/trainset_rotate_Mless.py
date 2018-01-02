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
		imagelist = []
		getFileList(dir, imagelist)	
		imagelist.sort(key= lambda x:int(x.split('.')[1]))
		#print imagelist
		indexB = 0
		indexM = 0
		for imagepath in imagelist:
			if imagepath.split('/')[-2] in ('train0_flip','train1_flip','train2_flip','train3_flip','train4_flip'):
				print imagepath+'\n'
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
	
	degree=[0,90,180,270]
	
	for de in degree:	
	
		if de==90:
			rotatestate=Image.ROTATE_90		
		elif de==180:
			rotatestate=Image.ROTATE_180
		elif de==270:
			rotatestate=Image.ROTATE_270
		if de!=0:
			nim4 = im.transpose( rotatestate )
		else:
			nim4=im
		if not os.path.exists( os.path.join(outTrainingpath,imagepath.split('/')[-2]+'_rotate_mless')):    
			os.makedirs( os.path.join(outTrainingpath,imagepath.split('/')[-2]+'_rotate_mless'))
		nim4.save( os.path.join(outTrainingpath,imagepath.split('/')[-2]+'_rotate_mless','benign'+'.'+str(index)+'.'+imagepath.split('.')[-2]+'.tiff'))
		print "indexB:"+str(index)
		print "de:"+str(de)
		index += 1			
	
	return index
	
def MAN_AUG(imagepath,index):
	outTrainingpath = '$藍偉任/test_data/EBUSimagestraining_test/'
	im=Image.open(imagepath)
	
	degree=[0,180]
	
	for de in degree:	
	
		#if de==90:
			#rotatestate=Image.ROTATE_90		
		if de==180:
			rotatestate=Image.ROTATE_180
		#elif de==270:
			rotatestate=Image.ROTATE_270
		if de!=0:
			nim4 = im.transpose( rotatestate )
		else:
			nim4=im
		if not os.path.exists( os.path.join(outTrainingpath,imagepath.split('/')[-2]+'_rotate_mless')):    
			os.makedirs( os.path.join(outTrainingpath,imagepath.split('/')[-2]+'_rotate_mless'))
		nim4.save( os.path.join(outTrainingpath,imagepath.split('/')[-2]+'_rotate_mless','malignant'+'.'+str(index)+'.'+imagepath.split('.')[-2]+'.tiff'))
		print "indexM:"+str(index)
		print "de:"+str(de)
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
