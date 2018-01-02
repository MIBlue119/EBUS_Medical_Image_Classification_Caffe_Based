import dicom 
import os
from PIL import Image
import cv2
import numpy as np
import random
import math 


dataPath = "$藍偉任/test_data/EBUSimages_BM/"



def main():
	DirList = []
	getDirList(dataPath, DirList)
	mFilePathList = []
#	indexM=0
#	indexB=0
#	index_test=0
	for dir in DirList:
		imagelist = []
		getFileList(dir, imagelist)
		train={}
		test={}
		(train,test)=k_fold(imagelist,5)
		


		for ii in range(5):
			indexM=0
			indexB=0
			index_test=0
			print ii
			
			for imagepath in train[ii]:
				if imagepath.split('.')[-1] == 'tiff':
					(indexM, indexB) =TRAINCASE(imagepath,indexM,indexB,ii)
			for imagepath in test[ii]:
				if imagepath.split('.')[-1] == 'tiff':
					index_test =TESTCASE(imagepath,index_test,ii)
			
		

		


		
#		random.shuffle(imagelist)
		
		
#		trainlist=[]
#		testlist=[]
#		trainlist=random.sample(imagelist,len(imagelist)*1/3)
#		testlist=list(set(imagelist)-set(trainlist))
		
#		for imagepath in trainlist:
#			if imagepath.split('.')[-1] == 'tiff':
#				(indexM, indexB) =TRAINCASE(imagepath,indexM,indexB)
		
#		for imagepath in testlist:
#				index_test=TESTCASE(imagepath,index_test)
				

def k_fold(data,k):
    random.shuffle(data)
    len_part=int(math.ceil(len(data)/float(k)))
    train={}
    test={}
    for ii in range(k):
		test[ii]  = data[ii*len_part:ii*len_part+len_part]
		train[ii] = [jj for jj in data if jj not in test[ii]]
    return train, test    
				
				
	
				

			
def TRAINCASE(imagepath,indexM,indexB,fold):
	outTrainingpath = '$藍偉任/test_data/EBUSimagestraining_test/'
	im=Image.open(imagepath)
	if not os.path.exists(os.path.join(outTrainingpath,'train'+str(fold))):    
           os.makedirs(os.path.join(outTrainingpath,'train'+str(fold)))
	if imagepath.split('/')[-2] in ('M'):
		im.save( os.path.join(outTrainingpath,'train'+str(fold),'malignant'+'.'+str(indexM)+'.'+imagepath.split('.')[-2]+'.tiff'))
		indexM +=1
	else:
		im.save( os.path.join(outTrainingpath,'train'+str(fold),'benign'+'.'+str(indexB)+'.'+imagepath.split('.')[-2]+'.tiff'))
		indexB +=1
	return indexM,indexB
			
def TESTCASE(imagepath,index_test,fold):
	outTrainingpath = '$藍偉任/test_data/EBUSimagestraining_test/'
	im=Image.open(imagepath)
	if not os.path.exists(os.path.join(outTrainingpath,'test'+str(fold))):    
           os.makedirs(os.path.join(outTrainingpath,'test'+str(fold)))
	if imagepath.split('/')[-2] in ('M'):
		im.save( os.path.join(outTrainingpath,'test'+str(fold),'malignant'+'.'+str(index_test)+'.'+imagepath.split('.')[-2]+'.tiff'))
	else:
		im.save( os.path.join(outTrainingpath,'test'+str(fold),'benign'+'.'+str(index_test)+'.'+imagepath.split('.')[-2]+'.tiff'))	
	#im.save( os.path.join(outTrainingpath,'test',str(index_test)+'.tiff'))
	index_test +=1
	return index_test

	
	
	
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