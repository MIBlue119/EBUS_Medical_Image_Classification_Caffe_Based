import dicom 
import os
from PIL import Image
import cv2

#meta=dicom.read_file("219370150.dcm") 
#imHeight=meta.Rows
#imWidth=meta.Columns 
#imSize=(imWidth,imHeight)
#TT=Image.frombuffer("L",imSize,meta.PixelData,"raw","L",0,1)

#TT.save("testOUTPUT.tiff","TIFF",compression="none")


#test_image="testOUTPUT.tiff"
#original = Image.open(test_image)
#original.show()

#width, height = original.size   # Get dimensions
#print width ,height   #Tiff  640  480

#left = 148
#top=41
#right = 597
#bottom = 435
#cropped_example = original.crop((left, top, right, bottom))
#print cropped_example.size

#cropped_example.show()
#cropped_example.save("testOUTPUT2.tiff")

sourceimagepath = "$藍偉任/test_data/EBUSimages_programuse"


def main():
	DirList = []
	getDirList(sourceimagepath, DirList)
	mFilePathList = []

	for dir in DirList:
		imagelist = []
		getFileList(dir, imagelist)		
		index = 0
		for imagepath in imagelist:
			if imagepath.split('.')[-1] == 'dcm':
				Dicom2PNG(imagepath, index)
			index += 1

	

def Dicom2PNG(imagepath, index):

	#img = sitk.ReadImage (imagepath, sitk.sitkVectorInt16)
	
	#nda = np.zeros((1,512,512)) 
	#nda = np.zeros(img.shape, np.int16) 
	#nda = sitk.GetArrayFromImage(img)
	#print nda , nda.shape
	#print imagepath.split('/')[-2]
	#sitk.WriteImage(img, os.path.join(dataPath+imagepath.split('/')[-2],  "train"+str(index)+'.png'))
	outputimagepath = '$藍偉任/test_data/EBUSimages_converttotiff/'
	#sitk.WriteImage(img, os.path.join(outTrainingpath,  imagepath.split('/')[-2]+'_dcmTraining_'+str(index)+'.tiff'))
	
	meta=dicom.read_file(imagepath) 
	imHeight=meta.Rows
	imWidth=meta.Columns 
	imSize=(imWidth,imHeight)
	TT=Image.frombuffer("L",imSize,meta.PixelData,"raw","L",0,1)
	TT.save("testOUTPUT.tiff","TIFF",compression="none")
	test_image="testOUTPUT.tiff"
	original = Image.open(test_image)
	#original.show()
	#width, height = original.size   # Get dimensions
	#print width ,height   #Tiff  640  480
     #central point x:380 y:240
	#for 3574370.dcm

	#centralx=362
	#centraly=248
	#left = 165
	#top=40
	#right = 559
	#bottom = 437
	
	centralx=380
	centraly=240
	left = 148
	top=41
	right = 597
	bottom = 435

	
	xl=centralx-left
	xr=right-centralx
	yd=bottom-centraly
	yt=centraly-top
	
	if xl > xr:
		left=centralx-xr
	else:
		right=centralx+xl
	
	if yd >yt:
		bottom=centraly+yt
	else:
		top=centraly-yd
	
	
	cropped_example = original.crop((left, top, right, bottom))
	
	cropped_example.save( os.path.join(outputimagepath,  imagepath.split('/')[-2],imagepath.split('/')[-2]+'_dcmTraining_'+str(index)+'.tiff'))
	#img = cv2.imread(imagepath, cv2.IMREAD_ANYCOLOR)
	#cv2.imwrite(imagepath.split('/')[-2]+'_dcmTraining_'+str(index)+'.tiff', nda)
	
	return

	
	
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


