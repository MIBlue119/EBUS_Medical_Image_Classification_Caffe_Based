import numpy as np
import os, sys, getopt
import glob
import cv2
import caffe
import lmdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from caffe.proto import caffe_pb2
from numpy import array
from skimage.feature import greycomatrix, greycoprops


import time

root='/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/'  

# Main path to your caffe installation


# Model prototxt file
deploy ='C:/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_2/caffenet_deploy_2.prototxt'


model_number=input('model number:')
model_number_s=str(model_number)

test_set=input('# of test set:')
test_set_s=str(test_set)
# Model caffemodel file
caffe_model=root + 'caffe_models/caffe_model_2/t'+test_set_s+'_210.caffemodel'

# File containing the class labels


# Path to the mean image (used for input processing)
MEAN_FILE ='/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/input/mean.binaryproto'


feature_file = 'C:/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_2/features.txt'

# Name of the layer we want to extract
layer= 'fc7'


CSV_FILE=root+'caffe_models/caffe_model_2/cross_fusionglcm.'+model_number_s+'.'+test_set_s+'.csv'
SVM_FILE=root+'caffe_models/caffe_model_2/svm_fusionglcm.'+model_number_s+'.'+test_set_s+'.csv'
RF_FILE=root+'caffe_models/caffe_model_2/rf_fusionglcm.'+model_number_s+'.'+test_set_s+'.csv'
caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

'''
Image processing helper function
'''
def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    #img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    #img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    #img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


	
	
def svc(traindata,trainlabel,testdata,testlabel,file,test_ids):
    print("Start training SVM...")
    start_svm=time.time()
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000,gamma="auto",probability=True)
    svcClf.fit(traindata,trainlabel)
    pred_testlabel = svcClf.predict(testdata)
    pred_value=svcClf.predict_proba(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print(pred_testlabel)
    print(pred_value[:,1])
    end_svm=time.time()  
    elapsed=end_svm-start_svm	
    print "SVM traing time taken:",elapsed,"seconds."    	
    with open(file,"w") as f:
		f.write("id,label\n")
		for i in range(len(test_ids)):
			f.write(str(test_ids[i])+","+str(pred_value[i,1])+"\n")
    f.close()
	
    TP,FN =countTPFN(pred_testlabel,test_ids)
    TN,FP =countTNFP(pred_testlabel,test_ids)
    
    print("TP of cnn-svm",TP)
    print("FN of cnn-svm",FN)
    print("TN of cnn-svm",TN)
    print("FP of cnn-svm",FP)
    Sensitivity,Specificity=countSFSP(TP,FN,TN,FP)	
    print("cnn-svm Accuracy:",accuracy)
    print(" cnn-svm Sensitivity:",Sensitivity)
    print("cnn-svm Specificity:",Specificity)

def rf(traindata,trainlabel,testdata,testlabel,file,test_ids):
    print("Start training Random Forest...")
    start_rf=time.time()     
    rfClf = RandomForestClassifier(n_estimators=1000,criterion='gini')
    rfClf.fit(traindata,trainlabel)
    
    pred_testlabel = rfClf.predict(testdata)
    pred_value=rfClf.predict_proba(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print(pred_testlabel)
	
    end_rf=time.time()  
    elapsed=end_rf-start_rf
    print "RF traing time taken:",elapsed,"seconds."   
    
    with open(file,"w") as f:
		f.write("id,label\n")
		for i in range(len(test_ids)):
			f.write(str(test_ids[i])+","+str(pred_value[i,1])+"\n")
    f.close()	

    TP,FN =countTPFN(pred_testlabel,test_ids)
    TN,FP =countTNFP(pred_testlabel,test_ids)
    
    print("TP of cnn-rf",TP)
    print("FN of cnn-rf",FN)
    print("TN of cnn-rf",TN)
    print("FP of cnn-rf",FP)
    Sensitivity,Specificity=countSFSP(TP,FN,TN,FP)	
    print("cnn-rf Accuracy:",accuracy)
    print(" cnn-rf Sensitivity:",Sensitivity)
    print(" cnn-rf Specificity:",Specificity)	

def countTPFN(pr_label,test_ids):    
	TP=0
	FN=0
	for i in range(len(test_ids)):
		if'malignant'in test_ids[i]:
			if pr_label[i]==1:
				TP=TP+1
			else:
				FN=FN+1
	return TP,FN

def countTNFP(pr_label,test_ids):    
	TN=0
	FP=0
	for i in range(len(test_ids)):
		if'benign'in test_ids[i]:
			if pr_label[i]==0:
				TN=TN+1
			else:
				FP=FP+1
	return TN,FP

def countSFSP(TP,FN,TN,FP):
	
	Sensitivity=TP/float(TP+FN)
	Specificity=TN/float(TN+FP)
	return Sensitivity,Specificity	
	
'''
Reading mean image, caffe model and its weights 
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
#with open('/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/input/mean.binaryproto') as f:
#mean_blob.ParseFromString(f.read())
mean_blob.ParseFromString(open(MEAN_FILE, 'rb').read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
net = caffe.Net(deploy,caffe_model,caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
##Reading image paths
#test_img_paths = [img_path for img_path in glob.glob("../input/test/*tiff")]
test_img_paths = [img_path for img_path in glob.glob("C:/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/input/test/*tiff")]
train_img_paths =  [img_path for img_path in glob.glob("C:/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/input/train/*tiff")]
#test_img_paths = [img_path for img_path in glob.glob("C:/Users/repon/Desktop/EBUS_training_test_noaug/test2/*tiff")]


test_ids = []
preds = []
#features=[]
len_b=0
len_m=0
len_image=len(test_img_paths)
features = np.empty((len_image,4144),dtype="float32")
or_label = np.empty((len_image,),dtype="uint8")
pr_label = np.empty((len_image,),dtype="uint8")
pr_value = np.empty((len_image,),dtype="float32")
j=0
for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  
	#print img.shape
	#print img.dtype
	#px=img[11,11]
	#print px
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)    
    data=np.rollaxis(img,2)
    arr = array(data[0])
    xs = []
    ys = []
    zs=[]
    ws=[]
    us=[]
    vs=[]
    
    glcm = greycomatrix(arr, [1,2], [0,np.pi/2,np.pi/4,3*np.pi/4], 256, symmetric=True, normed=True)
 
    xs.append(greycoprops(glcm, 'contrast'))
    ys.append(greycoprops(glcm, 'correlation'))
    zs.append(greycoprops(glcm, 'homogeneity'))
    ws.append(greycoprops(glcm, 'energy'))
    us.append(greycoprops(glcm, 'dissimilarity'))
    vs.append(greycoprops(glcm, 'ASM'))
    yz=np.concatenate((xs[0],ys[0],zs[0],ws[0],us[0],vs[0])).reshape(1,48)
    print xs[0]
    
	
    
    
	
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']
	#print out['fc8-cats-dogs']
	#print( float(out['prob'][:,1])) # python2.7 print float(out['prob'][:,1])
	#print float(out['prob'][:,1])
	#print type(out['prob'])
	#print pred_probas.argmin()

    with open(feature_file, 'w') as f:
		np.savetxt(f, net.blobs[layer].data[0], fmt='%.12f', delimiter='\n')
		f.write("OK\n")
    #print("another image\n")
	#features = features + [net.blobs[layer].data[0]]
	#np.concatenate((features,net.blobs[layer].data[0]))
    features[j,0:4096]=net.blobs[layer].data[0]
    features[j,4096:4144]=yz
	#print yz
	#print features[j,4096:4128]
	#print(net.blobs[layer].data[0])
	#print(len(net.blobs[layer].data[0]))
	#print(type(net.blobs[layer].data[0]))

	#print("\n")
	
    if 'benign' in img_path:
			or_label[j]= 0
			len_b=len_b+1
    else:
			or_label[j]= 1
			len_m=len_m+1
	

    test_ids = test_ids + [img_path.split('/')[-1][:-4]]
    pr_label[j]=pred_probas.argmax()
    pr_value[j]=float(out['prob'][:,1])
    j=j+1
    preds = preds + [pred_probas.argmax()]

	#print img_path
	#print pred_probas.argmax()
	#print preds
	#print '-------'


	
	
train_ids = []	

#features=[]

len_train=len(train_img_paths)
features_train = np.empty((len_train,4144),dtype="float32")
or_label_tr = np.empty((len_train,),dtype="uint8")
pr_label_tr = np.empty((len_train,),dtype="uint8")
k=0
for img_path in train_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  
	#print img.shape
	#print img.dtype
	#px=img[11,11]
	#print px
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
	#img_crop=img[50:176,50:176,0] crop image to get precision features
    data=np.rollaxis(img,2)
    arr = array(data[0])
    xs = []
    ys = []
    zs=[]
    ws=[]
    us=[]
    vs=[]
    
    glcm = greycomatrix(arr, [1,2], [0,np.pi/2,np.pi/4,3*np.pi/4], 256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'contrast'))
    ys.append(greycoprops(glcm, 'correlation'))
    zs.append(greycoprops(glcm, 'homogeneity'))
    ws.append(greycoprops(glcm, 'energy'))
    us.append(greycoprops(glcm, 'dissimilarity'))
    vs.append(greycoprops(glcm, 'ASM' ))
    
    yz=np.concatenate((xs[0],ys[0],zs[0],ws[0],us[0],vs[0])).reshape(1,48)
	
    
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
	
	#print out['fc8-cats-dogs']
	#print( float(out['prob'][:,1])) # python2.7 print float(out['prob'][:,1])
	#print float(out['prob'][:,1])
	#print type(out['prob'])
	#print pred_probas.argmin()


	#print("another image\n")
	#features = features + [net.blobs[layer].data[0]]
	#np.concatenate((features,net.blobs[layer].data[0]))
    features_train[k,0:4096]=net.blobs[layer].data[0]
    features_train[k,4096:4144]=yz
     
	#print("\n")
	
    if 'benign' in img_path:
			or_label_tr[k]= 0
			
    else:
			or_label_tr[k]= 1
			
	

    train_ids = train_ids + [img_path.split('/')[-1][:-4]]
    pr_label_tr[k]=pred_probas.argmax()
    k=k+1
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
#for i in range(len(features)):
#	print("the "+str(i)+"image features:\n")
#	print(features[i])	

#print(len(features))
print(or_label)	
print(pr_label)


accuracy = len([1 for i in range(len_image) if or_label[i]==pr_label[i]])/float(len_image)
#print([1 for i in range(len_image) if or_label[i]==pr_label[i]])
#sensitivity=
TP,FN =countTPFN(pr_label,test_ids)
TN,FP =countTNFP(pr_label,test_ids)

print("TP of Fine tune",TP)
print("FN of Fine tune",FN)
print("TN of Fine tune",TN)
print("FP of Fine tune",FP)

Sensitivity,Specificity=countSFSP(TP,FN,TN,FP)



#print(" Origin_model Accuracy:",accuracy)


#accuracy_tr = len([1 for i in range(len_train) if or_label_tr[i]==pr_label_tr[i]])/float(len_train)
#print([1 for i in range(len_train) if or_label_tr[i]==pr_label_tr[i]])
#sensitivity=
#print(" Training Accuracy:",accuracy_tr)

print(" Fine tune Accuracy:",accuracy)
print(" Fine tune Sensitivity:",Sensitivity)
print(" Fine tune Specificity:",Specificity)



#CSV_FILE
#with open("../caffe_models/caffe_model_2/cross_19_0.csv","w") as f:
'''
Making submission file
'''
with open(CSV_FILE,"w") as f:
    f.write("id,label\n")
    for i in range(len(test_ids)):
        f.write(str(test_ids[i])+","+str(round(pr_value[i],9))+"\n")
		
f.close()



#scaler= MinMaxScaler()
#features = scaler.fit_transform(features)
#features_train = scaler.fit_transform(features_train)
scaler= MaxAbsScaler()
features = scaler.fit_transform(features)
features_train = scaler.fit_transform(features_train)

print("---------------------------------------")
svc(features_train,or_label_tr,features,or_label,SVM_FILE,test_ids)
print("---------------------------------------")
rf(features_train,or_label_tr,features,or_label,RF_FILE,test_ids)

