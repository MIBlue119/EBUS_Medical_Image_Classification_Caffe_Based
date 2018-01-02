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
from sklearn import preprocessing
from caffe.proto import caffe_pb2
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
#from svmutil import *
import time

root='/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/'  

# Main path to your caffe installation


# Model prototxt file
deploy ='C:/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_3/deploy.prototxt'


model_number=input('model number:')
model_number_s=str(model_number)

test_set=input('# of test set:')
test_set_s=str(test_set)
# Model caffemodel file
#caffe_model=root + 'caffe_models/caffe_model_4/VGG_16_iter_'+model_number_s+'.caffemodel'

caffe_model=root + 'caffe_models/caffe_model_3/gnet_iter_'+model_number_s+'.caffemodel'

# File containing the class labels


# Path to the mean image (used for input processing)
MEAN_FILE ='/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/input/mean.binaryproto'


feature_file = 'C:/caffe/Build/x64/Release/pycaffe/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_3/features.txt'

# Name of the layer we want to extract
layer= 'pool5/7x7_s1'


CSV_FILE=root+'caffe_models/caffe_model_3/gnetcross_'+model_number_s+'.'+test_set_s+'.csv'
SVM_FILE=root+'caffe_models/caffe_model_3/gnetsvm_'+model_number_s+'.'+test_set_s+'.csv'
RF_FILE=root+'caffe_models/caffe_model_3/gnetrf_'+model_number_s+'.'+test_set_s+'.csv'
GBC_FILE=root+'caffe_models/caffe_model_3/gbc_'+model_number_s+'.'+test_set_s+'.csv'
caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

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
    #C_range = np.logspace(0, 0, 1)
    #gamma_range = np.logspace(-3.6, -2.7, 10)
    #param_grid = dict(gamma=gamma_range, C=C_range)
    #cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    #svcClf = GridSearchCV(SVC(), param_grid=param_grid)
    #svcClf.fit(traindata, trainlabel)
    #C_2d_range = [1.0]
    #gamma_2d_range =np.logspace(-3.7, -3.1, 20)
    #gamma_2d_range =np.logspace(-3.2, -2.6, 40)
    #classifiers = []
    #for C in C_2d_range:
	#	for gamma in gamma_2d_range:
	#		svcClf = SVC(C=C, gamma=gamma,cache_size=3000)
	#		svcClf.fit(traindata,trainlabel)
	#		pred_testlabel = svcClf.predict(testdata)
	#		num = len(pred_testlabel)
	#		accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
	#		classifiers.append((C, gamma, svcClf))
	#		print (str(accuracy)+"  gamma="+str(gamma))
			
    #print(classifiers)
	
	
	
	
	
	#before using libsvm
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000,gamma="auto",probability=True)
    svcClf.fit(traindata,trainlabel)
    pred_testlabel = svcClf.predict(testdata)
    pred_value=svcClf.predict_proba(testdata)
    end_svm=time.time()  
    elapsed=end_svm-start_svm
    print "SVM traing time taken:",elapsed,"seconds."    
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print(pred_testlabel)
    
    print(pred_value[:,1])
    
    with open(file,"w") as f:
        f.write("id,label\n")
        for i in range(len(test_ids)):
            f.write(str(test_ids[i])+","+str(round(pred_value[i,1],9))+"\n")
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
    end_rf=time.time()  
    elapsed=end_rf-start_rf
    print "RF traing time taken:",elapsed,"seconds."    
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print(pred_testlabel)
    print(pred_value[:,1])
    
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

def gbc(traindata,trainlabel,testdata,testlabel,file,test_ids):
    print("Start training Gradient boosting classifier...")
    GBC = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0,max_depth=1,random_state=0)
    GBC.fit(traindata,trainlabel)
    pred_testlabel = GBC.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print(pred_testlabel)
    print("cnn-GBC Accuracy:",accuracy)	
	

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
#mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
#    (mean_blob.channels, mean_blob.height, mean_blob.width))

mean_array = np.array([104, 117, 123])   # BGR

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
features = np.empty((len_image,1024),dtype="float32")
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
    features[j,:]=net.blobs[layer].data[0,:,0,0]
    #print (net.blobs[layer].data[0,:,0,0]).shape
    #print ((np.rollaxis(net.blobs[layer].data, 2))[0].shape)
    

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

#print float(out['prob'][:,1])
	
	
train_ids = []	

#features=[]

len_train=len(train_img_paths)
features_train = np.empty((len_train,1024),dtype="float32")
or_label_tr = np.empty((len_train,),dtype="uint8")
pr_label_tr = np.empty((len_train,),dtype="uint8")
pr_value_tr= np.empty((len_train,),dtype="float32")
k=0
for img_path in train_img_paths:
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)  
	#print img.shape
	#print img.dtype
	#px=img[11,11]
	#print px
	img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    
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
	features_train[k,:]=net.blobs[layer].data[0,:,0,0]
	#print net.blobs[layer].data[0]

	#print("\n")
	
	if 'benign' in img_path:
			or_label_tr[k]= 0
			
	else:
			or_label_tr[k]= 1
			
	

	train_ids = train_ids + [img_path.split('/')[-1][:-4]]
	pr_label_tr[k]=pred_probas.argmax()
	pr_value_tr=float(out['prob'][:,1])
	k=k+1
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
#for i in range(len(features)):
#	print("the "+str(i)+"image features:\n")
#	print(features[i])	

#print(len(features))
print(or_label)	
print(pr_label)


accuracy = len([1 for i in range(len_image) if or_label[i]==pr_label[i]])/float(len_image)
#specificity=
#sensitivity=

#TP=len([1 for i in range(test_ids) if test_ids[i]==pr_label[i]])

TP,FN =countTPFN(pr_label,test_ids)
TN,FP =countTNFP(pr_label,test_ids)

print("TP of Fine tune",TP)
print("FN of Fine tune",FN)
print("TN of Fine tune",TN)
print("FP of Fine tune",FP)

Sensitivity,Specificity=countSFSP(TP,FN,TN,FP)

#if 'benign' in test_ids:

#TN=
#FP=
#FN=

#print([1 for i in range(len_image) if or_label[i]==pr_label[i]])
#sensitivity=
print(" Fine tune Accuracy:",accuracy)
print(" Fine tune Sensitivity:",Sensitivity)
print(" Fine tune Specificity:",Specificity)

accuracy_tr = len([1 for i in range(len_train) if or_label_tr[i]==pr_label_tr[i]])/float(len_train)
#print([1 for i in range(len_train) if or_label_tr[i]==pr_label_tr[i]])
#sensitivity=
#print(" Training Accuracy:",accuracy_tr)


print pr_value




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

#scaler = preprocessing.StandardScaler().fit(features_train)
#features = scaler.transform(features)
#features_train = scaler.transform(features_train)

print("---------------------------------------")
svc(features_train,or_label_tr,features,or_label,SVM_FILE,test_ids)
#print("---------------------------------------")
#rf(features_train,or_label_tr,features_train,or_label_tr,SVM_FILE,train_ids)
#rf(features_train,or_label_tr,features,or_label,RF_FILE,test_ids)
#gbc(features_train,or_label_tr,features,or_label,GBC_FILE,test_ids)

#print("---------------------------------------")
#NN(features_train,or_label_tr,features,or_label,NN_FILE,test_ids)

