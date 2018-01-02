#

使用Caffe框架，並運用開源ImageNet pretrained model做finetune，並串接CNN+SVM做肺部支氣管超音波影像良惡性診斷

* 環境: Caffe on Windows10 pro eng+CUDA8.0+VS2013 pro+CUDA8.0
* [環境架設說明](
https://medium.com/@willylan/caffe-on-windows-windows-10-pro-vs2013-pro-update5-eng-cuda-8-0-cudnn-v5-1rc-31a9307dfa37)
* 設備：GPU: GTX 1070 8GB

### Dataset
良性：56
惡性：108

## ./program/code/*.py


### Step1:資料整理
* convertdicomtotiff_crop.py:超音波影像從dicom轉至tiff檔，並去掉非相關影像。
* classify_BM.py：原來影像檔以肺部疾病名稱做資料夾分類，經此步驟分為良性、惡性。	
* cross_validate_produce.py:5-fold cross validation，影像分為5個fold test0~4與train0~4。


### Step2:Data Augementation

* trainset_flip.py：影像上下翻轉 	
* trainset_rotate_Mless.py：影像旋轉90、180、270度	


### Step3:Data Augementation
* create_lmdb_224.py/create_lmdb_227.py:影像調整為model的input size，並產生training時所需的lmdb檔案。
* do_training_script.sh：跑create_lmdb_224(7).py及 finetuning command


### Step4:Predict Result
* caffenet_result.py: 執行caffenet/caffenet+svm/caffenet+rf 
* caffenet_result_combinedglcm.py：結合glcm features與caffenet fc8 的features後，在丟入svm/rf 	
* googlenet_result.py:執行googlenet/googlenet+svm/googlenet+rf 	
* resnet50_result.py:執行resnet/resnet+svm/resnet+rf	
* vgg16_result.py:執行vggnet/vggnet+svm/vggnet+rf		


---
BestResult: Fine-tuned CaffeNet-SVM
Accuracy (%): 85.4 (140/164)
Sensitivity (%): 87.0 (94/108)
Specificity (%): 82.1 (46/56)
PPV (%): 90.4 (94/104)
NPV (%): 76.7 (46/60)
AUC: 0.8705
---

