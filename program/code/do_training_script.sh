#!/bin/bash

dir=$藍偉任/program/input
test_sr=$藍偉任/test_data/EBUSimagestraining_test/test2
train_sr=$藍偉任/test_data/EBUSimagestraining_test/train2_flip_rotate_mless
test_d=$藍偉任/program/input/test
train_d=$藍偉任/program/input/train
create_lmdb=$藍偉任/program/code/create_lmdb_227.py


filelist=$(ls ${dir})
for filename in ${filelist}
    do  
		echo $dir/$filename
		rm -rf $dir/$filename
        
    done

cp -R $train_sr $train_d
cp -R $test_sr $test_d

#echo "Hi, I'm sleeping for 3 seconds..."
#sleep 7

python2 $create_lmdb


#echo "Hi, I'm sleeping for 3 seconds..."
#sleep 18

C:/caffe/Build/x64/Release/compute_image_mean.exe -backend=lmdb  $藍偉任/program/input/train_lmdb  $藍偉任/program/input/mean.binaryproto
$藍偉任/ program/caffe_models/caffenet/solver_2.prototxt --weights $藍偉任/program/pretrained_models/bvlc_reference_caffenet.caffemodel  2>&1 | tee $藍偉任/ program/caffe_models/caffenet/model_2_train_trasfer.log

read -p "Press any key..."