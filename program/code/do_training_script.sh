#!/bin/bash

dir=$thesis/program/input
test_sr=$thesis/test_data/EBUSimagestraining_test/test2
train_sr=$thesis/test_data/EBUSimagestraining_test/train2_flip_rotate_mless
test_d=$thesis/program/input/test
train_d=$thesis/program/input/train
create_lmdb=$thesis/program/code/create_lmdb_227.py


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

C:/caffe/Build/x64/Release/compute_image_mean.exe -backend=lmdb  $thesis/program/input/train_lmdb  $thesis/program/input/mean.binaryproto
$thesis/ program/caffe_models/caffenet/solver_2.prototxt --weights $thesis/program/pretrained_models/bvlc_reference_caffenet.caffemodel  2>&1 | tee $thesis/ program/caffe_models/caffenet/model_2_train_trasfer.log

read -p "Press any key..."
