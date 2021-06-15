# Improvements-of-SCNN-on-lane-detection-in-harsh-scenarios
This project is based on the work of [Pan, X., Shi, J., Luo, P., Wang, X., & Tang, X. ( Spatial as Deep: Spatial CNN for Traffic Scene Understanding)](https://github.com/cardwing/Codes-for-Lane-Detection/tree/master/SCNN-Tensorflow). 

## Dataset
The dataset we use is [CULane](https://xingangpan.github.io/projects/CULane.html), a large scale challenging dataset comprised ofurban, rural and highway scenes collected in Beijing for traffic lane detection. This dataset extracts 133235 frames from more than 55 hours of videos frames, and the images are divided into 88880 images for training set (66.7%), 9675 images for validation set (7.3%) and 34680 images for test set(26.0%). 

Test set is further divided into normal and 8 challengingcategories, including crowded, night, no line, shadow, arrow, dazzle light, curve, crossroad. Eachframe has a resolution of 1640 x 590 and is distorted. Lane segmentation labels, i.e.per-pixel labelsgenerated from original annotations, are provided.

<img width="821" alt="Screen Shot 2021-06-15 at 3 03 30 PM" src="https://user-images.githubusercontent.com/55666152/122129424-ed2b2080-cdea-11eb-958b-897e581d5968.png">

## How to use
```
conda create -n tensorflow_gpu pip python=3.5 
source activate tensorflow_gpu
pip install --upgrade tensorflow-gpu==1.3.0
pip3 install -r lane-detection-model/requirements.txt 
```
Download the [vgg16.npy](https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg16.py) and put it in lane-detection-model/data.

## Train
```
cd lane-detection-model
CUDA_VISIBLE_DEVICES="0" python tools/train_lanenet.py --net vgg --dataset_dir path/to/CULane-dataset/
```

## Test
```
cd lane-detection-model
CUDA_VISIBLE_DEVICES="0" python tools/test_lanenet.py --weights_path path/to/model_weights_file --image_path path/to/image_name_list
```
