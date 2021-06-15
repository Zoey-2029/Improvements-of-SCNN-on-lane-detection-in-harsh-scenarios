# Improvements-of-SCNN-on-lane-detection-in-harsh-scenarios
This project is based on the work of [Pan, X., Shi, J., Luo, P., Wang, X., & Tang, X. ( Spatial as Deep: Spatial CNN for Traffic Scene Understanding)](https://github.com/cardwing/Codes-for-Lane-Detection/tree/master/SCNN-Tensorflow). 

## Dataset
The dataset we use is [CULane](https://xingangpan.github.io/projects/CULane.html), a large scale challenging dataset comprised ofurban, rural and highway scenes collected in Beijing for traffic lane detection. This dataset extracts 133235 frames from more than 55 hours of videos frames, and the images are divided into 88880 images for training set (66.7%), 9675 images for validation set (7.3%) and 34680 images for test set(26.0%). 

Test set is further divided into normal and 8 challengingcategories, including crowded, night, no line, shadow, arrow, dazzle light, curve, crossroad. Eachframe has a resolution of 1640 x 590 and is distorted. Lane segmentation labels, i.e.per-pixel labelsgenerated from original annotations, are provided.

<p align="center">
<img width="821" alt="Screen Shot 2021-06-15 at 3 03 30 PM" src="https://user-images.githubusercontent.com/55666152/122129424-ed2b2080-cdea-11eb-958b-897e581d5968.png">
</p>

## How to use

```
conda create -n tensorflow_gpu pip python=3.5 
source activate tensorflow_gpu
pip install --upgrade tensorflow-gpu==1.3.0
pip3 install -r lane-detection-model/requirements.txt 
```
* VGG16 (**Model_vgg**): download [vgg16.npy](https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg16.py) and put it in lane-detection-model/data.

* Loss function and optimizer (**Model_bg0.2**): The loss function is defined as:
<p align="center">
<img width="250" alt="Screen Shot 2021-06-15 at 3 37 13 PM" src="https://user-images.githubusercontent.com/55666152/122132388-9b38c980-cdef-11eb-8374-66e3022125a5.png">
</p>

The binary segmentation loss is a weighted sum of cross entropy loss of lane and background pixels. The coefficient(w0) of background pixels for the segmentation loss is 0.4 and the coefficients(w1, w2, w3, w4) of lane marks are 1. g calculates sigmoid cross entropy loss, which represents the loss estimating existence label of each lane. The coefficient(β) of existence loss is 0.1.
Since the number of background pixels is much larger than that of lane pixels, we want to lower their influence in the loss function. We decrease w0 from 0.4 to 0.2. 

* Data augmentation (**Model_data_aug**): run `data_aug.py`, it would expand the dataset from 88880 to 115701 by decreasing brightness, adding Gaussian blur and deceasing contrast to existing images. The proportion of images generated with different method is: decreased brightness (40%), Gaussian blur (30%) and decreased contrast (20%).

* Replacing VGG16 with ResNet(**Model_resnet**):


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

## Results
* training accuracy:
<p align="center">
<img alt="training_accuracy" src="https://user-images.githubusercontent.com/55666152/122131893-bd7e1780-cdee-11eb-9178-c7abcf8b736f.png" width="800">
</p>

* F1 score:
<p align="center">
<img alt="F1" src="https://user-images.githubusercontent.com/55666152/122132452-b86d9800-cdef-11eb-9756-f4d0f9b57e4d.png" width="800" height="275">
</p>

* accuracy:
<p align="center">
<img alt="accuracy" src="https://user-images.githubusercontent.com/55666152/122132468-c58a8700-cdef-11eb-9056-8902ecd1e449.png" width="800" height="275">
</p>

* callback:
<p align="center">
<img alt="callback" src="https://user-images.githubusercontent.com/55666152/122132505-d6d39380-cdef-11eb-8e01-c959a3272c29.png" width="800" height="275">
</p>

* performance:
<p align="center">
<img width="800" alt="lanes" src="https://user-images.githubusercontent.com/55666152/122132909-8f99d280-cdf0-11eb-9897-7bb739676a00.png">
</p>


