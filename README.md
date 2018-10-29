# Exploit fully automatic low-level segmented PET Data for training high-level Deep Learning Algorithms for the corresponding CT Data Code for automatic urinary bladder segmentation using Python and Tensorflow.

A framework for urinary bladder segmentation in CT images using deep learning.

Contains code to train and test two different segmentation network architectures using training and testing data obtained from combined PET/CT scans. 

## Requirements
To use the framework, you need:

1. [Python](https://www.python.org/download/releases/3.5/) 3.5
2. [TensorFlow](https://www.tensorflow.org/versions/r1.3/) 1.3
3. [TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) library


Furthermore, you may find the [MeVisLab](https://www.mevislab.de/download/) network to preprocess training and testing data useful:

[Exploit 18F-FDG enhanced urinary bladder in PET data for Deep Learning Ground Truth Generation in CT scans](https://github.com/cgsaxner/DataPrep_UBsegmentation)

## Data
Our networks were trained and tested on the publically available [RIDER Lung PET CT Dataset](https://wiki.cancerimagingarchive.net/display/Public/RIDER+Lung+PET-CT). 


The data was preprocessed and prepared using a [MeVisLab network](https://github.com/cgsaxner/DataPrep_UBsegmentation).


This software produces ground-truth segmentations of the urinary bladder in CT using the co-registered PET data. PET radiotracer 18F-FDG accumulates in the urinary bladder, therefore, this organ can be distinguished using simple thresholding. Furthermore, data augmentation is applied using MeVisLab software. For further information, please refer to the paper:


[Exploit 18F-FDG Enhanced Urinary Bladder in PET Data for Deep Learning Ground Truth Generation in CT Scans.](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10578/105781Z/Exploit-sup18-supF-FDG-enhanced-urinary-bladder-in-PET-data/10.1117/12.2292706.short?SSO=1)


## Functionalities

- **Creating TFRecords files for training and testing data.** 
The script [`make_tfrecords_dataset.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/make_tfrecords_dataset.py) contains code to convert a directory of image files to the TensorFlow recommended file format TFRecords. TFRecords files are easy and fast to process in TensorFlow.

- **Training networks.**
The scripts [`FCN_training.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/FCN_training.py) and [`ResNet_training.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/ResNet_training.py) contain code for training deep neural networks using your own data. 
FCN is based on [FCN-8s by Long et al.](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) using [pre-trained VGG](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg.py).
ResNet is based on [DeepLab by Chen et al.](https://arxiv.org/pdf/1606.00915.pdf]) using [pre-trained ResNet V2](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v2.py).

- **Testing networks.**
The scripts [`FCN_testing.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/FCN_testing.py) and [`ResNet_testing.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/ResNet_testing.py) contain code for testing the previously trained networks.

- **Evaluation metrics.**
The file [`metrics.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/metrics.py) contains functions to calculate following metrics for evaluating segmentation results:
  - True Positive Rate (TPR)
  - True Negative Rate (TNR)
  - Intersection over union (Jaccard Index, IoU)
  - Dice-Sorensen coefficient (DSC)
  - Hausdorff distance (HD)


