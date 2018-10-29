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

## Functionalities

- **Creating TFRecords files for training and testing data.** 
The script [`make_tfrecords_dataset.py`](https://github.com/cgsaxner/UB_Segmentation/blob/master/make_tfrecords_dataset.py) contains code to convert a directory of image files to the TensorFlow recommended file format TFRecords. TFRecords files are easy and fast to process in TensorFlow.


