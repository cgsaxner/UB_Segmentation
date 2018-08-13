
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from scipy.misc import imresize


def _bytes_feature(value):
    """helper function to create byte feature
    reference: https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/how_tos/reading_data/
    convert_to_records.py
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """helper function to create int64 feature
    reference: https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/how_tos/reading_data/
    convert_to_records.py
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_to_tfrecords(image_path, label_path, tfrecords_filename, image_height, image_width):
    """converts image and label data to a tfrecord

    Parameters
    ----------
    image_path : string
        path to a directory of image files
    label_path : string
        path to a directory of label files
    tfrecords_filename : string
        name of the created tfrecords file
    image_height : int
        height to which images and labels will be rescaled
    image_width : int
        width to which images and labels will be rescaled
    """

    print("writing to tfrecords file ", tfrecords_filename)

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    filename_images = []
    filename_labels = []

    counter = 0

    # get filename pairs of labels and images
    for root_img, dir_img, files_img in os.walk(image_path):
        for name_img in files_img:
            image_name = os.path.join(root_img, name_img)
            filename_images.append(image_name)

    for root_labels, dir_labels, files_labels in os.walk(label_path):
        for name_labels in files_labels:
            image_name = os.path.join(root_labels, name_labels)
            filename_labels.append(image_name)

    filename_pairs = zip(filename_images, filename_labels)

    # write image-label pairs to tfrecords
    for img_path, lb_path in filename_pairs:
        img = np.array(Image.open(img_path))
        label = np.array(Image.open(lb_path))
        counter = counter + 1
        if counter % 100 == 0:
            print(counter)

        img = imresize(img, (image_height, image_width, 1))
        label = imresize(label, (image_height, image_width, 1))

        img_raw = img.tostring()
        label_raw = label.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(image_height),
            'width': _int64_feature(image_width),
            'image_raw': _bytes_feature(img_raw),
            'label_raw': _bytes_feature(label_raw)
        }))

        writer.write(example.SerializeToString())

    print("done. %d examples written to %s." % (counter, tfrecords_filename))
    writer.close()


def read_and_decode(tfrecords_filename):
    """reads and decodes image an label data from tfrecords file

    Parameters
    ----------
    tfrecords_filename : string
        path to the tfrecords file to decode

    Returns
    ----------
    (image, label) : tuple (tensor, tensor)
        image is a tensor of shape {height, width, num_channels] containing the image and label is a tensor of shape
        {height, width] containing the corresponding ground truth label
    """
    print("reading from file %s ..." % tfrecords_filename)
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=1)
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string)
    })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.decode_raw(features['label_raw'], tf.uint8)

    record_it = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    for str_record in record_it:
        example = tf.train.Example()
        example.ParseFromString(str_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])

    image = tf.reshape(image, tf.stack([height, width, 1]))
    label = tf.reshape(label, tf.stack([height, width]))

    image = tf.image.grayscale_to_rgb(image)

    return image, label
