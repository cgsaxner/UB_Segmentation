import tf_records
import os

data_path = "F:/TestingDataNoAug"
images_path = os.path.join(data_path, "Images")
labels_path = os.path.join(data_path, "Labels")

height, width = 512, 512

tfrecords_filename = os.path.join(data_path, "TestingData_512.tfrecords")

tf_records.write_to_tfrecords(images_path, labels_path, tfrecords_filename, height, width)
