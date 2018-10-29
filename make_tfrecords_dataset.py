import tf_records
import os

data_path = "PATH_TO_IMAGE_DATA"
outfile_name = "Data.tfrecords"
height, width = 512, 512

tfrecords_filename = os.path.join(data_path, outfile_name)
images_path = os.path.join(data_path, "Images")
labels_path = os.path.join(data_path, "Labels")

tf_records.write_to_tfrecords(images_path, labels_path, tfrecords_filename, height, width)
