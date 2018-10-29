import tensorflow as tf
import numpy as np
import os
import sys
import scipy.misc
from matplotlib import pyplot as plt
import csv
from six.moves import xrange

import metrics
import tf_records

slim = tf.contrib.slim

os.environ["CUDA_VISIBLE_DEVICES"] = " "

#####################################################################
#
# specify paths and filenames here!
#
#####################################################################

# define path to tf-slim here
sys.path.append("PATH_TO_SLIM/models-master/slim")

# define the path to your project here:
project_path = "PATH_TO_PYTHON_PROJECT"

# path to testing data in tfrecords file format:
testing_data_filename = "DATA_PATH\TestingData.tfrecords"

# trained model checkpoint to test:
model_checkpoint_path = "CHECKPOINT_PATH\checkpoint_file.ckpt"

#####################################################################

from networks import FCN

checkpoints_path = os.path.join(project_path, "Checkpoints")
log_path = os.path.join(project_path, "Logs")
data_save_path = os.path.join(project_path, "Results")

number_of_classes = 2

image, label = tf_records.read_and_decode(testing_data_filename)

image_batch = tf.expand_dims(image, axis=0)
label_batch = tf.expand_dims(label, axis=0)

label_batch = tf.cast(tf.equal(label_batch, 0), dtype=tf.int64)

# perform inference
logits, var_mapping = FCN(image_batch, num_classes=number_of_classes,
                          is_training=False)

# calculate prediction and probabilities
pred = tf.argmax(logits, axis=3)
pred = tf.cast(tf.equal(pred, 0), dtype=tf.float32)

prob = tf.nn.softmax(logits, dim=3)

# calculate metrics
label_batch = tf.cast(tf.equal(label_batch, 0), dtype=tf.float32)

tpr_coeff = metrics.tpr(label_batch, pred)
tnr_coeff = metrics.tnr(label_batch, pred)
dice_coeff = metrics.dsc_coeff(label_batch, pred)

initializer = tf.local_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(initializer)
    saver.restore(sess, model_checkpoint_path)

    summary_writer = tf.summary.FileWriter(log_path, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    hd_vec = []
    dice_vec = []
    tpr_vec = []
    tnr_vec = []

    iterations = 215

    for i in xrange(iterations):
        current_image, current_label, current_pred, tpr, tnr, dice = sess.run([image, label_batch, pred, tpr_coeff,
                                                                               tnr_coeff, dice_coeff],
                                                                              options=run_options,
                                                                              run_metadata=run_metadata)

        summary_writer.add_run_metadata(run_metadata, 'step%d' % i)

        if np.count_nonzero(current_label) > 0 and np.count_nonzero(current_pred) > 0:
            hd = metrics.hd_distance(current_label, current_pred)
            hd_vec.append(hd)

        dice_vec.append(dice * 100)
        tpr_vec.append(tpr * 100)
        tnr_vec.append(tnr * 100)

        image_save_path = os.path.join(data_save_path, "%d_image.jpg" % i)
        label_save_path = os.path.join(data_save_path, "%d_label.jpg" % i)
        pred_save_path = os.path.join(data_save_path, "%d_pred.jpg" % i)


        scipy.misc.imsave(image_save_path, current_image)
        scipy.misc.imsave(label_save_path, current_label[0, :, :])
        scipy.misc.imsave(pred_save_path, current_pred[0, :, :])

    coord.request_stop()
    coord.join(threads)

    mean_hd = np.mean(hd_vec)
    mean_dice = np.mean(dice_vec)
    mean_tpr = np.mean(tpr_vec)
    mean_tnr = np.mean(tnr_vec)

    with open('FCN_TransAug_512.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(hd_vec)
        filewriter.writerow(dice_vec)
        filewriter.writerow(tpr_vec)
        filewriter.writerow(tnr_vec)

    print("Mean Hausdorff Distance: ", mean_hd)
    print("Mean Dice: ", mean_dice)
    print("Mean TPR: ", mean_tpr)
    print("Mean TNR: ", mean_tnr)

summary_writer.close()
