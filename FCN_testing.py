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

# define slim path
sys.path.append("C:/Users/gsaxner/Documents/Code/models-master/slim")
# sys.path.append("D:/Dokumente/Uni/Master/Masterarbeit/Documents/Code/models-master/slim")

# sys.path.append("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/extras/CUPTI/libx64")

from networks import FCN

# checkpoints_path = "D:/Dokumente/Uni/Master/Masterarbeit/PycharmProjects/MastersThesis/Checkpoints"
checkpoints_path = "C:/Users/gsaxner/PycharmProjects/MastersThesis/Checkpoints"

# log_path = "D:/Dokumente/Uni/Master/Masterarbeit/PycharmProjects/MastersThesis/Logs"
log_path = "C:/Users/gsaxner/PycharmProjects/MastersThesis/Logs"

testing_data_filename = "F:\TestingDataNoAug\TestingData_512.tfrecords"

data_save_path = "F:\Results\FCN_TransAug_256"

model_checkpoint_path = os.path.join(checkpoints_path, "FCN_TransAug_256.ckpt")

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

        # if i % 500 == 0:
        #     f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        #     ax1.imshow(current_image)
        #     ax1.set_title('Input image')
        #     ax2.imshow(current_label[0, :, :])
        #     ax2.set_title('Input Ground-Truth Annotation')
        #     ax3.imshow(current_pred[0, :, :])
        #     ax3.set_title('Prediction')
        #     plt.show()

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

    # hd_save_path = os.path.join(data_save_path, "hausdorff.jpg")
    # dice_save_path = os.path.join(data_save_path, "dice.jpg")
    # tpr_save_path = os.path.join(data_save_path, "tpr.jpg")
    # tnr_save_path = os.path.join(data_save_path, "tnr.jpg")
    #
    # plt.hist(dice_vec, edgecolor='black', linewidth=1.2)
    # plt.xlabel('Dice Coefficient')
    # plt.ylabel('Frequency')
    # plt.axis([0, 100, 0, 150])
    # plt.grid(True)
    # plt.savefig(dice_save_path)
    # plt.clf()
    #
    # plt.hist(tpr_vec, edgecolor='black', linewidth=1.2)
    # plt.xlabel('True Positive Rate')
    # plt.ylabel('Frequency')
    # plt.axis([0, 100, 0, 150])
    # plt.grid(True)
    # plt.savefig(tpr_save_path)
    # plt.clf()
    #
    # plt.hist(tnr_vec, edgecolor='black', linewidth=1.2)
    # plt.xlabel('True Negative Rate')
    # plt.ylabel('Frequency')
    # plt.axis([60, 100, 0, 150])
    # plt.grid(True)
    # plt.savefig(tnr_save_path)
    # plt.clf()
    #
    # plt.hist(hd_vec, edgecolor='black', linewidth=1.2)
    # plt.xlabel('Hausdorff Distance')
    # plt.ylabel('Frequency')
    # plt.axis([0, 60, 0, 150])
    # plt.grid(True)
    # plt.savefig(hd_save_path)
    # plt.clf()

    print("Mean Hausdorff Distance: ", mean_hd)
    print("Mean Dice: ", mean_dice)
    print("Mean TPR: ", mean_tpr)
    print("Mean TNR: ", mean_tnr)

summary_writer.close()
