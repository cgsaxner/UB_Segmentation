import sys
import os
import tensorflow as tf
from six.moves import xrange
import tf_records
slim = tf.contrib.slim

# define CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
training_data_filename = "DATA_PATH\TrainingData.tfrecords"

#####################################################################


from networks import upsampled_ResNet

checkpoints_dir = os.path.join(project_path, "Checkpoints")
log_path = os.path.join(project_path, "Logs")
data_save_path = os.path.join(project_path, "Results")

model_checkpoint_path = os.path.join(checkpoints_dir, 'ResNet.ckpt')
model_checkpoint_final_path = os.path.join(checkpoints_dir, 'ResNet_final.ckpt')
resnet_checkpoint_path = os.path.join(checkpoints_dir, 'resnet_v2_152.ckpt')


# define parameters
number_of_classes = 2
batch_size = 1

# load image and annotations
image, label = tf_records.read_and_decode(training_data_filename)

# obtain random batches from images and annotations
image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                       capacity=3000, num_threads=2,
                                                       min_after_dequeue=1000)

# create valid labels
class_labels = tf.not_equal(label_batch, 0)
background_labels = tf.equal(label_batch, 0)

combined_labels_batch = tf.stack(axis=3, values=[tf.to_float(class_labels),
                                          tf.to_float(background_labels)])

# perform inference
logits_batch, var_mapping = upsampled_ResNet(image_batch,
                                             num_classes=number_of_classes,
                                             is_training=False)

# loss function: cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                               logits=logits_batch,
                               labels=combined_labels_batch))

# calculate prediction and probabilities
prediction = tf.argmax(logits_batch, dimension=3)
probabilities = tf.nn.softmax(logits_batch)

with tf.variable_scope("optimizer_variables"):
    train_step = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cross_entropy)

# remove last layer from variable mapping (the last layer defines the number of classes)
var_keys = var_mapping.keys()
var_keys_without_logits = []

for key in var_keys:
    if 'logits' not in key:
        var_keys_without_logits.append(key)

updated_mapping = {key: var_mapping[key] for key in var_keys_without_logits}

# define & create summary
tf.summary.scalar('cross_entropy_loss', cross_entropy)
merged_summary_op = tf.summary.merge_all()
summary_string_writer = tf.summary.FileWriter(log_path)

# load vgg checkpoint variables an map them to new variable names
init_fn = slim.assign_from_checkpoint_fn(resnet_checkpoint_path, updated_mapping)

# variable initializers
global_vars_init_op = tf.global_variables_initializer()
local_vars_init_op = tf.local_variables_initializer()
combined_op = tf.group(local_vars_init_op, global_vars_init_op)

model_variables = slim.get_model_variables()
saver = tf.train.Saver(model_variables)

# start session
with tf.Session() as sess:
    sess.run(combined_op)
    init_fn(sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # 10 epochs of training
    for i in xrange(17010 * 2):

        loss, summary_string, _ = sess.run([cross_entropy,
                                            merged_summary_op, train_step])
        if i % 100 == 0:
            print("Step: %d Current loss: %f" %(i, loss))

        summary_string_writer.add_summary(summary_string, i)
        if i % 1000 == 0:
            save_path = saver.save(sess, model_checkpoint_path)
            print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)

    save_path = saver.save(sess, model_checkpoint_path)
    print("Model saved in file: %s" % save_path)

summary_string_writer.close()
