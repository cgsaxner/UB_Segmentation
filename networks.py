import tensorflow as tf
from upsampling import bilinear_upsampling_weights
from nets import vgg, resnet_v2

slim = tf.contrib.slim

# network definitions are based on the code found at
# https://github.com/warmspringwinds/tf-image-segmentation

def FCN(image_batch, num_classes, is_training):
    """FCN-8s model architecture as introduced by Long et al. in "Fully Convolutional Networks for Semantic
    Segmentation". The VGG 16 Network downsamples an input byfactor 32, then bilinear interpolation and skip connections
    are used to upsample theprediction to the size of the original input.

    Parameters
    ----------
    image_batch : tensor
        tensor of shape [batch_size, height, width, depth] containing the image batch on which inference should be
        performed
    num_classes : int
        specifies the number of classes to predict
    is_training : boolean
        argument defining if the network is trained or tested. required by the vgg 16 network definition.

    Returns
    ----------
    fcn_8_logits : tensor
        tensor of shape [batch_size, height, width, num_classes] containing the unscaled log probabilities from vgg 16
        network, upsampled using FCN8s network architecture
    fcn_var_mapping : dict {string: variable}
        dictionary to map variables from VGG 16 name scope to FCN name scope
    """

    image_batch_float = tf.to_float(image_batch)

    with tf.variable_scope("FCN_vars") as fcn_variable_scope:
        # vgg model definition
        with slim.arg_scope(vgg.vgg_arg_scope()):
            vgg_logits, end_points = vgg.vgg_16(image_batch_float, num_classes=num_classes,
                                                is_training=is_training, spatial_squeeze=False,
                                                fc_conv_padding='SAME')

            vgg_logits_shape = tf.shape(vgg_logits)

            # upsampling filters
            with tf.variable_scope("Upsampling_filt_2"):
                upsampling_filter_2_val = bilinear_upsampling_weights(2, num_classes)
                upsampling_filter_2 = tf.constant(upsampling_filter_2_val)

            with tf.variable_scope("Upsampling_filt_8"):
                upsampling_filter_8_val = bilinear_upsampling_weights(8, num_classes)
                upsampling_filter_8 = tf.constant(upsampling_filter_8_val)

            # FCN 8s
            # upsample last layer prediction by factor two
            with tf.variable_scope("2xUpsampling"):
                vgg_logits_upsample2_shape = tf.stack([vgg_logits_shape[0], vgg_logits_shape[1] * 2,
                                                   vgg_logits_shape[2] * 2, vgg_logits_shape[3]])

                vgg_logits_upsample2 = tf.nn.conv2d_transpose(vgg_logits, upsampling_filter_2,
                                                          vgg_logits_upsample2_shape, [1, 2, 2, 1], name="Upsampling_2")

            # create the pool4 skip connection
            pool4_endpoint = end_points['FCN_vars/vgg_16/pool4']
            pool4_logits = slim.conv2d(pool4_endpoint, num_classes, [1, 1],
                                       activation_fn=None, normalizer_fn=None,
                                       weights_initializer=tf.zeros_initializer,
                                       scope='pool4_skip')

            # combine
            with tf.variable_scope("combine"):
                comb1 = pool4_logits + vgg_logits_upsample2
                comb1_shape = tf.shape(comb1)

            # upsample combination by factor 2
            with tf.variable_scope("2xUpsampling"):
                comb1_upsample2_shape = tf.stack([comb1_shape[0], comb1_shape[1] * 2,
                                              comb1_shape[2] * 2, comb1_shape[3]])

                comb1_upsample2 = tf.nn.conv2d_transpose(comb1, upsampling_filter_2,
                                                     comb1_upsample2_shape, [1, 2, 2, 1], name="Upsampling_2")

            # create the pool3 skip connection
            pool3_endpoint = end_points['FCN_vars/vgg_16/pool3']
            pool3_logits = slim.conv2d(pool3_endpoint, num_classes, [1, 1],
                                       activation_fn=None, normalizer_fn=None,
                                       weights_initializer=tf.zeros_initializer,
                                       scope='pool3_skip')

            # combine & upsample by factor 8
            with tf.variable_scope("combine"):
                comb2 = pool3_logits + comb1_upsample2
                comb2_shape = tf.shape(comb2)

            with tf.variable_scope("8xUpsampling"):
                fcn_8_logits_shape = tf.stack([comb2_shape[0], comb2_shape[1] * 8,
                                           comb2_shape[2] * 8, comb2_shape[3]])
                fcn_8_logits = tf.nn.conv2d_transpose(comb2, upsampling_filter_8,
                                                  fcn_8_logits_shape, [1, 8, 8, 1], name="Upsampling_8")

            # create a variable mapping from vgg_16 namescope to fcn namescope
            fcn_var_mapping = {}
            fcn_vars = slim.get_variables(fcn_variable_scope)

            for var in fcn_vars:

                if 'pool4_skip' in var.name:
                    continue

                if 'pool3_skip' in var.name:
                    continue

                vgg_16_vars = var.name[len(fcn_variable_scope.name)+1:-2]
                fcn_var_mapping[vgg_16_vars] = var

    return fcn_8_logits, fcn_var_mapping


def upsampled_ResNet(image_batch, num_classes, is_training):
    """Upsampled ResNet V2 152 architecture as introduced by Chen et al. in "DeepLab:Semantic Image Segmentation with
    Deep Convolutional Nets, Atrous Convolution, andFully Connected CRFs". The ResNet V2 152 network downsamples an
    input by afactor specifyable by the user, then bilinear interpolation is used to upsample theprediction to the size
    of the original input.

    Parameters
    ----------
    image_batch : tensor
        tensor of shape [batch_size, height, width, depth] containing the image batch on which inference should be
        performed
    num_classes : int
        specifies the number of classes to predict
    is_training : boolean
        argument defining if the network is trained or tested. required by the ResNet network definition. must always
        be set to false to run ResNet in segmentation mode!

    Returns
    ----------
    upsampled_logits : tensor
        tensor of shape [batch_size, height, width, num_classes] containing the upsampled unscaled log probabilities
        from resnet_V2_152
    up_resnet_var_mapping : dict {string: variable}
        dictionary to map variables from ResNet V2 152 name scope to upsampled ResNet name scope
    """

    image_batch_float = tf.to_float(image_batch)

    with tf.variable_scope("up_ResNet_vars") as up_resnet_var_scope:
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            # resnet model definition
            resnet_logits, end_points = resnet_v2.resnet_v2_152(image_batch_float,
                                                                num_classes,
                                                                is_training=is_training,
                                                                global_pool=False,
                                                                spatial_squeeze=False,
                                                                output_stride=8)

            resnet_logits_shape = tf.shape(resnet_logits)

            # upsampling filters
            with tf.variable_scope("Upsampling_filt_8"):
                upsampling_filter_8_val = bilinear_upsampling_weights(8, num_classes)
                upsampling_filter_8 = tf.constant(upsampling_filter_8_val)

            # perform upsampling
            with tf.variable_scope("8xUpsampling"):
                upsampled_logits_shape = tf.stack([resnet_logits_shape[0],
                                               resnet_logits_shape[1] * 8,
                                               resnet_logits_shape[2] * 8,
                                               resnet_logits_shape[3]])

                upsampled_logits = tf.nn.conv2d_transpose(resnet_logits, upsampling_filter_8,
                                                      upsampled_logits_shape, [1, 8, 8, 1], name="Upsampling_8")

            # create a variable mapping from resnet namescope to upsampled resnet namescope
            up_resnet_var_mapping = {}
            up_resnet_vars = slim.get_variables(up_resnet_var_scope)

            for var in up_resnet_vars:

                resnet_vars = var.name[len(up_resnet_var_scope.name)+1:-2]
                up_resnet_var_mapping[resnet_vars] = var

    return upsampled_logits, up_resnet_var_mapping
