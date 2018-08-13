import tensorflow as tf
import numpy as np


def tpr(label, prediction):
    """calculate true positive rate between label and prediction
    TPR = TP / (TP + FN)

    Parameters
    ----------
    label : tensor
        tensor of shape [batch_size, height, width, depth] containing the ground truth label
    prediction : tensor
        tensor of shape [batch_size, height, width, depth] containing the class prediction made by the network

    Returns
    ----------
    coeff : float
        calculated true negative rate
    """
    with tf.variable_scope("calc_tpr"):
        smoothing = 1e-5
        axis = [1, 2]

        tp = tf.reduce_sum(tf.multiply(prediction, label), axis=axis)
        fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(label, tf.ones_like(label)),
                                              tf.equal(prediction, tf.zeros_like(prediction))), "float"), axis=axis)

        coeff = tf.reduce_mean((tp + smoothing) / (tp + fn + smoothing))
    return coeff


def tnr(label, prediction):
    """calculate true negative rate between label and prediction
    TNR = TN / (TN + FP)

    Parameters
    ----------
    label : tensor
        tensor of shape [batch_size, height, width, depth] containing the ground truth label
    prediction : tensor
        tensor of shape [batch_size, height, width, depth] containing the class prediction made by the network

    Returns
    ----------
    coeff : float
        calculated true negative rate
    """

    with tf.variable_scope("calc_tnr"):
        smoothing = 1e-5
        axis = [1, 2]

        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(label, tf.zeros_like(label)),
                                              tf.equal(prediction, tf.zeros_like(prediction))), "float"), axis=axis)
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(label, tf.zeros_like(label)),
                                              tf.equal(prediction, tf.ones_like(prediction))), "float"), axis=axis)

        coeff = tf.reduce_mean((tn + smoothing) / (tn + fp + smoothing))
    return coeff


def iou_coeff(label, prediction):
    """calculate intersection over union (jaccard index) between label and prediction
    IOU = Intersection / Union = |Label AND Prediction| / |Label OR Prediction|

    Parameters
    ----------
    label : tensor
        tensor of shape [batch_size, height, width, depth] containing the ground truth label
    prediction : tensor
        tensor of shape [batch_size, height, width, depth] containing the class prediction made by the network

    Returns
    ----------
    coeff : float
        calculated intersection over union coefficient (jaccard index)
    """

    smoothing = 1e-5
    axis = [1, 2]

    intersec = tf.reduce_sum(tf.multiply(prediction, label), axis=axis)
    union = tf.reduce_sum(tf.cast(tf.add(prediction, label) >= 1, dtype=tf.float32), axis=axis)
    batch_iou = (intersec + smoothing) / (union + smoothing)
    coeff = tf.reduce_mean(batch_iou)
    return coeff


def dsc_coeff(label, prediction):
    """calculate dice sorensen coefficient between label and prediction
    DSC = Intersection / (number of label elements + number of prediction elements)

    Parameters
    ----------
    label : tensor
        tensor of shape [batch_size, height, width, depth] containing the ground truth label
    prediction : tensor
        tensor of shape [batch_size, height, width, depth] containing the class prediction made by the network

    Returns
    ----------
    coeff : float
        calculated dice sorensen coefficient
    """

    with tf.variable_scope("calc_dsc"):
        smoothing = 1e-5
        axis = [1, 2]

        intersec = tf.reduce_sum(tf.multiply(prediction, label), axis=axis)
        x = tf.reduce_sum(prediction, axis=axis)
        y = tf.reduce_sum(label, axis=axis)
        batch_dice = (2.0 * intersec + smoothing) / (x + y + smoothing)
        coeff = tf.reduce_mean(batch_dice)
    return coeff


def directed_hausdorff(A, B):
    """calculate directed hausdorff distance h(A,B) between point sets A and B
    h(A, B) = max(min(d(a, b)))
    where d(a, b) is L2 Norm between points a and b.
    Point sets A and B may have a different number of rows, but must have the same dimension (number of columns)

    Parameters
    ----------
    A : ndarray
       array of shape [m, dim] where m denotes the number of points in point set A and dim the dimension of the point
       set
    B : ndarray
        array of shape [n, dim] where n denotes the number of points in point set B and dim the dimension of the point
        set

    Returns
    ----------
    distance : float
        calculated directed hausdorff distance
    """

    m = np.shape(A)[0]
    n = np.shape(B)[0]
    dim = np.shape(A)[1]

    dist = []

    for k in range(m):
        C = np.ones([n, 1]) * A[k, :]
        D = np.multiply((C - B), (C - B))
        D = np.sqrt(D.dot(np.ones([dim, 1])))
        dist.append(np.min(D))

    distance = np.max(dist)
    return distance


def hd_distance(label, prediction):
    """calculate hausdorff distance between label and prediction
    HD = max(h(A,B), h(B,A))
    where A and B are point sets containing the indices of positive class labels and predictions and h(A,B) beeing the
    directed hausdorff distance calculated by function "directed_hausdorff(A, B)"

    Parameters
    ----------
    label : numpy array
       array of shape [batch_size, height, width] containing the ground truth label
    prediction : tensor
        array of shape [batch_size, height, width] containing the class prediction made by the network

    Returns
    ----------
    coeff : float
        calculated hausdorff distance
    """

    A_indices = np.argwhere(label[0, :, :] > 0)
    B_indices = np.argwhere(prediction[0, :, :] > 0)

    AB = directed_hausdorff(A_indices, B_indices)
    BA = directed_hausdorff(B_indices, A_indices)

    coeff = max(AB, BA)

    return coeff
