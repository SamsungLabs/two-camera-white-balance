"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Abdelrahman Abdelhamed (a.abdelhamed@samsung.com; abdoukamel@gmail.com)

Loss functions for white balance estimation.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import losses
from tensorflow.python.keras.losses import LossFunctionWrapper, mean_absolute_error, cosine_similarity
from tensorflow.python.keras.utils import losses_utils


@tf.function
def l2_normalize(x, axis, epsilon=1e-8):
    x = tf.sign(x) * tf.maximum(tf.abs(x), epsilon)
    norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keepdims=True))
    return x / tf.maximum(norm, epsilon)


@tf.function
def cos_two_vectors(y_true, y_pred):
    """cosine between two vectors, in [-1, 1]"""
    y_true_n = l2_normalize(y_true, axis=-1)
    y_pred_n = l2_normalize(y_pred, axis=-1)
    cos = tf.reduce_sum(y_true_n * y_pred_n, axis=-1)  # keep batch dim
    return tf.clip_by_value(cos, -1.0, 1.0)


@tf.function
def cos_distance(y_true, y_pred):
    """cosine distance between two vectors, multiplied by -1, to be minimized"""
    cos = cos_two_vectors(y_true, y_pred)
    cos_dist = -1. * cos  # [-1, 1] => [1, -1]
    return cos_dist


@tf.function
def angular_error(y_true, y_pred):
    # According to tensorflow docs, cosine_similarity() is a negative quantity between -1 and 0,
    # where 0 indicates orthogonality and values closer to -1 indicate greater similarity.
    # in some cases, cosine_similarity() above returns values out of [-1, 0], so clip
    # after some experimentation, cosine_proximity() has some problems and produces NaNs in loss

    # cos_prox = cosine_similarity(y_true, y_pred)
    # cos_prox = tf.clip_by_value(cos_prox, -1.0, 0.0)
    # ang_errs = tf.math.acos(-1 * cos_prox) * 180 / np.pi

    cos = cos_two_vectors(y_true, y_pred)
    ang_errs = tf.math.acos(cos) * 180 / np.pi
    return ang_errs


@tf.function
def mean_angular_error_two_vectors(y_true, y_pred):
    wb1_pred = y_pred[:, :3]
    wb2_pred = y_pred[:, 3:6]
    wb1_true = y_true[:, :3]
    wb2_true = y_true[:, 3:6]
    ang_errs_1 = angular_error(wb1_true, wb1_pred)
    ang_errs_2 = angular_error(wb2_true, wb2_pred)
    mean_ang_err = tf.reduce_mean([ang_errs_1, ang_errs_2], axis=0)  # keep batch dim
    return mean_ang_err


@tf.function
def cosine_similarity_two_vectors(y_true, y_pred):
    wb1_pred = y_pred[:, :3]
    wb2_pred = y_pred[:, 3:6]
    wb1_true = y_true[:, :3]
    wb2_true = y_true[:, 3:6]

    cos_sim_1 = cosine_similarity(wb1_true, wb1_pred)
    cos_sim_2 = cosine_similarity(wb2_true, wb2_pred)

    cos_sim = tf.reduce_mean([cos_sim_1, cos_sim_2], axis=0)  # keep batch dim
    return cos_sim


@tf.function
def mean_cos_distance_two_vectors(y_true, y_pred):
    wb1_pred = y_pred[:, :3]
    wb2_pred = y_pred[:, 3:6]
    wb1_true = y_true[:, :3]
    wb2_true = y_true[:, 3:6]

    cos_dist_1 = cos_distance(wb1_true, wb1_pred)
    cos_dist_2 = cos_distance(wb2_true, wb2_pred)

    cos_dist = tf.reduce_mean([cos_dist_1, cos_dist_2], axis=0)  # keep batch dim
    return cos_dist


@tf.function
def mean_absolute_error_two_vectors(y_true, y_pred):
    wb1_pred = y_pred[:, :3]
    wb2_pred = y_pred[:, 3:6]
    wb1_true = y_true[:, :3]
    wb2_true = y_true[:, 3:6]
    mae1 = mean_absolute_error(wb1_true, wb1_pred)
    mae2 = mean_absolute_error(wb2_true, wb2_pred)
    mae = tf.reduce_mean([mae1, mae2], axis=0)  # keep batch dim
    return mae


@tf.function
def mean_angular_error_with_transform(y_true, y_pred):
    wb1_pred = y_pred[:, :3]
    wb2_pred = y_pred[:, 3:6]
    trans = y_pred[:, 6:]
    trans = tf.reshape(trans, (-1, 3, 3))
    wb1_true = y_true[:, :3]
    wb2_true = y_true[:, 3:6]

    ang_errs_1 = angular_error(wb1_true, wb1_pred)
    ang_errs_2 = angular_error(wb2_true, wb2_pred)

    ang_errs = tf.reduce_mean([ang_errs_1, ang_errs_2], axis=0)  # keep batch dim
    trans_err = transform_error(wb1_pred, wb2_pred, trans)  # keep batch dim
    return tf.reduce_mean([ang_errs + trans_err], axis=0)  # keep batch dim


@tf.function
def mean_absolute_error_with_transform(y_true, y_pred):
    mae = mean_absolute_error_two_vectors(y_true, y_pred)
    trans_err = transform_error_from_outputs(y_true, y_pred)  # keep batch dim

    return tf.reduce_mean([mae + trans_err], axis=0)  # keep batch dim


def get_mean_absolute_error_with_transform_weight(trans_weight):
    @tf.function
    def mean_absolute_error_with_transform_weight(y_true, y_pred):
        mae = mean_absolute_error_two_vectors(y_true, y_pred)
        trans_err = transform_error_from_outputs(y_true, y_pred)  # keep batch dim
        return tf.reduce_mean([mae + trans_weight * trans_err], axis=0)  # keep batch dim

    return mean_absolute_error_with_transform_weight


@tf.function
def transform_error(wb1_pred, wb2_pred, trans):
    wb1_pred_trans = tf.reduce_sum(wb1_pred[:, :, tf.newaxis] * trans, axis=-1)  # matrix multiplication over batch
    trans_err = mean_absolute_error(wb1_pred_trans, wb2_pred)  # keep batch dim
    # trans_err = angular_error(wb1_pred_trans, wb2_pred)  # keep batch dim
    return trans_err


@tf.function
def transform_error_from_outputs(y_true, y_pred):
    wb1_pred = y_pred[:, :3]
    wb2_pred = y_pred[:, 3:6]
    trans = y_pred[:, 6:]
    trans = tf.reshape(trans, (-1, 3, 3))
    trans_err = transform_error(wb1_pred, wb2_pred, trans)  # keep batch dim
    return trans_err


@tf.function
def mean_angular_error_plus_mae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mean_ang_err = angular_error(y_true, y_pred)
    return mae + mean_ang_err  # keep batch dim


class MaeAndCos(LossFunctionWrapper):
    """mean absolute error + cosine distance"""

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='mae_cos_loss'):
        def mae_scc_loss(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            scc = cosine_similarity(y_true, y_pred)
            return tf.reduce_mean(mae + scc, axis=np.arange(1, len(mae.shape)))

        super(MaeAndCos, self).__init__(mae_scc_loss, name=name, reduction=reduction)


class AngularError(LossFunctionWrapper):
    """angular error loss"""

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='ang'):
        super(AngularError, self).__init__(angular_error, name=name, reduction=reduction)


class MeanAngularErrorTwoVectors(LossFunctionWrapper):
    """angular error loss"""

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='ang_two_vec'):
        super(MeanAngularErrorTwoVectors, self).__init__(mean_angular_error_two_vectors, name=name, reduction=reduction)


class AngularErrorPlusMae(LossFunctionWrapper):
    """angular error + mean absolute error"""

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='ang_mae'):
        super(AngularErrorPlusMae, self).__init__(mean_angular_error_plus_mae, name=name, reduction=reduction)


class MeanAbsoluteErrorTwoVectors(LossFunctionWrapper):
    """MAE where data include two WB vectors"""

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='mae_two_vec'):
        super(MeanAbsoluteErrorTwoVectors, self).__init__(mean_absolute_error_two_vectors, name=name,
                                                          reduction=reduction)


class CosineSimilarityTwoVectors(LossFunctionWrapper):
    """cosine similarity where data include two WB vectors"""

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='cos_two_vec'):
        super(CosineSimilarityTwoVectors, self).__init__(cosine_similarity_two_vectors, name=name, reduction=reduction)


class CosineDistanceTwoVectors(LossFunctionWrapper):
    """cosine distance [0, 1] where data include two WB vectors"""

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='cos_dist_two_vec'):
        super(CosineDistanceTwoVectors, self).__init__(mean_cos_distance_two_vectors, name=name, reduction=reduction)


class MeanAngularErrorWithTransform(LossFunctionWrapper):
    """Mean angular error where data include WB vectors and transforms as well"""

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='ang_trans'):
        super(MeanAngularErrorWithTransform, self).__init__(mean_angular_error_with_transform, name=name,
                                                            reduction=reduction)


class MeanAbsoluteErrorWithTransform(LossFunctionWrapper):
    """MAE where data include WB vectors and transforms as well"""

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='mae_trans'):
        super(MeanAbsoluteErrorWithTransform, self).__init__(mean_absolute_error_with_transform, name=name,
                                                             reduction=reduction)


class MeanAbsoluteErrorWithTransformWeight(LossFunctionWrapper):
    """MAE where data include WB vectors and transforms as well"""

    def __init__(self, trans_weight, reduction=losses_utils.ReductionV2.AUTO, name='mae_trans_wgt'):
        super(MeanAbsoluteErrorWithTransformWeight, self).__init__(
            get_mean_absolute_error_with_transform_weight(trans_weight), name=name, reduction=reduction)


class TransformError(LossFunctionWrapper):
    """transforms error between two WB vectors"""

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='trans_err'):
        super(TransformError, self).__init__(transform_error_from_outputs, name=name, reduction=reduction)


def get_loss_function(loss_name, trans_weight):
    if loss_name == 'mae':
        loss = losses.MeanAbsoluteError()
    elif loss_name == 'mse':
        loss = losses.MeanSquaredError()
    elif loss_name == 'cos':
        loss = losses.CosineSimilarity()
    elif loss_name == 'ang':
        loss = AngularError()
    elif loss_name == 'ang_mae':
        loss = AngularErrorPlusMae()
    elif loss_name == 'ang_two_vec':
        loss = MeanAngularErrorTwoVectors()
    elif loss_name == 'cos_two_vec':
        loss = CosineSimilarityTwoVectors()
    elif loss_name == 'cos_dist_two_vec':
        loss = CosineDistanceTwoVectors()
    elif loss_name == 'mae_two_vec':
        loss = MeanAbsoluteErrorTwoVectors()
    elif loss_name == 'ang_trans':
        loss = MeanAngularErrorWithTransform()
    elif loss_name == 'mae_trans':
        loss = MeanAbsoluteErrorWithTransform()
    elif loss_name == 'mae_trans_wgt':
        loss = MeanAbsoluteErrorWithTransformWeight(trans_weight)
    else:
        raise ValueError("Invalid loss: {}".format(loss_name))
    return loss


def get_metrics(model_type):
    if model_type in ['1in1out', '2in1out', '1in1out_trans']:
        metrics = [AngularError()]
    elif model_type in ['2in2out']:
        metrics = [MeanAbsoluteErrorTwoVectors(), MeanAngularErrorTwoVectors()]
    elif model_type in ['2in2out_trans', '3in2out_trans']:
        metrics = [MeanAbsoluteErrorTwoVectors(), MeanAngularErrorTwoVectors(), TransformError()]
    else:
        raise ValueError("Invalid model_type = {}".format(model_type))
    return metrics
