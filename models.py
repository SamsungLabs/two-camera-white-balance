"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Abdelrahman Abdelhamed (a.abdelhamed@samsung.com; abdoukamel@gmail.com)

A tf.keras model for white balance estimation.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, \
    Layer
from tensorflow.keras import Model


class Exp(Layer):
    def call(self, inputs, **kwargs):
        return tf.exp(inputs)


class MultiplyBy(Layer):
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs, **kwargs):
        return tf.multiply(inputs, self.scale)


class AdjustOutput1(Layer):
    def call(self, inputs, **kwargs):
        inputs = K.concatenate(
            [inputs[..., 0][..., tf.newaxis],
             K.ones_like(inputs[..., 0])[..., tf.newaxis],
             inputs[..., 1][..., tf.newaxis]],
            axis=-1)
        return inputs


class AdjustOutput2(Layer):
    def call(self, inputs, **kwargs):
        inputs = K.concatenate(
            [inputs[..., 0][..., tf.newaxis],
             K.ones_like(inputs[..., 0])[..., tf.newaxis],
             inputs[..., 1][..., tf.newaxis],
             inputs[..., 2][..., tf.newaxis],
             K.ones_like(inputs[..., 0])[..., tf.newaxis],
             inputs[..., 3][..., tf.newaxis]],
            axis=-1)
        return inputs


class WbEstimator(Model):
    def __init__(self, n_conv_layers, n_filters, n_outputs, model_type, n_dense_layers=2, n_dense_units=9,
                 last_layer='exp'):
        super(WbEstimator, self).__init__()

        self.n_conv_layers = n_conv_layers
        self.n_filters = n_filters
        self.n_outputs = n_outputs
        self.model_type = model_type
        self.n_dense_layers = n_dense_layers
        self.n_dense_units = n_dense_units
        self.last_layer = last_layer
        self.layers_ = []

        # if self.model_type in ['2in2out', '2in2out_trans', '3in2out_trans']:
        #     self.n_outputs = 2 * self.n_outputs

        if self.model_type == '1in1out_trans':
            self.set_transform_to_wb_layers()
        else:
            for i in range(self.n_conv_layers):
                conv_ = Conv2D(n_filters, 3, padding='same', activation='relu')
                self.layers_.append(conv_)
                bn_ = BatchNormalization()
                self.layers_.append(bn_)
                pool_ = MaxPool2D()
                self.layers_.append(pool_)
            self.layers_.append(Flatten())
            self.set_last_layer()

        # adjust outputs
        if self.model_type in ['1in1out', '2in1out', '1in1out_trans']:
            self.layers_.append(AdjustOutput1())
        elif self.model_type in ['2in2out', '2in2out_trans', '3in2out_trans']:
            self.layers_.append(AdjustOutput2())
        else:
            raise ValueError("Invalid model_type = {}".format(self.model_type))

    def call(self, x, **kwargs):
        if self.model_type in ['2in2out_trans', '3in2out_trans']:
            trans = x['trans']
            trans = tf.reshape(trans, (-1, trans.shape[-2] * trans.shape[-1]))  # flatten last 2 dims
            x = x['hists']
        # if self.model_type == '1in1out_trans':
        #     x = trans

        res = x
        for i in range(len(self.layers_)):
            res = self.layers_[i](res)
            # concatenate transform (model type: 3in2out_trans)
            if self.model_type == '3in2out_trans' and self.layers_[i].name == 'flatten':
                res = tf.concat([res, trans], axis=-1)

        if self.model_type in ['2in2out_trans', '3in2out_trans']:
            res = tf.concat([res, trans], axis=-1)

        return res

    def set_transform_to_wb_layers(self):
        self.layers_.append(Flatten())
        for i in range(self.n_dense_layers):
            self.layers_.append(Dense(self.n_dense_units, activation='relu'))
        self.set_last_layer()

    def set_last_layer(self):
        if self.last_layer == 'relu':
            self.layers_.append(Dense(self.n_outputs, activation='relu'))
        elif self.last_layer == 'sigmoid':
            self.layers_.append(Dense(self.n_outputs, activation='sigmoid'))
            self.layers_.append(MultiplyBy(3.0))
        elif self.last_layer == 'exp':
            self.layers_.append(Dense(self.n_outputs))
            self.layers_.append(Exp())


def set_network_params(args):
    if args.arch_id == 3:
        args.n_dense_layers = 2
        args.n_dense_units = 9
    elif args.arch_id == 4:
        args.n_dense_layers = 5
        args.n_dense_units = 9
    elif args.arch_id == 5:
        args.n_dense_layers = 16
        args.n_dense_units = 9
    elif args.arch_id == 6:
        args.n_dense_layers = 32
        args.n_dense_units = 9
    elif args.arch_id == 7:
        args.n_dense_layers = 64
        args.n_dense_units = 9
    elif args.arch_id == 8:
        args.n_dense_layers = 128
        args.n_dense_units = 9
    else:
        raise ValueError('Unknown architecture ID: args.arch_id = {}'.format(args.arch_id))
    return args
