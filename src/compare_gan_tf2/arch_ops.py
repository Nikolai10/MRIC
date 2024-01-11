# Copyright 2023 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://github.com/google/compare_gan/blob/master/compare_gan/architectures/arch_ops.py
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tensorflow_addons.layers import SpectralNormalization
import tensorflow as tf
from tensorflow.keras import layers


def conv2d(output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name="conv2d", use_sn=False, use_bias=True):
    """Performs 2D convolution of the input."""
    initializer = tf.keras.initializers.RandomNormal(stddev=stddev)
    if use_sn:
        # Apply spectral normalization
        conv = SpectralNormalization(layers.Conv2D(output_dim, (k_h, k_w), strides=(d_h, d_w),
                                                   padding="same", kernel_initializer=initializer,
                                                   use_bias=use_bias), name=name)
    else:
        conv = layers.Conv2D(output_dim, (k_h, k_w), strides=(d_h, d_w), padding="same",
                             kernel_initializer=initializer, use_bias=use_bias, name=name)

    return conv


def lrelu(inputs, leak=0.2, name="lrelu"):
    """Performs leaky-ReLU on the input."""
    return tf.maximum(inputs, leak * inputs, name=name)
