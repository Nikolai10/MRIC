# Copyright 2023 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://github.com/google-research/google-research/blob/master/vct/src/elic.py
# Copyright 2023 The Google Research Authors.
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
"""Building blocks from MRIC.

Blocks include the conditional synthesis transform presented in:
Multi-Realism Image Compression with a Conditional Generator
https://arxiv.org/abs/2212.13824
"""
import functools
import tensorflow as tf
from elic import build_conv, SimpleAttention


class ConditionalResidualBlock(tf.keras.layers.Layer):
    """Conditional Residual block based on (Cheng 2020) and ELIC.

    (Cheng 2020) = https://arxiv.org/abs/2001.01568

    Reference PyTorch code from CompressAI:
    https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/layers/layers.py#L208

    Reference TF code from Cheng:
    https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/network.py#L41

    See https://arxiv.org/pdf/2212.13824v2.pdf, Figure 5.
    Note we set use_bias = False for the projection layers, different to
    https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py#L49
    """

    def __init__(self, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self._activation = activation

    def build(self, input_shape):
        c = input_shape[0][-1]

        # Conv layers = [1x1 @ N/2, 3x3 @ N/2, 1x1 @ N].
        self._block1 = tf.keras.Sequential([
            build_conv(output_channels=c // 2, kernel_size=1, act=self._activation),
        ])
        self._proj1 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=c // 2, use_bias=False)
        ])

        self._block2 = tf.keras.Sequential([
            build_conv(output_channels=c // 2, kernel_size=3, act=self._activation),
        ])
        self._proj2 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=c // 2, use_bias=False)
        ])

        self._block3 = tf.keras.Sequential([
            build_conv(output_channels=c, kernel_size=1, act=None),
        ])
        self._proj3 = tf.keras.Sequential([
            tf.keras.layers.Dense(units=c, use_bias=False)
        ])

    def call(self, x):
        x, fourier_features_mlp = x

        x1 = self._block1(x)
        proj1 = self._proj1(fourier_features_mlp)
        x1 += proj1[:, None, None, :]

        x2 = self._block2(x1)
        proj2 = self._proj2(fourier_features_mlp)
        x2 += proj2[:, None, None, :]

        x3 = self._block3(x2)
        proj3 = self._proj3(fourier_features_mlp)
        x3 += proj3[:, None, None, :]

        x += x3
        return x


def build_elic_synthesis(num_residual_blocks=3,
                         channels=(192, 160, 128, 3),
                         output_channels=None,
                         name="ElicSynthesis"):
    """Synthesis transform from MRIC."""
    if len(channels) != 4:
        raise ValueError(f"ELIC uses 4 conv layers (not {channels}).")
    if output_channels is not None and output_channels != channels[-1]:
        raise ValueError("output_channels specified but does not match channels: "
                         f"{output_channels} vs. {channels}")

    # Keep activation separate from conv layer for clarity and because
    # second conv is followed by attention, not an activation.
    conv = functools.partial(
        build_conv, kernel_size=5, strides=2, act=None, up_or_down="up")
    convs = [conv(output_channels=c) for c in channels]

    crb = functools.partial(ConditionalResidualBlock, activation="relu")

    def build_act():
        return [crb() for _ in range(num_residual_blocks)]

    blocks = [
        SimpleAttention(),
        convs[0],
        *build_act(),
        convs[1],
        SimpleAttention(),
        *build_act(),
        convs[2],
        *build_act(),
        convs[3],
    ]
    blocks = list(filter(None, blocks))  # remove None elements

    x = tf.keras.layers.Input(
        shape=(None, None, 320), name="image_input"
    )

    fourier_features_mlp = tf.keras.layers.Input(
        shape=512, name="fourier_features_mlp"
    )

    y = x
    for block in blocks:
        if isinstance(block, ConditionalResidualBlock):
            y = block((y, fourier_features_mlp))
        else:
            y = block(y)

    y = y * 255.

    model = tf.keras.Model(inputs=[x, fourier_features_mlp], outputs=y, name=name)
    model.summary()

    return model
