# Copyright 2023 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://github.com/tensorflow/compression/blob/master/models/hific/archs.py#L300
# Copyright 2020 Google LLC.
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

import tensorflow as tf
from compare_gan_tf2.arch_ops import conv2d, lrelu
import collections

# Output of discriminator, where real and fake are merged into single tensors.
DiscOutAll = collections.namedtuple(
    "DiscOutAll",
    ["d_all", "d_all_logits"])

# Split each tensor in a DiscOutAll into 2.
DiscOutSplit = collections.namedtuple(
    "DiscOutSplit",
    ["d_real", "d_fake",
     "d_real_logits", "d_fake_logits"])


class PatchDiscriminator(tf.keras.Model):
    """PatchDiscriminator architecture."""

    def __init__(self,
                 name,
                 num_filters_base=64,
                 num_layers=3,
                 ):
        """Instantiate discriminator.

        Args:
          name: Name of the layer.
          num_filters_base: Number of base filters. will be multiplied as we go down in resolution.
          num_layers: Number of downscaling convolutions.
        """

        super(PatchDiscriminator, self).__init__(name=name)
        self._num_layers = num_layers
        self._num_filters_base = num_filters_base
        self._spectral_norm = True
        self._conv_blocks = []

    def build(self, input_shape):
        self._latent_prep = conv2d(12, 3, 3, 1, 1, name="latent",
                                   use_sn=self._spectral_norm)
        k = 4
        self._conv_blocks.append(
            conv2d(self._num_filters_base, k, k, 2, 2,
                   name="d_conv_head", use_sn=self._spectral_norm)
        )

        num_filters = self._num_filters_base
        for i in range(self._num_layers - 1):
            num_filters = min(num_filters * 2, 512)
            self._conv_blocks.append(conv2d(num_filters, k, k, 2, 2,
                                            name=f"d_conv_{i}", use_sn=self._spectral_norm))

        num_filters = min(num_filters * 2, 512)
        self._conv_blocks.append(conv2d(num_filters, k, k, 1, 1,
                                        name="d_conv_a", use_sn=self._spectral_norm))

        self._conv_blocks.append(conv2d(1, k, k, 1, 1,
                                        name="d_conv_b", use_sn=self._spectral_norm))

    def call(self, inputs):
        if not isinstance(inputs, tuple) or len(inputs) != 2:
            raise ValueError("Expected 2-tuple, got {}".format(inputs))

        x, latent = inputs
        x_shape = tf.shape(x)

        # Upscale and fuse latent.
        latent = self._latent_prep(latent)
        latent = lrelu(latent, leak=0.2)
        latent = tf.image.resize(latent, [x_shape[1], x_shape[2]],
                                 tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = tf.concat([x, latent], axis=-1)

        # The discriminator:
        net = self._conv_blocks[0](x)
        net = lrelu(net, leak=0.2)

        for i in range(self._num_layers - 1):
            net = self._conv_blocks[i + 1](net)
            net = lrelu(net, leak=0.2)

        net = self._conv_blocks[-2](net)
        net = lrelu(net, leak=0.2)

        # Final 1x1 conv that maps to 1 Channel
        net = self._conv_blocks[-1](net)

        bs = tf.shape(net)[0]
        # Reshape all into batch dimension.
        out_logits = tf.reshape(net, [bs, -1])
        out = tf.nn.sigmoid(out_logits)

        return DiscOutAll(out, out_logits)
