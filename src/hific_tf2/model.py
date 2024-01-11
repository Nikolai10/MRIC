# Copyright 2023 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://github.com/tensorflow/compression/blob/master/models/hific/model.py
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

import collections
import tensorflow as tf
from .helpers import ensure_lpips_weights_exist

Nodes = collections.namedtuple(
    "Nodes",  # Expected ranges for RGB:
    ["input_image",  # [0, 255]
     "input_image_scaled",  # [0, 1]
     "reconstruction",  # [0, 255]
     "reconstruction_scaled",  # [0, 1]
     "latent_quantized"])  # Latent post-quantization.


def pad(input_image, factor=256):
    """Pad `input_image` such that H and W are divisible by `factor`."""
    with tf.name_scope("pad"):
        height, width = tf.shape(input_image)[0], tf.shape(input_image)[1]
        pad_height = (factor - (height % factor)) % factor
        pad_width = (factor - (width % factor)) % factor
        return tf.pad(input_image,
                      [[0, pad_height], [0, pad_width], [0, 0]], "REFLECT")


def compute_perceptual_loss(x, x_hat, lpips_path):
    # [0, 255] -> [-1, 1]
    x = (x - 127.5) / 127.5
    x_hat = (x_hat - 127.5) / 127.5

    # First the fake images, then the real! Otherwise no gradients.
    return LPIPSLoss(lpips_path)(x_hat, x)


class LPIPSLoss(object):
    """
    Calcualte LPIPS loss based on:
    https://github.com/tensorflow/compression/blob/master/models/hific/model.py

    Note that for MRIC we preserve that batch dimension for the final output.

    call: lpips_loss = LPIPSLoss(_lpips_weight_path)
    """

    def __init__(self, weight_path):
        ensure_lpips_weights_exist(weight_path)

        def wrap_frozen_graph(graph_def, inputs, outputs):
            def _imports_graph_def():
                tf.graph_util.import_graph_def(graph_def, name="")

            wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
            import_graph = wrapped_import.graph
            return wrapped_import.prune(
                tf.nest.map_structure(import_graph.as_graph_element, inputs),
                tf.nest.map_structure(import_graph.as_graph_element, outputs))

        # Pack LPIPS network into a tf function
        graph_def = tf.compat.v1.GraphDef()
        with open(weight_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        self._lpips_func = tf.function(
            wrap_frozen_graph(
                graph_def, inputs=("0:0", "1:0"), outputs="Reshape_10:0"))

    def __call__(self, fake_image, real_image):
        """
        Assuming inputs are in [-1, 1].

        :param fake_image:
        :param real_image:
        :return:
        """

        # Move inputs to NCHW format.
        def _transpose_to_nchw(x):
            return tf.transpose(x, (0, 3, 1, 2))

        fake_image = _transpose_to_nchw(fake_image)
        real_image = _transpose_to_nchw(real_image)
        loss = self._lpips_func(fake_image, real_image)
        return tf.reduce_mean(loss, axis=[1, 2, 3])  # Loss is N111
