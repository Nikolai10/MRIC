# Copyright 2023 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://github.com/google/compare_gan/blob/master/compare_gan/gans/loss_lib.py
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

import tensorflow as tf
from .utils import call_with_accepted_args


def check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits):
    """Checks the shapes and ranks of logits and prediction tensors.

    Args:
      d_real: prediction for real points, values in [0, 1], shape [batch_size, 1].
      d_fake: prediction for fake points, values in [0, 1], shape [batch_size, 1].
      d_real_logits: logits for real points, shape [batch_size, 1].
      d_fake_logits: logits for fake points, shape [batch_size, 1].

    Raises:
      ValueError: if the ranks or shapes are mismatched.
    """

    def _check_pair(a, b):
        if a != b:
            raise ValueError("Shape mismatch: %s vs %s." % (a, b))
        if len(a) != 2 or len(b) != 2:
            raise ValueError("Rank: expected 2, got %s and %s" % (len(a), len(b)))

    if (d_real is not None) and (d_fake is not None):
        _check_pair(d_real.shape.as_list(), d_fake.shape.as_list())
    if (d_real_logits is not None) and (d_fake_logits is not None):
        _check_pair(d_real_logits.shape.as_list(), d_fake_logits.shape.as_list())
    if (d_real is not None) and (d_real_logits is not None):
        _check_pair(d_real.shape.as_list(), d_real_logits.shape.as_list())


def non_saturating(d_real_logits, d_fake_logits, d_real=None, d_fake=None):
    """Returns the discriminator and generator loss for Non-saturating loss.

    Note that for MRIC we preserve the batch dimension for g_loss.

    Args:
      d_real_logits: logits for real points, shape [batch_size, 1].
      d_fake_logits: logits for fake points, shape [batch_size, 1].
      d_real: ignored.
      d_fake: ignored.

    Returns:
      A tuple consisting of the discriminator loss, discriminator's loss on the
      real samples and fake samples, and the generator's loss.
    """
    with tf.name_scope("non_saturating_loss"):
        check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_real_logits, labels=tf.ones_like(d_real_logits),
            name="cross_entropy_d_real"))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits),
            name="cross_entropy_d_fake"))
        d_loss = d_loss_real + d_loss_fake
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake_logits, labels=tf.ones_like(d_fake_logits),
            name="cross_entropy_g"), axis=1)
        return d_loss, d_loss_real, d_loss_fake, g_loss


def get_losses(fn=non_saturating, **kwargs):
    """Returns the losses for the discriminator and generator."""
    return call_with_accepted_args(fn, **kwargs)
