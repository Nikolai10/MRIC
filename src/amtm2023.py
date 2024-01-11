# Copyright 2023 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://github.com/tensorflow/compression/blob/master/models/ms2020.py,
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

"""Nonlinear transform coder with hyperprior for RGB images.

This is a TensorFlow implementation of MRIC published in:
E. Agustsson and D. Minnen and G. Toderici and F. Mentzer:
"Multi-Realism Image Compression with a Conditional Generator"
Computer Vision and Pattern Recognition (CVPR), 2023
https://arxiv.org/pdf/2212.13824v2.pdf

This work is based on the image compression model published in:
D. Minnen and S. Singh:
"Channel-wise autoregressive entropy models for learned image compression"
Int. Conf. on Image Compression (ICIP), 2020
https://arxiv.org/abs/2007.08739

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.

This script requires TFC v2 (`pip install tensorflow-compression==2.*`).
"""

import sys
import os

# support both local environment + google colab (update as required)
sys.path.append('src')
sys.path.append('/content/MRIC/src')

import argparse
import functools
import glob
from absl import app
from absl.flags import argparse_flags
import tensorflow as tf

import tensorflow_compression as tfc
import tensorflow_datasets as tfds
import collections

from hific_tf2.model import compute_perceptual_loss, Nodes, pad
from hific_tf2.helpers import ensure_lpips_weights_exist
from elic import ElicAnalysis
from synthesis import build_elic_synthesis
from fourier_cond import build_global_conditioning
from hific_tf2.archs import PatchDiscriminator, DiscOutSplit

from config import ConfigMRIC as cfg_mric

from compare_gan_tf2 import loss_lib

# How many dataset preprocessing processes to use.
DATASET_NUM_PARALLEL = 8

# How many batches to prefetch.
DATASET_PREFETCH_BUFFER = 20

TFDSArguments = collections.namedtuple(
    "TFDSArguments", ["dataset_name", "features_key", "downloads_dir"])


def read_png(filename):
    """Loads a PNG image file."""
    string = tf.io.read_file(filename)
    return tf.image.decode_image(string, channels=3)


def write_png(filename, image):
    """Saves an image to a PNG file."""
    string = tf.image.encode_png(image)
    tf.io.write_file(filename, string)


class HyperAnalysisTransform(tf.keras.Sequential):
    """The analysis transform for the entropy model parameters."""

    def __init__(self, hyperprior_depth):
        super().__init__()
        conv = functools.partial(tfc.SignalConv2D, corr=True, padding="same_zeros")

        # See Appendix C.2 for more information on using a small hyperprior.
        layers = [
            conv(320, (3, 3), name="layer_0", strides_down=1, use_bias=True,
                 activation=tf.nn.relu),
            conv(256, (5, 5), name="layer_1", strides_down=2, use_bias=True,
                 activation=tf.nn.relu),
            conv(hyperprior_depth, (5, 5), name="layer_2", strides_down=2,
                 use_bias=False, activation=None),
        ]
        for layer in layers:
            self.add(layer)


class HyperSynthesisTransform(tf.keras.Sequential):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self):
        super().__init__()
        conv = functools.partial(
            tfc.SignalConv2D, corr=False, padding="same_zeros", use_bias=True,
            kernel_parameter="variable", activation=tf.nn.relu)

        # Note that the output tensor is still latent (it represents means and
        # scales but it does NOT hold mean or scale values explicitly). Therefore,
        # the final activation is ReLU rather than None or Exp). For the same
        # reason, it is not a requirement that the final depth of this transform
        # matches the depth of `y`.
        layers = [
            conv(192, (5, 5), name="layer_0", strides_up=2),
            conv(256, (5, 5), name="layer_1", strides_up=2),
            conv(640, (3, 3), name="layer_2", strides_up=1),
        ]
        for layer in layers:
            self.add(layer)


class SliceTransform(tf.keras.layers.Layer):
    """Transform for channel-conditional params and latent residual prediction."""

    def __init__(self, latent_depth, num_slices):
        super().__init__()
        conv = functools.partial(
            tfc.SignalConv2D, corr=False, strides_up=1, padding="same_zeros",
            use_bias=True, kernel_parameter="variable")

        # Note that the number of channels in the output tensor must match the
        # size of the corresponding slice. If we have 10 slices and a bottleneck
        # with 320 channels, the output is 320 / 10 = 32 channels.
        slice_depth = latent_depth // num_slices
        if slice_depth * num_slices != latent_depth:
            raise ValueError("Slices do not evenly divide latent depth (%d / %d)" % (
                latent_depth, num_slices))

        self.transform = tf.keras.Sequential([
            conv(224, (5, 5), name="layer_0", activation=tf.nn.relu),
            conv(128, (5, 5), name="layer_1", activation=tf.nn.relu),
            conv(slice_depth, (3, 3), name="layer_2", activation=None),
        ])

    def call(self, tensor):
        return self.transform(tensor)


class AMTM2023(tf.keras.Model):
    """Main model class."""

    def __init__(self, warm_up, lmbda, latent_depth, hyperprior_depth,
                 num_slices, max_support_slices,
                 num_scales, scale_min, scale_max):
        super().__init__()
        self.beta = tf.Variable(0.0, trainable=False)  # default PSNR-fav.
        self.warm_up = warm_up
        self.lmbda = lmbda
        self.num_scales = num_scales
        self.num_slices = num_slices
        self.slice_size = latent_depth // self.num_slices
        self.max_support_slices = max_support_slices
        offset = tf.math.log(scale_min)
        factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
                num_scales - 1.)
        self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
        self.global_cond = build_global_conditioning()
        self.analysis_transform = ElicAnalysis(channels=[256, 256, 256, 320])
        self.synthesis_transform = build_elic_synthesis(channels=[256, 256, 256, 3])
        self.hyper_analysis_transform = HyperAnalysisTransform(hyperprior_depth)
        self.hyper_synthesis_mean_scale_transform = HyperSynthesisTransform()
        self.cc_mean_transforms = [
            SliceTransform(latent_depth, num_slices) for _ in range(num_slices)]
        self.cc_scale_transforms = [
            SliceTransform(latent_depth, num_slices) for _ in range(num_slices)]
        self.lrp_transforms = [
            SliceTransform(latent_depth, num_slices) for _ in range(num_slices)]
        self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=[hyperprior_depth])

        self.disc = None
        if not warm_up:
            self.disc = PatchDiscriminator(name='disc')
            # Explicitly call the discriminator with some dummy data to create its weights
            _ = self.disc((tf.random.uniform((1, 256, 256, 3)), tf.random.uniform((1, 16, 16, 320))))

        self.gan_loss_function = loss_lib.non_saturating
        self.build([(None, None, None, 3), (None, 1)])
        # The call signature of decompress() depends on the number of slices, so we
        # need to compile the function dynamically.
        self.decompress = tf.function(
            input_signature=3 * [tf.TensorSpec(shape=(2,), dtype=tf.int32)] +
                            (num_slices + 1) * [tf.TensorSpec(shape=(1,), dtype=tf.string)]
        )(self.decompress)

    def call(self, x, training):
        """Computes rate and distortion losses."""

        x, betas = x
        x = tf.cast(x, self.compute_dtype)  # TODO(jonycgn): Why is this necessary?

        # obtain fourier features
        fourier_features_mlp = self.global_cond(betas)

        # Build the encoder (analysis) half of the hierarchical autoencoder.
        y = self.analysis_transform(x)
        y_shape = tf.shape(y)[1:-1]

        z = self.hyper_analysis_transform(y)

        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[1:-1]), tf.float32)

        # Build the entropy model for the hyperprior (z).
        em_z = tfc.ContinuousBatchedEntropyModel(
            self.hyperprior, coding_rank=3, compression=False,
            offset_heuristic=False)

        # When training, z_bpp is based on the noisy version of z (z_tilde).
        _, z_bits = em_z(z, training=training)
        z_bpp = tf.reduce_mean(z_bits) / num_pixels

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_hat = em_z.quantize(z)

        # Build the decoder (synthesis) half of the hierarchical autoencoder.
        latent_means_scales = self.hyper_synthesis_mean_scale_transform(z_hat)
        latent_means, latent_scales = tf.split(latent_means_scales, 2, axis=-1)

        # Build a conditional entropy model for the slices.
        em_y = tfc.LocationScaleIndexedEntropyModel(
            tfc.NoisyNormal, num_scales=self.num_scales, scale_fn=self.scale_fn,
            coding_rank=3, compression=False)

        # En/Decode each slice conditioned on hyperprior and previous slices.
        y_slices = tf.split(y, self.num_slices, axis=-1)
        y_hat_slices = []
        y_bpps = []
        y_bits_arr = []

        for slice_index, y_slice in enumerate(y_slices):
            # Model may condition on only a subset of previous slices.
            support_slices = (y_hat_slices if self.max_support_slices < 0 else
                              y_hat_slices[:self.max_support_slices])

            start_index = slice_index * self.slice_size
            end_index = slice_index * self.slice_size + self.slice_size
            latent_means_slice = latent_means[:, :, :, start_index:end_index]
            latent_scales_slice = latent_scales[:, :, :, start_index:end_index]

            # Predict mu and sigma for the current slice.
            mean_support = tf.concat([latent_means_slice] + support_slices, axis=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :y_shape[0], :y_shape[1], :]

            # Note that in this implementation, `sigma` represents scale indices,
            # not actual scale values.
            scale_support = tf.concat([latent_scales_slice] + support_slices, axis=-1)
            sigma = self.cc_scale_transforms[slice_index](scale_support)
            sigma = sigma[:, :y_shape[0], :y_shape[1], :]

            _, slice_bits = em_y(y_slice, sigma, loc=mu, training=training)
            slice_bpp = tf.reduce_mean(slice_bits) / num_pixels
            y_bpps.append(slice_bpp)

            y_bits_mean = tf.reduce_mean(slice_bits)
            y_bits_arr.append(y_bits_mean)

            # For the synthesis transform, use rounding. Note that quantize()
            # overrides the gradient to create a straight-through estimator.
            y_hat_slice = em_y.quantize(y_slice, loc=mu)

            # Add latent residual prediction (LRP).
            lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * tf.math.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        # Merge slices and generate the image reconstruction.
        y_hat = tf.concat(y_hat_slices, axis=-1)

        x_hat = self.synthesis_transform((y_hat, fourier_features_mlp))

        # Total bpp is sum of bpp from hyperprior and all slices.
        total_bpp = tf.add_n(y_bpps + [z_bpp])

        # Mean squared error across pixels per batch.
        # Don't clip or round pixel values while training.
        distortion_loss = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
        weighted_distortion_loss = cfg_mric.mse_weight * distortion_loss
        weighted_distortion_loss = tf.cast(weighted_distortion_loss, total_bpp.dtype)

        # LPIPS loss per image (will be reduced here)
        perceptual_loss = compute_perceptual_loss(x, x_hat, cfg_mric.lpips_path)
        weighted_perceptual_loss = tf.reduce_mean(betas * cfg_mric.lpips_weight * perceptual_loss)
        weighted_perceptual_loss = tf.cast(weighted_perceptual_loss, total_bpp.dtype)

        # rate loss per batch
        weighted_rate_loss = self.lmbda * cfg_mric.lmbda_weight * total_bpp

        # Calculate and return the rate-distortion loss
        loss = weighted_distortion_loss + weighted_perceptual_loss + weighted_rate_loss

        # we make use of the convenient wrapper structure from HiFiC
        input_image = x
        input_image_scaled = x / 255.
        reconstruction = x_hat
        reconstruction_scaled = reconstruction / 255.
        latent_quantized = y_hat

        nodes = Nodes(input_image, input_image_scaled, reconstruction, reconstruction_scaled, latent_quantized)
        return nodes, loss, total_bpp, distortion_loss, weighted_distortion_loss, weighted_perceptual_loss, weighted_rate_loss

    def compute_discriminator_out(self,
                                  nodes: Nodes,
                                  gradients_to_generator=True
                                  ) -> DiscOutSplit:
        """Get discriminator outputs."""
        with tf.name_scope("disc"):
            input_image = nodes.input_image_scaled
            reconstruction = nodes.reconstruction_scaled

            if not gradients_to_generator:
                reconstruction = tf.stop_gradient(reconstruction)

            discriminator_in = tf.concat([input_image, reconstruction], axis=0)

            # Condition D.
            latent = tf.stop_gradient(nodes.latent_quantized)
            latent = tf.concat([latent, latent], axis=0)

            discriminator_in = (discriminator_in, latent)

            disc_out_all = self.disc(discriminator_in)

        d_real, d_fake = tf.split(disc_out_all.d_all, 2)
        d_real_logits, d_fake_logits = tf.split(disc_out_all.d_all_logits, 2)
        disc_out_split = DiscOutSplit(d_real, d_fake,
                                      d_real_logits, d_fake_logits)

        return disc_out_split

    def create_gan_loss(self,
                        d_out: DiscOutSplit,
                        mode="g_loss"):
        """Create GAN loss using compare_gan."""
        if mode not in ("g_loss", "d_loss"):
            raise ValueError("Invalid mode: {}".format(mode))
        assert self.gan_loss_function is not None

        # Called within either train_disc or train_gen namescope.
        with tf.name_scope("gan_loss"):
            d_loss, _, _, g_loss = loss_lib.get_losses(
                # Note: some fn's need other args.
                fn=self.gan_loss_function,
                d_real=d_out.d_real,
                d_fake=d_out.d_fake,
                d_real_logits=d_out.d_real_logits,
                d_fake_logits=d_out.d_fake_logits)
            loss = d_loss if mode == "d_loss" else g_loss

        return loss

    def prepare_data(self, x):
        """split images into two batches for G and D."""
        x1, x2 = tf.split(x, 2)
        return x1, x2

    def train_step(self, x):

        bs = tf.shape(x)[0]
        betas = tf.random.uniform(minval=0, maxval=5.12, shape=(bs,), dtype=tf.float32)

        betas1, betas2 = tf.split(betas, 2)
        inputs, inputs_d_steps = self.prepare_data(x)

        # stage 2
        if not self.warm_up:
            # first disc update
            with tf.GradientTape() as disc_tape:
                nodes_disc, _, _, _, _, _, _ = self([inputs_d_steps, betas1], training=True)
                d_out = self.compute_discriminator_out(nodes_disc, gradients_to_generator=False)
                d_loss = self.create_gan_loss(d_out, mode="d_loss")

            variables = self.disc.trainable_variables
            gradients = disc_tape.gradient(d_loss, variables)
            self.d_optimizer.apply_gradients(zip(gradients, variables))

        # then generator update
        with tf.GradientTape() as gen_tape:
            nodes, loss, bpp, mse, w_mse, w_lpips, w_rate_loss = self([inputs, betas2], training=True)

            # stage 2
            if not self.warm_up:
                d_outs = self.compute_discriminator_out(nodes, gradients_to_generator=True)
                g_loss = self.create_gan_loss(d_outs, mode="g_loss")
                w_g_loss = tf.reduce_mean(betas2 * g_loss)
                loss += w_g_loss

        variables = [var for var in self.trainable_variables if not var.name.startswith('disc')]
        gradients = gen_tape.gradient(loss, variables)
        self.g_optimizer.apply_gradients(zip(gradients, variables))

        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.mse.update_state(mse)
        self.weighted_mse.update_state(w_mse)
        self.weighted_lpips.update_state(w_lpips)
        self.weighted_rate.update_state(w_rate_loss)
        self.used_betas.update_state(tf.reduce_mean(betas))

        # stage 1
        if self.warm_up:
            return {m.name: m.result() for m in
                    [self.loss, self.bpp, self.mse, self.weighted_mse, self.weighted_lpips,
                     self.weighted_rate, self.used_betas]}

        # stage 2
        self.d_loss.update_state(d_loss)
        self.g_loss.update_state(tf.reduce_mean(g_loss))
        self.weighted_g_loss.update_state(w_g_loss)

        return {m.name: m.result() for m in
                [self.loss, self.bpp, self.mse, self.weighted_mse, self.weighted_lpips, self.weighted_rate, self.d_loss,
                 self.g_loss, self.weighted_g_loss, self.used_betas]}

    def test_step(self, x):

        bs = tf.shape(x)[0]
        betas = tf.random.uniform(minval=0, maxval=5.12, shape=(bs,), dtype=tf.float32)
        x = tf.image.random_crop(value=x, size=(tf.shape(x)[0], 256, 256, 3))

        _, loss, bpp, mse, w_mse, w_lpips, w_rate = self([x, betas], training=False)
        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.mse.update_state(mse)
        self.weighted_mse.update_state(w_mse)
        self.weighted_lpips.update_state(w_lpips)
        self.weighted_rate.update_state(w_rate)
        self.used_betas.update_state(betas[0])
        return {m.name: m.result() for m in
                [self.loss, self.bpp, self.mse, self.weighted_mse, self.weighted_lpips,
                 self.weighted_rate, self.used_betas]}

    def predict_step(self, x):
        raise NotImplementedError("Prediction API is not supported.")

    def compile(self, d_optimizer, g_optimizer, **kwargs):
        super().compile(
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            **kwargs,
        )
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss = tf.keras.metrics.Mean(name="loss")
        self.bpp = tf.keras.metrics.Mean(name="bpp")
        self.mse = tf.keras.metrics.Mean(name="mse")
        self.weighted_mse = tf.keras.metrics.Mean(name="weighted_mse")
        self.weighted_lpips = tf.keras.metrics.Mean(name="weighted_lpips")
        self.weighted_rate = tf.keras.metrics.Mean(name="weighted_rate")
        self.d_loss = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss = tf.keras.metrics.Mean(name="g_loss")
        self.weighted_g_loss = tf.keras.metrics.Mean(name="weighted_g_loss")
        self.used_betas = tf.keras.metrics.Mean(name="used_betas")

    def fit(self, *args, **kwargs):
        retval = super().fit(*args, **kwargs)
        # After training, fix range coding tables.
        self.em_z = tfc.ContinuousBatchedEntropyModel(
            self.hyperprior, coding_rank=3, compression=True,
            offset_heuristic=False)
        self.em_y = tfc.LocationScaleIndexedEntropyModel(
            tfc.NoisyNormal, num_scales=self.num_scales, scale_fn=self.scale_fn,
            coding_rank=3, compression=True)
        return retval

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
    ])
    def compress(self, x):
        """Compresses an image."""

        # Add batch dimension and cast to float.
        x = tf.expand_dims(x, 0)
        x = tf.cast(x, dtype=self.compute_dtype)

        y_strings = []

        x_shape = tf.shape(x)[1:-1]

        # Build the encoder (analysis) half of the hierarchical autoencoder.
        y = self.analysis_transform(x)
        y_shape = tf.shape(y)[1:-1]

        z = self.hyper_analysis_transform(y)
        z_shape = tf.shape(z)[1:-1]

        z_string = self.em_z.compress(z)
        z_hat = self.em_z.decompress(z_string, z_shape)

        # Build the decoder (synthesis) half of the hierarchical autoencoder.
        latent_means_scales = self.hyper_synthesis_mean_scale_transform(z_hat)
        latent_means, latent_scales = tf.split(latent_means_scales, 2, axis=-1)

        # En/Decode each slice conditioned on hyperprior and previous slices.
        y_slices = tf.split(y, self.num_slices, axis=-1)
        y_hat_slices = []
        for slice_index, y_slice in enumerate(y_slices):
            # Model may condition on only a subset of previous slices.
            support_slices = (y_hat_slices if self.max_support_slices < 0 else
                              y_hat_slices[:self.max_support_slices])

            start_index = slice_index * self.slice_size
            end_index = slice_index * self.slice_size + self.slice_size
            latent_means_slice = latent_means[:, :, :, start_index:end_index]
            latent_scales_slice = latent_scales[:, :, :, start_index:end_index]

            # Predict mu and sigma for the current slice.
            mean_support = tf.concat([latent_means_slice] + support_slices, axis=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :y_shape[0], :y_shape[1], :]

            # Note that in this implementation, `sigma` represents scale indices,
            # not actual scale values.
            scale_support = tf.concat([latent_scales_slice] + support_slices, axis=-1)
            sigma = self.cc_scale_transforms[slice_index](scale_support)
            sigma = sigma[:, :y_shape[0], :y_shape[1], :]

            slice_string = self.em_y.compress(y_slice, sigma, mu)
            y_strings.append(slice_string)
            y_hat_slice = self.em_y.decompress(slice_string, sigma, mu)

            # Add latent residual prediction (LRP).
            lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * tf.math.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        return (x_shape, y_shape, z_shape, z_string) + tuple(y_strings)

    def decompress(self, x_shape, y_shape, z_shape, z_string, *y_strings):
        """Decompresses an image."""
        assert len(y_strings) == self.num_slices

        z_hat = self.em_z.decompress(z_string, z_shape)
        _, h, w, c = z_hat.shape

        # Build the decoder (synthesis) half of the hierarchical autoencoder.
        latent_means_scales = self.hyper_synthesis_mean_scale_transform(z_hat)
        latent_means, latent_scales = tf.split(latent_means_scales, 2, axis=-1)

        # En/Decode each slice conditioned on hyperprior and previous slices.
        y_hat_slices = []
        for slice_index, y_string in enumerate(y_strings):
            # Model may condition on only a subset of previous slices.
            support_slices = (y_hat_slices if self.max_support_slices < 0 else
                              y_hat_slices[:self.max_support_slices])

            start_index = slice_index * self.slice_size
            end_index = slice_index * self.slice_size + self.slice_size
            latent_means_slice = latent_means[:, :, :, start_index:end_index]
            latent_scales_slice = latent_scales[:, :, :, start_index:end_index]

            # Predict mu and sigma for the current slice.
            mean_support = tf.concat([latent_means_slice] + support_slices, axis=-1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :y_shape[0], :y_shape[1], :]

            # Note that in this implementation, `sigma` represents scale indices,
            # not actual scale values.
            scale_support = tf.concat([latent_scales_slice] + support_slices, axis=-1)
            sigma = self.cc_scale_transforms[slice_index](scale_support)
            sigma = sigma[:, :y_shape[0], :y_shape[1], :]

            y_hat_slice = self.em_y.decompress(y_string, sigma, loc=mu)

            # Add latent residual prediction (LRP).
            lrp_support = tf.concat([mean_support, y_hat_slice], axis=-1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * tf.math.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        # Merge slices and generate the image reconstruction.
        y_hat = tf.concat(y_hat_slices, axis=-1)

        bs = tf.shape(y_hat)[0]

        betas = tf.repeat(self.beta, repeats=bs)
        fourier_features_mlp = self.global_cond(betas)

        x_hat = self.synthesis_transform((y_hat, fourier_features_mlp))

        # Remove batch dimension, and crop away any extraneous padding.
        x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
        # Then cast back to 8-bit integer.
        return tf.saturate_cast(tf.round(x_hat), tf.uint8)


def check_image_size(image, patchsize):
    shape = tf.shape(image)
    return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def crop_image(image, patchsize):
    image = tf.image.random_crop(image, (patchsize, patchsize, 3))
    return tf.cast(image, tf.keras.mixed_precision.global_policy().compute_dtype)


def get_dataset(name, split, args):
    """Creates input data pipeline from a TF Datasets dataset."""
    with tf.device("/cpu:0"):
        dataset = tfds.load(name, split=split, shuffle_files=True)
        if split == "train":
            dataset = dataset.repeat()
        dataset = dataset.filter(
            lambda x: check_image_size(x["image"], args.patchsize))
        dataset = dataset.map(
            lambda x: crop_image(x["image"], args.patchsize))
        dataset = dataset.batch(args.batchsize, drop_remainder=True)
    return dataset


def get_custom_dataset(split, args):
    """Creates input data pipeline from custom PNG images."""
    with tf.device("/cpu:0"):
        files = glob.glob(args.train_glob)
        if not files:
            raise RuntimeError(f"No training images found with glob "
                               f"'{args.train_glob}'.")
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
        if split == "train":
            dataset = dataset.repeat()
        dataset = dataset.map(
            lambda x: crop_image(read_png(x), args.patchsize),
            num_parallel_calls=args.preprocess_threads)
        dataset = dataset.batch(args.batchsize, drop_remainder=True)
    return dataset


def train(args):
    """Instantiates and trains the model."""
    if args.precision_policy:
        tf.keras.mixed_precision.set_global_policy(args.precision_policy)
    if args.check_numerics:
        tf.debugging.enable_check_numerics()

    # download LPIPS weights if not exists
    ensure_lpips_weights_exist(cfg_mric.lpips_path)

    model = AMTM2023(
        args.warm_up, args.lmbda, args.latent_depth, args.hyperprior_depth, args.num_slices,
        args.max_support_slices, args.num_scales, args.scale_min, args.scale_max)

    total_steps = int(args.epochs * args.steps_per_epoch)
    switch_step = int(0.85 * total_steps)

    # if warm-up, decay lr for last 0.15 steps by factor 10
    if args.warm_up:
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[switch_step],
            values=[args.lr, args.lr / 10]
        )
    # use constant lr
    else:
        lr_schedule = args.lr

    model.compile(
        d_optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        g_optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, global_clipnorm=1.0))  # only on G

    if args.train_glob:
        train_dataset = get_custom_dataset("train", args)
        validation_dataset = get_custom_dataset("validation", args)
    else:
        train_dataset = get_dataset("clic", "train", args)
        validation_dataset = get_dataset("clic", "validation", args)
        validation_dataset = validation_dataset.take(args.max_validation_steps)

    model.fit(
        train_dataset.prefetch(args.batchsize),
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=validation_dataset.cache(),
        validation_freq=1,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.TensorBoard(
                log_dir=args.train_path,
                histogram_freq=1, update_freq="epoch"),
            tf.keras.callbacks.BackupAndRestore(args.train_path, delete_checkpoint=False),
        ],
        verbose=int(args.verbose),
    )
    model.save(args.model_path)


def compress(args):
    """Compresses an image."""
    # Load model and use it to compress the image.
    model = tf.keras.models.load_model(args.model_path)

    # assign beta [0, 2.56] - only required if args.verbose == True (!)
    # beta is meant to change the decoder logic only.
    model.beta.assign(args.beta)
    print(model.summary())

    x = read_png(args.input_file)
    x_padded = pad(x, factor=256)
    tensors = model.compress(x_padded)

    # Write a binary file with the shape information and the compressed string.
    packed = tfc.PackedTensors()
    packed.pack(tensors)
    with open(args.output_file, "wb") as f:
        f.write(packed.string)

    # If requested, decompress the image and measure performance.
    if args.verbose:
        x_hat = model.decompress(*tensors)

        # undo padding
        height, width = tf.shape(x)[0], tf.shape(x)[1]
        x_hat = x_hat[:height, :width, :]

        # Cast to float in order to compute metrics.
        x = tf.cast(x, tf.float32)
        x_hat = tf.cast(x_hat, tf.float32)
        mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
        psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
        msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
        msssim_db = -10. * tf.math.log(1 - msssim) / tf.math.log(10.)

        # The actual bits per pixel including entropy coding overhead.
        num_pixels = tf.reduce_prod(tf.shape(x_padded)[:-1])
        bpp = len(packed.string) * 8 / num_pixels

        print(f"Mean squared error: {mse:0.4f}")
        print(f"PSNR (dB): {psnr:0.2f}")
        print(f"Multiscale SSIM: {msssim:0.4f}")
        print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
        print(f"Bits per pixel: {bpp:0.4f}")

        pathname, _ = os.path.splitext(args.output_file)
        write_png(pathname + '_hat.png', tf.cast(x_hat, tf.uint8))


def decompress(args):
    """Decompresses an image."""
    # Load the model and determine the dtypes of tensors required to decompress.
    model = tf.keras.models.load_model(args.model_path)
    dtypes = [t.dtype for t in model.decompress.input_signature]

    # assign beta [0, 2.56]
    model.beta.assign(args.beta)

    # Read the shape information and compressed string from the binary file,
    # and decompress the image using the model.
    with open(args.input_file, "rb") as f:
        packed = tfc.PackedTensors(f.read())
    tensors = packed.unpack(dtypes)
    x_hat = model.decompress(*tensors)

    # Write reconstructed image out as a PNG file.
    write_png(args.output_file, x_hat)


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Report progress and metrics when training or compressing.")
    parser.add_argument(
        "--model_path", default="res/amtm2023",
        help="Path where to save/load the trained model.")
    parser.add_argument(
        "--beta", type=float, default=0.0,
        help="Beta conditioning as described in MRIC (Agustsson et al. CVPR 2023)")
    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="What to do: 'train' loads training data and trains (or continues "
             "to train) a new model. 'compress' reads an image file (lossless "
             "PNG format) and writes a compressed binary file. 'decompress' "
             "reads a binary file and reconstructs the image (in PNG format). "
             "input and output filenames need to be provided for the latter "
             "two options. Invoke '<command> -h' for more information.")

    # 'train' subcommand.
    train_cmd = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains (or continues to train) a new model. Note that this "
                    "model trains on a continuous stream of patches drawn from "
                    "the training image dataset. An epoch is always defined as "
                    "the same number of batches given by --steps_per_epoch. "
                    "The purpose of validation is mostly to evaluate the "
                    "rate-distortion performance of the model using actual "
                    "quantization rather than the differentiable proxy loss. "
                    "Note that when using custom training images, the validation "
                    "set is simply a random sampling of patches from the "
                    "training set.")
    train_cmd.add_argument(
        "--warm_up", type=int, default=0,
        help="{1: stage 1 (full learning objective w/o GAN), 0: stage 2 (full learning objective)}")
    train_cmd.add_argument(
        "--lambda", type=float, default=0.01, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    train_cmd.add_argument(
        "--lr", type=float, default=1e-4, dest="lr",
        help="Learning rate.")
    train_cmd.add_argument(
        "--train_glob", type=str, default=None,
        help="Glob pattern identifying custom training data. This pattern must "
             "expand to a list of RGB images in PNG format. If unspecified, the "
             "CLIC dataset from TensorFlow Datasets is used.")
    train_cmd.add_argument(
        "--num_filters", type=int, default=192,
        help="Number of filters per layer.")
    train_cmd.add_argument(
        "--latent_depth", type=int, default=320,
        help="Number of filters of the last layer of the analysis transform.")
    train_cmd.add_argument(
        "--hyperprior_depth", type=int, default=192,
        help="Number of filters of the last layer of the hyper-analysis "
             "transform.")
    train_cmd.add_argument(
        "--num_slices", type=int, default=10,
        help="Number of channel slices for conditional entropy modeling.")
    train_cmd.add_argument(
        "--max_support_slices", type=int, default=10,
        help="Maximum number of preceding slices to condition the current slice "
             "on. See Appendix C.1 of the paper for details.")
    train_cmd.add_argument(
        "--num_scales", type=int, default=64,
        help="Number of Gaussian scales to prepare range coding tables for.")
    train_cmd.add_argument(
        "--scale_min", type=float, default=.11,
        help="Minimum value of standard deviation of Gaussians.")
    train_cmd.add_argument(
        "--scale_max", type=float, default=256.,
        help="Maximum value of standard deviation of Gaussians.")
    train_cmd.add_argument(
        "--train_path", default="res/train_amtm2023",
        help="Path where to log training metrics for TensorBoard and back up "
             "intermediate model checkpoints.")
    train_cmd.add_argument(
        "--batchsize", type=int, default=16,
        help="Batch size for training and validation.")
    train_cmd.add_argument(
        "--patchsize", type=int, default=256,
        help="Size of image patches for training and validation.")
    train_cmd.add_argument(
        "--epochs", type=int, default=1000,
        help="Train up to this number of epochs. (One epoch is here defined as "
             "the number of steps given by --steps_per_epoch, not iterations "
             "over the full training dataset.)")
    train_cmd.add_argument(
        "--steps_per_epoch", type=int, default=1000,
        help="Perform validation and produce logs after this many batches.")
    train_cmd.add_argument(
        "--max_validation_steps", type=int, default=16,
        help="Maximum number of batches to use for validation. If -1, use one "
             "patch from each image in the training set.")
    train_cmd.add_argument(
        "--preprocess_threads", type=int, default=16,
        help="Number of CPU threads to use for parallel decoding of training "
             "images.")
    train_cmd.add_argument(
        "--precision_policy", type=str, default=None,
        help="Policy for `tf.keras.mixed_precision` training.")
    train_cmd.add_argument(
        "--check_numerics", action="store_true",
        help="Enable TF support for catching NaN and Inf in tensors.")

    # 'compress' subcommand.
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a PNG file, compresses it, and writes a TFCI file.")

    # 'decompress' subcommand.
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a TFCI file, reconstructs the image, and writes back "
                    "a PNG file.")

    # Arguments for both 'compress' and 'decompress'.
    for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
        cmd.add_argument(
            "input_file",
            help="Input filename.")
        cmd.add_argument(
            "output_file", nargs="?",
            help=f"Output filename (optional). If not provided, appends '{ext}' to "
                 f"the input filename.")

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    # Invoke subcommand.
    if args.command == "train":
        train(args)
    elif args.command == "compress":
        if not args.output_file:
            args.output_file = args.input_file + ".tfci"
        compress(args)
    elif args.command == "decompress":
        if not args.output_file:
            args.output_file = args.input_file + ".png"
        decompress(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
