# Copyright 2023 Nikolai Körber. All Rights Reserved.
#
# Fourier feature computation based on:
# https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py
# Copyright (c) 2020 bmild.
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

Fourier Conditioning is based on:
Nerf: Representing scenes as neural radiance fields for view synthesis
https://dl.acm.org/doi/abs/10.1145/3503250
"""
import tensorflow as tf


class BetaMlp(tf.keras.Model):
    def __init__(self, channels=512, act_layer=tf.keras.layers.ReLU):
        super(BetaMlp, self).__init__()
        self.fc1 = tf.keras.layers.Dense(channels)
        self.act = act_layer()
        self.fc2 = tf.keras.layers.Dense(channels)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        return x


def build_global_conditioning():
    beta = tf.keras.layers.Input(
        shape=(), name="beta"
    )

    embed_fn, input_ch = get_embedder(multires=10, i=0)
    beta_mlp = BetaMlp()
    fourier_features = embed_fn(beta)
    fourier_features_mlp = beta_mlp(fourier_features)

    model = tf.keras.Model(inputs=beta, outputs=fourier_features_mlp, name='global_cond')
    model.summary()

    return model


class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                        freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.transpose(tf.stack([fn(inputs) for fn in self.embed_fns]))


def get_embedder(multires, i=0):
    if i == -1:
        return tf.identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 1,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tf.math.sin, tf.math.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim
