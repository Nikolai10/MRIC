# Copyright 2023 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://github.com/tensorflow/compression/blob/master/models/hific/helpers.py
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

import os
import urllib.request

_LPIPS_URL = "http://rail.eecs.berkeley.edu/models/lpips/net-lin_alex_v0.1.pb"


def ensure_lpips_weights_exist(weight_path_out):
    if os.path.isfile(weight_path_out):
        return
    print("Downloading LPIPS weights:", _LPIPS_URL, "->", weight_path_out)
    urllib.request.urlretrieve(_LPIPS_URL, weight_path_out)
    if not os.path.isfile(weight_path_out):
        raise ValueError(f"Failed to download LPIPS weights from {_LPIPS_URL} "
                         f"to {weight_path_out}. Please manually download!")
