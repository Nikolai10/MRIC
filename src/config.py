# Copyright 2023 Nikolai Körber. All Rights Reserved.
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

# see https://arxiv.org/pdf/2212.13824, Section 3.2
class ConfigMRIC:
    mse_weight = 1 / 100
    lpips_weight = 4.26
    lmbda_weight = 100
    lpips_path = '/content/MRIC/res/data/model/lpips_weights'
