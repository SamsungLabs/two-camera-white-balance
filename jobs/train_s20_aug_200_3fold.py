"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Abdelrahman Abdelhamed (a.abdelhamed@samsung.com; abdoukamel@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import os


def train_s20_aug_200_3fold():
    command_str = 'python -m train ' \
                  '--experiment_name "s20_aug_200_cv0" ' \
                  '--dataset_dir "./data/Metadata_Image_Pairs_Augment_99" ' \
                  '--dataset_type real_trans_aug ' \
                  '--n_aug_real 100 ' \
                  '--arch_id 3 ' \
                  '--n_cross_val 3 ' \
                  '--cross_val_idx 0 ' \
                  '--loss mae ' \
                  '--trans_weight 1 ' \
                  '--n_epochs 1000000 ' \
                  '--batch_size 1024 ' \
                  '--n_ds_reps 1000 ' \
                  '--learn_rate 1e-4 '

    # 3-fold cross validation
    os.system(command_str)
    os.system(command_str.replace('aug_200_cv0', 'aug_200_cv1').replace('cross_val_idx 0', 'cross_val_idx 1'))
    os.system(command_str.replace('aug_200_cv0', 'aug_200_cv2').replace('cross_val_idx 0', 'cross_val_idx 2'))


if __name__ == '__main__':
    train_s20_aug_200_3fold()
