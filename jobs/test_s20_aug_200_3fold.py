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
import argparse
import numpy as np


def test_s20_aug_200_3fold(test_model_paths):
    test_model_paths = [t.strip() for t in test_model_paths.split(',')]

    # 3-fold cross validation
    for test_model_path in test_model_paths:
        print('Testing: {}'.format(test_model_path))

        command_str = 'python3 -m test ' \
                      '--dataset_dir ' \
                      '"./data/Metadata_Image_Pairs" ' \
                      '--dataset_type real_trans ' \
                      '--n_aug_real 1 ' \
                      '--model_type 1in1out_trans  ' \
                      '--arch_id 3 ' \
                      '--test_model_path "{}"'.format(test_model_path)

        os.system(command_str)

    # aggregate 3-fold results
    three_fold_results = np.zeros(shape=(4, 10))
    three_fold_results_fn = os.path.dirname(os.path.dirname(test_model_paths[0])) + '_3fold_results.csv'
    with open(three_fold_results_fn, 'w')as three_fold_results_file:
        three_fold_results_file.write(
            'fold,'
            'mean_ang_err,'
            'median_ang_err,'
            'best25_mean_ang_err,'
            'worst25_mean_ang_err,'
            'min_ang_err,'
            'quartile1_ang_err,'
            'quartile3_ang_err,'
            'max_ang_err,'
            'mean_pred_time'
            '\n'
        )
        for i, test_model_path in enumerate(test_model_paths):
            three_fold_results[i, 0] = i  # fold number
            results_fn = os.path.join(os.path.dirname(os.path.dirname(test_model_path)), 'results_s20',
                                      'aggregate_results.txt')
            with open(results_fn, 'r')as f:
                result_lines = f.readlines()
            fold_results_str = [line.strip().split(' = ')[1] for line in result_lines]
            fold_results_float = [float(val_str) for val_str in fold_results_str]
            three_fold_results[i, 1:] = fold_results_float
            three_fold_results_file.write(','.join([str(i)] + fold_results_str))
            three_fold_results_file.write('\n')
        # mean of 3 folds
        three_fold_results[-1, 0] = -1
        three_fold_results[-1, 1:] = np.mean(three_fold_results[:2, 1:], axis=0)
        print("three_fold_results: ")
        print(three_fold_results)
        three_fold_results_file.write(','.join(['-1'] + [str(x) for x in three_fold_results[-1, 1:]]))
        three_fold_results_file.write('\n')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--test_model_paths', type=str, help='comma-separated paths to models to be tested',
                            default="./experiments/2022_01_11_14_09_53_s20_aug_200_cv0/best_model/cross_val_0,"
                                    "./experiments/2022_01_11_14_11_03_s20_aug_200_cv1/best_model/cross_val_1,"
                                    "./experiments/2022_01_11_14_12_05_s20_aug_200_cv2/best_model/cross_val_2"
                            )
    args = arg_parser.parse_args()
    test_s20_aug_200_3fold(args.test_model_paths)
