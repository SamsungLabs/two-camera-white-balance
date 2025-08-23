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
import numpy as np

from time import time
from scipy.io import savemat

from utils import ang_err
from arg_parser import parse_args
from data_loader import load_dataset_switch, load_or_generate_cross_val_indices
from models import WbEstimator, set_network_params


def main():
    t0_t = time()
    np.set_printoptions(precision=4)

    # parse args, set parameters
    args = parse_args()

    print("testing: {}".format(args.test_model_path))

    # load data
    x, y, n_images, n_repetitions, fn_pairs = load_dataset_switch(args)

    # load testing indices
    idx_parts, idx_parts_no_rep = load_or_generate_cross_val_indices(args, n_images, n_repetitions)

    # do cross validation for NUS and S20 data
    # for radiometric data, use the 60/20/20, train/val/test split
    do_cross_val = True if args.dataset_name in ['nus', 's20'] else False
    if do_cross_val:
        # figure out which cross validation part to run from model directory name
        cross_val_idx = int(args.test_model_path[-1])
    else:
        cross_val_idx = 0

    # figure out test model checkpoint
    # ckpt_dir = os.path.join(args.best_model_dir, "cross_val_{}".format(cross_val_idx))
    ckpt_path = os.path.join(args.test_model_path, "best.ckpt")

    # network architecture
    args = set_network_params(args)

    # create model
    test_model = WbEstimator(args.n_conv_layers, args.n_filters, args.n_outputs, args.model_type, args.n_dense_layers,
                             args.n_dense_units)

    # load model
    test_model.load_weights(ckpt_path)

    # test
    n = len(idx_parts_no_rep[cross_val_idx])
    y_est_arr = np.zeros(shape=(n, 3))
    pred_time_arr = np.zeros(shape=(n,))
    ang_err_arr = np.zeros(shape=(n,))
    for i in range(n):
        test_idx = idx_parts_no_rep[cross_val_idx][i]
        pred_time = time()
        y_est = test_model.predict(x[test_idx][np.newaxis, ...]).flatten()
        pred_time = time() - pred_time
        y_est_arr[i, :] = y_est
        pred_time_arr[i] = pred_time
        ang_err_ = ang_err(y[test_idx].flatten(), y_est)
        ang_err_arr[i] = ang_err_

    # clean up
    ang_err_arr_clean = ang_err_arr[~np.isnan(ang_err_arr)]
    ang_err_arr_clean = ang_err_arr_clean[~np.isinf(ang_err_arr_clean)]

    # result stats
    mean_pred_time = np.mean(pred_time_arr)
    ang_err_arr_sorted = np.sort(ang_err_arr_clean)
    fourth = int(np.round(n / 4.0))
    mean_ang_err = np.mean(ang_err_arr_clean)
    median_ang_err = np.median(ang_err_arr_clean)
    min_ang_err = ang_err_arr_sorted[0]
    max_ang_err = ang_err_arr_sorted[-1]
    quartile1_ang_err = np.quantile(ang_err_arr_clean, 0.25)
    quartile3_ang_err = np.quantile(ang_err_arr_clean, 0.75)
    best25_mean_ang_err = np.mean(ang_err_arr_sorted[:fourth])
    worst25_mean_ang_err = np.mean(ang_err_arr_sorted[-fourth:])

    # Note:
    # 0 quartile = 0 quantile = 0 percentile
    # 1 quartile = 0.25 quantile = 25 percentile
    # 2 quartile = .5 quantile = 50 percentile (median)
    # 3 quartile = .75 quantile = 75 percentile
    # 4 quartile = 1 quantile = 100 percentile

    # save results
    res_dir = os.path.join(args.test_model_path, '..', '..', 'results_{}'.format(args.dataset_name))
    os.makedirs(res_dir, exist_ok=True)
    savemat(os.path.join(res_dir, 'wb_est_arr.mat'), {'wb_est_arr': y_est_arr})
    savemat(os.path.join(res_dir, 'ang_err_arr.mat'), {'ang_err_arr': ang_err_arr})
    savemat(os.path.join(res_dir, 'mean_ang_err.mat'), {'mean_ang_err': mean_ang_err})
    savemat(os.path.join(res_dir, 'median_ang_err.mat'), {'median_ang_err': median_ang_err})
    savemat(os.path.join(res_dir, 'best25_mean_ang_err.mat'), {'best25_mean_ang_err': best25_mean_ang_err})
    savemat(os.path.join(res_dir, 'worst25_mean_ang_err.mat'), {'worst25_mean_ang_err': worst25_mean_ang_err})
    savemat(os.path.join(res_dir, 'min_ang_err.mat'), {'min_ang_err': min_ang_err})
    savemat(os.path.join(res_dir, 'quartile1_ang_err.mat'), {'quartile1_ang_err': quartile1_ang_err})
    savemat(os.path.join(res_dir, 'quartile3_ang_err.mat'), {'quartile3_ang_err': quartile3_ang_err})
    savemat(os.path.join(res_dir, 'max_ang_err.mat'), {'max_ang_err': max_ang_err})
    savemat(os.path.join(res_dir, 'pred_time_arr.mat'), {'ang_err_arr': pred_time_arr})
    savemat(os.path.join(res_dir, 'mean_pred_time.mat'), {'mean_pred_time': mean_pred_time})
    with open(os.path.join(res_dir, 'aggregate_results.txt'), 'w')as f:
        f.write('mean_ang_err = {}\n'.format(mean_ang_err))
        f.write('median_ang_err = {}\n'.format(median_ang_err))
        f.write('best25_mean_ang_err = {}\n'.format(best25_mean_ang_err))
        f.write('worst25_mean_ang_err = {}\n'.format(worst25_mean_ang_err))
        f.write('min_ang_err = {}\n'.format(min_ang_err))
        f.write('quartile1_ang_err = {}\n'.format(quartile1_ang_err))
        f.write('quartile3_ang_err = {}\n'.format(quartile3_ang_err))
        f.write('max_ang_err = {}\n'.format(max_ang_err))
        f.write('mean_pred_time = {}\n'.format(mean_pred_time))

    t0_t = time() - t0_t
    tstr_1 = "Total test time = {} sec".format(t0_t)
    print(tstr_1)
    with open(os.path.join(res_dir, 'total_test_time.txt'), 'w') as t_file:
        t_file.write(tstr_1 + "\n")

    print("Done.")


if __name__ == '__main__':
    main()
