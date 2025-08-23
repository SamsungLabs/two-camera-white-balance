"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Abdelrahman Abdelhamed (a.abdelhamed@samsung.com; abdoukamel@gmail.com)

A data loader for white balance estimation.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""
import math
import os
import glob
import pickle

import scipy.io as sio
import pandas as pd
import numpy as np

from scipy.io import loadmat, savemat
from tensorflow.python.keras.utils.data_utils import Sequence

from utils import get_cross_validation_indices


def load_dataset_switch(args):
    if args.dataset_type == 'hist':
        args.dataset_name = 'nus'
        fn_pairs, histogram_pairs, wb_gt_pairs, transforms, n_images = load_data(args.dataset_dir, args.cam_p,
                                                                                 args.cam_f)
        x, y = prep_data(histogram_pairs, wb_gt_pairs, transforms, args.model_type, target_cam=args.target_cam)
        n_repetitions = n_images
    elif args.dataset_type == 'aug_hist':
        args.dataset_name = 'nus'
        fn_pairs, histogram_pairs, wb_gt_pairs, transforms, n_images = load_augmented_histograms(args.dataset_dir,
                                                                                                 args.n_images)
        x, y = prep_data(histogram_pairs, wb_gt_pairs, transforms, args.model_type, target_cam=args.target_cam)
        n_repetitions = args.n_aug_real
    elif args.dataset_type == 'real_trans':
        args.dataset_name = 's20'
        fn_pairs, histogram_pairs, wb_gt_pairs, transforms, n_images = load_data_real(args.dataset_dir,
                                                                                      args.n_images)
        x, y = prep_data(histogram_pairs, wb_gt_pairs, transforms, args.model_type, target_cam=args.target_cam)
        n_repetitions = 1
    elif args.dataset_type == 'real_trans_aug':
        args.dataset_name = 's20'
        fn_pairs, histogram_pairs, wb_gt_pairs, transforms, n_images = load_data_real(args.dataset_dir,
                                                                                      args.n_images)
        x, y = prep_data(histogram_pairs, wb_gt_pairs, transforms, args.model_type, target_cam=args.target_cam)
        n_repetitions = args.n_aug_real
        n_images //= n_repetitions
    elif args.dataset_type == 'syn_trans':
        args.dataset_name = 'radio'
        x, y = load_transform_wb_synthetic(args.dataset_dir)
        n_images = x.shape[0]
        n_repetitions = 1
        fn_pairs = None
    elif args.dataset_type == 'syn_trans_rad':
        args.dataset_name = 'radio'
        x, y = load_transform_wb_radiometric(args.dataset_dir, target_cam=args.target_cam)
        n_images = x.shape[0]
        n_repetitions = 1
        fn_pairs = None
    else:
        raise ValueError('Invalid: args.dataset_type = {}'.format(args.dataset_type))
    return x, y, n_images, n_repetitions, fn_pairs


def load_data(nus_dataset_dir, cam_p, cam_f):
    file_pattern_p = os.path.join(nus_dataset_dir, cam_p, 'histogram_rg_bg', '*.npy')
    file_pattern_f = os.path.join(nus_dataset_dir, cam_f, 'histogram_rg_bg', '*.npy')

    fns_p = sorted(glob.glob(file_pattern_p))
    fns_f = sorted(glob.glob(file_pattern_f))

    gt_p = sio.loadmat(os.path.join(nus_dataset_dir, cam_p, cam_p + '_gt'))
    gt_f = sio.loadmat(os.path.join(nus_dataset_dir, cam_f, cam_f + '_gt'))

    fn_map = pd.read_csv(os.path.join(nus_dataset_dir, cam_p, 'matched_with_' + cam_f + '.txt'), header=None)

    transforms_fn = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/transforms_image.pkl')
    with open(transforms_fn, 'rb') as f:
        transform_tuples = np.array(pickle.load(f), dtype=object)

    fn_pairs = []
    histogram_pairs = []
    wb_gt_pairs = []
    transforms = []

    for idx_p in range(len(fns_p)):

        # check if two images match
        imname_p = fn_map.iat[idx_p, 0]
        imname_f = fn_map.iat[idx_p, 1]
        # if idx_p == 154:
        #     x=1
        idx_f = np.where(gt_f['all_image_names'].flatten() == imname_f)[0]
        if len(idx_f) == 0:
            continue
        idx_f = idx_f[0]

        # file name pair
        fn_pairs.append([imname_p, imname_f])

        # histogram pair
        hist_p = np.load(fns_p[idx_p])
        hist_f = np.load(fns_f[idx_f])
        histogram_pairs.append([hist_p, hist_f])

        if np.isnan(hist_p).any() or np.isnan(hist_f).any() or np.isinf(hist_p).any() or np.isinf(hist_f).any():
            print("NaN/Inf in histogram!!!")
            import pdb
            pdb.set_trace()

        # WB GT pair
        wb_p = gt_p['groundtruth_illuminants'][idx_p].flatten()
        wb_p /= wb_p[1]
        wb_f = gt_f['groundtruth_illuminants'][idx_f].flatten()
        wb_f /= wb_f[1]
        wb_gt_pairs.append([wb_p, wb_f])

        if np.isnan(wb_p).any() or np.isnan(wb_f).any() or np.isinf(wb_p).any() or np.isinf(wb_f).any():
            print("NaN/Inf in WB GT!!!")
            import pdb
            pdb.set_trace()

        # transform
        trans_tuple = transform_tuples[np.where(transform_tuples[:, 0] == imname_p)]
        trans = trans_tuple[0, 4]
        wb_p_ = trans_tuple[0, 2]
        transforms.append(trans)
        # TODO
        assert np.mean(np.abs(wb_p_ - wb_p)) < 1e-6

    return fn_pairs, histogram_pairs, wb_gt_pairs, transforms, len(fns_p)


def prep_data(histogram_pairs, wb_gt_pairs, transforms, model_type, target_cam):
    # if model_type == '1in1out':
    #     x = np.concatenate(
    #         [histogram_pairs[i][0][np.newaxis, :, :, np.newaxis] for i in range(len(histogram_pairs))], axis=0)
    #     y = np.concatenate([wb_gt_pairs[i][0][np.newaxis, :] for i in range(len(wb_gt_pairs))], axis=0)
    # elif model_type == '2in1out':
    #     x = np.concatenate(
    #         [np.concatenate([histogram_pairs[i][0][np.newaxis, :, :, np.newaxis],
    #                          histogram_pairs[i][1][np.newaxis, :, :, np.newaxis]], axis=-1)
    #          for i in range(len(histogram_pairs))], axis=0)
    #     y = np.concatenate([wb_gt_pairs[i][0][np.newaxis, :] for i in range(len(wb_gt_pairs))], axis=0)
    # elif model_type == '2in2out':
    #     x = np.concatenate(
    #         [np.concatenate([histogram_pairs[i][0][np.newaxis, :, :, np.newaxis],
    #                          histogram_pairs[i][1][np.newaxis, :, :, np.newaxis]], axis=-1)
    #          for i in range(len(histogram_pairs))], axis=0)
    #     y = np.concatenate(
    #         [np.concatenate([wb_gt_pairs[i][0][np.newaxis, :],
    #                          wb_gt_pairs[i][1][np.newaxis, :]], axis=-1)
    #          for i in range(len(wb_gt_pairs))], axis=0)
    # elif model_type in ['2in2out_trans', '3in2out_trans']:
    #     x = np.array([(np.concatenate([histogram_pairs[i][0][:, :, np.newaxis],
    #                                    histogram_pairs[i][1][:, :, np.newaxis]], axis=-1),
    #                    transforms[i])
    #                   for i in range(len(histogram_pairs))
    #                   ], dtype=object)
    #     y = np.array([(np.concatenate([wb_gt_pairs[i][0][np.newaxis, :],
    #                                    wb_gt_pairs[i][1][np.newaxis, :]], axis=-1),
    #                    transforms[i])
    #                   for i in range(len(wb_gt_pairs))
    #                   ], dtype=object)
    # elif model_type == '1in1out_trans':

    x = np.array([transforms[i] for i in range(len(transforms))])
    y = np.concatenate([wb_gt_pairs[i][target_cam][np.newaxis, :] for i in range(len(wb_gt_pairs))], axis=0)

    # else:
    #     raise ValueError('Invalid model_type = {}'.format(model_type))

    return x, y


def check_nan_inf(x, s, i, j):
    if np.isnan(x).any() or np.isinf(x).any():
        print("{} [{}, {}] has NaN/Inf!!!".format(s, i, j))


def load_augmented_histograms(data_dir, n_images=0):
    fn_pairs = []
    histogram_pairs = []
    wb_gt_pairs = []
    transforms = []
    fns = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
    if n_images > 0:
        fns = fns[:n_images]
    n_files = len(fns)
    basenames = [os.path.splitext(os.path.basename(fn))[0] for fn in fns]
    data_list = [dict(np.load(fn, allow_pickle=True)[()]) for fn in fns]
    for i in range(len(data_list)):
        for j in range(len(data_list[i]['wb1'])):
            # TODO: get file names from data_list
            fn_pairs.append([basenames[i].replace('hist_data_', ''), basenames[j].replace('hist_data_', '')])
            if j < len(data_list[i]['hist1']):
                histogram_pairs.append([data_list[i]['hist1'][j], data_list[i]['hist2'][j]])
            wb_gt_pairs.append([data_list[i]['wb1'][j].squeeze(), data_list[i]['wb2'][j].squeeze()])
            transforms.append(data_list[i]['transform'][j])

            # check NaN/Inf
            if j < len(data_list[i]['hist1']):
                check_nan_inf(data_list[i]['hist1'][j], 'hist1', i, j)
                check_nan_inf(data_list[i]['hist2'][j], 'hist2', i, j)
            check_nan_inf(data_list[i]['wb1'][j], 'wb1', i, j)
            check_nan_inf(data_list[i]['wb2'][j], 'wb2', i, j)
    return fn_pairs, histogram_pairs, wb_gt_pairs, transforms, n_files


def load_data_real(data_dir, n_images=0):
    fn_pairs = []
    histogram_pairs = []
    wb_gt_pairs = []
    transforms = []
    fns = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
    if n_images > 0:
        fns = fns[:n_images]
    n_files = len(fns)
    data_list = [dict(np.load(fn, allow_pickle=True)[()]) for fn in fns]
    for i in range(len(data_list)):
        fn_pairs.append([data_list[i]['image_name_1'], data_list[i]['image_name_2']])
        histogram_pairs.append([None, None])  # TODO: generate histograms
        wb_gt_1 = data_list[i]['wb_gt_1'].squeeze()
        wb_gt_2 = data_list[i]['wb_gt_2'].squeeze()
        wb_gt_1 /= np.maximum(wb_gt_1[1], 1e-6)
        wb_gt_2 /= np.maximum(wb_gt_2[1], 1e-6)
        wb_gt_pairs.append([wb_gt_1, wb_gt_2])
        trans = data_list[i]['transform_image_1to2']
        transforms.append(trans)
        assert not np.isnan(wb_gt_1).any() and not np.isinf(wb_gt_1).any()
        assert not np.isnan(wb_gt_2).any() and not np.isinf(wb_gt_2).any()
        assert not np.isnan(trans).any() and not np.isinf(trans).any()
    return fn_pairs, histogram_pairs, wb_gt_pairs, transforms, n_files


def load_transform_wb_synthetic(fn):
    data = loadmat(fn)
    transforms = data['T1to2']
    wb_gt = data['Y'][:, :3]
    wb_gt /= np.maximum(wb_gt[:, 1:2], 1e-6)
    return transforms, wb_gt


def load_transform_wb_radiometric(fn, max_n=0, target_cam=0):
    data = loadmat(fn)
    transforms = data['transforms1to2']
    if target_cam == 0:
        wb_gt = data['wb_gts_1'][:, :3]
    else:
        wb_gt = data['wb_gts_2'][:, :3]
    if max_n != 0 and max_n < transforms.shape[0]:
        transforms = transforms[:max_n, ...]
        wb_gt = wb_gt[:max_n, ...]
    wb_gt /= np.maximum(wb_gt[:, 1:2], 1e-6)
    return transforms, wb_gt


def load_or_generate_cross_val_indices(args, n_images, n_repetitions):
    idx_parts = idx_parts_no_rep = None
    if args.dataset_name == 'radio':
        # use train/val/test split for radiometric data (60/20/20 percentages)
        idx_parts_fn = os.path.join(os.path.dirname(__file__), "data",
                                    "{}_idx_{}_parts.mat".format("radio", "train_val_test"))
        if os.path.exists(idx_parts_fn):
            # import pdb
            # pdb.set_trace()
            idx_parts = loadmat(idx_parts_fn)['idx_parts']
            idx_parts = idx_parts[0]
            idx_parts = [idx_parts[0].squeeze(), idx_parts[1].squeeze(), idx_parts[2].squeeze()]
        else:
            idx_parts, idx_parts_no_rep = get_cross_validation_indices(n_images, 5, repetitions=1, shuffle=True)
            idx_parts[0] = np.concatenate([idx_parts[0], idx_parts[1], idx_parts[2]])
            idx_parts[1] = idx_parts[3]
            idx_parts[2] = idx_parts[4]
            idx_parts = idx_parts[:3]
            savemat(idx_parts_fn, {"idx_parts": idx_parts})
        idx_parts_no_rep = idx_parts
    elif args.dataset_name in ['nus', 's20']:
        idx_parts_fn = os.path.join(os.path.dirname(__file__), "data",
                                    "{}_idx_{}_parts.mat".format(args.dataset_name, args.n_cross_val))
        idx_parts_no_rep_fn = idx_parts_fn.replace('_parts.mat', '_parts_no_rep.mat')
        if os.path.exists(idx_parts_fn) and os.path.exists(idx_parts_no_rep_fn):
            idx_parts = loadmat(idx_parts_fn)['idx_parts']
            idx_parts = [idx_parts[p] for p in range(idx_parts.shape[0])]
            idx_parts_no_rep = loadmat(idx_parts_no_rep_fn)['idx_parts_no_rep']
            idx_parts_no_rep = [idx_parts_no_rep[0].squeeze(), idx_parts_no_rep[1].squeeze(),
                                idx_parts_no_rep[2].squeeze()]
        else:
            idx_parts, idx_parts_no_rep = get_cross_validation_indices(n_images, args.n_cross_val,
                                                                       repetitions=n_repetitions,
                                                                       shuffle=True)
            savemat(idx_parts_fn, {"idx_parts": idx_parts})
            savemat(idx_parts_no_rep_fn, {"idx_parts_no_rep": idx_parts_no_rep})
    return idx_parts, idx_parts_no_rep


def get_callable_generator(x, y):
    def gen():
        for i in range(len(x)):
            yield x[i], y[i]

    return gen


class Generator(Sequence):

    def __init__(self, x_set, y_set, batch_size, shuffle):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x))

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[indexes]
        batch_y = self.y[indexes]
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
