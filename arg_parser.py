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

import argparse
import datetime
import os


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_dir', type=str, help='dataset directory to use for training/testing OR '
                                                            'dataset filename, in case of synthetic data.',
                            default='./S20-Two-Camera-Dataset/Metadata_Image_Pairs'
                            )
    arg_parser.add_argument('--dataset_type', type=str, default='real_trans',
                            help='dataset type ['
                                 '`hist`: individual histograms | '
                                 '`aug_hist`: augmented histograms | '
                                 '`syn_trans`: synthetic transforms | '
                                 '`syn_trans_rad`: synthetic transforms (radiometric) | '
                                 '`real_trans`: transforms from real images (S20) | '
                                 '`real_trans_aug`: transforms from real images (S20), augmented | '
                                 ']')
    arg_parser.add_argument('--n_aug_real', type=int, help='number of augmentations per image (for real data)',
                            default=1)
    arg_parser.add_argument('--cam_p', type=str, help='first camera ID (for NUS dataset)', default='NikonD5200')
    arg_parser.add_argument('--cam_f', type=str, help='second camera ID (for NUS dataset)', default='Canon1DsMkIII')
    arg_parser.add_argument('--target_cam', type=int, help='target camera for which to estimate illuminant ( 0 or 1)',
                            default=0)
    arg_parser.add_argument('--n_images', type=int,
                            help='number of images to use in training, 0 means to use all images', default=0)
    arg_parser.add_argument('--model_type', type=str, default='1in1out_trans',
                            help='model type. '
                                 '`1in1out`: input 1 histogram, output 1 WB. '
                                 '`2in1out`: input 2 histograms, output 1 WB. '
                                 '`2in2out`: input 2 histograms, output 2 WB. '
                                 '`2in2out_trans`: input 2 histograms, output 2 WB, use histogram in loss. '
                                 '`3in2out_trans`: input 2 histograms + transform, output 2 WB, use histogram in loss. '
                                 '`1in1out_trans`: input transform only, output 1 WB. '
                            )
    arg_parser.add_argument('--experiments_dir', type=str, help='directory to save all experiments\' outputs',
                            default='./experiments'
                            )
    arg_parser.add_argument('--experiment_dir', type=str, help='path to an existing experiment to continue training',
                            default='')
    arg_parser.add_argument('--test_model_path', type=str, help='path to model to be tested',
                            default=''
                            )
    arg_parser.add_argument('--experiment_name', type=str, help='experiment name', default='')
    arg_parser.add_argument('--tensorboard_dir', type=str, help='tensorboard log directory',
                            default=''
                            )
    arg_parser.add_argument('--n_cross_val', type=int, help='number of cross validation parts', default=3)
    arg_parser.add_argument('--cross_val_idx', type=int, help='0-based index of cross validation part to run, '
                                                              'if -1, all parts will be run', default=-1)
    arg_parser.add_argument('--random_seed', type=int, help='random seed for reproducibility', default=999)
    arg_parser.add_argument('--learn_rate', type=float, help='learning rate', default=1e-4)
    arg_parser.add_argument('--loss', type=str,
                            help='loss function [mae, mse, cos, ang, ang_mae, mae_trans, mae_trans_wgt]',
                            default='mae')
    arg_parser.add_argument('--trans_weight', type=float, help='weight of transform term in loss', default=1.0)
    arg_parser.add_argument('--n_epochs', type=int, help='number of epochs', default=2000)
    arg_parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    arg_parser.add_argument('--arch_id', type=int, help='architecture ID',
                            default=3
                            )
    arg_parser.add_argument('--n_conv_layers', type=int, help='number of convolutional layers', default=5)
    arg_parser.add_argument('--n_filters', type=int, help='number of filters in each layers', default=64)
    arg_parser.add_argument('--n_dense_layers', type=int, help='number of dense layers for dense models', default=2)
    arg_parser.add_argument('--n_dense_units', type=int, help='number of dense units per layer for dense models',
                            default=9)
    arg_parser.add_argument('--n_outputs', type=int, help='number of model outputs', default=2)
    arg_parser.add_argument('--n_workers', type=int, help='number of processes for data loading', default=8)
    arg_parser.add_argument('--n_ds_reps', type=int,
                            help='number of dataset repetitions, epochs will be divided by this number', default=10)
    arg_parser.add_argument('--last_layer', type=str,
                            default='exp',
                            help='choice of last layer activation (exp, sigmoid, relu, ...)')

    args = arg_parser.parse_args()
    return args


def set_experiment_params(args):
    # experiment's name, directory, etc.
    if args.experiment_dir != '':
        # an existing experiment (continue training or testing model)
        args.continue_training = True
    else:
        # a new experiment
        args.continue_training = False
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        args.experiment_dir = os.path.join(args.experiments_dir, timestamp)
        # append experiment name
        if args.experiment_name != '':
            args.experiment_dir = "{}_{}".format(args.experiment_dir, args.experiment_name)
        args.script_name = os.path.basename(__file__)
        print("args.experiment_dir = {}".format(args.experiment_dir))

    os.makedirs(args.experiments_dir, exist_ok=True)
    os.makedirs(args.experiment_dir, exist_ok=True)

    # directories to save training/testing sample images
    args.train_samples_dir = os.path.join(args.experiment_dir, 'train_samples')
    args.test_samples_dir = os.path.join(args.experiment_dir, 'test_samples')
    os.makedirs(args.train_samples_dir, exist_ok=True)
    os.makedirs(args.test_samples_dir, exist_ok=True)

    # model directory
    args.best_model_dir = os.path.join(args.experiment_dir, 'best_model')
    os.makedirs(args.best_model_dir, exist_ok=True)

    # log files
    args.log_file = os.path.join(args.experiment_dir, 'log.txt')
    args.best_model_info_file = os.path.join(args.experiment_dir, 'best_model_info.txt')
    args.model_summary_fn = os.path.join(args.experiment_dir, 'model_summary.txt')

    # tensorboard
    if args.tensorboard_dir == '' or args.tensorboard_dir is None:
        args.tensorboard_dir = os.path.join(args.experiment_dir, 'tensorboard')
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    # lastly, save experiment's arguments/parameters
    args_file = os.path.join(args.experiment_dir, 'args.txt')
    with open(args_file, 'w')as af:
        af.write(str(args))

    return args
