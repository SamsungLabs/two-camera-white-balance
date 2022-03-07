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
import math
import os
import pickle
import time
import numpy as np
import tensorflow as tf

from data_loader import load_dataset_switch, load_or_generate_cross_val_indices
from arg_parser import parse_args, set_experiment_params
from losses import get_loss_function, get_metrics
from models import WbEstimator, set_network_params
from plot import plot_model_history
from utils import save_best_metrics


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        logs["learning_rate"] = self.model.optimizer.lr


def lr_scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr / 10.0


def main():
    t0_t = time.time()
    np.set_printoptions(precision=4)

    # parse args, set parameters
    args = parse_args()
    args = set_experiment_params(args)

    # set seeds for reproducibility
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    # load data
    x, y, n_images, n_repetitions, fn_pairs = load_dataset_switch(args)

    # load/generate cross validation indices
    idx_parts, idx_parts_no_rep = load_or_generate_cross_val_indices(args, n_images, n_repetitions)

    if 'aug0' in args.dataset_dir:
        idx_parts = idx_parts_no_rep

    # adjust epochs vs. dataset repetitions, for efficiency
    args.n_epochs = args.n_epochs // args.n_ds_reps

    models = []

    # network architecture
    args = set_network_params(args)

    # cross validation parts to run
    if args.cross_val_idx != -1:
        cross_val_idxs = [args.cross_val_idx]
    else:
        cross_val_idxs = np.arange(args.n_cross_val)

    # don't use cross validation with radiometric dataset
    if args.dataset_name == 'radio':
        cross_val_idxs = [0]

    for k in cross_val_idxs:

        if args.dataset_name == 'radio':
            train_idx = idx_parts[0]
            val_idx = idx_parts[1]
        else:
            val_idx = idx_parts[k]
            train_idx = []
            for m in np.arange(args.n_cross_val):
                if m != k:
                    train_idx.extend(idx_parts[m])

        x_train = x[train_idx]
        x_val = x[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        print("x_train.shape = {}".format(x_train.shape))
        print("y_train.shape = {}".format(y_train.shape))
        print("x_val.shape = {}".format(x_val.shape))
        print("y_val.shape = {}".format(y_val.shape))

        # create model
        model = WbEstimator(args.n_conv_layers, args.n_filters, args.n_outputs, args.model_type, args.n_dense_layers,
                            args.n_dense_units, args.last_layer)

        optimizer = tf.keras.optimizers.Adam(lr=args.learn_rate)

        loss = get_loss_function(args.loss, args.trans_weight)

        def lr_step_decay(epoch, lr):
            drop_rate = 0.5
            epochs_drop = 50.0
            return args.learn_rate * math.pow(drop_rate, math.floor(epoch / epochs_drop))

        lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=1)

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=get_metrics(args.model_type)
                      )

        ckpt_dir = os.path.join(args.best_model_dir, "cross_val_{}".format(k))
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "best.ckpt")
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, save_best_only=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.tensorboard_dir, os.path.basename(args.experiment_dir), "cross_val_{}".format(k)))

        print("Preparing datasets...")
        t0_d = time.time()

        # using tf datasets
        autotune = tf.data.experimental.AUTOTUNE

        if args.model_type in ['1in1out', '2in1out', '2in2out', '1in1out_trans']:
            train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            val_set = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        elif args.model_type in ['2in2out_trans', '3in2out_trans']:
            x_tr_hists = [xt[0] for xt in x_train]
            x_tr_trans = [xt[1] for xt in x_train]
            y_tr_wbs = [yt[0].squeeze() for yt in y_train]
            # y_tr_trans = [yt[1] for yt in y_train]
            train_set = tf.data.Dataset.from_tensor_slices(({'hists': x_tr_hists, 'trans': x_tr_trans}, y_tr_wbs))
            x_vl_hists = [xv[0] for xv in x_val]
            x_vl_trans = [xv[1] for xv in x_val]
            y_vl_wbs = [yv[0].squeeze() for yv in y_val]
            # y_vl_trans = [yv[1] for yv in y_val]
            val_set = tf.data.Dataset.from_tensor_slices(({'hists': x_vl_hists, 'trans': x_vl_trans}, y_vl_wbs))
        else:
            raise ValueError("Invalid model_type = {}".format(args.model_type))

        train_set = train_set.repeat(args.n_ds_reps)
        train_set = train_set.shuffle(buffer_size=1000)
        train_set = train_set.batch(batch_size=args.batch_size)
        # train_set = train_set.interleave(map_func=map_fn, num_parallel_calls=autotune)  # slow!
        train_set = train_set.prefetch(autotune)

        val_set = val_set.repeat(args.n_ds_reps)
        val_set = val_set.batch(batch_size=args.batch_size)
        # val_set = val_set.interleave(map_func=map_fn, num_parallel_calls=autotune)  # slow!
        val_set = val_set.prefetch(autotune)

        print("done... time = {} sec".format(time.time() - t0_d))

        print('Starting model.fit()...')
        model.fit(train_set, validation_data=val_set, batch_size=args.batch_size, epochs=args.n_epochs,
                  verbose=2, workers=args.n_workers, use_multiprocessing=True, max_queue_size=100,
                  # steps_per_epoch=train_epoch_steps, validation_steps=val_epoch_steps,
                  callbacks=[LearningRateLogger(), ckpt_callback, tensorboard_callback,
                             # lr_schedule_callback
                             ])

        models.append(model)

        # save training history
        with open(os.path.join(args.experiment_dir, "history_cross_val_{}".format(k)), 'wb') as h_file:
            pickle.dump(model.history.history, h_file)

        plot_model_history(model.history, args.experiment_dir, args.loss, k + 1, args.n_cross_val)

        with open(args.model_summary_fn, 'w') as f:
            model.summary(print_fn=lambda arg: f.write(arg + '\n'))

        t0_t = time.time() - t0_t
        tstr_s = "Total time = {} sec".format(t0_t)
        tstr_h = "Total time = {:.3f} hrs".format(t0_t / 3600.0)
        print(tstr_s)
        print(tstr_h)
        with open(os.path.join(args.experiment_dir, 'total_time.txt'), 'w') as t_file:
            t_file.write(tstr_s + "\n")
            t_file.write(tstr_h + "\n")

    models[0].summary()

    save_best_metrics(models, args.experiment_dir)


if __name__ == '__main__':
    main()
