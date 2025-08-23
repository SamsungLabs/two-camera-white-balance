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
import matplotlib.pyplot as plt


def plot_model_history(history, experiment_dir, loss_name, model_num, n_cross_val):
    plt_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(plt_dir, exist_ok=True)

    keys = list(history.history.keys())
    train_keys = keys[:len(keys) // 2]
    for key in train_keys:
        fig = plt.figure()
        plt.plot(history.history[key])
        plt.plot(history.history['val_' + key])
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epoch')
        if key == 'loss':
            plt.ylabel(key + " ({})".format(loss_name))
        else:
            plt.ylabel(key)
        plt.title('Cross validation part {} / {}'.format(model_num, n_cross_val))
        # plt.show()
        plt.savefig(os.path.join(plt_dir, '{}_{}.png'.format(key, model_num)))
