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
import cv2
import pandas as pd
import matplotlib.pyplot as plt


def mae(x, y):
    return np.mean(np.abs(x - y))


def mse(x, y):
    return np.mean(np.power(x - y, 2))


def dot(x, y):
    return np.sum(x * y, axis=-1)


def norm(x):
    return np.sqrt(dot(x, x))


def ang_err(x, y):
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    cos_ = dot(x, y) / (norm(x) * norm(y) + 1e-8)
    ang_rad = np.arccos(cos_)
    ang_deg = ang_rad * 180.0 / np.pi
    return ang_deg


def mean_ang_err(x, y):
    return np.mean(ang_err(x, y))


def compute_transform(colors1, colors2):
    """compute transform from colors1 to colors2 (T = colors1^{-1} colors2)"""
    colors1_g = np.sum(colors1, axis=1, keepdims=True)
    colors2_g = np.sum(colors2, axis=1, keepdims=True)
    # avoid div by zero
    colors1_g = np.maximum(colors1_g, 1e-6)
    colors2_g = np.maximum(colors2_g, 1e-6)
    colors1_ = colors1 / colors1_g
    colors2_ = colors2 / colors2_g
    assert not np.isnan(colors1_).any()
    assert not np.isinf(colors1_).any()
    assert not np.isnan(colors2_).any()
    assert not np.isinf(colors2_).any()
    transform = np.matmul(np.linalg.pinv(colors1_), colors2_)
    return transform


def apply_transform(colors1, transform, clip):
    """apply transform on colors1 (output_colors = colors1 T)"""
    colors1_g = np.sum(colors1, axis=1, keepdims=True)
    colors1_g = np.maximum(colors1_g, 1e-6)  # avoid division by zero
    colors1_ = colors1 / colors1_g
    colors2 = np.matmul(colors1_, transform)
    colors2 = colors2 * colors1_g
    if clip:
        colors2 = np.clip(colors2, 0.0, 1.0)
    return colors2


def normalize(values, black_level, white_level):
    return (values - black_level) / (white_level - black_level)


def hist2d_to_colormap(hist2d, cmap='jet', upscale=8):
    cm = plt.get_cmap(cmap)
    hist2d_cm = cm(hist2d)
    hist2d_cm = (hist2d_cm[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)
    hist2d_cm = cv2.resize(hist2d_cm, dsize=(hist2d_cm.shape[1] * upscale, hist2d_cm.shape[0] * upscale))
    return hist2d_cm


def get_cross_validation_indices(n, n_parts, repetitions=1, shuffle=True):
    if shuffle:
        idx = np.random.permutation(n)
    else:
        idx = np.arange(n)
    avg_size = float(n) / n_parts
    idx_parts = []
    end = 0.0
    while end < n:
        idx_parts.append(idx[int(end):int(end + avg_size)])
        end += avg_size
    idx_parts_no_rep = idx_parts.copy()
    if repetitions > 1:
        for p in range(n_parts):
            idx_parts[p] = np.concatenate([np.arange(i * repetitions, (i + 1) * repetitions) for i in idx_parts[p]])
    return idx_parts, idx_parts_no_rep


def ratios_to_floats(ratio_list):
    return np.array([float(ratio_list[k].num) / ratio_list[k].den for k in range(len(ratio_list))])


def save_best_metrics(models, exper_dir):
    n_models = len(models)
    keys = list(models[0].history.history.keys())
    best_metrics = pd.DataFrame(columns=['model_num'] + keys)
    best_metrics.set_index('model_num')
    for m in range(n_models):
        min_val_loss_idx = np.argmin(models[m].history.history['val_loss'])
        vals = [m] + [models[m].history.history[key][min_val_loss_idx] for key in keys]
        best_metrics.loc[m] = vals
    best_metrics.loc[n_models] = ['avg'] + list(np.mean(best_metrics.values[:, 1:], axis=0))
    best_metrics.to_csv(os.path.join(exper_dir, 'best_metrics.csv'), index=False)
