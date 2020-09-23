# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def halfmoon(rad, width, d, numberof_samp):
    if numberof_samp % 2 != 0:
        numberof_samp += 1

    data = np.zeros((3, numberof_samp))

    aa = np.random.random((2, numberof_samp // 2))
    rad1 = (rad - width // 2) + width * aa[0, :]
    tita = np.pi * aa[1, :]

    x = rad1 * np.cos(tita)
    y = rad1 * np.sin(tita)
    label = np.ones((1, len(x)))  # label for Class 1

    x1 = rad1 * np.cos(-tita) + rad
    y1 = rad1 * np.sin(-tita) - d
    label1 = -1 * np.ones((1, len(x1)))  # label for Class 2

    data[0, :] = np.concatenate([x, x1])
    data[1, :] = np.concatenate([y, y1])
    data[2, :] = np.concatenate([label, label1], axis=1)

    return data


def halfmoon_shuffle(rad, width, d, n_samp):
    data = halfmoon(rad, width, d, n_samp)
    shuff_seq = np.random.permutation(np.arange(n_samp))
    data_shuff = data[:, shuff_seq]

    return data_shuff





