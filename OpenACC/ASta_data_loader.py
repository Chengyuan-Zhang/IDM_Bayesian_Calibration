import pandas as pd
import numpy as np
import pickle
from os import path
from pickle import UnpicklingError

import corner
import os
import sys

sys.path.append('../')

from config import Config

import warnings

warnings.filterwarnings("ignore")

from pykalman import KalmanFilter


def load_ASta(id=1):
    df = pd.read_csv('../OpenACC/ASta_040719_platoon{:d}.csv'.format(id), header=5)
    df = df[['Time', 'Speed1', 'Speed2', 'Speed3', 'Speed4', 'Speed5', 'IVS1', 'IVS2', 'IVS3', 'IVS4']]
    df['dv1'] = df['Speed2'] - df['Speed1']
    df['dv2'] = df['Speed3'] - df['Speed2']
    df['dv3'] = df['Speed4'] - df['Speed3']
    df['dv4'] = df['Speed5'] - df['Speed4']
    return df


def KF_tracks(df, dim=5):
    transition_matrices = np.block([[np.eye(dim), Config.dt * np.eye(dim), 0.5 * Config.dt ** 2 * np.eye(dim)],
                                    [np.zeros((dim, dim)), np.eye(dim), Config.dt * np.eye(dim)],
                                    [np.zeros((dim, dim)), np.zeros((dim, dim)), np.eye(dim)]])

    a = np.eye(dim)
    a[0:-1, 1:, ] -= np.eye(dim - 1)

    observation_matrices = np.block([[np.zeros((dim, dim)), np.eye(dim), np.zeros((dim, dim))],
                                     [a[:-1, :], np.zeros((dim - 1, dim)), np.zeros((dim - 1, dim))]])

    measurements = df[['Speed1', 'Speed2', 'Speed3', 'Speed4', 'Speed5', 'IVS1', 'IVS2', 'IVS3', 'IVS4']].to_numpy()

    initial_state_mean = np.array(
        [0, -df['IVS1'][0], -df['IVS1'][0] - df['IVS2'][0], -df['IVS1'][0] - df['IVS2'][0] - df['IVS3'][0],
         -df['IVS1'][0] - df['IVS2'][0] - df['IVS3'][0] - df['IVS4'][0], df['Speed1'][0], df['Speed2'][0],
         df['Speed3'][0], df['Speed4'][0], df['Speed5'][0], 0, 0, 0, 0, 0])

    kf = KalmanFilter(n_dim_state=3 * dim, n_dim_obs=2 * dim - 1,
                      transition_matrices=transition_matrices,
                      observation_matrices=observation_matrices,
                      initial_state_mean=initial_state_mean)
    kf.em(measurements)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
    df[['x1_kf', 'x2_kf', 'x3_kf', 'x4_kf', 'x5_kf']] = smoothed_state_means[:, :5]
    df[['v1_kf', 'v2_kf', 'v3_kf', 'v4_kf', 'v5_kf']] = smoothed_state_means[:, 5:10]
    df[['a1_kf', 'a2_kf', 'a3_kf', 'a4_kf', 'a5_kf']] = smoothed_state_means[:, 10:]
    return df


def load_df(id):
    cache = "../OpenACC/cache/ASta_040719_platoon{:d}.pkl".format(id)
    if path.exists(cache):
        try:
            fp = open(cache, 'rb')
            df = pickle.load(fp)
            fp.close()
        #             print("Load trace", cache, ": done!")
        except UnpicklingError:
            os.remove(cache)
            print('Removed broken cache:', cache)
    else:
        output_file = open(cache, 'wb')
        df = KF_tracks(load_ASta(id=id).iloc[::2])
        pickle.dump(df, output_file)
        output_file.close()
        print("Filtered and Saved", output_file, ": done!")
    return df


if __name__ == '__main__':
    for i in range(1, 11):
        load_df(i)
