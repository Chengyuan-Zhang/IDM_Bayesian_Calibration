import numpy as np
from config import Config

from data.read_csv import find_id_in_frame, read_dataset, read_track_meta_csv
import os
from os import path
import pickle
from pickle import UnpicklingError
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# from pykalman import KalmanFilter

pair_id = 0
PREPROCESSING_VERSION = 2


class CFPairList(list):
    """Support stable (recording ID, follower ID) keys as well as list indices."""

    def __getitem__(self, key: Union[int, slice, Tuple[int, int]]) -> Any:
        if isinstance(key, tuple):
            matches = [pair for pair in self
                       if (pair['recording_id'], int(pair['id_Foll'])) == key]
            if len(matches) != 1:
                raise KeyError("Expected one car-following pair for %r, found %d" % (key, len(matches)))
            return matches[0]
        return super().__getitem__(key)


def load_CF_pairs(tracks, LC_id, class_list, min_traj_lenth=Config.min_traj_lenth):
    global pair_id
    track_pair_list_temp = []
    tracks_by_id = {int(track['id']): track for track in tracks}
    for tracks_dict in tracks:
        if tracks_dict['id'] in LC_id:  # continue if this ID change lane
            continue

        valid_idx = np.nonzero(tracks_dict['precedingId'])[0]
        pre_ID_list, idx_list, count_list = np.unique(tracks_dict['precedingId'], return_index=True, return_counts=True)
        flag = 0
        if '0' in pre_ID_list:
            if len(pre_ID_list) > 2:
                flag = 1
        elif len(pre_ID_list) > 1:
            flag = 1

        if flag == 1:
            long_id = pre_ID_list[np.argmax(count_list[1:]) + 1]
            valid_idx = np.where(tracks_dict['precedingId'] == long_id)[0]

        len_valid = valid_idx.shape[0]
        if len_valid < min_traj_lenth:
            continue
        else:
            track_pair = {'pair_No': pair_id,
                          'xFollReal': tracks_dict['bbox'][valid_idx, 0],
                          'vFollReal': tracks_dict['xVelocity'][valid_idx],
                          'aFollReal': tracks_dict['xAcceleration'][valid_idx],
                          'vehicle_length': tracks_dict['bbox'][valid_idx, 2],
                          'precedingId_Foll': tracks_dict['precedingId'][valid_idx],
                          'frame_Foll': tracks_dict['frame'][valid_idx],
                          'id_Foll': tracks_dict['id'],
                          'class_foll': class_list.loc[int(tracks_dict['id'])]
                          }
            xLeaderReal = np.zeros(len_valid)
            vLeaderReal = np.zeros(len_valid)
            aLeaderReal = np.zeros(len_valid)
            for leader_id in np.unique(track_pair['precedingId_Foll']):
                leader_frame_idx = np.where(track_pair['precedingId_Foll'] == leader_id)[0]
                leader_frame = track_pair['frame_Foll'][leader_frame_idx]
                leader_tracks = tracks_by_id[int(leader_id)]
                assert leader_tracks['id'] == leader_id
                common_frames, _, idx_in_leader_tracks = np.intersect1d(leader_frame, leader_tracks['frame'],
                                                                        return_indices=True)
                assert len(leader_frame_idx) == len(idx_in_leader_tracks)

                flag = 0
                if np.unique(leader_tracks['followingId'][idx_in_leader_tracks])[0] != track_pair['id_Foll']:
                    flag = 1
                    break

                xLeaderReal[leader_frame_idx] = leader_tracks['bbox'][idx_in_leader_tracks, 0]
                vLeaderReal[leader_frame_idx] = leader_tracks['xVelocity'][idx_in_leader_tracks]
                aLeaderReal[leader_frame_idx] = leader_tracks['xAcceleration'][idx_in_leader_tracks]
            if flag == 1:
                continue
            track_pair['xLeaderReal'] = xLeaderReal
            track_pair['vLeaderReal'] = vLeaderReal
            track_pair['aLeaderReal'] = aLeaderReal

            # Kalman filter
            # tracks_for_KF = np.vstack((track_pair['xFollReal'], track_pair['vFollReal'], track_pair['aFollReal'])).T
            # tracks_for_KF = KF_tracks(tracks_for_KF, 3)
            # track_pair['xFollReal'] = tracks_for_KF[:, 0]
            # track_pair['vFollReal'] = tracks_for_KF[:, 1]
            # track_pair['aFollReal'] = tracks_for_KF[:, 2]
            #
            # tracks_for_KF = np.vstack(
            #     (track_pair['xLeaderReal'], track_pair['vLeaderReal'], track_pair['aLeaderReal'])).T
            # tracks_for_KF = KF_tracks(tracks_for_KF, 3)
            # track_pair['xLeaderReal'] = tracks_for_KF[:, 0]
            # track_pair['vLeaderReal'] = tracks_for_KF[:, 1]
            # track_pair['aLeaderReal'] = tracks_for_KF[:, 2]

            track_pair['dvReal'] = track_pair['vFollReal'] - track_pair['vLeaderReal']
            track_pair['sReal'] = track_pair['xLeaderReal'] - track_pair['xFollReal'] - track_pair['vehicle_length']

            track_pair['xLeaderReal_next'] = track_pair['xLeaderReal'][
                                             int(Config.frame_rate_orignial / Config.frame_rate):-1:int(
                                                 Config.frame_rate_orignial / Config.frame_rate)]
            track_pair['xFollReal_next'] = track_pair['xFollReal'][
                                           int(Config.frame_rate_orignial / Config.frame_rate):-1:int(
                                               Config.frame_rate_orignial / Config.frame_rate)]
            track_pair['sReal_next'] = track_pair['sReal'][
                                       int(Config.frame_rate_orignial / Config.frame_rate):-1:int(
                                           Config.frame_rate_orignial / Config.frame_rate)]
            track_pair['vFollReal_next'] = track_pair['vFollReal'][
                                           int(Config.frame_rate_orignial / Config.frame_rate):-1:int(
                                               Config.frame_rate_orignial / Config.frame_rate)]
            track_pair['vLeaderReal_next'] = track_pair['vLeaderReal'][
                                             int(Config.frame_rate_orignial / Config.frame_rate):-1:int(
                                                 Config.frame_rate_orignial / Config.frame_rate)]

            track_pair.update({
                'xFollReal': track_pair['xFollReal'][
                             :-int(Config.frame_rate_orignial / Config.frame_rate) - 1:int(
                                 Config.frame_rate_orignial / Config.frame_rate)],
                'vFollReal': track_pair['vFollReal'][
                             :-int(Config.frame_rate_orignial / Config.frame_rate) - 1:int(
                                 Config.frame_rate_orignial / Config.frame_rate)],
                'aFollReal': track_pair['aFollReal'][
                             :-int(Config.frame_rate_orignial / Config.frame_rate) - 1:int(
                                 Config.frame_rate_orignial / Config.frame_rate)],
                'vehicle_length': track_pair['vehicle_length'][
                                  :-int(Config.frame_rate_orignial / Config.frame_rate) - 1:int(
                                      Config.frame_rate_orignial / Config.frame_rate)],
                'frame_Foll': track_pair['frame_Foll'][
                              :-int(Config.frame_rate_orignial / Config.frame_rate) - 1:int(
                                  Config.frame_rate_orignial / Config.frame_rate)],
                'precedingId_Foll': track_pair['precedingId_Foll'][
                                    :-int(Config.frame_rate_orignial / Config.frame_rate) - 1:int(
                                        Config.frame_rate_orignial / Config.frame_rate)],
                'xLeaderReal': track_pair['xLeaderReal'][
                               :-int(Config.frame_rate_orignial / Config.frame_rate) - 1:int(
                                   Config.frame_rate_orignial / Config.frame_rate)],
                'vLeaderReal': track_pair['vLeaderReal'][
                               :-int(Config.frame_rate_orignial / Config.frame_rate) - 1:int(
                                   Config.frame_rate_orignial / Config.frame_rate)],
                'aLeaderReal': track_pair['aLeaderReal'][
                               :-int(Config.frame_rate_orignial / Config.frame_rate) - 1:int(
                                   Config.frame_rate_orignial / Config.frame_rate)],
                'dvReal': track_pair['dvReal'][
                          :-int(Config.frame_rate_orignial / Config.frame_rate) - 1:int(
                              Config.frame_rate_orignial / Config.frame_rate)],
                'sReal': track_pair['sReal'][
                         :-int(Config.frame_rate_orignial / Config.frame_rate) - 1:int(
                             Config.frame_rate_orignial / Config.frame_rate)],
            })
            assert track_pair['vFollReal_next'].shape[0] == track_pair['sReal'].shape[0]
            track_pair_list_temp.append(track_pair)
            pair_id += 1
    return track_pair_list_temp


# Load dataset with preprocessing
def read_training_data(base_path: str = '../data/highD/',
                       dataset_list: Optional[Sequence[int]] = None,
                       min_traj_lenth: int = Config.min_traj_lenth,
                       cache_dir: Optional[str] = None) -> CFPairList:
    """Read corrected pairs; preserve and never reuse historical pair caches.

    New caches use a versioned filename and payload. Stable recording/vehicle
    keys survive changes in extraction order; pair_No is only a list index.
    """
    if dataset_list is None:
        dataset_list = range(1, 61)
    if cache_dir is None:
        cache_dir = path.normpath(path.join(base_path, '../cache'))
    os.makedirs(cache_dir, exist_ok=True)
    track_pair_list = CFPairList()
    for dataset_id in dataset_list:
        cache = path.join(cache_dir, 'track_pair_list_temp{:02d}_MinLength{}_freq_{}_metadata_v2.pkl'.format(
            dataset_id, int(min_traj_lenth / Config.frame_rate_orignial), int(Config.frame_rate)))
        if path.exists(cache):
            with open(cache, 'rb') as fp:
                payload = pickle.load(fp)
            if not isinstance(payload, dict) or payload.get('preprocessing_version') != PREPROCESSING_VERSION:
                raise ValueError('Invalid preprocessing cache version: ' + cache)
            track_pair_list_temp = payload['pairs']
        else:
            tracks = read_dataset(base_path, dataset_id)
            meta_path = path.join(base_path, '{:02d}_tracksMeta.csv'.format(dataset_id))
            lane_change_ids, class_by_id = read_track_meta_csv(meta_path)
            track_pair_list_temp = load_CF_pairs(tracks, lane_change_ids, class_by_id, min_traj_lenth)
            for pair in track_pair_list_temp:
                pair['recording_id'] = int(dataset_id)
            with open(cache, 'wb') as output_file:
                pickle.dump({'preprocessing_version': PREPROCESSING_VERSION, 'pairs': track_pair_list_temp}, output_file)
            print('Saved', cache, ': done')
        for pair in track_pair_list_temp:
            pair['pair_No'] = len(track_pair_list)
            track_pair_list.append(pair)
    return track_pair_list


def read_id_60(base_path: str) -> Dict[str, Any]:
    """Return pair index 60 from the corrected dataset on every call."""
    return read_training_data(base_path=base_path)[60]


def KF_tracks(tracks, dim=3):
    dt = 1 / Config.frame_rate_orignial
    transition_matrices = np.array([[1, dt, 0],
                                    [0, 1, dt],
                                    [0, 0, 1]])

    observation_matrices = np.eye(dim)

    measurements = tracks

    initial_state_mean = np.array([measurements[0, 0], measurements[0, 1], measurements[0, 2]])

    transition_covariance = np.array([[1000, 0, 0],
                                      [0, 0.1, 0],
                                      [0, 0, 0.1]])
    observation_covariance = np.array([[1000, 0, 0],
                                       [0, 0.1, 0],
                                       [0, 0, 0.1]])

    kf = KalmanFilter(n_dim_state=dim, n_dim_obs=dim,
                      transition_matrices=transition_matrices,
                      transition_covariance=transition_covariance,
                      observation_matrices=observation_matrices,
                      observation_covariance=observation_covariance,
                      initial_state_mean=initial_state_mean)
    kf.em(measurements)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

    ###
    import matplotlib.pyplot as plt
    plt.plot(tracks[1:, 0] - tracks[:-1, 0], 'r')
    plt.plot(tracks[0:-1, 1] * dt, 'b')
    plt.show()

    tracks = smoothed_state_means
    plt.plot(tracks[1:, 0] - tracks[:-1, 0], 'r')
    plt.plot(tracks[0:-1, 1] * dt, 'b')
    plt.show()

    return tracks


if __name__ == '__main__':
    dataset_list = [*range(1, 61)]
    track_pair_list = read_training_data(base_path='./highD/', dataset_list=dataset_list,
                                         min_traj_lenth=50 * Config.frame_rate_orignial)
    track = read_id_60(base_path='./highD/')
    print("Total number of pairs:", len(track))
