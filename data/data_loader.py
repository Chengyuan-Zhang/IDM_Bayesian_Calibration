import numpy as np
from config import Config

from data.read_csv import find_id_in_frame, read_dataset, read_track_meta_csv
import os
from os import path
import pickle
from pickle import UnpicklingError

# from pykalman import KalmanFilter

pair_id = 0


def load_CF_pairs(tracks, LC_id, class_list, min_traj_lenth=Config.min_traj_lenth):
    global pair_id
    track_pair_list_temp = []
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
                          'class_foll': class_list[tracks_dict['id']]
                          }
            xLeaderReal = np.zeros(len_valid)
            vLeaderReal = np.zeros(len_valid)
            aLeaderReal = np.zeros(len_valid)
            for leader_id in np.unique(track_pair['precedingId_Foll']):
                leader_frame_idx = np.where(track_pair['precedingId_Foll'] == leader_id)[0]
                leader_frame = track_pair['frame_Foll'][leader_frame_idx]
                leader_tracks = tracks[leader_id - 1]
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
def read_training_data(base_path='../CF_data/highD', dataset_list=[*range(1, 61)],
                       min_traj_lenth=Config.min_traj_lenth):
    """
    Data are from highD
    :param dataset_list:
    :param base_path:
    :return:
    """
    track_pair_list = []
    track_pair_list_temp = []
    for dataset_id in dataset_list:
        cache = path.join(base_path,
                          "../cache/track_pair_list_temp" + str("{:02d}".format(dataset_id)) + '_MinLength' + str(
                              int(min_traj_lenth / Config.frame_rate_orignial)) + '_freq_' + str(
                              "{:d}".format(int(Config.frame_rate))) + '.pkl')
        if path.exists(cache):
            try:
                fp = open(cache, 'rb')
                track_pair_list_temp = pickle.load(fp)
                fp.close()
                # print("Load", cache, ": done")
            except UnpicklingError:
                os.remove(cache)
                print('Removed broken cache:', cache)
        else:
            tracks = read_dataset(base_path, dataset_id)
            meta_dataset_path = base_path + str("{:02d}".format(dataset_id)) + "_tracksMeta.csv"
            LC_id, class_list = read_track_meta_csv(meta_dataset_path)
            track_pair_list_temp = load_CF_pairs(tracks, LC_id, class_list, min_traj_lenth)
            # dataset_path = base_path + str("{:02d}".format(dataset_id)) + "_tracks.csv"
            # id_in_frame = find_id_in_frame(dataset_path, base_path, dataset_id)
            output_file = open(cache, 'wb')
            pickle.dump(track_pair_list_temp, output_file)
            print("Saved", cache, ": done")
        track_pair_list.extend(track_pair_list_temp)
    #     print("Total number of pairs:", len(track_pair_list))
    return track_pair_list


def read_id_60(base_path):
    cache = path.join(base_path, "../cache/track_ID_60" + str("{:.02f}".format(Config.dt)) + ".pkl")
    if path.exists(cache):
        try:
            fp = open(cache, 'rb')
            track = pickle.load(fp)
            fp.close()
            print("Load", cache, ": done")
        except UnpicklingError:
            os.remove(cache)
            print('Removed broken cache:', cache)
    else:
        track = read_training_data(base_path=base_path, dataset_list=[*range(1, 61)])
        output_file = open(cache, 'wb')
        pickle.dump(track[60], output_file)
        print("Saved ", cache, ": done")
    return track


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
