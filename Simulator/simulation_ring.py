import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy

from matplotlib.collections import LineCollection

from sklearn.metrics.pairwise import rbf_kernel

import pickle
from pickle import UnpicklingError
import os
from os import path

import warnings

warnings.filterwarnings("ignore")

from config import Config

np.random.seed(1116)


def a_IDM(VMAX, DSAFE, TSAFE, AMAX, AMIN, DELTA, s, vt, dv):
    sn = DSAFE + np.max((vt * TSAFE + vt * dv / (2 * np.sqrt(AMAX * AMIN)), 0))
    a = AMAX * (1 - (vt / VMAX) ** DELTA - (sn / s) ** 2)
    return a


def load_trace(cache):
    if path.exists(cache):
        fp = open(cache, 'rb')
        tr = pickle.load(fp)
        fp.close()
        print("Load trace", cache, ": done!")
    else:
        print("No file named ", cache)
    return tr


def RBF_kernel(XA, XB, l=33, A=0.2):
    X_dist = A ** 2 * rbf_kernel(XA[:, [0]], XB[:, [0]], gamma=0.5 / l ** 2)
    return X_dist


# Gaussian process posterior
def GP(X1, y1, X2, kernel_func):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1),
    and the prior kernel function.
    """
    # Kernel of the observations
    Σ11 = kernel_func(X1, X1)
    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X1, X2)
    # Solve
    solved = scipy.linalg.solve(Σ11, Σ12).T
    # Compute posterior mean
    μ2 = solved @ y1
    # Compute the posterior covariance
    Σ22 = kernel_func(X2, X2, l=int(33 * 0.04 / Config.dt), A=0.2)
    Σ2 = Σ22 - (solved @ Σ12)
    return μ2, Σ2  # mean, covariance


def ring_scenarios(GP_flag=True):
    cache = "../PGM/cache/GP_IDM_hierarchical-exp.pkl"
    tr = load_trace(cache)
    id_idx = np.random.choice(range(20000), Config.sim_ring_tracks, replace=False)
    driver_idx = np.random.choice(range(8), Config.sim_ring_tracks, replace=True)
    params = tr.posterior.mu_d[0, id_idx, driver_idx, :]

    tracks = [None] * Config.sim_ring_tracks
    init_pos_theta = np.linspace(0, 2 * np.pi, Config.sim_ring_tracks + 1)
    R = Config.sim_ring_radius
    N = Config.sim_ring_time
    for id in range(Config.sim_ring_tracks):
        tracks[id] = {'pos_s': np.zeros(N),
                      'pos_theta': np.zeros(N),
                      'a': np.zeros(N),
                      'v': np.zeros(N)}

    if GP_flag:
        # we select a window size of 3l = 99:
        sample_dim = int(3 * 33 * 0.04 / Config.dt)
        #         a_GP_sim = np.zeros(N)

        t = np.array(range(-sample_dim, N)).reshape(-1, 1)
        X1 = np.expand_dims(np.linspace(-sample_dim, 0, sample_dim), 1)
        y1 = np.zeros((sample_dim))
        X2 = np.expand_dims(np.linspace(0, N, N), 1)
        μ2, Σ2 = GP(X1, y1, X2, RBF_kernel)
        a_GP_sim = np.random.multivariate_normal(mean=μ2, cov=Σ2, size=Config.sim_ring_tracks)

    for t in range(N - 1):
        for id in range(Config.sim_ring_tracks):
            if id == Config.sim_ring_tracks - 1:
                id_pre = 0
            else:
                id_pre = id + 1

            if t == 0:
                tracks[id]['pos_s'][0] = init_pos_theta[id] * R
                tracks[id]['pos_theta'][0] = tracks[id]['pos_s'][0] / R
                tracks[id]['v'][0] = Config.sim_ring_init_speed

            s = tracks[id_pre]['pos_s'][t] - tracks[id]['pos_s'][t] - Config.sim_veh_length
            dv = tracks[id]['v'][t] - tracks[id_pre]['v'][t]
            v = tracks[id]['v'][t]
            if s < 0:
                s += Config.sim_ring_radius * 2 * np.pi
            if GP_flag:
                a_eps = a_GP_sim[id, t - 1] + np.random.normal(0, 0.1)
                a = np.max((a_IDM(params[id, id, 0], params[id, id, 1], params[id, id, 2], params[id, id, 3],
                                  params[id, id, 4], 4, s, v, dv) + a_eps, -6))
            else:
                a_eps = np.random.normal(0, 0.3)
                a = np.max((a_IDM(33.3, 2, 1.6, .73, 1.67, 4, s, v, dv) + a_eps, -6))
                # 33.3, 2, 1.6, .73, 1.67, 4
                # VMAX, DSAFE, TSAFE, AMAX, AMIN, DELTA
            v_new = np.max((v + a * Config.dt, 0))
            dx = 0.5 * (v_new + v) * Config.dt
            tracks[id]['v'][t + 1] = v_new
            tracks[id]['a'][t] = a
            tracks[id]['pos_s'][t + 1] = (tracks[id]['pos_s'][t] + dx) % (
                    2 * np.pi * R)
            tracks[id]['pos_theta'][t + 1] = tracks[id]['pos_s'][t] / R
        if (t + 1) % 100 == 0:
            print("Simulation:", t + 1, "/", N)
    return tracks


def plot_GP_IDM_simulation(tracks, GP_flag):
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "Times New Roman"
    matplotlib.rcParams['font.size'] = 26
    matplotlib.rcParams['font.family'] = 'Times New Roman'

    fig, ax = plt.subplots(figsize=(12, 4))
    # fig = plt.figure(figsize=(16, 10))

    tt = np.arange(Config.sim_ring_time)

    for id in range(Config.sim_ring_tracks):
        pos_s = tracks[id]['pos_s']
        pos_theta = tracks[id]['pos_theta']
        a = tracks[id]['a']
        v = tracks[id]['v']

        ###################################
        # ax = plt.subplot(3, 1, 1)
        points = np.array([tt * Config.dt, pos_s]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        idx_delete = np.where(segments[:, 0, 1] - segments[:, 1, 1] > 0)
        segments = np.delete(segments, idx_delete, axis=0)
        norm = plt.Normalize(0, 18)  # v.min(), v.max())
        lc = LineCollection(segments, cmap='jet_r', norm=norm)
        # Set the values used for colormapping
        lc.set_array(v)
        lc.set_linewidth(1)
        line = ax.add_collection(lc)
        plt.xlim((0, tt[-1] * Config.dt))
        plt.ylim((0, Config.sim_ring_radius * 2 * np.pi))

        # ###################################
        # ax2 = plt.subplot(3, 1, 2)
        # ax2.plot(tt[:-1] * Config.dt, pos_theta_mask)
        # plt.xlim((0, tt[-1] * Config.dt))
        # plt.ylim((0, 2 * np.pi))
        #
        # ax3 = plt.subplot(3, 1, 3)
        # ax3.plot(tt * Config.dt, a)
        # plt.xlim((0, tt[-1] * Config.dt))

    cax = ax.inset_axes([1.06, 0., 0.015, 1.0], transform=ax.transAxes)
    fig.colorbar(line, ax=ax, cax=cax)
    plt.text(1.01, 0.22, 'speed (m/s)', rotation='vertical', transform=ax.transAxes)
    plt.xlabel('time (s)')
    plt.ylabel('space (m)')

    ax.set_rasterized(True)
    plt.tight_layout()
    if GP_flag:
        fig.savefig('../Figs/Sim_GP_IDM.pdf', dpi=300)
    else:
        fig.savefig('../Figs/Sim_IDM.pdf', dpi=300)
    plt.show()


def fundamental_area(tracks, L, T):
    T_all = tracks[0]['a'].shape[0]
    N = round(T_all / T)
    N_s = len(range(0, 800 - L, L))
    density = np.zeros((N - 1, N_s))
    speed = np.zeros((N - 1, N_s))

    for count, lower in enumerate(range(0, 800 - L, L)):
        upper = lower + L
        for tt in range(N - 1):
            d_all = 0
            t_all = 0
            for id in range(Config.sim_ring_tracks):
                for t in range(T * tt, T * (tt + 1)):
                    if t == 8000:
                        print('a')
                    x = tracks[id]['pos_s'][t]
                    v = tracks[id]['v'][t]
                    if x < upper and x > lower:
                        d_all += v * Config.dt
                        t_all += Config.dt
            density[tt, count] = t_all / (L * T * Config.dt)
            if t_all == 0:
                speed[tt, count] = 100
            else:
                speed[tt, count] = d_all / t_all

    flow = density * speed
    return speed * 3.6, density * 1000, flow * 3600


def plot_speed_density(savefig):
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "Times New Roman"
    matplotlib.rcParams['font.size'] = 22
    matplotlib.rcParams['font.family'] = 'Times New Roman'

    alpha = 0.4
    L = 50
    T = int(60 / Config.dt)

    tracks = load_tracks(GP_flag=False)
    speed, density, flow = fundamental_area(tracks, L, T)
    tracks_GP = load_tracks(GP_flag=True)
    speed_GP, density_GP, flow_GP = fundamental_area(tracks_GP, L, T)

    speed = speed[1:]
    density = density[1:]
    flow = flow[1:]
    speed_GP = speed_GP[2:]
    density_GP = density_GP[2:]
    flow_GP = flow_GP[2:]

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(density, speed, marker='s', color='b', alpha=alpha)
    plt.scatter(density_GP, speed_GP, marker='o', color='r', alpha=alpha)
    plt.scatter(density[-1], speed[-1], marker='s', color='b', alpha=1, label='homogeneous IDM')
    plt.scatter(density_GP[-1], speed_GP[-1], marker='o', color='r', alpha=1, label='heterogeneous IDM')
    plt.xlabel('density (veh/km)')
    plt.ylabel('speed (km/h)')
    plt.legend(loc='best')
    plt.tight_layout()
    if savefig:
        fig.savefig('../Figs/Sim_vk_fundamental_diagram.pdf', dpi=300)
    plt.show()

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(density * speed,
                speed, marker='s',
                color='b',
                alpha=alpha)
    plt.scatter(density_GP * speed_GP,
                speed_GP, marker='o',
                color='r',
                alpha=alpha)
    plt.scatter(density[-1] * speed[-1],
                speed[-1], marker='s', color='b',
                alpha=1, label='homogeneous IDM')
    plt.scatter(density_GP[-1] * speed_GP[-1],
                speed_GP[-1], marker='o', color='r',
                alpha=1, label='heterogeneous IDM')
    plt.xlabel('flow (veh/h)')
    plt.ylabel('speed (km/h)')
    plt.legend(loc='best')
    plt.tight_layout()
    if savefig:
        fig.savefig('../Figs/Sim_qv_fundamental_diagram.pdf', dpi=300)
    plt.show()

    fig = plt.figure(figsize=(7, 5))
    plt.scatter(density,
                density * speed,
                marker='s', color='b', alpha=alpha)
    plt.scatter(density_GP,
                density_GP * speed_GP,
                marker='o',
                color='r',
                alpha=alpha)
    plt.scatter(density[-1], density[-1] * speed[-1],
                marker='s', color='b',
                alpha=1, label='homogeneous IDM')
    plt.scatter(density_GP[-1],
                density_GP[-1] * speed_GP[-1],
                marker='o', color='r',
                alpha=1, label='heterogeneous IDM')
    plt.ylabel('flow (veh/h)')
    plt.xlabel('density (veh/km)')

    plt.legend(loc='best')
    plt.tight_layout()
    if savefig:
        fig.savefig('../Figs/Sim_qk_fundamental_diagram.pdf', dpi=300)
    plt.show()


def load_tracks(GP_flag):
    if GP_flag:
        cache = "../Simulator/cache/GP_IDM_hierarchical-sim_37_8000_128_11_noisy.pkl"
    else:
        cache = "../Simulator/cache/IDM-sim_37_8000_128_11_noisy2.pkl"

    if path.exists(cache):
        try:
            fp = open(cache, 'rb')
            tracks = pickle.load(fp)
            fp.close()
            print("Load trace", cache, ": done!")
        except UnpicklingError:
            os.remove(cache)
            print('Removed broken cache:', cache)
    else:
        tracks = ring_scenarios(GP_flag)
        output_file = open(cache, 'wb')
        pickle.dump(tracks, output_file)
        output_file.close()
        print("Generated and Saved", output_file, ": done!")
    return tracks


if __name__ == '__main__':
    GP_flag = False
    savefig = False
    tracks = load_tracks(GP_flag)
    plot_GP_IDM_simulation(tracks, GP_flag)
    plot_speed_density(savefig)
