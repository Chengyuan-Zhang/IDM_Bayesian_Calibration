import numpy as np
from time import gmtime, strftime
import os


class Config:
    # HighD preprocessing

    frame_rate_orignial = 25
    frame_rate = 5
    dt = 1 / frame_rate
    min_traj_lenth = 50 * frame_rate_orignial

    # simulator
    t0 = 5
    steps = 5
    min_acc = -5

    sim_eps_sigma = .333
    sim_eps_sigma_GP = .1

    sim_ring_tracks = 32
    sim_ring_time = 15000
    sim_ring_radius = 137
    sim_ring_init_speed = 11.6
    sim_veh_length = 5

    # Preserve the original 20 vehicles, regrouped by their true highD class.
    # Keys are (recording ID, follower ID), independent of cache/list order.
    # Historical outputs used the old 10/10 groups; refit before using v2 traces.
    car_interactive_pair_list = [
        (25, 273), (25, 352), (25, 309), (25, 314), (25, 355), (25, 361),
        (25, 412), (25, 1005), (26, 404), (25, 211), (25, 287), (25, 398),
        (25, 747), (25, 2486), (25, 2502), (25, 2524),
    ]
    truck_interactive_pair_list = [(25, 216), (25, 221), (26, 816), (26, 343)]
