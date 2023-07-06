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

    car_interactive_pair_list = [14, 35, 23, 25, 36, 38, 60, 90, 228, 232]
    truck_interactive_pair_list = [3, 4, 18, 52, 81, 144, 153, 162, 5, 241]
