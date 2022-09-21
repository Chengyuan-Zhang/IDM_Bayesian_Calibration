import numpy as np
from time import gmtime, strftime
import os


class Config:
    # HighD preprocessing

    frame_rate = 25 #2
    dt = 1 / frame_rate
    min_traj_lenth = 50 / dt

    # simulator
    # dt = 0.04
    t0 = 5
    steps = 5
    min_acc = -5

    sim_eps_sigma = .333
    sim_eps_sigma_GP = .1

    sim_ring_tracks = 37  # 20
    sim_ring_time = 6000 # 80000
    sim_ring_radius = 128  # 33.5
    sim_ring_init_speed = 11.6  # 8

    car_interactive_pair_list = range(35, 45)
    car_free_pair_list = [27510, 3162, 48615, 51449, 13449, 8651, 11696, 16560]
    truck_interactive_pair_list = [5, 6, 20, 57, 66, 73, 79, 80, 86, 90, 96]
    truck_free_pair_list = [12, 100, 102, 193, 198, 310, 680, 1637]