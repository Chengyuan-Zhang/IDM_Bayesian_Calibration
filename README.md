# Bayesian Calibration of IDM

This repo provides the implementation of MA-IDM and Bayesian IDM in ''**Bayesian Calibration of Intelligent Driver Model**,'' as well as the dynamic IDM (AR+IDM) in our latest paper "**Calibrating Car-following Models via Bayesian Dynamic Regression.**" Besides, the repo provides the implementation of the multi-vehicle ring-road simulations.

<center>
  <img src="./Figs/idm_pgm.png" width="85%" />
</center>

## How to run

We calibrate our model on [highD dataset](https://www.highd-dataset.com/). The preprocessed data are stored
in ```data/cache/*.pkl```. To implement your preprocessing procedures, please download and store the original data in
the ```data/highD``` folder, e.g., it should contain ```data/highD/**_tracks.csv```
, ```data/highD/**_tracksMeta.csv```, and ```data/highD/**_recordingMeta.csv```.

We develop the probabilistic graphical models (PGMs) with [PyMC](https://github.com/pymc-devs/pymc). Please install
**PyMC4** by following their instructions:
```
conda create -c conda-forge -n pymc_env "pymc=4"
conda activate pymc_env
```

The PGMs in this work are implemented
in: ```PGM_highD/Bayesian_IDM_(hierarchy)_(driver_type).ipynb```, ```PGM_highD/MA_IDM_(hierarchy)_(driver_type).ipynb```,
and ```PGM_highD/AR_IDM_(hierarchy)_(driver_type).ipynb```;

To visualize the result and conduct the single-vehicle stochastic
simulations: ```PGM_highD(_joint)/Stochastic_simulation_GP.ipynb```
and ```PGM_highD(_joint)/Stochastic_simulation_AR.ipynb```;

To conduct the multi-vehicle ring-road simulations, run  ```Simulator/simulation_ring.py```

## Read More

- Access our project via: GP+IDM [[arXiv](https://arxiv.org/abs/2210.03571)] and AR+IDM [[arXiv](https://arxiv.org/pdf/2307.03340.pdf)].
- Presentation: [[recording](https://youtu.be/GIqcL6I7MsU)].
- What is LKJ Cholesky Covariance
  Priors: [[https://tomicapretto.com/posts/2022-06-12_lkj-prior/](https://tomicapretto.com/posts/2022-06-12_lkj-prior/)].

## Contact

**If you have any questions, please feel free to contact
us:  [Chengyuan Zhang](https://chengyuan-zhang.github.io/) (<enzozcy@gmail.com>)
and [Lijun Sun](https://lijunsun.github.io/) (<lijun.sun@mcgill.ca>).**
