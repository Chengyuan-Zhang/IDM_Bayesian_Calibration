## *(Update: the latest version will be uploaded in June 2023)*

# Bayesian Calibration of IDM 

### This repo provides the implementation of MA-IDM and Bayesian IDM in paper ''Bayesian Calibration of Intelligent Driver Model.''


<center>
  <img src="./Figs/idm_pgm.png" width="85%" />
</center>

## How to run

We calibrate our model on [highD dataset](https://www.highd-dataset.com/). The preprocessed data are stored
in ```.data/cache/*.pkl```. To implement your preprocessing procedures, please download and store the original data in
the ```./data/highD``` folder, e.g., it should contains ```./data/highD/**_tracks.csv```
, ```./data/highD/**_tracksMeta.csv```, and ```.data/highD/**_recordingMeta.csv```.

We develop the probabilistic graphical models (PGMs) with [PyMC](https://github.com/pymc-devs/pymc), please install
**PyMC4** by following their instructions. The PGMs in this work are implemented
in: ```./PGM/Bayesian_IDM_(hierarchy)_(driver_type).ipynb``` and ```./PGM/MA_IDM_(hierarchy)_(driver_type).ipynb```;

To visualize the result and conduct the single-vehicle stochastic simulations: ```./PGM/Results_analysis.ipynb.ipynb```;

To conduct the multi-vehicle ring-road simulations, run  ```./Simulator/simulation_ring.py```

## Read More
- Access our project via: [[arxiv](https://arxiv.org/abs/2210.03571)].
- Presentation: [[recording](https://youtu.be/GIqcL6I7MsU)].
- What is LKJ Cholesky Covariance Priors: [https://www.pymc.io/projects/examples/en/latest/case_studies/LKJ.html](https://www.pymc.io/projects/examples/en/latest/case_studies/LKJ.html).

## Contact

**If you have any questions please feel free to contact
us:  [Chengyuan Zhang](https://chengyuan-zhang.github.io/) (<enzozcy@gmail.com>)
and [Lijun Sun](https://lijunsun.github.io/) (<lijun.sun@mcgill.ca>).**
