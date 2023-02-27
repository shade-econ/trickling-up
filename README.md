This repository contains materials for "The Trickling Up of Excess Savings" (Auclert, Rognlie, Straub 2023, forthcoming AEA P&P). You can [download it as a zip by clicking here](https://github.com/shade-econ/trickling-up/archive/refs/heads/main.zip).

The main content is in the Jupyter notebook `trickling up model.ipynb`, which performs all calculations in the paper to generate Figure 4 and Table 1. You can easily edit this notebook to do new simulations with your own parameters—for instance, by editing the parameters for the main calibration in cell 4.

Note that the extensions of the baseline model rely on the support file `ct_re_solver.py`, which is a simple linear rational expectations solver in continuous time.

The `replication` folder contains all figures from the paper and also a `Figure1.ipynb` notebook, which uses data from the `Data` subfolder to generate our motivating Figure 1 on excess savings.
