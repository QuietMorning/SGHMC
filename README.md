# SGHMC

This repository contains the implementation of Stochastic Gradient Hamiltonian Monte Carlo sampling method in Python. The work is done by Weiting Miao, Chen An, and Ryan Fang. The implementation is based on the following paper: 

Chen, Tianqi, Emily Fox, and Carlos Guestrin. "Stochastic gradient hamiltonian monte carlo." International conference on machine learning. PMLR, 2014.


### Contents in the repository

`Airbnb/` contains data and code for the real world application on Airbnb dataset.

`Optimization/` contains code for numba optimization.

`bayesnn/` contains data and code for the real world application on MNIST dataset using Bayesian Neural Network.

`STA 663 First Try.ipynb` contains code for the SGHMC and relevant algorithms, as well as code for the applications of simulated data.

[To be completed]

### Package Installation and Usage 

The latest version (0.0.2) of the package could be installed using the command:  
`python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps maf==0.0.3`. 

Once you have installed the package, you could import the package and access the algorithms. There are four algorithms available in the package, `sghmc`, `hmc`, `sghmc_with_data`, and `sghmc_naive`. For example, you could use the following commands to use the algorithms in the package.  

```
from maf_sghmc import alg 
alg.sghmc(...)
alg.hmc(...)
alg.sghmc_with_data(...)
alg.sghmc_naive(...) 
``` 

Make sure that you have `numpy` package installed to use this package freely. And you could alwasy check the documentation and other specifications of the algorithms using `help(alg)` once you have imported `alg` from `maf_sghmc` using the command shown above.  


### Authors

Weiting Miao

Chen An

Ryan Fang

### Licence

This repository is licensed under the GNU General Public License v3.0.






