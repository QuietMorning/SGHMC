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

Once you have installed the package, you could import the package and access the algorithms. There are four algorithms available in the package, `sghmc_with_grad`, `hmc`, `sghmc_with_data`, and `sghmc_naive`. For example, you could use the following commands to use the algorithms in the package.  

```python
from maf_sghmc import alg 
alg.sghmc(...)
alg.hmc(...)
alg.sghmc_with_data(...)
alg.sghmc_naive(...) 
``` 
There are also two gradient algorithms available in the package, which are called `U_tilde` and `U_grad_tilde`. You could assess these two algorithms using the following command. 

```python
from maf_sghmc import grad 
grad.U_tilde(...)
grad.U_grad_tilde(...)
```
Make sure that you have `numpy` and `numba` packages installed to use this package freely. And you could alwasy check the documentation and other specifications of the functions using `help(alg)` or `help(grad)` once you have imported `alg` or `grad` modules from `maf_sghmc` using the command shown above. If you would like to access all the functions directly without mentioning `alg` or `grad`, you could use the following commmand. 

```python 
from maf_sghmc.alg import * 
from maf_sghmc.grad import *
sghmc(...)
U_tilde(...)
```



### Authors

Weiting Miao

Chen An

Ryan Fang

### Licence

This repository is licensed under the GNU General Public License v3.0.






