# SGHMC

This repository contains the implementation of Stochastic Gradient Hamiltonian Monte Carlo sampling method in Python. The work is done by Weiting Miao, Chen An, and Ryan Fang. The implementation is based on the following paper: 

Chen, Tianqi, Emily Fox, and Carlos Guestrin. "Stochastic gradient hamiltonian monte carlo." International conference on machine learning. PMLR, 2014.


### Contents in the repository

notebooks: contain the raw SGHMC algorithm and other relevant algorithms and functions as well as the optimizations.  
data: contain the data for the real-world rexamples using the developed algorithms, including the data for Bayes Neural Network Classifier application and the Airbnb dataset.  
src: contains the source code for all the algorithms and gradient functions. 
Tests: contains the test cases for the algorithm implementations. 

[To be completed]

### Package Installation and Usage 

The latest version (0.0.4) of the package could be installed using the command:  
`python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps maf==0.0.4`. 

Once you have installed the package, you could import the package and access the algorithms. There are four algorithms available in the package, `sghmc_with_grad`, `hmc`, `sghmc_with_data`, and `sghmc_naive`. For example, you could use the following commands to use the algorithms in the package.  

```python
from maf_sghmc import alg 
alg.sghmc_with_grad(...)
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
**Make sure that you have `numpy` package installed to use this package freely.** And you could alwasy check the documentation and other specifications of the functions using `help(alg)` or `help(grad)` once you have imported `alg` or `grad` modules from `maf_sghmc` using the command shown above. If you would like to access all the functions directly without mentioning `alg` or `grad`, you could use the following commmand. 

```python 
from maf_sghmc.alg import * 
from maf_sghmc.grad import *
sghmc_with_grad(...)
U_tilde(...)
```



### Authors

Weiting Miao

Chen An

Ryan Fang

### Licence

This repository is licensed under the GNU General Public License v3.0.






