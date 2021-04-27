import numpy as np


def U_grad_tilde(theta, data, logp_data_grad, logp_prior_grad, mb_size):
    """
    Stochastic gradient estimates of posterior density with respect to distribution parameters
    Based on a minibatch D_hat sampled uniformly at random from D
    ------------------------

    Dimensions
    -----------
    n: number of observations from the data
    m: dimension of the data

    Input
    -----
    D: n-by-m np array
        Dataset

    logp_data_grad: callable 'logp_data_grad(data, theta)'
        Gradient of likelihood of the data with respect to distribution parameters

    logp_prior_grad: callable 'logp_prior_grad(theta)'
        Gradient of prior with respect to distribution parameters

    mb_size: int
        Size of the minibatch

    theta: d-by-1 np array
        Distribution parameters

    Output
    -----
    U_tilde: d-by-1 np array
        Stochastic gradient estimates of posterior density with respect to distribution parameters
    """
    n = data.shape[0]
    data_hat = data[np.random.choice(range(n), size=mb_size, replace=False)]
    U_tilde = -(n / mb_size) * logp_data_grad(data_hat, theta) - logp_prior_grad(theta)
    return U_tilde


def U_tilde(theta, data, logp_data, logp_prior, mb_size):
    """
    Stochastic gradient estimates of posterior density with respect to distribution parameters
    Based on a minibatch D_hat sampled uniformly at random from D
    ------------------------

    Dimensions
    -----------
    n: number of observations from the data
    m: dimension of the data

    Input
    -----
    D: n-by-m np array
        Dataset

    logp_data: callable 'logp_data(data, theta)'
        Likelihood of the data with respect to distribution parameters

    logp_prior: callable 'logp_prior(theta)'
        Prior with respect to distribution parameters

    mb_size: int
        Size of the minibatch

    theta: d-by-1 np array
        Distribution parameters

    Output
    -----
    U_tilde: d-by-1 np array
        Stochastic gradient estimates of posterior density with respect to distribution parameters
    """
    n = data.shape[0]
    data_hat = data[np.random.choice(range(n), size=mb_size, replace=False)]
    U_tilde = -(n / mb_size) * logp_data(data_hat, theta) - logp_prior(theta)
    return U_tilde