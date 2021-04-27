import numpy as np
import numba


def sghmc_with_grad(U_grad, theta_init, M, C, V_hat, epsilon, T, m):
    """
    Stochastic Gradient Hamiltonian Monte Carlo Sampling.
    Based on Chen, Tianqi, Emily Fox, and Carlos Guestrin (2014)
    --------------------

    Dimensions
    -----------
    d: number of parameters
    T: length of samples

    Input
    ------
    U_grad: callable
        Stochastic gradient estimates of posterior density with respect to distribution parameters
        'U_grad_tilde(D, logp_data_grad, logp_prior_grad, mb_size, theta)' when the gradient is unknown

    theta_init: d-by-1 np array
        The inital sampling point

    M: d-by-d np array
        A mass matrix

    C: d-by-d np array
        A user specified friction term, should be greater than B_hat = 0.5*epsilon*V_hat
        in the sense of in the sense of positive semi-definiteness

    V_hat: d-by-d np array
        Empirical Fisher information of theta

    epsilon: float
        Step size

    T: int
        Number of samples drawn from the desired distribution

    m: int
        Number of steps for each draw


    Output
    ------
    theta_s: T-by-d np array
        Draws sampled from the desired distrition

    r_s: T-by-d np array
        Draws of the momentun variables

    """
    # sampling environment parameters setup
    d = len(theta_init)
    theta_s = np.zeros((T, d))
    r_s = np.zeros((T, d))
    theta_s[0] = theta_init
    M_inv = np.linalg.inv(M)
    B_hat = 0.5 * epsilon * V_hat

    if d > 1:
        assert np.all(np.linalg.eigvals(C-B_hat) >= 0), "Check Documentation of C and update C to make sure that C - B_hat is positive semidefinite."
        assert M.shape[0] == M.shape[1], "M must be a square matrix."
        assert C.shape[0] == C.shape[1], "M must be a square matrix."
        assert V_hat.shape[0] == V_hat.shape[1], "M must be a square matrix."
        sd = np.linalg.cholesky(2 * epsilon * (C - B_hat))
        r_s = np.random.multivariate_normal(np.zeros(d), M, size=T)
    if d == 1:
        assert C-B_hat >= 0, "Check Documentation of C and update C to make sure that C - B_hat is positive semidefinite."
        sd = np.sqrt(2 * epsilon * (C - B_hat))
        r_s = np.sqrt(M) * np.random.randn(T).reshape(T, 1)

    # U_grad_vec = numba.jit(float64[:](float64[:]), nopython=True)(U_grad)
    U_grad_jit = numba.njit(U_grad)

    # update parameters
    @numba.jit(nopython=True)
    def update(theta_s, r_s):
        for t in range(T - 1):
            theta0 = theta_s[t]
            r0 = r_s[t]
            for i in range(m):
                theta0 = theta0 + epsilon * M_inv @ r0
                r0 = r0 - epsilon * U_grad_jit(theta0) - epsilon * C @ M_inv @ r0 + sd @ np.random.randn(d)
            theta_s[t + 1] = theta0
        return theta_s

    theta_s = update(theta_s, r_s)

    return [theta_s, r_s]


def sghmc_with_data(data, logp_data_grad, logp_prior_grad, mb_size, theta_init, M, C, V_hat, epsilon, T, m):
    """
    Stochastic Gradient Hamiltonian Monte Carlo Sampling when the Gradient Function is Unknown.
    Based on Chen, Tianqi, Emily Fox, and Carlos Guestrin (2014)
    ----------------------------

    Dimensions:
    -----------
    d: number of parameters
    T: length of samples
    p: dimension of the data

    Input:
    ------
    data: T-by-p np array
        Training data.

    logp_data_grad: callable function
        Calculate the gradient of log likelihood function of data given the parameters.

    logp_prior_grad: callable function
        Calculate the gradient of the log prior.

    mb_size: int,
        mini batch size.

    theta_init: d-by-1 np array
        The inital sampling point

    M: d-by-d np array
        A mass matrix

    C: d-by-d np array
        A user specified friction term, should be greater than B_hat = 0.5*epsilon*V_hat
        in the sense of in the sense of positive semi-definiteness

    V_hat: d-by-d np array
        Empirical Fisher information of theta

    epsilon: float
        Step size

    T: int
        Number of samples drawn from the desired distribution

    m: int
        Number of steps for each draw

    Output
    ------
    theta_s: T-by-d np array
        Draws sampled from the desired distrition

    r_s: T-by-d np array
        Draws of the momentun variables

    """

    # Sampling environment parameters setup
    d = theta_init.shape[0]
    theta_s = np.zeros((T, d))
    r_s = np.zeros((T, d))
    theta_s[0] = theta_init
    M_inv = np.linalg.inv(M)
    B_hat = 0.5 * epsilon * V_hat

    n = data.shape[0]
    data_hat = data[np.random.choice(range(n), size=mb_size, replace=False)]

    if d > 1:
        assert np.all(np.linalg.eigvals(
            C - B_hat) >= 0), "Check Documentation of C and update C to make sure that C - B_hat is positive semidefinite."
        assert M.shape[0] == M.shape[1], "M must be a square matrix."
        assert C.shape[0] == C.shape[1], "M must be a square matrix."
        assert V_hat.shape[0] == V_hat.shape[1], "M must be a square matrix."
        sd = np.linalg.cholesky(2 * epsilon * (C - B_hat))
        r_s = np.random.multivariate_normal(np.zeros(d), M, size=T)
    if d == 1:
        assert C - B_hat >= 0, "Check Documentation of C and update C to make sure that C - B_hat is positive semidefinite."
        sd = np.sqrt(2 * epsilon * (C - B_hat))
        r_s = np.sqrt(M) * np.random.randn(T).reshape(T, 1)

    # Update Parameters
    for t in range(T - 1):
        theta0 = theta_s[t]
        r0 = r_s[t]
        for i in range(m):
            theta0 = theta0 + epsilon * M_inv @ r0
            r0 = r0 + epsilon * ((n / mb_size) * logp_data_grad(data_hat, theta0) + logp_prior_grad(
                theta0)) - epsilon * C @ M_inv @ r0 + sd @ np.random.randn(d)
        theta_s[t + 1] = theta0

    return [theta_s, r_s]


def sghmc_naive(U_grad, U, theta_init, M, V_hat, epsilon, T, m, MH=True, resample=False):
    """
    Naive Stochastic Gradient Hamiltonian Monte Carlo Sampling.
    --------------------

    Input
    ------
    U_grad: callable
        Stochastic gradient estimates of posterior density with respect to distribution parameters
        'U_grad_tilde(D, logp_data_grad, logp_prior_grad, mb_size, theta)' when the gradient is unknown

    U: callable
        Potential energy function

    theta_init: d-by-1 np array
        The inital sampling point

    M: d-by-d np array
        A mass matrix

    V_hat: d-by-d np array
        Empirical Fisher information of theta

    epsilon: float
        Step size

    T: int
        Number of samples drawn from the desired distribution

    m: int
        Number of steps for each draw

    MH: Boolean
        Whether to incorporate a Metropolis-Hasting correction

    resample: Boolean
        Whether to resample the momentum variables

    Output
    ------
    theta_s: T-by-d np array
        Draws sampled from the desired distrition

    r_s: T-by-d np array
        Draws of the momentun variables

    """

    d = len(theta_init)
    theta_s = np.zeros((T, d))
    theta_s[0] = theta_init
    M_inv = np.linalg.inv(M)
    B_hat = 0.5 * epsilon * V_hat

    if d > 1:
        sd = np.linalg.cholesky(2 * epsilon * B_hat)
        r_s = np.random.multivariate_normal(np.zeros(d), M, size=T)
    elif d == 1:
        sd = np.sqrt(2 * epsilon * B_hat)
        r_s = np.sqrt(M) * np.random.randn(T).reshape(T, 1)

    for t in range(T - 1):
        theta0 = theta_s[t]
        r0 = r_s[t]

        for i in range(m):
            theta0 = theta0 + epsilon * np.dot(M_inv, r0)
            r0 = r0 - epsilon * U_grad(theta0) + np.dot(sd, np.random.randn(d))

        ## M-H correction
        if MH == True:
            u = np.random.rand(1)
            H1 = U(theta_s[t]) + 0.5 * np.dot(np.dot(r_s[t], M_inv), r_s[t])
            H2 = U(theta0) + 0.5 * np.dot(np.dot(r0, M_inv), r0)
            rho = np.exp(H1 - H2)
            if u < np.minimum(1, rho):
                theta_s[t + 1] = theta0
                if resample == False:
                    r_s[t + 1] = r0
            else:
                theta_s[t + 1] = theta_s[t]
                if resample == False:
                    r_s[t + 1] = r_s[t]
        else:
            theta_s[t + 1] = theta0
            if resample == False:
                r_s[t + 1] = r0

    return [theta_s, r_s]


def hmc(U_grad, U, theta_init, M, epsilon, T, m, MH=True, resample=False):
    """
    Hamiltonian Monte Carlo Sampling.
    --------------------

    Input
    ------
    U_grad: callable
        Gradient of U with respect to distribution parameters

    U: callable
        Potential energy function

    theta_init: d-by-1 np array
        The inital sampling point

    M: d-by-d np array
        A mass matrix

    epsilon: float
        Step size

    T: int
        Number of samples drawn from the desired distribution

    m: int
        Number of steps for each draw

    MH: Boolean
        Whether to incorporate a Metropolis-Hasting correction

    resample: Boolean
        Whether to resample the momentum variables

    Output
    ------
    theta_s: T-by-d np array
        Draws sampled from the desired distrition

    r_s: T-by-d np array
        Draws of the momentun variables

    """
    d = len(theta_init)
    theta_s = np.zeros((T, d))
    theta_s[0] = theta_init
    M_inv = np.linalg.inv(M)

    if d > 1:
        r_s = np.random.multivariate_normal(np.zeros(d), M, size=T)
    elif d == 1:
        r_s = np.sqrt(M) * np.random.randn(T).reshape(T, 1)

    for t in range(T - 1):

        theta0 = theta_s[t]
        r0 = r_s[t]

        ## leapfrog
        r0 = r0 - 0.5 * epsilon * U_grad(theta0)
        for i in range(m - 1):
            theta0 = theta0 + epsilon * np.dot(M_inv, r0)
            r0 = r0 - epsilon * U_grad(theta0)
        theta0 = theta0 + epsilon * np.dot(M_inv, r0)
        r0 = r0 - 0.5 * epsilon * U_grad(theta0)

        ## M-H correction
        if MH == True:
            u = np.random.rand(1)
            H1 = U(theta_s[t]) + 0.5 * np.dot(np.dot(r_s[t], M_inv), r_s[t])
            H2 = U(theta0) + 0.5 * np.dot(np.dot(r0, M_inv), r0)
            rho = np.exp(H1 - H2)
            if u < np.minimum(1, rho):
                theta_s[t + 1] = theta0
                if resample == False:
                    r_s[t + 1] = r0
            else:
                theta_s[t + 1] = theta_s[t]
                if resample == False:
                    r_s[t + 1] = r_s[t]
        else:
            theta_s[t + 1] = theta0
            if resample == False:
                r_s[t + 1] = r0

    return [theta_s, r_s]

