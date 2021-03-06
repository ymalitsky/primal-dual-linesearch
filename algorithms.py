
import numpy as np
import scipy as sp
import numpy.linalg as LA
from itertools import count
from time import process_time


# Proximal Gradient Methods ###

def prox_grad(J,  d_f, prox_g, x0, la, numb_iter=100):
    """
    Minimize function F(x) = f(x) + g(x) by proximal gradient method
    with fixed stepsize la.  Takes J as some evaluation function for
    comparison.
    """
    begin = process_time()
    values = [J(x0)]
    time_list = [process_time() - begin]
    x = x0
    for i in range(numb_iter):
        x = prox_g(x - la * d_f(x), la)
        values.append(J(x))
        time_list.append(process_time() - begin)

    end = process_time()
    print("---- PGM ----")
    print("Time execution:", end - begin)
    return time_list, values, x


def prox_grad_linesearch(J, f, d_f, prox_g, x0, la0=1, numb_iter=100):
    """
    Minimize function F(x) = f(x) + g(x) by proximal gradient method
    with linesearch.  Takes J as some evaluation function for
    comparison.
    """
    begin = process_time()
    values = [J(x0)]
    time_list = [process_time() - begin]
    fx0 = f(x0)
    iterates = [time_list, values, x0, fx0, la0, 0, 0, 0]

    def iter_T(time_list, values, x, fx, la, n_f, n_df, n_prox):
        dfx = d_f(x)
        sigma = 0.7
        for j in count(0):
            x1 = prox_g(x - la * dfx, la)
            fx1 = f(x1)
            if fx1 <= fx + np.vdot(dfx, x1 - x) + 0.5 / la * LA.norm(x1 - x)**2:
                break
            else:
                la *= sigma

        values.append(J(x1))
        time_list.append(process_time() - begin)
        # compute the number of all the operations
        n_f += j + 1
        n_df += j + 1
        n_prox += j + 1
        ans = [time_list, values, x1, fx1,  2 * la, n_f, n_df, n_prox]
        return ans

    for i in range(numb_iter):
        iterates = iter_T(*iterates)\

    end = process_time()
    print("---- PGM ----")
    print("Number of iterations:", numb_iter)
    print("Number of function, n_f:", iterates[-3])
    print("Number of gradients, n_grad:", iterates[-2])
    print("Number of prox_g:", iterates[-1])
    print("Time execution:", end - begin)
    return iterates[:3]

# =============================================================== ###
# SpaRSA #


def sparsa(J, f, d_f, prox_g, x0, la=1, numb_iter=100, time_bound=3000):
    """
    Minimize function F(x) = f(x) + g(x) using SpaRSA.
    J = f + g
    In the paper stepsize a was like Lipschitz constant, hence la = 1/a
    """
    begin = process_time()
    mu = 0.7
    a_min = 1e-30
    a_max = 1e30
    sigma = 0.01
    M = 5
    # make one iteration of the proximal gradient with standard linesearch
    x = x0
    for j in count(0):
        dfx = d_f(x)
        fx = f(x)
        x1 = prox_g(x - la * dfx, la)
        fx1 = f(x1)
        if fx1 <= fx + np.vdot(dfx, x1 - x) + 0.5 / la * LA.norm(x1 - x)**2:
            break
        else:
            la *= mu

    Jx1 = J(x1)
    values = [Jx1]
    time_list = [process_time() - begin]
    iterates = [time_list, values, x1, x, dfx]

    def iter_T(time_list, values, x, x_old, dfx_old):
        dfx = d_f(x)
        s = x - x_old
        r = dfx - dfx_old
        s_norm2 = np.dot(s, s)
        a = np.dot(s, r) / s_norm2 if s_norm2 > a_min else a_min
        # print(s.dot(s), a)
        a = np.clip(a, a_min, a_max)
        la = 1. / a
        # print(LA.norm(s), la)
        J_max = max(values[-M:])
        for j in count(0):
            x1 = prox_g(x - la * dfx, la)
            Jx1 = J(x1)
            if Jx1 <= J_max - sigma / (2 * la) * LA.norm(x1 - x)**2:
                break
            else:
                la *= mu
        ans = [time_list, values, x1, x, dfx]
        values.append(Jx1)
        time_list.append(process_time() - begin)
        return ans

    for i in range(numb_iter):
        if iterates[0][-1] <= time_bound:
            iterates = iter_T(*iterates)
        else:
            break

    end = process_time()
    return iterates[:3]
# Accelerated Methods ###


def fista(J, d_f, prox_g, x0, la, numb_iter=100):
    """
    Minimize function F(x) = f(x) + g(x) using FISTA.
    Takes J as some evaluation function for comparison.
    """
    begin = process_time()
    values = [J(x0)]
    time_list = [process_time() - begin]
    res = [time_list, values, x0, x0, 1]

    def iter_T(time_list, values, x, y, t):
        x1 = prox_g(y - la * d_f(y), la)
        t1 = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        y1 = x1 + (t - 1) / t1 * (x1 - x)
        values.append(J(x1))
        time_list.append(process_time() - begin)
        return [time_list, values, x1, y1, t1]

    for i in range(numb_iter):
        res = iter_T(*res)

    end = process_time()
    print("---- FISTA----")
    print("Time execution:", end - begin)
    return res


def fista_linesearch(J, f, d_f, prox_g, x0, la0=1, numb_iter=100):
    """
    Minimize function F(x) = f(x) + g(x) by FISTA with linesearch.
    Takes J as some evaluation function for comparison.
    """
    begin = process_time()
    values = [J(x0)]
    time_list = [process_time() - begin]
    iterates = [time_list, values, x0, x0, 1, la0, 0, 0, 0]

    def iter_T(time_list, values, x, y, t, la, n_f, n_df, n_prox):
        dfy = d_f(y)
        fy = f(y)
        sigma = 0.7
        for j in count(0):
            x1 = prox_g(y - la * dfy, la)
            if f(x1) <= fy + np.vdot(dfy, x1 - y) + 0.5 / la * LA.norm(x1 - y)**2:  # F(x1) < Q(x1, y)
                break
            else:
                la *= sigma

        t1 = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        y1 = x1 + (t - 1) / t1 * (x1 - x)
        n_f += j + 2
        n_df += j + 1
        n_prox += j + 1
        values.append(J(x1))
        time_list.append(process_time() - begin)
        ans = [time_list, values, x1, y1, t1, la, n_f, n_df, n_prox]
        return ans

    for i in range(numb_iter):
        iterates = iter_T(*iterates)

    end = process_time()
    print("---- FISTA with linesearch ----")
    print("Number of iterations:", numb_iter)
    print("Number of function, n_f:", iterates[-3])
    print("Number of gradients, n_grad:", iterates[-2])
    print("Number of prox_g:", iterates[-1])
    print("Time execution:", end - begin)
    return iterates[:3]

# =============================================================== ###


# Tseng forward-backward-forward methods ###

def tseng_fbf(J, F, prox_g, x0, la, numb_iter=100):
    """
    Solve monotone inclusion $0 \in F + \partial g by Tseng
    forward-backward-forward method with a fixed step. In particular,
    minimize function F(x) = f(x) + g(x) with convex smooth f and
    convex g. Takes J as some evaluation function for comparison.
    """
    begin = process_time()
    time_list = [0]
    res = [time_list, [J(x0)], x0]

    def iter_T(time_list, values, x):
        Fx = F(x)
        z = prox_g(x - la * Fx, la)
        x = z - la * (F(z) - Fx)
        values.append(J(x))
        values.append(J(x, y))
        return [time_list, values, x]

    for i in range(numb_iter):
        res = iter_T(*res)
    return res


def tseng_fbf_linesearch(J, F, prox_g, x0, delta=2, numb_iter=100):
    """
    Solve monotone inclusion $0 \in F + \partial g by Tseng
    forward-backward-forward algorithm with a linesearch. In
    particular, minimize function F(x) = f(x) + g(x) with convex
    smooth f and convex g. Takes J as some evaluation function for
    comparison.
    """
    begin = process_time()
    beta = 0.7
    theta = 0.99
    la0 = initial_lambda(F, x0, theta)[3]
    time_list = [process_time() - begin]
    iterates = [time_list, [J(x0)], x0, la0, 1, 0]

    def iter_T(time_list, values, x, la, n_F, n_prox):
        Fx = F(x)
        la *= delta
        for j in count(0):
            z = prox_g(x - la * Fx, la)
            Fz = F(z)
            if la * LA.norm(Fz - Fx) <= theta * LA.norm(z - x):
                break
            else:
                la *= beta
        x1 = z - la * (Fz - Fx)
        values.append(J(z))
        time_list.append(process_time() - begin)
        # n_f += j+1
        n_F += j + 2
        n_prox += j + 1
        ans = [time_list, values, x1, la,  n_F, n_prox]
        return ans

    for i in range(numb_iter):
        iterates = iter_T(*iterates)

    end = process_time()
    print("---- FBF ----")
    print("Number of iterations:", numb_iter)
    print("Number of gradients, n_grad:", iterates[-2])
    print("Number of prox_g:", iterates[-1])
    print("Time execution:", end - begin)
    return iterates[:3]

# =============================================================== ###

# Extragradient methods ###


def prox_method_korpelevich(J, F, prox, x0, la, numb_iter=100):
    """find a solution VI(F,Set) by Korpelevich method."""
    res = [[J(x0)], x0]

    def iter_T(values, x):
        y = prox(x - la * F(x), la)
        x = prox(x - la * F(y), la)
        values.append(J(x))
        return [values, x]

    for i in range(numb_iter):
        res = iter_T(*res)
    return res

# =============================================================== ###

# Algorithms from paper PEGM ########################


def initial_lambda(F, x0, a, prec=1e-10):
    """
    Finds initial stepsize.

    With given map F and starting point x0 compute la_0 as an
    approximation of local Lipschitz constant of F in x0. This helps to
    make a better choice for the initial stepsize.  The resulting
    output is the full iterate information for the algorithm.

    """
    gen = 100  # random generator
    np.random.seed(gen)
    x1 = x0 + np.random.random(x0.shape) * prec
    Fx0 = F(x0)
    # need to fix division in case of zero in the denominator
    la0 = a * np.sqrt(np.vdot(x1 - x0, x1 - x0) /
                      np.vdot(F(x1) - F(x0), F(x1) - F(x0)))
    res = [x1, x0, x0, la0, Fx0]
    return res


def alg_VI_prox(J, F, prox, x0, numb_iter=100):
    """
    Implementation of the Algorithm 2 from the paper.

    Parameters
    ----------
    J: function that checks the progress of iterates. It may be
    ||x-x*||, or a gap function for VI, or just f(x)+g(x) for
    minimization problem. Takes x0-like array as input and gives
    scalar value.
    F: operator in the VI (or a gradient in minimization problems).
    prox: proximal operator prox_g. Takes x-array and positive scalar
    value rho and gives another x-array.
    x0: array. Initial point.
    numb_iter: positive integer, optional. Number of iteration. In
    general should be replaced by some stopping criteria.

    Returns:
    list of 3 elements: [values, x1, n_F]. Where
    values collect information for comparison.
    """
    begin = process_time()
    # Initialization  ########
    a = 0.41
    tau = 1.618
    sigma = 0.7
    la_max = 1e7
    iterates = [[J(x0)]] + initial_lambda(F, x0, a) + [tau, 2]
    #

    # Main iteration  #######
    def T(values, x, x_old, y_old, la_old, Fy_old, tau_old, n_F):
        tau = np.sqrt(1 + tau_old)
        for j in count(0):
            y = x + tau * (x - x_old)
            Fy = F(y)
            la = tau * la_old
            # if la**2*np.vdot(Fy-Fy_old, Fy-Fy_old) <= a**2*np.vdot(y-y_old,
            # y-y_old):
            if la * LA.norm(Fy - Fy_old) <= a * LA.norm(y - y_old):
                break
            else:
                tau *= sigma

        x1 = prox(x - la * Fy, la)
        n_F += j + 1
        values.append(J(x1))
        res = [values, x1, x, y, la, Fy, tau, n_F]
        return res

    # Running x_n+1 = T(x_n)
    for i in range(numb_iter):
        iterates = T(*iterates)
    end = process_time()
    print("---- Alg. 2 ----")
    print("Number of iterations:", numb_iter)
    print("Number of gradients, n_grad:", iterates[-1])
    print("Number of prox_g:", numb_iter)
    print("Time execution:", end - begin)
    return [iterates[i] for i in [0, 1, -1]]


def alg_VI_prox_affine(J, F, prox, x0, numb_iter=100):
    """
    Implementation of the Algorithm 2 from the paper in case when F is affine.
    The docs are the same as for the function alg_VI_prox.
    """
    begin = process_time()
    # Initialization  ########
    a = 0.41
    tau = 1.618
    sigma = 0.7
    init = initial_lambda(F, x0, a)
    time_list = [process_time() - begin]
    iterates = [time_list, [J(x0)]] + init + [init[-1]] + [tau, 2]
    #

    # Main iteration  #######
    def T(time_list, values, x, x_old, y_old, la_old, Fx_old, Fy_old, tau_old, n_F):
        tau = np.sqrt(1 + tau_old)
        Fx = F(x)
        for j in count(0):
            y = x + tau * (x - x_old)
            Fy = (1 + tau) * Fx - tau * Fx_old
            la = tau * la_old
            if la**2 * np.vdot(Fy - Fy_old, Fy - Fy_old) <= a**2 * np.vdot(y - y_old, y - y_old):
                break
            else:
                tau *= sigma
        x1 = prox(x - la * Fy, la)
        n_F += 1
        values.append(J(x1))
        time_list.append(process_time() - begin)
        res = [time_list, values, x1, x, y, la, Fx, Fy, tau, n_F]
        return res

    # Running x_n+1 = T(x_n)
    for i in range(numb_iter):
        iterates = T(*iterates)

    end = process_time()
    print("---- Alg. 2, affine operator ----")
    print("Number of iterations:", numb_iter)
    print("Number of gradients, n_grad:", iterates[-1])
    print("Number of prox_g:", numb_iter)
    print("Time execution:", end - begin)
    return iterates[:3]

# =============================================================== ###
