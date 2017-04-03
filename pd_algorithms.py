import numpy as np
import numpy.linalg as LA
from itertools import count
from time import process_time, clock


def pd(J, prox_g, prox_f_conj, K,  x0, y0, sigma, tau, numb_iter=100):
    """
    Primal-dual algorithm of Pock and Chambolle for the problem min_x
    max_y [<Kx,y> + g(x) - f*(y)]
    J denotes some function which we compute in every iteration to
    study perfomance. It may be energy, primal-dual gap, etc.
    """
    begin = clock()  # process_time()
    theta = 1.0
    x, y, z = x0, y0, x0
    values = [J(x0, y0)]
    time_list = [process_time() - begin]
    for i in range(numb_iter):
        x1 = prox_g(x - tau * K.T.dot(y), tau)
        z = x1 + theta * (x1 - x)
        y = prox_f_conj(y + sigma * K.dot(z), sigma)
        x = x1
        values.append(J(x, y))
        # time_list.append(process_time() - begin)
        time_list.append(clock() - begin)
    end = clock()  # process_time()
    print("----- Primal-dual method -----")
    print("Time execution:", end - begin)
    return [time_list, values, x, y]


def pd_accelerated_primal(J, prox_g, prox_f_conj, K,  x0, y0, sigma, tau, gamma, numb_iter=100):
    """
    Accelerated primal-dual algorithm of Pock and Chambolle for problem
    min_x max_y [<Kx,y>  + g(x) - f*(y)], where g is gamma-strongly convex
    """
    begin = process_time()
    theta = 1.0
    x, y, z = x0, y0, x0
    values = [J(x0, y0)]
    time_list = [process_time() - begin]
    for i in range(numb_iter):
        y = prox_f_conj(y + sigma * K.dot(z), sigma)
        x1 = prox_g(x - tau * K.T.dot(y), tau)
        theta = 1. / np.sqrt(1 + gamma * tau)
        tau *= theta
        sigma *= 1. / theta
        z = x1 + theta * (x1 - x)
        x = x1
        values.append(J(x, y))
        time_list.append(process_time() - begin)
    end = process_time()
    print(
        "----- Accelerated primal-dual method (g(x) is strongly convex)-----")
    print("Time execution:", end - begin)
    return [time_list, values, x, y]


def pd_general(J, prox_g, prox_f_conj, dh, K,  x0, y0, sigma, tau, numb_iter=100):
    """
    Primal-dual algorithm of Condat for problem min_x
    max_y [<Kx,y> + g(x) - h(y)- f*(y)]
    J denotes some function which we compute in every iteration to
    study perfomance. It may be energy, primal-dual gap, etc.
    """
    begin = process_time()
    theta = 1.0
    x, y, z = x0, y0, x0
    values = [J(x0, y0)]
    time_list = [process_time() - begin]

    for i in range(numb_iter):
        x1 = prox_g(x - tau * K.T.dot(y), tau)
        z = x1 + theta * (x1 - x)
        y = prox_f_conj(y + sigma * (K.dot(z) - dh(y)), sigma)
        x = x1
        values.append(J(x, y))
        time_list.append(process_time() - begin)
    end = process_time()
    print("----- General primal-dual method -----")
    print("Time execution:", end - begin)
    return [time_list, values, x, y]


def pd_linesearch(J, prox_g, prox_f_conj, K,  x0, y1, tau, beta, numb_iter=100):
    """
    Primal-dual method with linesearch for problem min_x max_y [
    <Kx,y> + g(x) - f*(y)].  Corresponds to Alg.1 in the paper.
    beta denotes sigma/tau from a classical primal-dual algorithm.
    """
    begin = process_time()
    theta = 1
    values = [J(x0, y1)]
    time_list = [process_time() - begin]
    mu = 0.7
    delta = 0.99
    iterates = [time_list, values, x0, y1, theta, tau, K.dot(x0)]
    sqrt_b = np.sqrt(beta)

    # function T is an operator that makes one iteration of the algorithm:
    # (x1, y1) = T(x,y, history)

    def T(time_list, values, x_old, y, th_old, tau_old, Kx_old):
        x = prox_g(x_old - tau_old * K.T.dot(y), tau_old)
        Kx = K.dot(x)
        th = np.sqrt(1 + th_old)
        for j in count(0):
            tau = tau_old * th
            z = x + th * (x - x_old)
            Kz = Kx + th * (Kx - Kx_old)
            y1 = prox_f_conj(y + tau * beta * Kz, tau * beta)
            if sqrt_b * tau * LA.norm(K.T.dot(y1 - y)) <= delta * LA.norm(y1 - y):
                break
            else:
                th *= mu
        values.append(J(x, y1))
        time_list.append(process_time() - begin)
        res = [time_list, values, x,  y1, th, tau, Kx]
        return res

    for i in range(numb_iter):
        iterates = T(*iterates)

    end = process_time()
    print("----- Primal-dual method with linesearch-----")
    print("Time execution:", end - begin)
    return iterates[:4]


def pd_linesearch_acceler_primal(J, prox_g, prox_f_conj, K,  x0, y1, tau, beta, gamma, numb_iter=100):
    """
    Accelerated primal-dual algorithm with linesearch for problem
    min_x max_y [<Kx,y>  + g(x) - f*(y)],
    where g is gamma-strongly convex
    """
    begin = process_time()
    theta = 1.0
    values = [J(x0, y1)]
    time_list = [process_time() - begin]
    mu = 0.7
    delta = 1.0
    iterates = [time_list, values, x0, y1, theta, tau, beta, K.dot(x0)]

    def T(time_list, values, x_old, y, th_old, tau_old, beta_old, Kx_old):
        x = prox_g(x_old - tau_old * K.T.dot(y), tau_old)
        Kx = K.dot(x)
        beta = (1.0 + gamma * tau_old) * beta_old
        sqrt_b = np.sqrt(beta)
        tau = tau_old * np.sqrt(beta_old / beta * (1. + th_old))
        # Linesearch starts
        for j in count(0):
            th = tau / tau_old
            z = x + th * (x - x_old)
            Kz = Kx + th * (Kx - Kx_old)
            y1 = prox_f_conj(y + tau * beta * Kz, tau * beta)
            if sqrt_b * tau * LA.norm(K.T.dot(y1 - y)) <= delta * LA.norm(y1 - y):
                break
            else:
                tau *= mu

        values.append(J(x, y1))
        time_list.append(process_time() - begin)
        res = [values, time_list, x, y1, th, tau, beta, Kx]
        return res

    for i in range(numb_iter):
        iterates = T(*iterates)

    end = process_time()
    print(
        "----- Accelerated primal-dual method with linesearch (g is strongly convex) -----")
    print("Time execution:", end - begin)
    return iterates[:4]


def pd_linesearch_acceler_dual(J, prox_g, prox_f_conj, K,  x0, y1, tau, beta, gamma, numb_iter=100):
    """
    Accelerated primal-dual algorithm with linesearch for problem
    min_x max_y [<Kx,y>  + g(x) - f*(y)],
    where f* is gamma-strongly convex
    """
    begin = process_time()
    theta = 1.0
    values = [J(x0, y1)]
    time_list = [process_time() - begin]
    mu = 0.7
    delta = 1.0
    iterates = [time_list, values, x0, y1, theta, tau, beta, K.dot(x0)]

    def T(time_list, values, x_old, y, th_old, tau_old, beta_old, Kx_old):
        x = prox_g(x_old - tau_old * K.T.dot(y), tau_old)
        Kx = K.dot(x)
        beta = beta_old / (1.0 + gamma * beta_old * tau_old)
        sqrt_b = np.sqrt(beta)
        tau = tau_old * np.sqrt(1. + th_old)
        # Linesearch starts
        for j in count(0):
            th = tau / tau_old
            sigma = beta * tau
            z = x + th * (x - x_old)
            Kz = Kx + th * (Kx - Kx_old)
            y1 = prox_f_conj(y + sigma * Kz, sigma)
            if sqrt_b * tau * LA.norm(K.T.dot(y1 - y)) <= delta * LA.norm(y1 - y):
                break
            else:
                tau *= mu
        values.append(J(x, y1))
        time_list.append(process_time() - begin)
        res = [time_list, values, x, y1, th, tau, beta, Kx]
        return res

    for i in range(numb_iter):
        iterates = T(*iterates)

    end = process_time()
    print(
        "----- Accelerated primal-dual method with linesearch (f^* is strongly convex) -----")
    print("Time execution:", end - begin)
    return iterates[:4]


def pd_linesearch_dual_is_square_norm(J, prox_g, b, K,  x0, y1, tau, beta, numb_iter=100):
    """
    Primal-dual method with linesearch for min_x max_y [ <Kx,y> + g(x) - f*(y) ]
    for the case when f*(y) = 0.5||y-b||^2
    """
    begin = process_time()
    theta = 1
    values = [J(x0, y1)]
    time_list = [process_time() - begin]
    mu = 0.7
    delta = 0.99
    Kx0 = K.dot(x0)
    iterates = [time_list, values, x0, y1,
                theta, tau, Kx0, K.T.dot(Kx0), K.T.dot(y1)]
    sqrt_beta = np.sqrt(beta)
    KTb = K.T.dot(b)

    def T(time_list, values, x_old, y, th_old, tau_old, Kx_old, KTKx_old, KTy):
        # Note that KTy = K.T.dot(y)
        x = prox_g(x_old - tau_old * KTy, tau_old)
        Kx = K.dot(x)
        KTKx = K.T.dot(Kx)
        th = np.sqrt(1 + th_old)
        for j in count(0):
            tau = tau_old * th
            sigma = tau * beta
            z = x + th * (x - x_old)
            Kz = Kx + th * (Kx - Kx_old)
            y1 = (y + sigma * (Kz + b)) / (1. + sigma)
            KTKz = (1 + th) * KTKx - th * KTKx_old
            KTy1 = (KTy + sigma * (KTKz + KTb)) / (1. + sigma)
            if sqrt_beta * tau * LA.norm(KTy1 - KTy) <= delta * LA.norm(y1 - y):
                break
            else:
                th *= mu
        values.append(J(x, y1))
        time_list.append(process_time() - begin)
        res = [time_list, values, x, y1, th, tau, Kx, KTKx,  KTy1]
        return res

    for i in range(numb_iter):
        iterates = T(*iterates)

    end = process_time()
    print(
        "----- Primal-dual method with  linesearch. f^*(y)=0.5*||y-b||^2-----")
    print("Time execution:", end - begin)
    return iterates[:4]


def pd_linesearch_dual_is_linear(J, prox_g, c, K,  x0, y1, tau, beta, numb_iter=100):
    """
    Primal-dual method with linesearch for min_x max_y [ <Kx,y> + g(x) - f*(y) ]
    for the case when f*(y) = (c,y)
    """
    begin = process_time()
    theta = 1
    values = [J(x0, y1)]
    time_list = [process_time() - begin]
    mu = 0.7
    delta = 0.99
    Kx0 = K.dot(x0)
    iterates = [time_list, values, x0, y1,
                theta, tau, Kx0, K.T.dot(Kx0), K.T.dot(y1)]
    sqrt_beta = np.sqrt(beta)
    KTc = K.T.dot(c)

    def T(time_list, values, x_old, y, th_old, tau_old, Kx_old, KTKx_old, KTy):
        # Note that KTy = K.T.dot(y)
        x = prox_g(x_old - tau_old * KTy, tau_old)
        Kx = K.dot(x)
        KTKx = K.T.dot(Kx)
        th = np.sqrt(1 + th_old)
        for j in count(0):
            tau = tau_old * th
            sigma = tau * beta
            z = x + th * (x - x_old)
            Kz = Kx + th * (Kx - Kx_old)
            y1 = y + sigma * (Kz - c)
            KTKz = (1 + th) * KTKx - th * KTKx_old
            KTy1 = KTy + sigma * (KTKz - KTc)
            if sqrt_beta * tau * LA.norm(KTy1 - KTy) <= delta * LA.norm(y1 - y):
                break
            else:
                th *= mu
        values.append(J(x, y1))
        time_list.append(process_time() - begin)
        res = [time_list, values, x, y1, th, tau, Kx, KTKx,  KTy1]
        return res

    for i in range(numb_iter):
        iterates = T(*iterates)

    end = process_time()
    print("----- Primal-dual method with linesearch. f^*(y)=(c,y)----")
    print("Time execution:", end - begin)
    return iterates[:4]


def pd_linesearch_general(J, prox_g, prox_f_conj, h, dh, K,  x0, y1, tau, beta, numb_iter=100):
    """
    Primal-dual method with  linesearch for problem min_x max_y [
    <Kx,y> + g(x) - f*(y)-h(y)].  Corresponds to Alg.4 in the paper.
    """
    begin = process_time()
    theta = 1
    values = [J(x0, y1)]
    time_list = [process_time() - begin]
    mu = 0.7
    delta = 0.99
    iterates = [time_list, values, x0, y1, theta, tau, K.dot(x0)]
    sqrt_b = np.sqrt(beta)

    # function T is an operator that makes one iteration of the algorithm:
    # (x1, y1) = T(x,y, history)

    def T(time_list, values, x_old, y, th_old, tau_old, Kx_old):
        x = prox_g(x_old - tau_old * K.T.dot(y), tau_old)
        Kx = K.dot(x)
        th = np.sqrt(1 + th_old)
        dhy = dh(y)
        hy = h(y)
        for j in count(0):
            tau = tau_old * th
            sigma = beta * tau
            z = x + th * (x - x_old)
            Kz = Kx + th * (Kx - Kx_old)
            y1 = prox_f_conj(y + sigma * (Kz - dhy), sigma)
            hy1 = h(y1)
            if sigma * tau * (LA.norm(K.T.dot(y1 - y)))**2 + 2 * sigma * (hy1 - hy - dhy.dot(y1 - y)) <= delta * np.dot(y1 - y, y1 - y):
                break
            else:
                th *= mu
        # print(j, tau)
        values.append(J(x, y1))
        time_list.append(process_time() - begin)
        res = [time_list, values, x,  y1, th, tau, Kx]
        return res

    for i in range(numb_iter):
        iterates = T(*iterates)

    end = process_time()
    print("----- General primal-dual method with linesearch-----")
    print("Time execution:", end - begin)
    return iterates[:4]


def pd_linesearch_general_particular(J, prox_g, c, H, b, gamma, K,  x0, y1, tau, beta, numb_iter=100):
    """
    Primal-dual method with linesearch for problem min_x max_y [
    <Kx,y> + g(x) - f^*(y) -h(y)].  where f*(y) = \lr{c,y}, h(y) =
    gamma/2 * ||Hy-b||^2
    """
    begin = process_time()
    theta = 1
    # x, y = x0, y1
    values = [J(x0, y1)]
    time_list = [process_time() - begin]
    mu = 0.7
    delta = 0.99
    Kx0 = K.dot(x0)
    KTc = K.T.dot(c)
    HTb = H.T.dot(b)
    iterates = [time_list, values, x0, y1, theta, tau, Kx0, K.T.dot(
        Kx0), K.T.dot(y1), H.dot(y1), H.dot(Kx0)]
    sqrt_b = np.sqrt(beta)

    # function T is an operator that makes one iteration of the algorithm:
    # (x1, y1) = T(x,y, history)

    def T(time_list, values, x_old, y, th_old, tau_old, Kx_old, KTKx_old, KTy, Hy, HKx_old):
        x = prox_g(x_old - tau_old * K.T.dot(y), tau_old)
        Kx = K.dot(x)
        KTKx = K.T.dot(Kx)
        HTHy = H.T.dot(Hy)
        HKx = H.dot(Kx)
        expr = gamma * (HTHy - HTb) - c
        H_expr = H.dot(expr)
        KT_expr = K.T.dot(expr)

        th = np.sqrt(1 + th_old)
        for j in count(0):
            tau = tau_old * th
            sigma = beta * tau
            z = x + th * (x - x_old)
            Kz = (1 + th) * Kx - th * Kx_old
            KTKz = (1 + th) * KTKx - th * KTKx_old
            HKz = (1 + th) * HKx - th * HKx_old
            t = Kz - expr
            Ht = HKz - H_expr
            KTt = KTKz - KT_expr
            # y1 = y + sigma * (Kz - gamma*(HTHy -HTb) -c)

            if sigma * tau * LA.norm(KTt)**2 + 2 * sigma * gamma * LA.norm(Ht)**2 <= delta * LA.norm(t)**2:
                y1 = y + sigma * t
                Hy1 = Hy + sigma * Ht
                KTy1 = KTy + sigma * KTt
                break
            else:
                th *= mu
        # print j, tau * sigma
        values.append(J(x, y1))
        time_list.append(process_time() - begin)
        res = [time_list, values, x,  y1, th, tau, Kx, KTKx, KTy1, Hy1, HKx]
        return res

    for i in range(numb_iter):
        iterates = T(*iterates)

    end = process_time()
    print("----- Primal-dual method with linesearch-----")
    print("Time execution:", end - begin)
    return iterates[:4]  # fixed this, previously was wrong


def pd_linesearch_acceler_dual_is_square_norm(J, prox_g, b, K,  x0, y1, tau, beta, gamma=1, numb_iter=100):
    """
    Accelerated primal-dual method with linesearch for min_x max_y [
    <Kx,y> + g(x) - f*(y) ] for the case when f*(y) = 0.5||y-b||^2
    """
    begin = process_time()
    theta = 1
    values = [J(x0, y1)]
    time_list = [process_time() - begin]
    mu = 0.7
    delta = 0.99
    Kx0 = K.dot(x0)
    iterates = [time_list, values, x0, y1,
                theta, tau, beta, Kx0, K.T.dot(Kx0), K.T.dot(y1)]
    KTb = K.T.dot(b)

    def T(time_list, values, x_old, y, th_old, tau_old, beta_old, Kx_old, KTKx_old, KTy):
        # Note that KTy = K.T.dot(y)
        x = prox_g(x_old - tau_old * KTy, tau_old)
        Kx = K.dot(x)
        KTKx = K.T.dot(Kx)
        beta = beta_old / (1.0 + gamma * beta_old * tau_old)
        sqrt_beta = np.sqrt(beta)
        tau = tau_old * np.sqrt(1. + th_old)
        # Linesearch starts
        for j in count(0):
            th = tau / tau_old
            sigma = beta * tau
            z = x + th * (x - x_old)
            Kz = Kx + th * (Kx - Kx_old)
            y1 = (y + sigma * (Kz + b)) / (1. + sigma)
            KTKz = (1 + th) * KTKx - th * KTKx_old
            KTy1 = (KTy + sigma * (KTKz + KTb)) / (1. + sigma)
            if sqrt_beta * tau * LA.norm(KTy1 - KTy) <= delta * LA.norm(y1 - y):
                break
            else:
                tau *= mu
        values.append(J(x, y1))
        time_list.append(process_time() - begin)
        res = [time_list, values, x, y1, th, tau, beta, Kx, KTKx, KTy1]
        return res

    for i in range(numb_iter):
        iterates = T(*iterates)

    end = process_time()
    print(
        "----- Accelerated primal-dual method with  linesearch for dual f^*(y)=0.5*||y-b||^2-----")
    print("Time execution:", end - begin)
    return iterates[:4]
