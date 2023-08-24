import teneva
import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter as tpc
from celluloid import Camera

def cross(f, Y0, m=None, e=None, nswp=None, tau=1.1, dr_min=1, dr_max=1,
          tau0=1.05, k0=100, info={}, cache=None, I_vld=None, y_vld=None,
          e_vld=None, cb=None, func=None, log=False, step_cb=None):
    """
    This is a modified version of the TT-cross algorithm,
    lifted directly from the Teneva package: 
    https://teneva.readthedocs.io/_modules/teneva/cross.html#cross
    """
    if m is None and e is None and nswp is None:
        if I_vld is None or y_vld is None:
            raise ValueError('One of arguments m/e/nswp should be set')
        elif e_vld is None:
            raise ValueError('One of arguments m/e/e_vld/nswp should be set')
    if e_vld is not None and (I_vld is None or y_vld is None):
        raise ValueError('Validation dataset is not set')

    _time = tpc()
    info.update({'r': teneva.erank(Y0), 'e': -1, 'e_vld': -1, 'nswp': 0,
        'stop': None, 'm': 0, 'm_cache': 0, 'm_max': int(m) if m else None,
        'with_cache': cache is not None})

    d = len(Y0)
    n = teneva.shape(Y0)
    Y = teneva.copy(Y0)

    Ig = [teneva._reshape(np.arange(k, dtype=int), (-1, 1)) for k in n]
    Ir = [None for i in range(d+1)]
    Ic = [None for i in range(d+1)]

    R = np.ones((1, 1))
    for i in range(d):
        G = np.tensordot(R, Y[i], 1)
        Y[i], R, Ir[i+1] = _iter(G, Ig[i], Ir[i], tau0=tau0, k0=k0, ltr=True)
        step_cb(Y, i, R, direction="left")
    Y[d-1] = np.tensordot(Y[d-1], R, 1)
<<<<<<< HEAD
    step_cb(Y, d-1, R, direction="right")
=======
    step_cb(Y, d-1, R, direction="right", animation_frame=False)
>>>>>>> refs/remotes/origin/tensor_train

    R = np.ones((1, 1))
    for i in range(d-1, -1, -1):
        G = np.tensordot(Y[i], R, 1)
        Y[i], R, Ic[i] = _iter(G, Ig[i], Ic[i+1], tau0=tau0, k0=k0, ltr=False)        
        step_cb(Y, i, R, direction="right")
    Y[0] = np.tensordot(R, Y[0], 1)
<<<<<<< HEAD
    step_cb(Y, 0, R, direction="right")
=======
    step_cb(Y, 0, R, direction="right", animation_frame=False)
>>>>>>> refs/remotes/origin/tensor_train

    info['e_vld'] = teneva.accuracy_on_data(Y, I_vld, y_vld)
    teneva._info_appr(info, _time, nswp, e, e_vld, log)

    while True:
        Yold = teneva.copy(Y)

        R = np.ones((1, 1))
        for i in range(d):
            Z = (func or _func)(f, Ig[i], Ir[i], Ic[i+1], info, cache)
            if Z is None:
                #Y[i] = np.tensordot(R, Y[i], 1)
                info['r'] = teneva.erank(Y)
                info['e'] = teneva.accuracy(Y, Yold)
                info['e_vld'] = teneva.accuracy_on_data(Y, I_vld, y_vld)
                info['stop'] = 'm'
                teneva._info_appr(info, _time, nswp, e, e_vld, log)
                return Y
            Y[i], R, Ir[i+1] = _iter(Z, Ig[i], Ir[i],
                tau, dr_min, dr_max, tau0, k0, ltr=True)
            
            if i < d - 1:
                Y[i+1] = np.tensordot(R, Y[i+1], 1)

            step_cb(Y, i, R, direction="left")

        Y[d-1] = np.tensordot(Y[d-1], R, 1)
        step_cb(Y, d-1, R, direction="right")

        R = np.ones((1, 1))
        for i in range(d-1, -1, -1):
            Z = (func or _func)(f, Ig[i], Ir[i], Ic[i+1], info, cache)
            if Z is None:
                Y[i] = np.tensordot(Y[i], R, 1)
                info['r'] = teneva.erank(Y)
                info['e'] = teneva.accuracy(Y, Yold)
                info['e_vld'] = teneva.accuracy_on_data(Y, I_vld, y_vld)
                info['stop'] = 'm'
                teneva._info_appr(info, _time, nswp, e, e_vld, log)
                return Y
            Y[i], R, Ic[i] = _iter(Z, Ig[i], Ic[i+1],
                tau, dr_min, dr_max, tau0, k0, ltr=False) 

            if i > 0:
                Y[i-1] = np.tensordot(Y[i-1], R, 1)

            step_cb(Y, i, R, direction="right")

        Y[0] = np.tensordot(R, Y[0], 1)
        step_cb(Y, 0, R, direction="right")

        info['nswp'] += 1
        info['r'] = teneva.erank(Y)
        info['e'] = teneva.accuracy(Y, Yold)
        info['e_vld'] = teneva.accuracy_on_data(Y, I_vld, y_vld)

        if info['m_cache'] > 5 * info['m']:
            info['stop'] = 'conv'

        if cb:
            opts = {'Yold': Yold, 'Ir': Ir, 'Ic': Ic, 'cache': cache}
            if cb(Y, info, opts) is True:
                info['stop'] = info['stop'] or 'cb'

        if teneva._info_appr(info, _time, nswp, e, e_vld, log):
            return Y


def _func(f, Ig, Ir, Ic, info, cache=None):
    n = Ig.shape[0]
    r1 = Ir.shape[0] if Ir is not None else 1
    r2 = Ic.shape[0] if Ic is not None else 1

    I = np.kron(np.kron(teneva._ones(r2), Ig), teneva._ones(r1))

    if Ir is not None:
        Ir_ = np.kron(teneva._ones(n * r2), Ir)
        I = np.hstack((Ir_, I))

    if Ic is not None:
        Ic_ = np.kron(Ic, teneva._ones(r1 * n))
        I = np.hstack((I, Ic_))

    y = _func_eval(f, I, info, cache)

    if y is not None:
        return teneva._reshape(y, (r1, n, r2))


def _func_eval(f, I, info, cache=None):
    if cache is None:
        if info['m_max'] is not None and info['m'] + len(I) > info['m_max']:
            return None
        info['m'] += len(I)
        return f(I)

    I_new = np.array([i for i in I if tuple(i) not in cache])
    if len(I_new):
        if info['m_max'] is not None and info['m'] + len(I_new) > info['m_max']:
            return None
        Y_new = f(I_new)
        for k, i in enumerate(I_new):
            cache[tuple(i)] = Y_new[k]

    info['m'] += len(I_new)
    info['m_cache'] += len(I) - len(I_new)

    return np.array([cache[tuple(i)] for i in I])


def _iter(Z, Ig, I, tau=1.1, dr_min=0, dr_max=0, tau0=1.05, k0=100, ltr=True):
    r1, n, r2 = Z.shape

    if ltr:
        Z = teneva._reshape(Z, (r1 * n, r2))
    else:
        Z = teneva._reshape(Z, (r1, n * r2)).T

    Q, R = np.linalg.qr(Z)

    ind, B = teneva._maxvol(Q, tau, dr_min, dr_max, tau0, k0)

    if ltr:
        G = teneva._reshape(B, (r1, n, -1))
        R = Q[ind, :] @ R
        I_new = np.kron(Ig, teneva._ones(r1))
        if I is not None:
            I_old = np.kron(teneva._ones(n), I)
            I_new = np.hstack((I_old, I_new))

    else:
        G = teneva._reshape(B.T, (-1, n, r2))
        R = (Q[ind, :] @ R).T
        I_new = np.kron(teneva._ones(r2), Ig)
        if I is not None:
            I_old = np.kron(I, teneva._ones(n))
            I_new = np.hstack((I_new, I_old))

    return G, R, I_new[ind, :]



if __name__=='__main__':
    from function_tensor import *
    from tensor_train import TensorTrain
    from quantization_plots import create_plot
    from functions import *

    lbound = 0.001
    ubound = 25
    func = sin2_test
    N = 1
    grid_bounds = np.array([[lbound, ubound] for _ in range(N)], dtype=np.double)
    subdivs = [2 ** 10] * N

    m         = 8.E+3  # Number of calls to target function
    e         = None   # Desired accuracy
    nswp      = 1      # Sweep number
    tt_rank   = 4      # TT-rank of the initial tensor
    dr_min    = 0      # Cross parameter (minimum number of added rows)
    dr_max    = 0      # Cross parameter (maximum number of added rows)

    quantization = Power2Quantization(subdivs, ordering="canonical")
    ground_truth = FunctionTensor(grid_bounds, subdivs, func, quantization=quantization, track_evals=True)
    tt_approx = TensorTrain(quantization.qdim_sizes, [tt_rank] * (quantization.qdim - 1))

    n = quantization.qdim_sizes

    def wrapped_func(I):
        return ground_truth.evaluate(I)

    # Number of test points:
    m_tst = int(1.E+4)

    # Random multi-indices for the test points:
    I_tst = np.vstack([np.random.choice(k, m_tst) for k in n]).T

    # Function values for the test points:
    y_tst = wrapped_func(I_tst)

    fig, ax = plt.subplots()
    camera = Camera(fig)

    ground_truth.initialize_accuracy_estimation(method="randomized", 
                                                rsample_count=10000)

    def step_callback(Y, i, R, direction, animation_frame=True):
        # Note - I switched the order of tensordot
        if direction == "left":
            #tt_approx.U[i] = np.tensordot(Y[i], R, 1)
            tt_approx.U[i] = Y[i].copy()
            tt_approx.update_internal_sampler(i, direction, False)
            if i < len(Y) - 1:
                tt_approx.U[i+1] = Y[i+1].copy()
                tt_approx.update_internal_sampler(i+1, direction, False)

        elif direction == "right":
            #tt_approx.U[i] = np.tensordot(R, Y[i], 1)
            tt_approx.U[i] = Y[i].copy()
            tt_approx.update_internal_sampler(i, direction, False)

            if i > 0: 
                tt_approx.U[i-1] = Y[i-1].copy()
                tt_approx.update_internal_sampler(i-1, direction, False)

        if animation_frame:
            create_plot(func, lbound, ubound, tt_approx, ground_truth, None,
                    name=None, animate=(ax, camera)) 

            approx_fit = ground_truth.compute_approx_tt_fit(tt_approx)
            print(f'Fit: {approx_fit}') 


    t = tpc()
    info, cache = {}, {}
    #Y = teneva.rand(n, r)
    Y = tt_approx.U
    tt_approx.build_fast_sampler(0, 100)
    Y = cross(wrapped_func, Y, m, e, nswp, dr_min=dr_min, dr_max=dr_max,
        info=info, cache=cache, step_cb=step_callback)

    animation = camera.animate()  
    animation.save('plotting/quantization_experiments/cross_animated.gif', writer = 'pillow')

    #tt_approx.U = Y
    #tt_approx.build_fast_sampler(0, 100)
    #Y = teneva.truncate(Y, 1.E-4) # We round the result at the end
    t = tpc() - t

    create_plot(func, lbound, ubound, tt_approx, ground_truth, None,
                    name=f"cross_result.png", animate=None) 

    print(f'Build time           : {t:-10.2f}')
    print(f'Evals func           : {info["m"]:-10d}')
    print(f'Cache uses           : {info["m_cache"]:-10d}')
    print(f'Iter accuracy        : {info["e"]:-10.2e}')
    print(f'Sweep number         : {info["nswp"]:-10d}')
    print(f'Stop condition       : {info["stop"]:>10}')
    print(f'TT-rank of pure res  : {info["r"]:-10.1f}')
    print(f'TT-rank of trunc res : {teneva.erank(Y):-10.1f}')

    approx_fit = ground_truth.compute_approx_tt_fit(tt_approx)
    print(f'Fit: {approx_fit}') 
