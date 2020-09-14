import numpy as np
from scipy.signal import lfilter


def filter_boundary(fil_num, fil_den, input, condition, npoints):
    """[summary]

    Arguments:
        fil_num {[type]} -- [description]
        fil_den {[type]} -- [description]
        input {[type]} -- [description]
        condition {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    fil_num = np.squeeze(fil_num)

    if condition == 'transpose':
        out = np.flipud(lfilter(fil_num, 1, np.flipud(input)))
    else:
        out = lfilter(fil_num, 1, input)

    out = np.squeeze(out)

    # Denominator
    if len(fil_den) == 2:
        causal = np.squeeze(fil_den[0])
        non_causal = np.squeeze(fil_den[1])

        if (causal.size + non_causal.size) > 2:
            shiftnc = non_causal.size - 1
            out = np.hstack((np.hstack((np.zeros(shiftnc), out)), np.zeros(shiftnc)))

            if condition == 'normal':
                out = lfilter(np.atleast_1d(1), causal, out)
                if non_causal.size == 1:
                    out = np.flipud(lfilter(np.atleast_1d(1), non_causal, np.flipud(out))) * non_causal
                else:
                    out = np.flipud(lfilter(np.atleast_1d(1), non_causal, np.flipud(out))) * non_causal[-1]
                out = out[2*shiftnc:]
            elif condition == 'transpose':
                out = np.flipud(lfilter(np.atleast_1d(1), causal, np.flipud(out)))
                if non_causal.size == 1:
                    out = lfilter(np.atleast_1d(1), non_causal, out) * non_causal
                else:
                    out = lfilter(np.atleast_1d(1), non_causal, out) * non_causal[-1]
                if shiftnc != 0:
                    out = out[0:-(2*shiftnc)]

    if out.size == 0:
        breakpoint()
        out = np.zeros(npoints)

    return out


def temporal_ta(y, params):
    """[summary]

    Arguments:
        y {[type]} -- [description]
        params {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    n = params['f_analyze']['num']
    d = params['f_analyze']['den']
    maxeig = params['maxeig']
    N = params['nscans']
    Nit = params['NitTemp']

    # If estimate before, take previous lambdas
    # if 'noise_estimate_fin' in params and len(params['noise_estimate_fin']) >= (params['vxl_ind'] + 1):
    #     lambd = params['noise_estimate_fin'][params['vxl_ind']]
    # else:
    lambd = params['LambdaTemp'][params['vxl_ind']]

    noise_estimate = params['LambdaTemp'][params['vxl_ind']]

    nv = np.zeros((Nit, 1))
    Lambd = np.zeros((Nit, 1))
    precision = noise_estimate / 100000

    if params['cost_save']:
        cost = np.zeros((Nit, 1))
    else:
        cost = None

    z = np.zeros(N)
    k = 0
    t = 1
    s = np.zeros(N)

    while (k < Nit):
        # Estimate for y
        z_l = z
        z = 1 / (lambd * maxeig) * filter_boundary(n, d, y, 'normal', N) + s - filter_boundary(n, d, filter_boundary(n, d, s, 'transpose', N), 'normal', N) / maxeig

        # Clipping
        z = np.maximum(np.minimum(z, 1), -1)
        t_l = t
        t = (1 + np.sqrt(1 + 4 * (t ** 2))) / 2
        s = z + (t_l - 1) / t * (z - z_l)
        if params['cost_save']:
            temp = y - lambd * filter_boundary(n, d, z, 'transpose', N)
            cost[k] = np.sum((temp - y) ** 2) / 2 + lambd * np.sum(abs(filter_boundary(n, d, temp, 'normal', N)))
            nv[k] = np.sqrt(np.sum((temp - y) ** 2) / N)
        else:
            nv[k] = np.sqrt(np.sum((lambd * filter_boundary(n, d, z, 'transpose', N)) ** 2) / N)

        if params['update']:
            if abs(nv[k] - noise_estimate) > precision:
                lambd = lambd * noise_estimate / nv[k]

        Lambd[k] = lambd
        k += 1

    # Update x
    x = y - lambd * filter_boundary(n, d, z, 'transpose', N)
    x[abs(x) < 1e-6] = 0

    u = filter_boundary(n, d, x, 'normal', N)

    return u, nv[-1], Lambd[-1], cost
