import numpy as np
from itertools import combinations as combs
from scipy.signal import freqz
from Scripts.temporal_ta import temporal_ta
from scipy.stats import median_absolute_deviation
from pywt import wavedec


def cons_filter(root):
    """[summary]

    Arguments:
        root {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    # breakpoint()
    if isinstance(root, float):
        n = 1
    else:
        n = len(root)
    fil = np.zeros((n + 1, 1))
    fil[0] = 1

    if isinstance(root, float):
        fil[1] = (-1) * np.exp(root)
    else:
        for i in range(n):
            combs_result = np.array(list(combs(root, i + 1)))
            fil[i + 1] = (-1) ** (i + 1) * np.sum(np.exp(np.sum(combs_result, axis=1)))

    return fil


def hrf_filters(tr, condition='spike', condition2='spmhrf'):

    if condition2 == 'bold':
        eps = 0.54
        ts = 1.54
        tf = 2.46
        t0 = 0.98
        alpha = 0.33
        E0 = 0.34
        V0 = 1
        k1 = 7 * E0
        k2 = 2
        k3 = 2 * E0 - 0.2

        c = (1 + (1 - E0) * np.log(1 - E0) / E0) /t0

        # Zeros
        a1 = -1 / t0
        a2 = -1 / (alpha * t0)
        a3 = -complex(1, np.sqrt(4 * ts ** 2 / tf - 1)) / (2 * ts)
        a4 = -complex(1, -np.sqrt(4 * ts ** 2 / tf - 1)) / (2 * ts)

        # Pole
        psi = -((k1 + k2) * ((1 - alpha)/alpha/t0 - c/alpha) - (k3 - k2)/t0) / (-(k1 + k2) * c * t0 - k3 + k2)

    elif condition2 == 'spmhrf':
        a1 = -0.27
        a2 = -0.27
        a3 = complex(-0.4347, -0.3497)
        a4 = complex(-0.4347, 0.3497)
        psi = np.float64(-0.1336)

    else:
        raise ValueError('Unknown filter')

    fil_zeros = np.array([a1, a2, a3, a4]) * tr
    fil_poles = psi * tr

    cons = 1
    hnum = cons_filter(fil_zeros) * cons
    hden = cons_filter(fil_poles)

    causal = fil_poles[fil_poles.real < 0]
    n_causal = fil_poles[fil_poles.real > 0]

    # Shortest filter, 1st order aproximation
    h_dc = cons_filter(causal)
    h_dnc = cons_filter(n_causal)

    h_d = [h_dc, h_dnc]

    filter_reconstruct = {'num': hnum, 'den': h_d}

    if condition == 'spike':
        d2, d1 = freqz(hnum, hden, 1024)
        maxeig = np.max(abs(d1) ** 2)
        filter_analyze = filter_reconstruct

    elif condition == 'block':
        fil_zeros_2 = np.append(fil_zeros, 0)
        # Shortest Filter, 1st order approximation
        hnum2 = cons_filter(fil_zeros_2)*cons
        # second order approximation
        #         hnum2 = cons_filter2(FilZeros2)*cons;

        d2, d1 = freqz(hnum2, hden, 1024)
        maxeig = np.max(abs(d1) ** 2)

        filter_analyze = {'num': hnum2, 'den': h_d}
    else:
        raise ValueError('No other conditions implemented yet')

    return filter_analyze, filter_reconstruct, maxeig


def total_activation(data, params, lambd=None, update_lambda=False):
    """[summary]

    Arguments:
        data {[type]} -- [description]
        params {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    nscans = data.shape[0]
    nvoxels = data.shape[1]
    params['nscans'] = nscans
    params['update'] = update_lambda

    # Creates filter with zeros and poles
    params['f_analyze'], params['f_recons'], params['maxeig'] = hrf_filters(params['tr'], condition=params['model'])
    params['NitTemp'] = 200
    maxeig = params['maxeig']
    # print(f'Maxeig = {maxeig}')

    # Initiates variables
    tc_out = np.zeros((nscans, nvoxels))
    params['LambdaTemp'] = np.zeros((nvoxels, 1))
    costtemp = np.zeros((params['NitTemp'], nvoxels))

    # Computes temporal regularization for each voxel
    for vox_idx in range(nvoxels):
        # print(f'Voxel {vox_idx + 1}/{nvoxels}')
        if lambd is None:
            _, cD1 = wavedec(data[:, vox_idx], 'db3', level=1, axis=0)
            params['LambdaTemp'][vox_idx] = np.median(abs(cD1 - np.median(cD1))) / 0.6745 # 0.8095
        else:
            params['LambdaTemp'][vox_idx] = lambd
        params['vxl_ind'] = vox_idx
        tc_out[:, vox_idx], noise_est_fin, lambd_fin, cost_temp = temporal_ta(data[:, vox_idx], params)
        params['lambda_temp_fin'] = lambd_fin
        params['noise_estimate_fin'] = noise_est_fin
        costtemp[:, vox_idx] = np.squeeze(cost_temp)

    params['cost_temp'] = costtemp

    return tc_out, params['lambda_temp_fin']
