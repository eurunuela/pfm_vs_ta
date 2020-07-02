import numpy as np
from scipy import linalg
from sklearn.utils.validation import check_array


def proximal_operator_lasso(y, lambd, weights=0):
    weights = np.ones(y.shape)
    x = y*np.maximum(np.zeros(y.shape), 1-(weights*lambd/abs(y)))
    x[np.abs(x) < np.finfo(float).eps] = 0

    return(x)


def debiasing(x, y, beta):

    beta_out = np.zeros(beta.shape)
    fitts_out = np.zeros(y.shape)

    index_voxels = np.unique(np.where(abs(beta) > 10 * np.finfo(float).eps)[1])

    for voxidx in range(len(index_voxels)):
        index_events_opt = np.where(
            abs(beta[:, index_voxels[voxidx]]) > 10 * np.finfo(float).eps)[0]

        X_events = x[:, index_events_opt]
        beta2save = np.zeros((beta.shape[0], 1))

        coef_LSfitdebias, residuals, rank, s = linalg.lstsq(
            X_events, y[:, index_voxels[voxidx]], cond=None)
        beta2save[index_events_opt, 0] = coef_LSfitdebias

        beta_out[:, index_voxels[voxidx]] = beta2save.reshape(len(beta2save))

    return beta_out


def fista_update(X, y, max_iter, lambd, update_lambda, precision=None):

    nscans = y.shape[0]
    nvoxels = y.shape[1]

    X_tilde_tt = np.dot(X.T, X)
    c_ist = 1/(linalg.norm(X) ** 2)

    y_fista = np.zeros((nscans, nvoxels), dtype=np.float32)
    prox_z = y_fista.copy()
    beta = y_fista.copy()
    nv = np.zeros((max_iter, 1))

    v = np.dot(X.T, y)

    t_fista = 1
    num_iter = 0
    noise_estimate = lambd
    if precision is None:
        precision = noise_estimate / 100000

    for num_iter in range(max_iter):

        prox_z_old = prox_z.copy()
        y_ista = y_fista.copy()

        # Forward-Backward step
        z_ista = y_ista + c_ist*(v - np.dot(X_tilde_tt, y_ista.astype(np.float32)))

        prox_z = proximal_operator_lasso(z_ista, c_ist*lambd)

        t_fista_old = t_fista
        t_fista = 0.5 * (1 + np.sqrt(1 + 4 * (t_fista_old ** 2)))

        beta = prox_z.copy()
        beta_deb = debiasing(X, y, prox_z)

        y_fista = prox_z + (prox_z - prox_z_old) * (t_fista_old-1) / t_fista

        nv[num_iter] = np.sqrt(np.sum((np.dot(X, beta_deb) - y) ** 2) / nscans)

        if abs(nv[num_iter] - noise_estimate) > precision and update_lambda:
            lambd = lambd * noise_estimate / nv[num_iter]
        else:
            break

        num_iter += 1

    # Save estimated bold signals for current iteration
    betafitts = np.dot(X, beta.astype(np.float32))

    return beta, betafitts, lambd
