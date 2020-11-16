import numpy as np
from pywt import wavedec
from scipy.stats import median_absolute_deviation
from sklearn.linear_model import lars_path


def stability_selection(Y, X, nsurrogates=100):
    """[summary]

    Parameters
    ----------
    Y : ndarray
        Input data.
    X : ndarray
        Design matrix
    nsurrogates : int, optional
        Number of surrogates for stability selection, by default 100

    Returns
    -------
    auc : ndarray
        Area under the curve of the stability paths.
    """
    nscans = Y.shape[0]
    nvoxels = Y.shape[1]
    nlambdas = nscans + 1

    auc = np.empty((nscans, nvoxels))

    for vox_idx in range(nvoxels):
        lambdas = np.zeros((nsurrogates, nlambdas), dtype=np.float32)
        coef_path = np.zeros((nsurrogates, nscans, nlambdas), dtype=np.float32)
        sur_idxs = np.zeros((nsurrogates,int(0.6 * nscans)))

        for surrogate_idx in range(nsurrogates):
            idxs = np.sort(np.random.choice(range(nscans), int(0.6*nscans), 0)) # 60% of timepoints are kept
            sur_idxs[surrogate_idx, :] = idxs
            y_sub = Y[idxs, vox_idx]
            X_sub = X[idxs, :]

            _, cD1 = wavedec(y_sub, 'db6', level=1, axis=0)
            lambda_min = median_absolute_deviation(cD1) / 0.6745
            lambda_min = lambda_min / y_sub.shape[0]

            # LARS path
            lambdas_temp, _, coef_path_temp = lars_path(X_sub, np.squeeze(y_sub), method='lasso',
                                                        Gram=np.dot(X_sub.T, X_sub),
                                                        Xy=np.dot(X_sub.T, np.squeeze(y_sub)),
                                                        max_iter=int(np.ceil(0.3*nscans)), eps=1e-6,
                                                        alpha_min=lambda_min)

            lambdas[surrogate_idx, :len(lambdas_temp)] = lambdas_temp
            n_coefs = (coef_path_temp != 0).shape[1]
            coef_path[surrogate_idx, :, :n_coefs] = coef_path_temp != 0


        # Sorting and getting indexes
        lambdas_merged = lambdas.copy()
        lambdas_merged = lambdas_merged.reshape((nlambdas * nsurrogates,))
        sort_idxs = np.argsort(-lambdas_merged)
        lambdas_merged = -np.sort(-lambdas_merged)
        nlambdas_merged = len(lambdas_merged)

        temp = np.zeros((nscans, nsurrogates * nlambdas), dtype=np.float64)

        for surrogate_idx in range(nsurrogates):
            if surrogate_idx == 0:
                first = 0
                last = nlambdas - 1
            else:
                first = last + 1
                last = first + nlambdas - 1

            same_lambda_idxs = np.where((first <= sort_idxs) & (sort_idxs <= last))[0]

            # Find indexes of changes in value (0 to 1 changes are expected).
            coef_path_temp = np.squeeze(coef_path[surrogate_idx, :, :])
            if len(coef_path_temp.shape) == 1:
                coef_path_temp = coef_path_temp[:, np.newaxis] 
            diff = np.diff(coef_path_temp)
            nonzero_change_scans, nonzero_change_idxs = np.where(diff)
            nonzero_change_idxs = nonzero_change_idxs + 1

            coef_path_squeezed = np.squeeze(coef_path[surrogate_idx, :, :])
            coef_path_merged = np.full((nscans, nlambdas * nsurrogates), False, dtype=bool)
            coef_path_merged[:, same_lambda_idxs] = coef_path_squeezed.copy()

            for i in range(len(nonzero_change_idxs)):
                coef_path_merged[nonzero_change_scans[i], same_lambda_idxs[nonzero_change_idxs[i]]:] = True

            # Sum of non-zero coefficients
            temp += coef_path_merged

        auc_temp = np.zeros((nscans, ), dtype=np.float64)
        lambda_sum = np.sum(lambdas_merged)

        for lambda_idx in range(nlambdas_merged):
            auc_temp += temp[:,lambda_idx]/nsurrogates*lambdas_merged[lambda_idx]/lambda_sum

        auc[:, vox_idx] = np.squeeze(auc_temp)

    return auc