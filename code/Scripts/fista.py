import numpy as np
from scipy import linalg
from sklearn.utils.validation import check_array
from Scripts.debiasing import debiasing


def _impose_f_order(X):
    """Helper Function"""
    # important to access flags instead of calling np.isfortran,
    # this catches corner cases.
    if X.flags.c_contiguous:
        return check_array(X.T, copy=False, order='F'), True
    else:
        return check_array(X, copy=False, order='F'), False


def fast_dot(A, B):
    """Compute fast dot products directly calling BLAS.
    This function calls BLAS directly while warranting Fortran contiguity.
    This helps avoiding extra copies `np.dot` would have created.
    For details see section `Linear Algebra on large Arrays`:
    http://wiki.scipy.org/PerformanceTips
    Parameters
    ----------
    A, B: instance of np.ndarray
        input matrices.
    """
    if A.dtype != B.dtype:
        raise ValueError('A and B must be of the same type.')
    if A.dtype not in (np.float32, np.float64):
        raise ValueError('Data must be single or double precision float.')

    dot = linalg.get_blas_funcs('gemm', (A, B))
    A, trans_a = _impose_f_order(A)
    B, trans_b = _impose_f_order(B)
    return dot(alpha=1.0, a=A, b=B, trans_a=trans_a, trans_b=trans_b)


def norm_lasso(y, weights):
    weights = np.ones(y.shape)
    x = np.sum(weights*np.abs(y))
    return(x)


def norm_group_lasso(y, weights):
    weights = np.ones(y.shape)
    x = np.sum(np.power(weights*(y*y), 0.5))
    return(x)


def norm_mixed(y, weights, rho):
    weights = np.ones(y.shape)
    x = rho*norm_lasso(y, weights) + (1-rho)*norm_group_lasso(y, weights)
    return(x)


def proximal_operator_lasso(y, lambda_, weights):
    weights = np.ones(y.shape)
    x = y*np.maximum(np.zeros(y.shape), 1-(weights*lambda_/abs(y)))
    x[np.abs(x) < np.finfo(float).eps] = 0

    return(x)


def proximal_operator_group_lasso(y, lambda_, weights, groups):

    nscans = y.shape[0]
    nvoxels = y.shape[1]

    weights = np.ones(nscans*nvoxels)

    if groups == 'space':
        temp = np.power(np.sum(np.power(np.abs(y), 2), axis=1), 0.5)
        temp = temp.reshape(len(temp), 1)
    else:
        temp = np.power(np.sum(np.power(np.abs(y), 2), axis=0), 0.5)
        temp = temp.reshape(1, len(temp))

    # Reshapes weights array into matrix
    weights = weights.reshape(nscans, nvoxels)

    if groups == 'space':
        temp = np.matmul(temp, np.ones((1, nvoxels)))
    else:
        temp = np.matmul(np.ones((nscans, 1)), temp)

    temp = weights-(lambda_*weights/temp)

    x = y*np.maximum(np.zeros((nscans, nvoxels)), temp)
    x[np.abs(x) < np.finfo(float).eps] = 0

    return(x)


def proximal_operator_mixed_norm(y, lambda_, rho, weights, groups):

    # Initialize weights matrix
    weights = np.ones(y.shape)

    # Division parameter of proximal operator
    div = y/np.abs(y)
    div[np.isnan(div)] = 0

    # First parameter of proximal operator
    p_one = np.maximum(np.zeros(y.shape), (np.abs(y) - weights*lambda_*rho))

    # Second parameter of proximal operator
    if groups == 'space':
        foo = np.sum((np.maximum(np.zeros(y.shape), np.abs(y)
                                 - weights*lambda_*rho) ** 2), axis=1)
        foo = foo[:, np.newaxis]
        foo = np.tile(foo, y.shape[1])
    else:
        foo = np.sum((np.maximum(np.zeros(y.shape), np.abs(y)
                                 - weights*lambda_*rho) ** 2), axis=0)
        foo = foo[np.newaxis, :]
        foo = np.tile(foo, (y.shape[0], 1))

    p_two = np.maximum(np.zeros(y.shape),
                       weights-weights*lambda_*(1-rho)/np.sqrt(foo))

    # Proximal operation
    x = div*p_one*p_two

    # Return result
    return(x)


def proximal_nuclear(B, threshold, group):
    B_1 = B

    for g in range(1, np.max(group)):
        u, s, vh = np.linalg.svd(B[group == g, :], full_matrices=True)
        D = (s - threshold)*(s - threshold > 0)
        B_1[group == g, ] = np.matmul(u, (D * vh.T).T)

    return(B_1)


class Fista:

    def __init__(self, maxiter=5000, proximal='lasso', conv_criteria=1e-6,
                 lambda_=None, rho=0.75, weights=None, groups='space',
                 bic=None, miniter=100, max_noupdates=100, mfista=True,
                 max_lambda=None, betas=None, update_lambda=False):
        self.maxiter = maxiter
        self.miniter = miniter
        self.proximal = proximal
        self.conv_criteria = conv_criteria
        self.lambda_ = lambda_
        self.rho = rho
        self.weights = weights
        self.groups = groups
        self.bic = bic
        self.max_noupdates = max_noupdates
        self.mfista = mfista
        self.max_lambda = max_lambda
        self.beta = betas
        self.update_lambda = update_lambda

    def fit(self, x, y, r2only=True):

        if self.max_lambda is None:
            self.max_lambda = abs(np.matmul(np.transpose(x), y)).max()

        print('Running FISTA with maxiter {}, miniter {}, {}, '
              'conv_criteria {}, max_noupdates {}, lambda {}, '
              'max_lambda {} and rho {}...'
              .format(self.maxiter, self.miniter, self.proximal,
                      self.conv_criteria, self.max_noupdates, self.lambda_,
                      self.max_lambda, self.rho))

        if r2only:
            nscans = x.shape[1]
        else:
            nscans = x.shape[1]/2

        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)

        nvoxels = y.shape[1]

        X_hrf_norm = x.astype(np.float32)
        data_tilde = y.astype(np.float32)

        # Gets data ready for FISTA
        X_tilde = np.squeeze(x).astype(np.float32)
        X_tilde_trans = np.transpose(X_tilde).astype(np.float32)

        X_tilde_tt = fast_dot(X_tilde_trans, X_tilde)
        c_ist = 1/(linalg.norm(X_hrf_norm) ** 2)

        v = fast_dot(X_tilde_trans, data_tilde)

        # Initializes empty matrices for FISTA
        if self.beta is None:
            self.beta = np.zeros((X_tilde.shape[1], nvoxels), dtype=np.float32)
        self.Y_fit = np.zeros((data_tilde.shape[0], nvoxels), dtype=np.float32)
        self.s_zero = None
        nv = np.zeros((self.maxiter, 1))

        y_fista = np.zeros((X_tilde.shape[1], nvoxels), dtype=np.float32)
        prox_z = y_fista.copy()

        # MFISTA
        if self.mfista:
            Jcost_sparsenorm = 0
            Jcost_datafit = (0.5
                             * np.power(np.linalg.norm(data_tilde
                                                       - self.Y_fit, 2), 2))
            Jcost = Jcost_datafit + Jcost_sparsenorm
            max_MFISTAnoupdates = self.max_noupdates
            MFISTAnoupdates = 0

        t_fista = 1
        convergence_criteria = np.inf

        noise_estimate = self.lambda_.copy()
        precision = noise_estimate / 100000

        # Both conditions must be true in order to continue inside the loop.
        # If one of the condition is not met, the loop breaks.
        for niter in range(self.maxiter):

            # Updating of variabless
            beta_old = self.beta.copy()
            prox_z_old = prox_z.copy()
            y_ista = y_fista.copy()
            if self.mfista:
                Jcost_old = Jcost.copy()

            # Forward-Backward step
            z_ista = y_ista + c_ist*(v - fast_dot(X_tilde_tt, y_ista.astype(np.float32)))

            # Only R2* indexes are passed through the proximal operator
            z_ista_hrf = z_ista[0:nscans, ].copy()

            # Proximal methods
            if self.proximal == 'lasso':
                prox_z = proximal_operator_lasso(z_ista_hrf,
                                                 c_ist*self.lambda_,
                                                 self.weights)
            elif self.proximal == 'glasso':
                prox_z = proximal_operator_group_lasso(z_ista_hrf,
                                                       c_ist*self.lambda_,
                                                       self.weights,
                                                       self.groups)
            elif self.proximal == 'mixed':
                prox_z = proximal_operator_mixed_norm(z_ista_hrf,
                                                      c_ist*self.lambda_,
                                                      self.rho,
                                                      self.weights,
                                                      self.groups)
            else:  # LASSO
                prox_z = proximal_operator_lasso(z_ista_hrf,
                                                 c_ist*self.lambda_,
                                                 self.weights)

            if self.mfista:
                if self.proximal == 'lasso':
                    Jcost_sparsenorm = self.lambda_*norm_lasso(prox_z, None)
                if self.proximal == 'glasso':
                    Jcost_sparsenorm = self.lambda_*norm_group_lasso(prox_z,
                                                                     None)
                if self.proximal == 'mixed':
                    Jcost_sparsenorm = self.lambda_*norm_mixed(prox_z, None,
                                                               self.rho)

                Jcost_datafit = 0.5*np.power(np.linalg.norm(
                                             data_tilde
                                             - fast_dot(X_tilde, prox_z.astype(np.float32)),
                                             2), 2)
                Jcost = Jcost_datafit + Jcost_sparsenorm

            t_fista_old = t_fista
            t_fista = 0.5*(1+np.sqrt(1+4*np.power(t_fista_old, 2)))

            if self.mfista:
                if Jcost < Jcost_old or niter == 0:
                    self.beta = prox_z.copy()
                    y_fista = (prox_z
                               + (prox_z - prox_z_old)
                               * (t_fista_old-1)
                               / t_fista)
                    MFISTAnoupdates = 0
                else:
                    self.beta = prox_z_old.copy()
                    y_fista = (self.beta
                               + (t_fista_old/t_fista)
                               * (prox_z-self.beta))
                    MFISTAnoupdates = MFISTAnoupdates + 1
            else:
                self.beta = prox_z.copy()
                y_fista = (prox_z
                           + (prox_z - prox_z_old)
                           * (t_fista_old-1)
                           / t_fista)

            beta_deb = debiasing(X_tilde, y, prox_z)['beta']
            nv[niter] = np.sqrt(np.sum((np.dot(X_tilde, beta_deb) - y) ** 2) / nscans)

            if abs(nv[niter] - noise_estimate) > precision and self.update_lambda:
                self.lambda_ = self.lambda_ * noise_estimate / nv[niter]
            else:
                break

            # Calculates error between current and previous iteration.
            if niter > self.miniter:
                try:
                    nonzero_idxs_rows, nonzero_idxs_cols = \
                        np.where(np.abs(self.beta) > 10 * np.finfo(float).eps)
                    diff = np.abs(self.beta[nonzero_idxs_rows,
                                            nonzero_idxs_cols]
                                  - beta_old[nonzero_idxs_rows,
                                             nonzero_idxs_cols])
                    convergence_criteria = \
                        np.abs(diff / beta_old[nonzero_idxs_rows,
                                               nonzero_idxs_cols])
                except:
                    convergence_criteria = self.conv_criteria
            else:
                convergence_criteria = np.inf

            if self.mfista:
                if MFISTAnoupdates > max_MFISTAnoupdates \
                   and niter > self.miniter:
                    print('Maximum MFISTA non-updates.'
                          'Stopping convergence...')
                    break

            if np.all(convergence_criteria <= self.conv_criteria) \
               and niter > self.miniter:
                break

        # Save estimated bold signals for current iteration
        self.betafitts = fast_dot(X_tilde, self.beta.astype(np.float32))

        print('Fista completed after %i iterations.' % (niter))

        return(self)
