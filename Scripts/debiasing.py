#!/usr/bin/python
import sys, argparse, os, socket, getpass, datetime
import numpy as np
import csv
import nibabel as nib
import subprocess
import shutil
import time
import scipy as sci
from sklearn.linear_model import RidgeCV
from scipy.signal import find_peaks
from Scripts.hrf_matrix import HRFMatrix

def read_volumes(paths, n_noise_voxels=10000, discard_pcg=1):
    # Input checking
    if paths:
        if not 'datasets' in paths:
            raise ValueError(
                'No echo volumens were inserted. Please insert the path(s) to your echo volume(s)')
        if not 'mask' in paths:
            raise ValueError(
                'No mask volumens was inserted. Please insert the path to your mask volume')
        if not 'te' in paths:
            print('No 1d file was inserted. Data will be treated as Single Echo data.')
    else:
        raise ValueError(
            'No dictionary containing the paths to the files was inserted. Please insert a dictionary with the paths to your files.')

    print('Reading files...')

    if paths['te'] is not None:
        # Get TE values
        with open(paths['dir'] + paths['te'], 'r') as f:
            reader = csv.reader(f)
            te_list = list(reader)[0]

        te_values = [float(i)/1000 for i in te_list]
        nTE = len(te_values)
    else:
        nTE = 1
        te_values = np.array([0])

    if type(paths['mask']) is list:
        nMasks = len(paths['mask'])
        mask_data = nib.load(paths['dir'] + paths['mask']).get_fdata()
        for maskidx in range(nMasks-1):
            temp_mask = nib.load(paths['dir'] + paths['mask'][maskidx+1]).get_fdata()
            mask_data = mask_data + temp_mask
    else:
        mask_data = nib.load(paths['dir'] + paths['mask']).get_fdata()

    if 'thr_mask' in paths:
        if type(paths['thr_mask']) is list:
            nthrMasks = len(paths['thr_mask'])
            thr_mask_data = nib.load(paths['dir'] + paths['thr_mask'][0]).get_fdata()
            for thrmaskidx in range(nthrMasks-1):
                temp_thr_mask = nib.load(paths['dir'] + paths['thr_mask'][thrmaskidx+1]).get_fdata()
                thr_mask_data = thr_mask_data + temp_thr_mask
        else:
            thr_mask_data = nib.load(paths['dir'] + paths['thr_mask']).get_fdata()

    if nTE != 1:
        echo_hdr = nib.load(paths['dir'] + paths['auc'][0]).header
    else:
        echo_hdr = nib.load(paths['dir'] + paths['auc']).header

    for echo_idx in range(nTE):
        # Reads echo file
        if nTE != 1:
            # Add slash at the start if it did not have it
            if not '/' in paths['datasets'][echo_idx]:
                paths['datasets'][echo_idx] = '/' + paths['datasets'][echo_idx]
            echo_data = nib.load(paths['dir'] + paths['datasets'][echo_idx]).get_fdata()
        else:
            # Add slash at the start if it did not have it
            if not '/' in paths['datasets']:
                paths['datasets'][echo_idx] = '/' + paths['datasets']
            echo_data = nib.load(paths['dir'] + paths['datasets']).get_fdata()

        # Gets dimensions of data
        if len(mask_data.shape) < len(echo_data.shape):
            dims = echo_data.shape
        else:
            dims = mask_data.shape
        # Checks nscans are the same among different time echos
        if echo_idx > 1 and nscans != dims[3]:
            raise ValueError(
                'Introduced files do not contain same number of scans. \n')
        else:
            nscans = dims[3]
        # Checks nvoxels are the same among different time echos
        if echo_idx > 1 and nvoxels_orig != np.prod(echo_hdr['dim'][1:4]):
            raise ValueError(
                'Introduced files do not contain same number of voxels. \n')
        else:
            nvoxels_orig = np.prod(echo_hdr['dim'][1:4])
        # Checks TR is the same among different time echos
        if echo_idx > 1 and tr_value != echo_hdr['pixdim'][4]:
            raise ValueError(
                'Introduced files do not contain same TR value. \n')
        else:
            tr_value = echo_hdr['pixdim'][4]

        # Initiates masked_data to make loop faster
        masked_data = np.zeros((dims[0], dims[1], dims[2], dims[3]))
        if 'thr_mask' in paths:
            thr_masked_data = np.zeros((dims[0], dims[1], dims[2], dims[3]))

        # Masks data
        if len(mask_data.shape) < 4:
            for i in range(nscans):
                masked_data[:, :, :, i] = np.squeeze(echo_data[:, :, :, i]) * mask_data
                if 'thr_mask' in paths:
                    thr_masked_data[:, :, :, i] = np.squeeze(echo_data[:, :, :, i] * thr_mask_data)
        else:
            masked_data = echo_data * mask_data
            if 'thr_mask' in paths:
                thr_masked_data = echo_data * thr_mask_data

        # Initiates data_restruct to make loop faster
        data_restruct_temp = np.reshape(np.moveaxis(masked_data, -1, 0), (nscans, nvoxels_orig))
        mask_idxs = np.unique(np.nonzero(data_restruct_temp)[1])
        data_restruct = data_restruct_temp[:, mask_idxs]

        if 'thr_mask' in paths:
            thr_data_restruct_temp = np.reshape(thr_masked_data, (nscans, nvoxels_orig))
            thr_mask_restruct_temp = np.reshape(thr_mask_data, (1, nvoxels_orig))
            noise_idxs = np.unique(np.nonzero(thr_mask_restruct_temp)[1])
            noise_restruct = thr_data_restruct_temp[:, noise_idxs]
        else:
            noise_idxs = np.where(data_restruct_temp == 0)
            noise_cols = np.unique(noise_idxs[1])[0:n_noise_voxels]
            noise_restruct = data_restruct_temp[:, noise_cols]

        # Concatenates different time echo signals to (nTE nscans x nvoxels)
        if echo_idx == 0:
            signal = data_restruct.astype('float32')
            noise = noise_restruct.astype('float32')
        else:
            signal = np.concatenate((signal, data_restruct), axis=0)
            noise = np.concatenate((noise, noise_restruct), axis=0)

        if signal.shape[1] == 0 or noise.shape[1] == 0:
            sys.exit('The discarding percentage is too low. Please try again with a higher threshold.')

        if echo_idx == 0:
            print(signal.shape[0], 'scans and',
                    signal.shape[1], 'voxels inside the mask')
        print(str(np.round((echo_idx + 1) / nTE * 100)) + '%')

    print('Files successfully read')

    output = {'signal': signal, 'noise': noise, 'mask': mask_idxs,
            'te': te_values, 'tr': tr_value, 'dims': dims, 'header': echo_hdr}

    return(output)


def reshape_and_mask2Dto4D(signal2d, dims, mask_idxs):
    signal4d = np.zeros((dims[0] * dims[1] * dims[2], signal2d.shape[0]))

    # Merges signal on mask indices with blank image
    for i in range(dims[3]):
        if len(mask_idxs.shape) > 3:
            idxs = np.where(mask_idxs[:, :, :, i] != 0)
        else:
            idxs = mask_idxs

        signal4d[idxs, i] = signal2d[i, :]

    # Reshapes matrix from 2D to 4D double
    signal4d = np.reshape(signal4d, (dims[0], dims[1], dims[2], signal2d.shape[0]))
    del signal2d, idxs, dims, mask_idxs
    return(signal4d)


def export_volumes(volumes, paths, header, nscans, history):

    if not 'output_dir' in paths:
        paths['output_dir'] = paths['dir']

    if 'LARS' in paths['auc']:
        lars_key = 'LARS.'
    else:
        lars_key = ''

    # R2* estimates
    if 'beta' in volumes:
        print('Exporting BETA estimates...')
        img = nib.nifti1.Nifti1Image(volumes['beta'], None, header=header)
        filename = paths['output_dir'] + '/' + paths['auc'][1:len(paths['auc']) - 7] + '.DR2.' + paths['key'] + 'nii.gz'
        nib.save(img, filename)
        print('BETA estimates correctly exported.')
        subprocess.run('3dcopy {} {} -overwrite'.format(filename, filename), shell=True)
        if history is not None:
            print('Updating file history...')
            subprocess.run('3dNotes -h "' + history + '" ' + filename, shell=True)
            print('File history updated.')

    # Fitted signal estimates
    if 'betafitts' in volumes:
        print('Exporting BETAFIITS estimates...')
        if paths['te'] is not None:
            # Get TE values
            with open(paths['dir'] + paths['te'], 'r') as f:
                reader = csv.reader(f)
                te_list = list(reader)[0]

            te_values = [float(i)/1000 for i in te_list]
            nTE = len(te_values)
        else:
            nTE = 1

        if nTE > 1:
            for teidx in range(nTE):
                img = nib.nifti1.Nifti1Image(volumes['betafitts'][teidx*nscans+1:nscans*(teidx+1),:], None, header=header)
                filename = paths['output_dir'] + '/' + paths['auc'][1:len(paths['auc']) - 7] + '.dr2HRF_E0' + str(teidx+1) + '.' + paths['key'] + 'nii.gz'
                nib.save(img, filename)
                print('BETAFIITS estimates correctly exported.')
                subprocess.run('3dcopy {} {} -overwrite'.format(filename, filename), shell=True)
                if history is not None:
                    print('Updating file history...')
                    subprocess.run('3dNotes -h "' + history + '" ' + filename, shell=True)
                    print('File history updated.')
        else:
            img = nib.nifti1.Nifti1Image(volumes['betafitts'], None, header=header)
            filename = paths['output_dir'] + '/' + paths['auc'][1:len(paths['auc']) - 7] + '.dr2HRF.' + paths['key'] + 'nii.gz'
            nib.save(img, filename)
            print('BETAFIITS estimates correctly exported.')
            subprocess.run('3dcopy {} {} -overwrite'.format(filename, filename), shell=True)
            if history is not None:
                print('Updating file history...')
                subprocess.run('3dNotes -h "' + history + '" ' + filename, shell=True)
                print('File history updated.')

    # S0 estimates
    if 's_zero' in volumes:
        if volumes['s_zero'].size != 0:
            print('Exporting S0 estimates...')
            img = nib.nifti1.Nifti1Image(volumes['s_zero'], None, header=header)
            filename = paths['output_dir'] + '/' + paths['auc'][1:len(paths['auc']) - 7] + '.s0.' + paths['key'] + 'nii.gz'
            nib.save(img, filename)
            print('S0 estimates correctly exported.')
            subprocess.run('3dcopy {} {} -overwrite'.format(filename, filename), shell=True)
            if history is not None:
                print('Updating file history...')
                subprocess.run('3dNotes -h "' + history + '" ' + filename, shell=True)
                print('File history updated.')
        else:
            print('No S0 estimates to export.')

    # AUC timecourses
    if 'auc' in volumes:
        if volumes['auc'].size != 0:
            print('Exporting AUC timecourses...')
            img = nib.nifti1.Nifti1Image(volumes['auc'], None, header=header)
            filename = paths['output_dir'] + '/' + paths['auc'][1:len(paths['auc']) - 7] + '.AUC.THR.' + paths['key'] + 'nii.gz'
            nib.save(img, filename)
            print('AUC timecourses correctly exported in {}'.format(filename))
            subprocess.run('3dcopy {} {} -overwrite'.format(filename, filename), shell=True)
            if history is not None:
                print('Updating file history...')
                subprocess.run('3dNotes -h "' + history + '" ' + filename, shell=True)
                print('File history updated.')
        else:
            print('No AUC timecourses to export.')

def read_auc(paths, nscans, mask_idxs):

    auc_4d = nib.load(paths['dir'] + paths['auc']).get_fdata()
    auc_4d_hdr = nib.load(paths['dir'] + paths['auc']).header
    mask_data = nib.load(paths['dir'] + paths['mask']).get_fdata()

    nvoxels_orig = np.prod(auc_4d_hdr['dim'][1:4])

    # Gets dimensions of data
    if len(mask_data.shape) < len(auc_4d.shape):
        dims = auc_4d.shape
    else:
        dims = mask_data.shape

    # Initiates masked_data to make loop faster
    masked_data = np.zeros((dims[0], dims[1], dims[2], dims[3]))

    # Masks data
    if len(mask_data.shape) < 4:
        for i in range(nscans):
            masked_data[:, :, :, i] = np.squeeze(auc_4d[:, :, :, i]) * mask_data
    else:
        masked_data = auc_4d * mask_data

    # Initiates data_restruct to make loop faster
    auc_2d_temp = np.reshape(np.moveaxis(masked_data, -1, 0), (nscans, nvoxels_orig))
    # mask_idxs = np.unique(np.nonzero(auc_2d_temp)[1])
    auc_2d = auc_2d_temp[:, mask_idxs]

    return(auc_2d)


# Performs the debiasing step on an AUC timeseries obtained considering the integrator model
def debiasing_int(auc, hrf, y, is_ls):

    # Find indexes of nonzero coefficients
    nonzero_idxs = np.where(auc != 0)[0]
    n_nonzero = len(nonzero_idxs) # Number of nonzero coefficients

    # Initiates beta
    beta = np.zeros((y.shape))
    S = 0

    if n_nonzero != 0:
        # Initiates matrix S and array of labels
        S = np.zeros((y.shape[0], n_nonzero+1))
        labels = np.zeros((y.shape[0]))

        # Gives values to S design matrix based on nonzeros in AUC
        # It also stores the labels of the changes in the design matrix
        # to later generate the debiased timeseries with the obtained betas
        for idx in range(n_nonzero+1):
            if idx == 0:
                S[0:nonzero_idxs[idx], idx] = 1
                labels[0:nonzero_idxs[idx]] = idx
            elif idx == n_nonzero:
                S[nonzero_idxs[idx-1]:, idx] = 1
                labels[nonzero_idxs[idx-1]:] = idx
            else:
                S[nonzero_idxs[idx-1]:nonzero_idxs[idx], idx] = 1
                labels[nonzero_idxs[idx-1]:nonzero_idxs[idx]] = idx

        # Performs the least squares to obtain the beta amplitudes
        if is_ls:
            beta_amplitudes, _, _, _ = np.linalg.lstsq(np.dot(hrf, S), y, rcond=None) # b-ax --> returns x
        else:
            clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10]).fit(np.dot(hrf, S), y)
            beta_amplitudes = clf.coef_

        # Positions beta amplitudes in the entire timeseries
        for amp_change in range(n_nonzero):
            beta[labels == amp_change] = beta_amplitudes[amp_change]

    return(beta, S)


def debiasing(x, y, beta):

    beta_out = np.zeros(beta.shape)
    fitts_out = np.zeros(y.shape)

    index_voxels = np.unique(np.where(abs(beta) > 10 * np.finfo(float).eps)[1])

    for voxidx in range(len(index_voxels)):
        index_events_opt = np.where(
            abs(beta[:, index_voxels[voxidx]]) > 10 * np.finfo(float).eps)[0]

        X_events = x[:, index_events_opt]
        beta2save = np.zeros((beta.shape[0], 1))

        coef_LSfitdebias, residuals, rank, s = sci.linalg.lstsq(
            X_events, y[:, index_voxels[voxidx]], cond=None)
        beta2save[index_events_opt, 0] = coef_LSfitdebias

        beta_out[:, index_voxels[voxidx]] = beta2save.reshape(len(beta2save))
        fitts_out[:, index_voxels[voxidx]] = np.dot(X_events, coef_LSfitdebias)

    return(beta_out, fitts_out)


def main(argv):

    parser = argparse.ArgumentParser(description='Debiasing function')
    parser.add_argument('--datasets', type=str, help='Original datasets that were analysed.', default=None, nargs='+')
    parser.add_argument('--mask', type=str, help='Mask of voxels to analyse.', default=None, nargs=1)
    parser.add_argument('--mask_dwm', type=str, help="File name of the mask of voxels in the Deep White Matter. It is used for thresholdind purposes.", default=None, nargs=1)
    parser.add_argument('--te', type=str, help='1D file with echo times (default=None). If no TE is given, the single echo version will be run.', default=None, nargs=1)
    parser.add_argument('--dir', type=str, help='Main directory with datasets (default=cwd).', default=None, nargs=1)
    parser.add_argument('--auc', type=str, help='File name of the AUC results.', default=None, nargs=1)
    parser.add_argument('--auc_dwm', type=str, help="File name of the Deep White Matter's AUC results. It is used for thresholdind purposes.", default=None, nargs=1)
    parser.add_argument('--key', type=str, help='String that the output filenames will contain at their end. (default="").', default=None, nargs=1)
    parser.add_argument('--hrfmodel', type=str, help='HRF model for 3dDeconvolve from AFNI (default="SPMG1").', default=None, nargs=1)
    parser.add_argument('--thr', type=int, help='Percentil value (integer) of the DWM AUC to apply the threshold with (default=95).', default=95, nargs=1)
    parser.add_argument('--afni', help='Whether the HRF matrix is generated using 3dDeconvolve from AFNI (default=False).', default=False, action='store_true')
    parser.add_argument('--max', help='Whether to only keep maximum values in AUC peaks (default=False).', default=False, action='store_true')
    parser.add_argument('--ls', help='Whether to use least squares to apply debiasing (default=False). It is the fastest option. \
                        Otherwise, ridge regression with cross validation is used for further stability.', default=False, action='store_true')
    parser.add_argument('--dist', type=int, help='Minimum distance between two AUC peaks (default=2).',default=3, nargs=1)
    parser.add_argument('--stc', help='Whether to compute Spatio-Temporal Clustering on the thresholded AUC before debiasing.', default=False, action='store_true')
    parser.add_argument('--cluster_size', type=int, help='Cluster size in STC (default=5).', default=5, nargs=1)
    parser.add_argument('--integrator', help='Whether the AUC was calculated with the integrator formulation (default=False).', default=False, action='store_true')
    args = parser.parse_args()

    # Data parameters
    if args.datasets is not None:
        datasets = args.datasets
    else:
        sys.exit('Error: No datasets were provided. \nExited pySPFM.')
    if args.mask is not None:
        mask = ' '.join(args.mask)
    else:
        sys.exit('Error: No mask was provided. \nExited pySPFM.')
    if args.mask_dwm is not None:
        mask_dwm = ' '.join(args.mask_dwm)
    else:
        print('No thresholding mask was given.')
    if args.te is not None:
        args.te = ' '.join(args.te)
    else:
        datasets = ' '.join(datasets)
        # args.te = [1]
    if args.dir is not None:
        args.dir = ' '.join(args.dir)
    else:
        args.dir = os.getcwd()
    if args.auc is not None:
        args.auc = ' '.join(args.auc)
    else:
        sys.exit('Error: No AUC dataset was provided. \nExited pySPFM.')
    if args.auc_dwm is not None:
        auc_dwm = ' '.join(args.auc_dwm)
    if args.key is not None:
        key = ' '.join(args.key) + '.'
    else:
        key = ''
    if args.hrfmodel is not None:
        lop_hrf = args.hrfmodel
    else:
        lop_hrf = 'SPMG1'
    if type(args.thr) is list:
        thr_pcl = args.thr[0]
    else:
        thr_pcl = args.thr
    if type(args.afni) is list:
        is_afni = args.afni[0]
    else:
        is_afni = args.afni
    if type(args.max) is list:
        max_only = args.max[0]
    else:
        max_only = args.max
    if type(args.ls) is list:
        is_ls = args.ls[0]
    else:
        is_ls = args.ls
    if type(args.dist) is list:
        dist = args.dist[0]
    else:
        dist = args.dist
    if type(args.stc) is list:
        compute_stc = args.stc[0]
    else:
        compute_stc = args.stc
    if type(args.cluster_size) is list:
        cluster_size = args.cluster_size[0]
    else:
        cluster_size = args.cluster_size
    if type(args.integrator) is list:
        is_integrator = args.integrator[0]
    else:
        is_integrator = args.integrator

    paths = {'datasets': datasets, 'mask': mask, 'te': args.te, 'dir': args.dir, 'auc': args.auc, 'key': key}

    # Command to save in file's history to be seen with the 3dinfo command
    args_str = ' '.join(argv)
    history_str = '[{username}@{hostname}: {date}] python debiasing_int.py {arguments}'.format(username=getpass.getuser(),\
                    hostname=socket.gethostname(), date=datetime.datetime.now().strftime('%c'), arguments=args_str)

    # If the paths only contains one / it means it's a local path
    if paths['dir'].count('/') == 1:
        paths['dir'] = os.getcwd() + paths['dir']
    
    input_data = read_volumes(paths)
    orig_signal = input_data['signal']
    nscans = input_data['dims'][3]
    nvoxels = orig_signal.shape[1]

    # Generates HRF matrix
    hrf_matrix = HRFMatrix(TR=input_data['tr'], TE=input_data['te'], nscans=nscans, r2only=True, has_integrator=False, is_afni=is_afni, lop_hrf=lop_hrf)
    hrf_matrix.generate_hrf()
    hrf = hrf_matrix.X_hrf_norm

    # Reads AUC
    auc = read_auc(paths, nscans, input_data['mask'])

    # Reads DWM AUC if both MASK and AUC of DWM are given
    if thr_pcl is not None:
        if args.mask_dwm is not None and args.auc_dwm is not None:
            print('Thresholding AUC...')
            paths['auc'] = auc_dwm
            paths['mask'] = mask_dwm
            auc_dwm = read_auc(paths, nscans, input_data['mask'])
            paths['mask'] = mask

            # Thresholding step
            thr_val = np.percentile(auc_dwm, thr_pcl)
            auc[auc <= thr_val] = 0
            print('Reshaping thresholded AUC timecourses from 2D to 4D...')
            auc_out = reshape_and_mask2Dto4D(auc, input_data['dims'], input_data['mask'])
            print('Thresholded AUC timecourses reshaping completed.')
            output_vols = {'auc': auc_out}
            export_volumes(output_vols, paths, input_data['header'], nscans, history_str)
        # Extracts DWM AUC from main dataset
        if args.mask_dwm is not None and args.auc_dwm is None:
            print('Thresholding AUC...')
            paths['mask'] = mask_dwm
            auc_dwm = read_auc(paths, nscans, input_data['mask'])
            paths['mask'] = mask
        
            # Thresholding step
            thr_val = np.percentile(auc_dwm, thr_pcl)
            auc[auc <= thr_val] = 0
            print('Reshaping thresholded AUC timecourses from 2D to 4D...')
            auc_out = reshape_and_mask2Dto4D(auc, input_data['dims'], input_data['mask'])
            print('Thresholded AUC timecourses reshaping completed.')
            output_vols = {'auc':auc_out}
            export_volumes(output_vols, paths, input_data['header'], nscans, history_str)

    if compute_stc:
        print('Computing spatio-temporal clustering...')
        if 'LARS' in paths['auc']:
            lars_key = 'LARS.'
        else:
            lars_key = ''
        auc_filename = paths['output_dir'] + paths['datasets'][1:len(paths['datasets']) - 7] + '.pySPFM.' + lars_key + 'AUC.THR.' + paths['key'] + 'nii.gz'
        paths['key'] = 'STC.' + paths['key']
        stc_filename = paths['datasets'][1:len(paths['datasets']) - 7] + '.pySPFM.' + lars_key + 'AUC.THR.' + paths['key'] + 'nii.gz'
        stc_filename_full = paths['output_dir'] + stc_filename
        stc_command = 'python /export/home/eurunuela/public/MEPFM/spatio-temporal-clustering/stc.py --input {} --output {} --clustsize {} --thr {}'.format(
                    auc_filename, stc_filename_full, cluster_size, thr_val)
        subprocess.run(stc_command, shell=True)
        del auc, auc_out, auc_dwm
        paths['auc'] = '/' + stc_filename
        print('Reading AUC after spatio-temporal clustering...')
        auc = read_auc(paths, nscans, input_data['mask'])

    if is_integrator:
        # Initiates beta matrix
        beta = np.zeros((nscans, nvoxels))
        percentage_old = 0

        if is_ls and not max_only:
            print('Distance selected for peak finding: {}'.format(dist))

        print('Starting debiasing step...')
        print('0% debiased...')
        # Performs debiasing
        for vox_idx in range(nvoxels):
            # Keep only maximum values in AUC peaks
            if is_ls:
                # print('Finding peaks...')
                temp = np.zeros((auc.shape[0],))
                if max_only:
                    for timepoint in range(nscans):
                        if timepoint != 0:
                            if auc[timepoint, vox_idx] != 0:
                                if auc[timepoint - 1, vox_idx] != 0:
                                    if auc[timepoint, vox_idx] > auc[timepoint - 1, vox_idx]:
                                        max_idx = timepoint
                                else:
                                    max_idx = timepoint
                            else:
                                if auc[timepoint - 1, vox_idx] != 0:
                                    temp[max_idx] = auc[max_idx, vox_idx].copy()
                        else:
                            if auc[timepoint, vox_idx] != 0:
                                max_idx = timepoint
                else:
                    peak_idxs, _ = find_peaks(auc[:,vox_idx], prominence=thr_val, distance=dist)
                    temp[peak_idxs] = auc[peak_idxs, vox_idx].copy()

                auc[:, vox_idx] = temp.copy()

            beta[:, vox_idx], S = debiasing_int(auc[:, vox_idx], hrf, orig_signal[:, vox_idx], is_ls)

            percentage = np.ceil((vox_idx+1)/nvoxels*100)
            if percentage > percentage_old:
                print('{}% debiased...'.format(int(percentage)))
                percentage_old = percentage

        # Calculates fitted signal
        betafitts = np.dot(hrf, beta)

        if max_only:
            if 'LARS' in paths['auc']:
                lars_key = 'LARS.'
            else:
                lars_key = ''
            max_filename = paths['datasets'][1:len(paths['datasets']) - 7] + '.pySPFM.' + lars_key + 'AUC.THR.MAX.' + paths['key'] + 'nii.gz'
            paths['auc'] = '/' + max_filename
            auc_out = reshape_and_mask2Dto4D(auc, input_data['dims'], input_data['mask'])
            print('Thresholded AUC peaks timecourses reshaping completed.')
            output_vols = {'auc':auc_out}
            export_volumes(output_vols, paths, input_data['header'], nscans, history_str)
    else:
        print('Starting debiasing...')
        deb_output = debiasing(hrf, orig_signal, auc)
        beta = deb_output['beta']
        betafitts = deb_output['betafitts']
        print('Debiasing finished.')

    # Transforms 2D to 4D
    print('Reshaping BETA timecourses from 2D to 4D...')
    beta_out = reshape_and_mask2Dto4D(beta, input_data['dims'], input_data['mask'])
    print('BETA timecourses reshaping completed.')

    print('Reshaping BETAFIITS timecourses from 2D to 4D...')
    betafitts_out = reshape_and_mask2Dto4D(betafitts, input_data['dims'], input_data['mask'])
    print('BETAFITTS timecourses reshaping completed.')

    # Dictionary to pass args to WriteVolumes
    output_vols = {'beta': beta_out, 'betafitts': betafitts_out}

    # Exports results
    export_volumes(output_vols, paths, input_data['header'], nscans, history_str)

    print('Debiasing finished. Your results were successfully exported.')

if __name__ == "__main__":
   main(sys.argv[1:])
