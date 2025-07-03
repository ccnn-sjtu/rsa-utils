import time 
import pickle
import numpy as np 
from tqdm import tqdm
import multiprocessing as mp 
import pingouin as pg 

import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, false_discovery_control
from scipy.spatial.distance import squareform, pdist
from scipy.stats import ttest_1samp, linregress, zscore

from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from skimage.measure import label

import nibabel as nib
from nilearn.image import smooth_img

eps_ = 1e-16

tqdm_bar_format="{l_bar}{bar:30} {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"

def get_t_stats(args):
    i, j, slice_data, baseline = args
    # If baseline is not 0, perform paired t-test against baseline
    diff = slice_data - baseline
    t_stats, pval = ttest_1samp(diff, 0, axis=0, alternative="greater")
    return i, j, np.vstack([t_stats, pval]).T

def second_lvl_stats(corrs, sl_baseline=0, fisher_z=True, n_cpu=1):

    # Check if sl_baseline is an int, and if so, convert it to an array
    if isinstance(sl_baseline, (int, float)):
        sl_baseline = np.ones_like(corrs) * sl_baseline

    if fisher_z is True: 
        corrs=.5*np.log((1+corrs)/(1-corrs+eps_)+eps_)
        sl_baseline=.5*np.log((1+sl_baseline)/(1-sl_baseline))

    jobs = []
    for i in range(corrs.shape[1]):
        for j in range(corrs.shape[2]):
            jobs.append((i, j, corrs[:, i, j], sl_baseline[:, i, j]))
    
    # initialize the data for save 
    stats = np.full(corrs.shape[1:]+(2,), np.nan)
    print(f'\n\tParallel computing using {n_cpu} cores:')

    with mp.Pool(processes=n_cpu) as pool:
        with tqdm(total=len(jobs), desc="\tGetting group stats",
                  bar_format=tqdm_bar_format,
                  ascii=True) as pbar:
            for i, j, corr in pool.imap_unordered(get_t_stats, jobs, 
                                        chunksize=max(1, len(jobs)//(n_cpu*100))):
                    stats[i, j] = corr
                    pbar.update()  

    return stats

def cluster_fdr(pvals, p_voxel=.01, p_cluster=.05,
                       n_connect=2, n_permutations=1000,
                       seed=1234):
    '''Cluster-wise FDR correction

    Args:
        pvals: p-value maps
        n_connect: which lvl connectivity
        p_theta_voxel: the voxel-wise p threshold
        p_theta_cluster: the cluster-wise p threshold
    '''
    rng = np.random.RandomState(seed)   
    px, py, pz = pvals.shape 

    # Get the valid voxels 
    pval_valid = pvals < p_voxel

    # Get the permutation baselines
    pval_flat = pval_valid.reshape([-1]) 
    perm_baslines = []
    for _ in tqdm(range(n_permutations), desc="\tCorrecting voxels", 
                  bar_format=tqdm_bar_format,
                  ascii=True):
        # shuffle the pval matrix 
        pi = np.random.permutation(pval_flat).reshape([px, py, pz])
        # calculate the cluster,
        labels = label(pi, connectivity=n_connect)
        # count n_voxels within a cluster
        cluster_ind, voxels_in_cluster = np.unique(labels, return_counts=True)
        # store the size of the largest non-background cluster 
        perm_baslines.append(voxels_in_cluster[1:].max() if len(cluster_ind)>1 else 0)
    perm_baslines = np.array(perm_baslines)

    # Get the 1-cluster-wise p percentile
    cluster_threshold = np.percentile(perm_baslines, 100*(1-p_cluster))

    # Get the cluster-level p values
    labels = label(pval_valid, connectivity=n_connect)
    cluster_ind, voxels_in_cluster = np.unique(labels, return_counts=True)
    voxels_in_cluster = voxels_in_cluster[1:] # exclude the background 0
    def get_cluster_p(x):
        tmp = np.sort(np.append(perm_baslines, x))
        pos = np.searchsorted(tmp, x, side='right')
        return 1-pos/len(tmp)
    cluster_p = np.array(list(map(get_cluster_p, voxels_in_cluster)))
    # conduct false rate discovery-BH, correction 
    cluster_fdr_p = false_discovery_control(cluster_p, method='bh')

    # Apply thresholds and create final map
    cluster_fdr_p_map = np.full((px, py, pz), np.nan)
    mask = (
        ~np.isnan(pvals) & # should be the valid voxels that pass the voxel-wise p threshold
        (labels!=0) & # label 0 is the background  
        (cluster_fdr_p[labels-1] < p_voxel) & # label here return the corrected p for each cluster, use labels-1 to index the cluster
        (voxels_in_cluster[labels-1] >= cluster_threshold) # use labels-1 to index the cluster
    )
    cluster_fdr_p_map[mask] = cluster_fdr_p[labels[mask] - 1]

    return cluster_fdr_p_map

def stats_to_nii(corrs, affine, size=[60, 60, 60], radius=4, 
                 stride=1, corr_mask=False, header=False,
                 smooth=True):
    
    # Unpack sizes and strides
    r, s = radius, stride

    # Map the correlation back to the pre-convolved sapce
    i, j, k = np.meshgrid(np.arange(corrs.shape[0]), 
                          np.arange(corrs.shape[1]), 
                          np.arange(corrs.shape[2]), 
                          indexing='ij')
    x, y, z = i*s+r, j*s+r, k*s+r

    # Extract p-values and t-values
    img_nii = np.full(size, np.nan)
    img_nii[x, y, z] = corrs
 
    # Masking to remove data out of the brian area
    pre_masked = (~np.isnan(img_nii)).sum()
    if corr_mask is not False: 
        img_nii = np.where(corr_mask==1, img_nii, np.nan)
        post_masked = (~np.isnan(img_nii)).sum()
        print(f'\t{post_masked} voxels remains after masked, {pre_masked-post_masked} was outside the brain area')
        
    file = nib.Nifti1Image(img_nii, affine, header=header)
    
    return smooth_img(file, fwhm='fast') if smooth else file

class BasicKernel:

    def __init__(self, estimator, cv=False, mask='unmasked', 
                 scoring='pearsonr', **kwargs):
        self.masks = None if mask=='unmasked' else pickle.load(open(mask, 'rb'))
        self.cv = cv if cv else None 
        self.scoring = eval(scoring)
        self.estimator = estimator
        for k, v in kwargs.items(): setattr(self, k, v)

    def standardize_and_mask(self, x):
        if self.masks is None: return zscore(x)
        return zscore(np.hstack([x[mask] for mask in self.masks]))
    
    def fit(self, x, y):
        raise NotImplementedError("The 'fit' method must be implemented.")
    
class RDM_regress(BasicKernel):
    
    def __init__(self, estimator, cv=False, mask='unmasked', 
                 scoring='pearsonr', pdist_metric='correlation',
                 **kwargs):
        super().__init__(estimator, cv, mask, scoring, **kwargs)
        self.pdist_metric = pdist_metric
        self.verbose = f'RDM {estimator["fn"]}'
    
    def fit(self, args):
        sub_id, x, y, z, x_neural, model_rdms = args
        x_neural = x_neural.reshape([x_neural.shape[0], -1])
        sl_rdm = squareform(pdist(x_neural, self.pdist_metric))
        neural_z = self.standardize_and_mask(sl_rdm)
        if np.isnan(neural_z).any(): 
            nan_beta = np.zeros([len(model_rdms)])+np.nan
            return sub_id, x, y, z, nan_beta
        models_z = np.vstack([self.standardize_and_mask(m) 
                        for m in model_rdms]).T
        beta = self.regress(neural_z, models_z)
        return sub_id, x, y, z, beta
    
    def standardize_and_mask(self, x):
        if self.masks is None: return zscore(squareform(x, checks=False))
        return zscore(np.hstack([x[mask].reshape([-1]) for mask in self.masks]))
    
    def regress(self, y_neural, x_model):
        df = pg.linear_regression(x_model, y_neural, remove_na=True)
        return df.query("names!='Intercept'")['coef'].values
    
class Decoder(BasicKernel):
    def __init__(self, estimator, cv=False, mask='unmasked', 
                 scoring='pearsonr', **kwargs):
        super().__init__(estimator, cv, mask, scoring, **kwargs)
        self.verbose = f'{estimator["fn"]} decoding'

    def fit(self, args):
        sub_id, x, y, z, x_neural, y_models = args
        x_neural = x_neural.reshape([x_neural.shape[0], -1])
        x_stand = self.standardize_and_mask(x_neural)
        valid_rows = ~np.isnan(x_stand).any(axis=1)
        if valid_rows.sum()<int(x_stand.shape[0]/4): 
            return sub_id, x, y, z, np.nan
        x_stand_valid = x_stand[valid_rows]
        rs = []
        for y_model in y_models:
            y_model_valid = y_model[valid_rows]
            y_hat = self.decode(x_stand_valid, y_model_valid)
            r, _ = self.scoring(y_model_valid, y_hat)
            rs.append(r)
        return sub_id, x, y, z, np.array(rs) 
        
    def decode(self, x_neural, y_model):
        
        if self.cv is None:
            est = eval(self.estimator['fn'])(**self.estimator['hyper']) 
            est.fit(x_neural, y_model)
            y_hat = est.predict(x_neural)
            # Check if y_hat is constant
            if np.allclose(y_hat, y_hat[0]):
                y_hat += np.random.randn(*y_hat.shape)*1e-16
            return y_hat 
        
        spliter = eval(self.cv['fn'])(**self.cv['hyper'])
        y_hat = np.zeros_like(y_model)
        for train_index, test_index in spliter.split(x_neural):
            X_train, X_test = x_neural[train_index], x_neural[test_index]
            y_train = y_model[train_index]
            est = eval(self.estimator['fn'])(**self.estimator['hyper']) 
            est.fit(X_train, y_train)
            y_hat[test_index] = est.predict(X_test)
        return y_hat

class Searchlight:

    def __init__(self, sl_kernel, radius=4, strides=1, n_paral=10):
        self.sl_kernel = sl_kernel
        self.k = radius*2+1
        self.s = strides
        self.n_paral = n_paral

    def search_and_fit(self, fmri_data, model_preds):

        if len(np.shape(fmri_data))!=5:
            print("\nThe shape of input for fmridata must be [n_phi, n_sub, nx, ny, nz].\n")
            return "Invalid input!"

        # get the number of conditions, subjects and the size of the fMRI-img
        _, n_sub, nx, ny, nz = np.shape(fmri_data)

        # calculate the number of the calculation units in the x, y, z directions
        n_x = int((nx - self.k) / self.s)+1
        n_y = int((ny - self.k) / self.s)+1
        n_z = int((nz - self.k) / self.s)+1

        # jobs
        x_coords, y_coords, z_coords = np.meshgrid(np.arange(n_x), 
                                                   np.arange(n_y), 
                                                   np.arange(n_z), 
                                                   indexing='ij')
        coords = np.stack((x_coords, y_coords, z_coords), axis=-1).reshape(-1, 3)
        jobs = []
        for sub_id in range(n_sub):
            for x, y, z in coords:
                slice_data = fmri_data[:, sub_id, 
                                        x*self.s:x*self.s+self.k, 
                                        y*self.s:y*self.s+self.k, 
                                        z*self.s:z*self.s+self.k]
                jobs.append((sub_id, x, y, z, slice_data, model_preds))
        
        # initialize the data for save 
        sl_stats = np.full([n_sub, n_x, n_y, n_z, len(model_preds)], np.nan)
        print(f'\n\tParallel computing using {self.n_paral} cores:')

        with mp.Pool(processes=self.n_paral) as pool:
            with tqdm(total=len(jobs), desc=f"\tSearchlight {self.sl_kernel.verbose}",
                        bar_format=tqdm_bar_format, ascii=True) as pbar:
                for sub_id, x, y, z, stats in pool.imap_unordered(
                        self.sl_kernel.fit, jobs, 
                        chunksize=max(1, len(jobs)//(self.n_paral*100))):
                        # Efficiently update corrs array
                        sl_stats[sub_id, x, y, z] = stats
                        pbar.update()  

        return sl_stats
