import json
import pickle
import argparse
import numpy as np 
import nibabel as nib

from sl_tools import second_lvl_stats, stats_to_nii, cluster_fdr

# pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--sub_lst', help='all scan sessions', nargs='+', required=True)
parser.add_argument('--pth', help='the base pth', type=str, required=True)
parser.add_argument('--stats', help='the base pth', type=str, default='beta')
parser.add_argument('--voi', help='the variable of interest', type=str, default='penalty1')
parser.add_argument('--config', help='configurations', type=str, default='config')
args = parser.parse_args()

# # --------------- For debugging --------------- #
# class args:
#     pth = '/home/data/analyses/CLearn/penalty1'
#     stats = 'beta'
#     voi = 'penalty1'
#     config = 'config/config.json'
# # --------------- For debugging --------------- #

# load configuration 
with open(args.config, 'r') as handle: config = json.load(handle)[args.voi]

# get the model name 
model_name_str = '-'.join(config["models"])
mask_str = 'unmasked' if config["sl_kernel"]['masked']=='unmasked' else 'masked'
sl_stats_name = f'{args.pth}/m3_result/{args.stats}_{args.voi}-{mask_str}'
sl_stats_name+= f'-{config["sl_kernel"]["estimator"]["fn"]}-{model_name_str}.pkl'
print(sl_stats_name)
with open(sl_stats_name, 'rb')as handle: sl_stats = pickle.load(handle)

# # --------------- For debugging --------------- #
# args.sub_lst = list(sl_stats.keys())
# # --------------- For debugging --------------- #

 # get the group level analysis
if config['sl_baseline']:
    with open(args.config, 'r') as handle: bconfig = json.load(handle)[f'{args.voi}-baseline']
    sl_base_name = f'{args.pth}/m3_result/{args.stats}_{args.voi}-baseline-{mask_str}'
    sl_base_name+= f'-{bconfig["sl_kernel"]["estimator"]["fn"]}-{model_name_str}.pkl'
    sl_baselines = pickle.load(open(sl_base_name, 'rb'))
else: 
    sl_baselines = 0

# get the target
p_voxel, p_cluster = config['p_voxel'], config['p_cluster']
for i, model in enumerate(config["models"]):

    print(f'\nCalulating group lvl stats for {model}, p voxel={p_voxel}, p cluster={p_cluster}')

    # get the model data
    corrs_lvl_1 = np.vstack([sl_stats[sub_id][..., i].copy() 
                             for sub_id in args.sub_lst])
    sl_baseline = np.vstack([sl_baselines[sub_id][..., i].copy() 
                             for sub_id in args.sub_lst]
                             ) if config['sl_baseline'] else \
                                np.zeros_like(corrs_lvl_1)
    
    corrs_lvl_2 = second_lvl_stats(corrs_lvl_1, 
                            fisher_z=config['fisher_z'], 
                            sl_baseline=sl_baseline,
                            n_cpu=10)
    p = corrs_lvl_2[...,1]
    n_sig_pre = (p[~np.isnan(p)]<p_voxel).sum()
    print(f'\t{n_sig_pre} voxels to be corrected')

    # mutiple test correction
    correct_p = cluster_fdr(corrs_lvl_2[...,1], p_voxel=p_voxel, 
                            p_cluster=p_cluster, n_connect=1
                            ) if n_sig_pre > 0 else np.full_like(p, np.nan)
    correct_mask_p = ~np.isnan(correct_p)
    corr_corrected = np.where(correct_mask_p, corrs_lvl_2[...,0], np.nan)
    n_sig_post = np.sum(correct_mask_p)
    print(f'\t{n_sig_post} valid voxels remained after FDR correction, {n_sig_pre-n_sig_post} voxels have been removed')
    
    # mask out the data outside the sculp
    fname = f'{args.pth}/src/utils/group_mask.7.nii'
    # fname = '/home/data/analyses/CLearn/RSA2/m1_first_lvl/stats.CN078.nii'
    f = nib.load(fname)
    corr_mask, affine, header = f.get_fdata(), f.affine, f.header
    nii_file = stats_to_nii(corr_corrected, affine, 
                            corr_mask=corr_mask, 
                            header=header,
                            size=corr_mask.shape, 
                            radius=config['radius'], 
                            smooth=True)

    n_model = len(config["models"])
    mask_str = 'unmasked' if config["sl_kernel"]['masked']=='unmasked' else 'masked'
    fname = f'{args.stats}_{args.voi}.{mask_str}.group_lvl_stats'
    fname+= f'.{config["sl_kernel"]["estimator"]["fn"]}.{n_model}model-{model}.nii'
    nib.save(nii_file, f'{args.pth}/m3_result/{fname}')
