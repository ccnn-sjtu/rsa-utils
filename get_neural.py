import os 
import pickle
import argparse
import subprocess
import nibabel as nib
import numpy as np  
from tqdm import tqdm

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--sub_lst', help='all scan sessions', nargs='+', required=True)
parser.add_argument('--stats', help='the base pth', type=str, default='beta')
parser.add_argument('--pth', help='the base pth', type=str, required=True)
args = parser.parse_args()

tqdm_bar_format="{l_bar}{bar:30} {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"

sanity_lst = []
for sub_id in tqdm(args.sub_lst, desc=f"Acquiring {args.stats}".rjust(15), 
                unit="subject",
                bar_format=tqdm_bar_format):

    # get the name for the output file 
    beta_name = f'{args.pth}/m2_betas_and_ts/{sub_id}.fmri_volume_{args.stats}.pkl'
    if not os.path.exists(beta_name): 

        # Find all files containing sub_id in the input path
        flst = ['.'.join(f.split('.')[:-1]) for f in os.listdir(f'{args.pth}/m1_first_lvl') if sub_id in f]
        flst = np.sort(np.unique(flst)).tolist()
        stats_lst = []
        for fn in flst:
            sanity_lst.append(fn)
            # covert the dset data to 1d.dset  
            fname = f'{args.pth}/m1_first_lvl/{fn}.BRIK'
            if os.path.exists(fname):
                oname = f'{args.pth}/m1_first_lvl/rm.{fn}.nii.gz'  
                print(f'\n\t{fn}: BRIK to PKL...')
                inlines = ['3dAFNItoNIFTI', '-prefix', oname, fname]
                process = subprocess.run(inlines)
                # convert to csv adta 
                result_data = nib.load(oname).get_fdata().copy()
                n_stats = result_data.shape[-1]-2
                # start from the 5th column
                # 1st column is R^2
                # 2nd column is Fstats
                # 3rd column is beta0
                # 4th column is t0
                # 5th column is beta1
                start_idx = 4 if args.stats=='beta' else 5 
                ind = np.arange(start_idx, n_stats, 2).tolist()
                stats_lst.append(result_data[..., ind].copy())
                process = subprocess.run(['rm', oname])
            else:
                print(f'\t{fname} not existed, check the 3ddeconvole')
        stats = np.concatenate(stats_lst, axis=-1) if len(stats_lst)>1 else stats_lst[0]
        stats = np.transpose(stats, (4, 3, 0, 1, 2))
        print(f'\t{sub_id} stats shape: {stats.shape}')
        with open(beta_name, 'wb')as handle: pickle.dump(stats, handle)
