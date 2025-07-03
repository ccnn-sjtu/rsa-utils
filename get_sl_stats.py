import os 
import json
import argparse
import pickle
import numpy as np 

from sl_tools import Searchlight, Decoder, RDM_regress

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--sub_lst', help='all scan sessions', nargs='+', required=True)
parser.add_argument('--pth', help='the base pth', type=str, required=True)
parser.add_argument('--stats', help='the base pth', type=str, default='t')
parser.add_argument('--voi', help='the variable of interest', type=str, default='penalty1')
parser.add_argument('--config', help='configurations', type=str, default='config')
args = parser.parse_args()

# class args:
#     sub_lst = ['CN046']
#     pth = '/home/data/analyses/CLearn/RSA_omega'
#     stats = 'beta'
#     voi = 'omega'
#     config = 'config/RSA.json'

# load configuration 
with open(args.config, 'r') as handle: config = json.load(handle)[args.voi]

# The file to save searchlight stats
model_name_str = '-'.join(config["models"])
mask_str = 'unmasked' if config["sl_kernel"]['masked']=='unmasked' else 'masked'
sl_stats_name = f'{args.pth}/m3_result/{args.stats}_{args.voi}-{mask_str}'
sl_stats_name+= f'-{config["sl_kernel"]["estimator"]["fn"]}-{model_name_str}.pkl'
print(f'\nSaving to {sl_stats_name}')

# Prepare for the searchlight model predictions
print(f'Runing {args.voi} for: {config["models"]}')
model_pred_lst = {}
for sub_id in args.sub_lst:
    model_rdms = []
    for m in config['models']:
        voi = args.voi.split('-')[0]
        model_rdm_name = f'{args.pth}/src/{m}/{voi}-mle.pkl'
        with open(model_rdm_name, 'rb')as handle: 
            m_rdms = pickle.load(handle)
        model_rdms.append(m_rdms[sub_id].copy())
    model_pred_lst[sub_id] = model_rdms

# Initilaize the searchlight kernel
kconfig = config["sl_kernel"]
sl_kernel = eval(kconfig['kernel_type'])(
    estimator=kconfig['estimator'],
    cv=kconfig['cv'],
    mask=kconfig['masked'],
    scoring=kconfig['scoring']
)
print(f'\nSL Estimator: {sl_kernel.estimator}, CV: {sl_kernel.cv}')

# Initialize the searchlighter 
sl = Searchlight(sl_kernel, radius=config['radius'], n_paral=10)

# create a dict to store sl_stats
if os.path.exists(sl_stats_name):
    with open(sl_stats_name, 'rb')as handle: sl_stats = pickle.load(handle)
    done_sub_lst = [k for k in sl_stats.keys()]
else:
    done_sub_lst = []
    sl_stats = {}
n_done = len(done_sub_lst)

for i, sub_id in enumerate(args.sub_lst):
    
    if sub_id not in done_sub_lst:

        # get beta and model rdm
        stats_name = f'{args.pth}/m2_betas_and_ts/{sub_id}.fmri_volume_{args.stats}.pkl'
        with open(stats_name, 'rb')as handle: sub_fmri_data = pickle.load(handle)

        print(f'\nSub_id: {sub_id}, #Features: {sub_fmri_data.shape[0]}, Progress: {(n_done*100)/len(args.sub_lst):.2f}%')
        print(f'')

        # get the model prediction
        model_preds = model_pred_lst[sub_id]

        # searchlight: search and fit
        sl_paral = sl.search_and_fit(sub_fmri_data, model_preds)
        print(f'\tSL stats dims: {sl_paral.shape}')
        
        # save sl stats
        sl_stats[sub_id] = sl_paral
        with open(sl_stats_name, 'wb')as handle: pickle.dump(sl_stats, handle)
        n_done += 1

