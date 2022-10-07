
import numpy as np
import pandas as pd
import os
import sys
import multiprocessing as mp
from functools import partial
import pickle



# add package to sys.path if it's not there already
# - so can import from comp0158
try:
    # python package (comp0158) location - one level up from this file
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if src_path not in sys.path:
        sys.path.extend([src_path])
except NameError:
    print('issue with adding to path, probably due to __file__ not being defined')
    src_path = None


from comp0158 import get_path
from comp0158 import system as ss
from comp0158 import prep_train_test_data as prep
from comp0158 import stock_classification as cl







if __name__ == '__main__':
    
     
    ### Build Predictions
    
    train_size=0.8
    
    start_year_loc_total=np.array([339,336,333,330,327,324,321,318,315,312,
                           309,306,303,300,297,294,291,288,285,282,279,276,273,
                           270,267,264,261,258,255,252,249,246,243,240,237,234,231])
    for start_year_loc in start_year_loc_total:
        
        ### Load and prepare data
        
        # Load data
        eps_act=pd.read_pickle(os.path.join(get_path('data/refined_data'), 'eps_reported_refined.pkl'))
        eps_act_lev=pd.read_pickle(os.path.join(get_path('data/refined_data'), 'eps_reported_level_refined.pkl'))
        eps_est_pch=pd.read_pickle(os.path.join(get_path('data/refined_data'), 'eps_estimated_pch.pkl'))
        eps_est_ach=pd.read_pickle(os.path.join(get_path('data/refined_data'), 'eps_estimated_ach.pkl'))
        price_pch=pd.read_pickle(os.path.join(get_path('data/refined_data'), 'price_pch.pkl'))
        price_ach=pd.read_pickle(os.path.join(get_path('data/refined_data'), 'price_ach.pkl'))
        macro=pd.read_pickle(os.path.join(get_path('data/refined_data'), 'macro_eps_refined.pkl'))
        change=np.load(os.path.join(get_path('data/refined_data'), 'eps_changes.npy'))
        sp_proxy=pd.read_pickle(os.path.join(get_path('data/refined_data'), 'sp500_proxy.pkl'))
    
        # Find stock isins, dates and macro variable names
        isins=np.array(eps_act.columns)
        dates=np.array(eps_act.index)
        macro_var=np.array(macro.columns)
    
        # Convert data to a numpy array
        eps_act=np.array(eps_act)
        eps_act_lev=np.array(eps_act_lev)
        eps_est_pch=np.array(eps_est_pch)
        eps_est_ach=np.array(eps_est_ach)
        price_pch=np.array(price_pch)
        price_ach=np.array(price_ach)
        macro=np.array(macro)
    
    
    
    
        ### Separate data by wanted date
        
        # Find end date
        end_date=str(dates[start_year_loc])
    
        # Find stock universe for given quarter
        sp_stocks=sp_proxy[sp_proxy.index==pd.to_datetime(end_date)]
        sp_stocks=sp_stocks[sp_stocks==1]
        sp_stocks.dropna(axis=1, how="all", inplace=True)
        wanted_isins=np.array(sp_stocks.columns)
        wanted_isins=wanted_isins[np.where(np.in1d(wanted_isins,isins))]
    
        # Find the ISIN location of the wanted stocks
        wanted_isins_loc=[int(np.where(wanted_isins[i]==isins)[0]) for i in range(len(wanted_isins)) if wanted_isins[i] in isins]
    
        # Reduce data down to just the stocks of the given stock universe and remove any data after the date of interest
        wanted_act=eps_act[:start_year_loc,wanted_isins_loc]
        wanted_act_lev=eps_act_lev[:start_year_loc,wanted_isins_loc]
        wanted_est_pch=eps_est_pch[:start_year_loc,wanted_isins_loc]
        wanted_est_ach=eps_est_ach[:start_year_loc,wanted_isins_loc]
        wanted_prc_pch=price_pch[:start_year_loc,wanted_isins_loc]    
        wanted_prc_ach=price_ach[:start_year_loc,wanted_isins_loc]
        wanted_mac=macro[:start_year_loc,:]
        wanted_dates=dates[:start_year_loc]
        wanted_change=change[wanted_isins_loc]
    
        # Remove any gaps
        wanted_act=prep.rm_gaps(wanted_act)
    
        # For the TL method, determine which stocks are the potential source domain and
        # which are the targets
        train_cutoff=int(((len(wanted_act)-3)/3)*train_size)*3
        full_cols=np.where(np.sum(~np.isnan(wanted_act[:train_cutoff,:]),axis=0)==len(wanted_act[:train_cutoff,:]))[0]
        less_cols=np.where((np.sum(~np.isnan(wanted_act[:train_cutoff,:]),axis=0)<120) & (np.sum(~np.isnan(wanted_act[:train_cutoff,:]),axis=0)>12))[0]
        parents_isins=wanted_isins[full_cols]
        children_isins=wanted_isins[less_cols]
        
        # Identify the source domain series for each TL target
        clusters=cl.assign_multi_clusters(parents_df=wanted_act[:train_cutoff,full_cols], children_df=wanted_act[:train_cutoff,less_cols],
                                          parents_isins=parents_isins, children_isins=children_isins)
        village_isins_np=[clusters[i] for i in list(clusters.keys())]
        village_isins_np=np.stack(village_isins_np)
        
        
       
        
       
        ### Build predictions for each stock
        
        pool = mp.Pool(5)
        part_stock=partial(ss.build_individual_stock, act=wanted_act, act_lev=wanted_act_lev, full_est_pch=wanted_est_pch, 
                           full_est_ach=wanted_est_ach, full_prc_pch=wanted_prc_pch, full_prc_ach=wanted_prc_ach, 
                           mac=wanted_mac, mac_vars=macro_var, wanted_change=wanted_change, full_isins=wanted_isins, 
                           village_isins_np=village_isins_np, children_isins=children_isins, wanted_dates=wanted_dates, 
                           bootstrap_num=500, subset_feat=0.5, keep_est=True, train_size=0.8)
        results = pool.map(part_stock, wanted_isins)
    
        # Save prediction results
        with open(os.path.join(get_path('data/results/raw_outputs/output_'+end_date+'.pkl')), 'wb') as fp:
            pickle.dump(results, fp)
        
    
    
     
    