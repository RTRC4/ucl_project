
import numpy as np
import pandas as pd
import os
import sys


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





def find_child_isins(results):
    
    # Identify the transfer learning targets
    child_isins_lt=list()
    for i in range(len(results)):
        temp_results=results[i]
        temp_isins=list(temp_results.keys())
        
        if len(temp_isins)>1:
            temp_isins=temp_isins[1:]
            for j in range(len(temp_isins)):
                child_isins_lt.append(temp_isins[j][6:])
    
    child_isins=np.array(child_isins_lt)
    child_isins=np.unique(child_isins)
    
    return child_isins






class extract_results():
    """Class to define the object to hold the model and results of a single model
    for a given ensemble model."""
    
    def __init__(self, child_isins, find_child_stocks):
        
        self.stock_isins_lt=list()
        
        self.ibes_val_mae_lt=list()
        self.best1_val_mae_lt=list()
        self.ensemble_val_mae_lt=list()
        self.baseline_val_mae_lt=list()
        
        self.act_test_lt=list()
        self.ibes_test_lt=list()
        self.best1_test_lt=list()
        self.ensemble_test_lt=list()
        self.baseline_test_lt=list()
        
        self.child_isins=child_isins
        self.find_child_stocks=find_child_stocks
        
    
    def find_results(self, results):
        
        for i in range(len(results)):
            
            # Extract the given stock's results
            temp_results=results[i]
            isin_keys=list(temp_results.keys())
            stock_direct_results=temp_results[isin_keys[0]]
            
            # If the given stock is not a transfer learning target, skip to next iteration
            if self.find_child_stocks and isin_keys[0] not in self.child_isins:
                continue
            
            # If predictions are present, extract results
            if type(stock_direct_results[list(stock_direct_results.keys())[0]]) is not int:
                
                # Extract the RE ensemble results
                temp_ensemble=stock_direct_results['ensemble_test_lev'][0,:]
                temp_ensemble=temp_ensemble.reshape(3,1)
                temp_act=stock_direct_results['act_test_lev']
                
                # If reported values are missing, skip to next iteration
                if all(np.isnan(temp_act)):
                    continue
                
                # If there are any missing gaps in the 3 months for the quarterly reported values,
                # we fill them as replicates of the existing data for that quarter. These will
                # be the same regardless, given that the reported data is quarterly converted to monthly
                if any(np.isnan(temp_act)):
                    temp_act=temp_act[~np.isnan(temp_act)][[0]]
                    temp_act=np.repeat(temp_act,3)
                    temp_act=temp_act.reshape(3,1)
                
                # Extract the single best model/model of best source domain
                temp_best1=stock_direct_results['best_1_test_lev'][0,:]
                temp_best1=temp_best1.reshape(3,1)
                
                # If missing values in the best model, skip
                if any(np.isnan(temp_best1)):
                    continue
                
                # If missing values in the ensemble, skip
                if any(np.isnan(temp_ensemble)):
                    continue
                
                # Add predictions and stock ISIN name to lists
                self.ensemble_test_lt.append(temp_ensemble)
                self.best1_test_lt.append(temp_best1)
                self.stock_isins_lt.append(isin_keys[0])
                
                # Add the MAE values to the lists
                self.ibes_val_mae_lt.append(stock_direct_results['ibes_val_mae'].reshape(3,1))    
                self.best1_val_mae_lt.append(stock_direct_results['best_1_val_mae'].reshape(3,1))
                self.ensemble_val_mae_lt.append(stock_direct_results['ensemble_pred_val_mae'].reshape(3,1))
                self.baseline_val_mae_lt.append(stock_direct_results['baseline_pred_val_mae'])
            
                # Add none prediction values to their respective lists
                self.act_test_lt.append(stock_direct_results['act_test_lev'])
                self.ibes_test_lt.append(stock_direct_results['ibes_test_lev'])
                self.baseline_test_lt.append(stock_direct_results['baseline_test_lev'])
                
        
        
        
    def tabulate_and_save_results(self, file_path, file_name):
        
        ### Collate results
        
        # Collate predicted, reported and estimated values
        act_test=np.hstack(self.act_test_lt)
        ibes_test=np.hstack(self.ibes_test_lt)
        best1_test=np.hstack(self.best1_test_lt)
        ensemble_test=np.hstack(self.ensemble_test_lt)
        baseline_test=np.hstack(self.baseline_test_lt)
        
        # Collate stock ISINs
        stock_isins=np.array(self.stock_isins_lt)
        
        # Identify the location of the RE model targets
        if self.find_child_stocks: isins_loc=np.where(~np.in1d(stock_isins,self.child_isins))[0]
        if not self.find_child_stocks: isins_loc=np.where(np.in1d(stock_isins,self.child_isins))[0]
        
        # Reduce the results arrays down to just the direct modelling targets
        subset_act_test=act_test[:,isins_loc]
        subset_ibes_test=ibes_test[:,isins_loc]
        subset_best1_test=best1_test[:,isins_loc]
        subset_ensemble_test=ensemble_test[:,isins_loc]
        subset_baseline_test=baseline_test[:,isins_loc]
        
        # Collate the MAE values
        ibes_val_mae=np.hstack(self.ibes_val_mae_lt)
        best1_val_mae=np.hstack(self.best1_val_mae_lt)
        ensemble_val_mae=np.hstack(self.ensemble_val_mae_lt)
        baseline_val_mae=np.array(self.baseline_val_mae_lt)
        baseline_val_mae=baseline_val_mae.reshape(1,len(baseline_val_mae))
        
        # Reduce the MAE arrays down to just the direct modelling targets
        subset_ibes_val_mae=ibes_val_mae[:,isins_loc]
        subset_best1_val_mae=best1_val_mae[:,isins_loc]
        subset_ensemble_val_mae=ensemble_val_mae[:,isins_loc]
        subset_baseline_val_mae=baseline_val_mae[:,isins_loc]
        
        # Combine the results into numpy arrays by month of prediction
        results_m1=np.vstack((subset_act_test[[0],:],subset_ibes_test[[0],:],subset_best1_test[[0],:],
                              subset_ensemble_test[[0],:],subset_baseline_test[[0],:],subset_ibes_val_mae[[0],:],
                              subset_best1_val_mae[[0],:],subset_ensemble_val_mae[[0],:],subset_baseline_val_mae)).T
        
        results_m2=np.vstack((subset_act_test[[1],:],subset_ibes_test[[1],:],subset_best1_test[[1],:],
                              subset_ensemble_test[[1],:],subset_baseline_test[[1],:],subset_ibes_val_mae[[1],:],
                              subset_best1_val_mae[[1],:],subset_ensemble_val_mae[[1],:],subset_baseline_val_mae)).T
        
        results_m3=np.vstack((subset_act_test[[2],:],subset_ibes_test[[2],:],subset_best1_test[[2],:],
                              subset_ensemble_test[[2],:],subset_baseline_test[[2],:],subset_ibes_val_mae[[2],:],
                              subset_best1_val_mae[[2],:],subset_ensemble_val_mae[[2],:],subset_baseline_val_mae)).T
        
        # Convert results arrays to pandas dataframes
        col_names=['reported','ibes','best1','ensemble','baseline','ibes_mae',
                   'best1_mae','ensemble_mae','baseline_mae']
        results_m1=pd.DataFrame(results_m1,columns=col_names,index=stock_isins[isins_loc])
        results_m2=pd.DataFrame(results_m2,columns=col_names,index=stock_isins[isins_loc])
        results_m3=pd.DataFrame(results_m3,columns=col_names,index=stock_isins[isins_loc])
        
        # Save dataframes
        results_m1.to_pickle(os.path.join(get_path(file_path), 'results_m1_'+file_name))
        results_m2.to_pickle(os.path.join(get_path(file_path), 'results_m2_'+file_name))
        results_m3.to_pickle(os.path.join(get_path(file_path), 'results_m3_'+file_name))
        
        
                
        



class tl_extract_results():
    """Class to define the object to hold the model and results of a single model
    for a given ensemble model."""
    
    def __init__(self, child_isins):
        
        self.village_ibes_val_mae_dt=dict.fromkeys(child_isins)
        self.village_best_1_val_mae_dt=dict.fromkeys(child_isins)
        self.village_ensemble_val_mae_dt=dict.fromkeys(child_isins)
        self.village_baseline_val_mae_dt=dict.fromkeys(child_isins)
        
        self.village_act_test_dt=dict.fromkeys(child_isins)
        self.village_ibes_test_dt=dict.fromkeys(child_isins)
        self.village_best_1_test_dt=dict.fromkeys(child_isins)
        self.village_ensemble_test_dt=dict.fromkeys(child_isins)
        self.village_baseline_test_dt=dict.fromkeys(child_isins)
        
        
        self.child_isins=child_isins
        
    
    def tl_find_results(self, results):
        
        for i in range(len(results)):
            
            # Extract the given stock's results
            temp_results=results[i]
            isin_keys=list(temp_results.keys())
            dir_isin=isin_keys[0]
            temp_child_isin_keys=[x for x in isin_keys if x != dir_isin]
            
            # If the given stock is not a transfer learning target, skip to next iteration
            if len(isin_keys)==1:
                continue
            
            # Extract the transfer learning predictions
            for j in temp_child_isin_keys:
                temp_child_results=temp_results[j]
                temp_child_isin=j[6:]
            
                # If predictions are present, extract results
                if type(temp_child_results[list(temp_child_results.keys())[0]]) is not int:
                    
                    # Extract ensemble predictions
                    temp_ensemble=temp_child_results['ensemble_test_lev'][0,:]
                    temp_ensemble=temp_ensemble.reshape(3,1)
                    temp_act=temp_child_results['act_test_lev']
                
                    # If reported results is empty, skip
                    if all(np.isnan(temp_act)):
                        continue
                    
                    # If there are any missing gaps in the 3 months for the quarterly reported values,
                    # we fill them as replicates of the existing data for that quarter. These will
                    # be the same regardless, given that the reported data is quarterly converted to monthly
                    if any(np.isnan(temp_act)):
                        temp_act=temp_act[~np.isnan(temp_act)][[0]]
                        temp_act=np.repeat(temp_act,3)
                        temp_act=temp_act.reshape(3,1)
                
                    # Extract the single best model
                    temp_best1=temp_child_results['best_1_test_lev'][0,:]
                    temp_best1=temp_best1.reshape(3,1)
                    
                    # If missing values in the single best model, skip
                    if any(np.isnan(temp_best1)):
                        continue
                    
                    # If missing values in the ensemble model, skip
                    if any(np.isnan(temp_ensemble)):
                        continue
                
                    # Store ensemble and single best predictions in dictionaries
                    if self.village_ensemble_test_dt[temp_child_isin] is None:
                        self.village_ensemble_test_dt[temp_child_isin]=temp_ensemble
                        self.village_best_1_test_dt[temp_child_isin]=temp_best1
                            
                    elif self.village_ensemble_test_dt[temp_child_isin] is not None:
                        self.village_ensemble_test_dt[temp_child_isin]=np.hstack((self.village_ensemble_test_dt[temp_child_isin],
                                                                            temp_ensemble))
                            
                        self.village_best_1_test_dt[temp_child_isin]=np.hstack((self.village_best_1_test_dt[temp_child_isin],
                                                                             temp_best1))
                            
                    # Store MAE values and reported and estimated values
                    if self.village_ibes_val_mae_dt[temp_child_isin] is None:
                        self.village_ibes_val_mae_dt[temp_child_isin]=temp_child_results['ibes_val_mae'].reshape(3,1)
                        self.village_best_1_val_mae_dt[temp_child_isin]=temp_child_results['best_1_val_mae'].reshape(3,1)
                        self.village_ensemble_val_mae_dt[temp_child_isin]=temp_child_results['ensemble_pred_val_mae'].reshape(3,1)
                        self.village_baseline_val_mae_dt[temp_child_isin]=temp_child_results['baseline_pred_val_mae']
                        
                        self.village_act_test_dt[temp_child_isin]=temp_child_results['act_test_lev'][:3]
                        self.village_ibes_test_dt[temp_child_isin]=temp_child_results['ibes_test_lev'][:3]
                        self.village_baseline_test_dt[temp_child_isin]=temp_child_results['baseline_test_lev'][:3]
                        
                    elif self.village_ibes_val_mae_dt[temp_child_isin] is not None:
                        self.village_best_1_val_mae_dt[temp_child_isin]=np.hstack((self.village_best_1_val_mae_dt[temp_child_isin],
                                                                              temp_child_results['best_1_val_mae'].reshape(3,1)))
                        self.village_ensemble_val_mae_dt[temp_child_isin]=np.hstack((self.village_ensemble_val_mae_dt[temp_child_isin],
                                                                                temp_child_results['ensemble_pred_val_mae'].reshape(3,1)))
                        
                
        
        
        
    def tl_tabulate_and_save_results(self, file_path, file_name):
        """Function to tabulate the predictions currently in list form before saving"""
        
        ### Collate results
        
        # Collate reported results
        vil_act_test=[self.village_act_test_dt[i] for i in list(self.village_act_test_dt.keys())]
        vil_act_test=np.hstack(vil_act_test)
        
        # Collate estimated results
        vil_ibes_test=[self.village_ibes_test_dt[i] for i in list(self.village_ibes_test_dt.keys())]
        vil_ibes_test=np.hstack(vil_ibes_test)
                
        # Collate baseline results
        vil_baseline_test=[self.village_baseline_test_dt[i] for i in list(self.village_baseline_test_dt.keys())]
        vil_baseline_test=np.hstack(vil_baseline_test)
        
        # Collate Ensemble results
        vil_ens_test=[np.mean(self.village_ensemble_test_dt[i], axis=1).reshape(3,1) for i in list(self.village_ensemble_test_dt.keys())]
        vil_ens_test=np.hstack(vil_ens_test)
        
        # Collate single best results
        vil_tl_best_test=[self.village_ensemble_test_dt[i][:,[0]] for i in list(self.village_ensemble_test_dt.keys())]
        vil_tl_best_test=np.hstack(vil_tl_best_test)
        
        
        # Collate estimates MAE
        vil_ibes_mae=[self.village_ibes_val_mae_dt[i] for i in list(self.village_ibes_val_mae_dt.keys())]
        vil_ibes_mae=np.hstack(vil_ibes_mae)
        
        # Collate baseline MAE
        vil_baseline_mae=np.array([self.village_baseline_val_mae_dt[i] for i in list(self.village_baseline_val_mae_dt.keys())])
        vil_baseline_mae=vil_baseline_mae.reshape(1,len(vil_baseline_mae))
        
        # Collate ensemble MAE
        vil_ens_mae=[np.nanmean(self.village_ensemble_val_mae_dt[i], axis=1).reshape(3,1) for i in list(self.village_ensemble_val_mae_dt.keys())]
        vil_ens_mae=np.hstack(vil_ens_mae)
        
        # Collate single best MAE
        vil_best1_mae=[np.nanmean(self.village_best_1_val_mae_dt[i], axis=1).reshape(3,1) for i in list(self.village_best_1_val_mae_dt.keys())]
        vil_best1_mae=np.hstack(vil_best1_mae)
        
        
        # Combine results into arrays by month of prediction
        results_m1=np.vstack((vil_act_test[[0],:],vil_ibes_test[[0],:],vil_tl_best_test[[0],:],
                              vil_ens_test[[0],:],vil_baseline_test[[0],:],vil_ibes_mae[[0],:],
                              vil_best1_mae[[0],:],vil_ens_mae[[0],:],vil_baseline_mae)).T
        
        results_m2=np.vstack((vil_act_test[[1],:],vil_ibes_test[[1],:],vil_tl_best_test[[1],:],
                              vil_ens_test[[1],:],vil_baseline_test[[1],:],vil_ibes_mae[[1],:],
                              vil_best1_mae[[1],:],vil_ens_mae[[1],:],vil_baseline_mae)).T
        
        results_m3=np.vstack((vil_act_test[[2],:],vil_ibes_test[[2],:],vil_tl_best_test[[2],:],
                              vil_ens_test[[2],:],vil_baseline_test[[2],:],vil_ibes_mae[[2],:],
                              vil_best1_mae[[2],:],vil_ens_mae[[2],:],vil_baseline_mae)).T
        
        
        # Convert results arrays into pandas dataframes
        col_names=['reported','ibes','best1','ensemble','baseline','ibes_mae',
                   'best1_mae','ensemble_mae','baseline_mae']
        results_m1=pd.DataFrame(results_m1,columns=col_names,index=self.child_isins)
        results_m2=pd.DataFrame(results_m2,columns=col_names,index=self.child_isins)
        results_m3=pd.DataFrame(results_m3,columns=col_names,index=self.child_isins)
        
        # Save results pandas dataframes
        results_m1.to_pickle(os.path.join(get_path(file_path), 'results_m1_'+file_name))
        results_m2.to_pickle(os.path.join(get_path(file_path), 'results_m2_'+file_name))
        results_m3.to_pickle(os.path.join(get_path(file_path), 'results_m3_'+file_name))
    
    
    





def clean_output(file_name, file_path, find_tl_results=False, find_child_stocks=True):
    """Function to organise and collate the results of the nowcasting model for
    the RE model targets. That is, just the target series that are long enough
    to model directly.
    
    Note, the final results are saved as pandas dataframes and nothing is then
    returned.
    
    Parameters
    - file_name: A single character value giving the name of the file path"""
    
    
    # Read in the raw output from the nowcasting   
    results=pd.read_pickle(os.path.join(get_path('data/results/eps/raw_outputs_v2'), file_name))
    
    # Identify child stocks
    child_isins=find_child_isins(results=results)
    
    
    # Extract, tabulate and save results
    if not find_tl_results:
        final_results=extract_results(child_isins=child_isins, find_child_stocks=find_child_stocks)
        final_results.find_results(results=results)
        final_results.tabulate_and_save_results(file_path=file_path, file_name=file_name)
        
    if find_tl_results:
        final_results=tl_extract_results(child_isins=child_isins)
        final_results.tl_find_results(results=results)
        final_results.tl_tabulate_and_save_results(file_path=file_path, file_name=file_name)
                       

    
    



if __name__ == '__main__':
    
    
    ### Collate predictions    
    
    files=os.listdir(get_path('data/results/eps/raw_outputs_v2')) 
   
   
    for i in files:
        
        # RE predictions for parent stocks
        clean_output(file_name=i, file_path='data/results/eps/ensemble_model_v4', 
                     find_tl_results=False, find_child_stocks=False)
        
        # RE predictions for child stocks
        clean_output(file_name=i, file_path='data/results/eps/transfer_learning/direct_results', 
                     find_tl_results=False, find_child_stocks=True)
        
        # TL predictions for child stocks
        clean_output(file_name=i, file_path='data/results/eps/transfer_learning/tl_results', 
                     find_tl_results=True)
        
    
    
    