
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt


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






def calc_error_prop(results, col_names):
    """Function to calculate the aggregate hit rate. 
    
    Parameters
    - results: A numpy array containing the stock level predictions
    - col_names: A numpy array containing the column names of results
        
    Return
    - results_agg: A numpy array containing the aggregate hit rateaverage error size for the IBES,
                   persistent baseline, single best and ensemble predictions"""
    
    # Find the error for each stock for the estimates, baseline and predictions
    ibes_err=np.abs(results[:,np.where(col_names=='reported')[0]]-results[:,np.where(col_names=='ibes')[0]])
    baseline_err=np.abs(results[:,np.where(col_names=='reported')[0]]-results[:,np.where(col_names=='baseline')[0]])
    best1_err=np.abs(results[:,np.where(col_names=='reported')[0]]-results[:,np.where(col_names=='best1')[0]])
    ensemble_err=np.abs(results[:,np.where(col_names=='reported')[0]]-results[:,np.where(col_names=='ensemble')[0]])
    
    # Find the proportion of stocks for the ensemble model that are more accurate than the baseline and estimates
    ens_ibes_err_prop=(np.sum(ensemble_err-ibes_err<=0)/(np.sum(ensemble_err-ibes_err<=0)+np.sum(ensemble_err-ibes_err>0)))*100
    ens_baseline_err_prop=(np.sum(ensemble_err-baseline_err<=0)/(np.sum(ensemble_err-baseline_err<=0)+np.sum(ensemble_err-baseline_err>0)))*100
    
    # Find the proportion of stocks for the single best model that are more accurate than the baseline and estimates
    best1_ibes_err_prop=(np.sum(best1_err-ibes_err<=0)/(np.sum(best1_err-ibes_err<=0)+np.sum(best1_err-ibes_err>0)))*100
    best1_baseline_err_prop=(np.sum(best1_err-baseline_err<=0)/(np.sum(best1_err-baseline_err<=0)+np.sum(best1_err-baseline_err>0)))*100
    
    # Collate proportion results into a numpy array
    results_agg=np.array([ens_ibes_err_prop,ens_baseline_err_prop,
                         best1_ibes_err_prop,best1_baseline_err_prop])
    
    return results_agg



def calc_error_size(results, col_names):
    """Function to calculate the aggregate error size. 
    
    Parameters
    - results: A numpy array containing the stock level predictions
    - col_names: A numpy array containing the column names of results
        
    Return
    - results_agg: A numpy array containing the average error size for the IBES,
                   persistent baseline, single best and ensemble predictions"""
    
    # Remove rows with missing predictions
    results=results[np.where(results[:,7]<1000)[0],:]
    results=results[np.where(np.abs(results[:,3])<1000)[0],:]
    results=results[np.where(~np.isnan(results[:,0]))[0],:]
    results=results[np.where(~np.isnan(results[:,1]))[0],:]
    results=results[np.where(~np.isnan(results[:,2]))[0],:]
    results=results[np.where(~np.isnan(results[:,3]))[0],:]
    results=results[np.where(~np.isnan(results[:,4]))[0],:]
    results=results[np.where(results[:,0]!=0)[0],:]
    
    # Find the error for each stock for the estimates, baseline and predictions
    ibes_err=np.abs(results[:,np.where(col_names=='reported')[0]]-results[:,np.where(col_names=='ibes')[0]])
    baseline_err=np.abs(results[:,np.where(col_names=='reported')[0]]-results[:,np.where(col_names=='baseline')[0]])
    best1_err=np.abs(results[:,np.where(col_names=='reported')[0]]-results[:,np.where(col_names=='best1')[0]])
    ensemble_err=np.abs(results[:,np.where(col_names=='reported')[0]]-results[:,np.where(col_names=='ensemble')[0]])
    
    # Find the relative error
    ibes_rel_err=ibes_err/np.abs(results[:,np.where(col_names=='reported')[0]])
    baseline_rel_err=baseline_err/np.abs(results[:,np.where(col_names=='reported')[0]])
    best1_rel_err=best1_err/np.abs(results[:,np.where(col_names=='reported')[0]])
    ensemble_rel_err=ensemble_err/np.abs(results[:,np.where(col_names=='reported')[0]])
    
    # Calculate the average error
    ibes_avg_err=np.mean(ibes_rel_err)
    baseline_avg_err=np.mean(baseline_rel_err)
    best1_avg_err=np.mean(best1_rel_err)
    ensemble_avg_err=np.mean(ensemble_rel_err)
    
    results_agg=np.array([ibes_avg_err, baseline_avg_err, 
                          best1_avg_err, ensemble_avg_err])
    
    return results_agg




def agg_results(file_name, file_path, calc_type = "error_prop"):
    """Function to aggregate the results by either hit rate or error size over all
    stocks. This is carried out over the M1, M2, and M3 predictions. The results 
    are then aggregated into one numpy array.
    
    Parameters
    - file_name: A single character value giving the file name
    - file_path: A single character value giving the file path
    - calc_type: A single character value that indicates if it is the hit rate
                 or error size that we are calculating and aggregating over. If
                 hit rate, then this parameter is defined as 'error_prop'. If error
                 size, then this parameter is defined as 'error_size'.
    
    Return
    - final_results: A numpy array containing the aggregated results for the M1,
                     M2, and M3 results vertically stacked."""
    
    # Load the stock results data
    results_m1=pd.read_pickle(os.path.join(get_path(file_path),'results_m1_'+file_name))
    results_m2=pd.read_pickle(os.path.join(get_path(file_path),'results_m2_'+file_name))
    results_m3=pd.read_pickle(os.path.join(get_path(file_path),'results_m3_'+file_name))
    
    # Extract the stock ISINs
    col_names=np.array(results_m1.columns)
    
    # Convert results to numpy and round
    results_m1=np.round(np.array(results_m1),2)
    results_m2=np.round(np.array(results_m2),2)
    results_m3=np.round(np.array(results_m3),2)
    
    # Calculate the aggregate hit rate for each of the M1, M2, and M3 predictions
    if calc_type == "error_prop":
        results_agg_m1=calc_error_prop(results=results_m1, col_names=col_names)
        results_agg_m2=calc_error_prop(results=results_m2, col_names=col_names)
        results_agg_m3=calc_error_prop(results=results_m3, col_names=col_names)
        
    # Calculate the aggregate error size for each of the M1, M2, and M3 predictions    
    if calc_type == "error_size":
        results_agg_m1=calc_error_size(results=results_m1, col_names=col_names)
        results_agg_m2=calc_error_size(results=results_m2, col_names=col_names)
        results_agg_m3=calc_error_size(results=results_m3, col_names=col_names)

    # Collate results
    final_results=np.vstack((results_agg_m1,results_agg_m2,results_agg_m3))
    
    return final_results




    








def avg_num_stocks(file_name, file_path):
    """Function to find the average number of stocks modelled for a given quarter.
    
    Parameters
    - file_name: A character value giving the file path name
    - pred_type: A single character value giving the type of predictions we are
                 calculating the hit rate. This can be either 'tl' if we want the
                 transfer learning results or 'dir' if we want the direct RE model
                 results for the transfer learning targets
    
    Return
    - tt_results: A 1xn numpy array containing the average number of stocks"""
        
    
    # Load results
    results_m1=pd.read_pickle(os.path.join(get_path(file_path),'results_m1_'+file_name))
    results_m2=pd.read_pickle(os.path.join(get_path(file_path),'results_m2_'+file_name))
    results_m3=pd.read_pickle(os.path.join(get_path(file_path),'results_m3_'+file_name))
    
    # Find the total number of stocks
    num_stocks_m1=len(results_m1)
    num_stocks_m2=len(results_m2)
    num_stocks_m3=len(results_m3)
    
    # Find the number of stocks modelled
    num_mods_m1=np.sum(~np.isnan(np.array(results_m1)[:,3]))
    num_mods_m2=np.sum(~np.isnan(np.array(results_m2)[:,3]))
    num_mods_m3=np.sum(~np.isnan(np.array(results_m3)[:,3]))
    
    # Collate results into a numpy array
    tt_results=np.array([num_stocks_m1,num_stocks_m2,num_stocks_m3,
                         num_mods_m1,num_mods_m2,num_mods_m3])
    tt_results=tt_results.reshape(1,len(tt_results))
    
    return tt_results

    


def plot_results(x_vals, y_vals_1, y_vals_2, xaxis_labels, xaxis_ticks,
                 leg_labels, output_path, y_vals_3=None):
    """Function to plot and save the results.
    
    Note, nothing is returned.
    
    Parameters
    - x_vals: The x axis values
    - y_vals_1: The y axis values for the first series
    - y_vals_2: The y axis values for the second series
    - xaxis_labels: A list of character values containing the x axis labels
    - xaxis_ticks: A numpy array containing the x axis tick values
    - leg_labels: A list of character values containing the legend labels
    - output_path: A single character value giving the output file path
    - y_vals_3: The y axis values for the third series, if defined. If None, the
                third series is not included. The default is None."""
    
    plt.plot(x_vals,y_vals_1)
    plt.plot(x_vals,y_vals_2)
    if y_vals_3 is not None: plt.plot(x_vals,y_vals_3)
    plt.ylabel('(%)')
    plt.axhline(y=50, color='black')
    plt.xticks(xaxis_ticks, xaxis_labels)
    plt.legend(leg_labels, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
    plt.savefig(output_path, bbox_inches='tight')
    plt.clf()





if __name__ == '__main__':
    
    
    ### Load file names and extract dates from file names 
    
    files=os.listdir(get_path('data/results/eps/raw_outputs_v2'))
    dates=np.array([np.datetime64(x[7:17]) for x in files])
    dates=np.sort(dates)
    dates=[str(dates[x]) for x in range(len(dates))]
    
    dates_np=np.array(dates)
    files_np=np.array(files)
   
    
   
    
    ### Calculate the hit rate, size of the error and number of stocks predicted
    
    # Calculate hit rate
    tl_lt=list()
    dr_lt=list()
    for i in range(len(files)):
        loc=int(np.where(np.array([dates[i] in files[x] for x in range(len(files))]))[0])
        tl_lt.append(agg_results(file_name=files[loc], 
                                 file_path='data/results/eps/transfer_learning/tl_results', 
                                 calc_type = "error_prop"))
        dr_lt.append(agg_results(file_name=files[loc], 
                                 file_path='data/results/eps/transfer_learning/direct_results', 
                                 calc_type = "error_prop"))
    
    # Calculate error size   
    tl_size_lt=list()
    dr_size_lt=list()
    for i in range(len(files)):
        loc=int(np.where(np.array([dates[i] in files[x] for x in range(len(files))]))[0])
        tl_size_lt.append(agg_results(file_name=files[loc], 
                                 file_path='data/results/eps/transfer_learning/tl_results', 
                                 calc_type = "error_size"))
        dr_size_lt.append(agg_results(file_name=files[loc], 
                                 file_path='data/results/eps/transfer_learning/direct_results', 
                                 calc_type = "error_size"))
     
    # Find the total number of stocks and the number we have predictions for
    tl_num_lt=list()
    dr_num_lt=list()
    for i in range(len(files)):
        loc=int(np.where(np.array([dates[i] in files[x] for x in range(len(files))]))[0])
        tl_num_lt.append(avg_num_stocks(file_name=files[loc], 
                                        file_path='data/results/eps/transfer_learning/tl_results'))
        dr_num_lt.append(avg_num_stocks(file_name=files[loc],
                                        file_path='data/results/eps/transfer_learning/direct_results'))
    
    
    
    # Convert the number of stocks and error sizes to numpy array
    tl_num_stocks=np.vstack(tl_num_lt)
    dr_num_stocks=np.vstack(dr_num_lt)
    tl_size_results=np.vstack(tl_size_lt)
    dr_size_results=np.vstack(dr_size_lt)
   
   
    # Convert the hit rate to numpy arrays
    tl_lt_m1=list()
    tl_lt_m2=list()
    tl_lt_m3=list()
    dr_lt_m1=list()
    dr_lt_m2=list()
    dr_lt_m3=list()
    for i in range(len(tl_lt)):
        tl_temp=tl_lt[i]
        tl_lt_m1.append(tl_temp[[0],:])
        tl_lt_m2.append(tl_temp[[1],:])
        tl_lt_m3.append(tl_temp[[2],:])
        
        dr_temp=dr_lt[i]
        dr_lt_m1.append(dr_temp[[0],:])
        dr_lt_m2.append(dr_temp[[1],:])
        dr_lt_m3.append(dr_temp[[2],:])
    
    tl_results_m1=np.vstack(tl_lt_m1)
    tl_results_m2=np.vstack(tl_lt_m2)
    tl_results_m3=np.vstack(tl_lt_m3)
    
    dr_results_m1=np.vstack(dr_lt_m1)
    dr_results_m2=np.vstack(dr_lt_m2)
    dr_results_m3=np.vstack(dr_lt_m3)
    
    
    
    
    
    ### Plot results
    
    
    ## Hit rate over persistent baseline
    
    # Construct x axis labels and series labels
    xaxis=np.arange(0,41,1)
    xaxis_labels=['11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    labels=['MSETS-TL', 'Direct RE', 'SSETS-TL']
    xaxis_ticks=np.arange(0,41,4.5)
    
    # plot hit rate for M1
    plot_results(x_vals=xaxis, y_vals_1=tl_results_m1[:,1], y_vals_2=dr_results_m1[:,1], 
                 y_vals_3=tl_results_m1[:,3], xaxis_labels=xaxis_labels, 
                 xaxis_ticks=xaxis_ticks, leg_labels=labels, 
                 output_path=get_path('data/results/eps/plots/tl_ensemble_vs_baseline_m1.pdf'))
    
    # plot hit rate for M2
    plot_results(x_vals=xaxis, y_vals_1=tl_results_m2[:,1], y_vals_2=dr_results_m2[:,1], 
                 y_vals_3=tl_results_m2[:,3], xaxis_labels=xaxis_labels, 
                 xaxis_ticks=xaxis_ticks, leg_labels=labels, 
                 output_path=get_path('data/results/eps/plots/tl_ensemble_vs_baseline_m2.pdf'))
    
    # plot hit rate for M3
    plot_results(x_vals=xaxis, y_vals_1=tl_results_m3[:,1], y_vals_2=dr_results_m3[:,1], 
                 y_vals_3=tl_results_m3[:,3], xaxis_labels=xaxis_labels, 
                 xaxis_ticks=xaxis_ticks, leg_labels=labels, 
                 output_path=get_path('data/results/eps/plots/tl_ensemble_vs_baseline_m3.pdf'))
    
   
    
    ## Hit rate over IBES
    
    # Construct x axis labels and series labels
    xaxis=np.arange(0,41,1)
    xaxis_labels=['11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    labels=['MSETS-TL', 'Direct RE', 'SSETS-TL']
    xaxis_ticks=np.arange(0,41,4.5)
    
    # plot hit rate for M1
    plot_results(x_vals=xaxis, y_vals_1=tl_results_m1[:,0], y_vals_2=dr_results_m1[:,0], 
                 y_vals_3=tl_results_m1[:,2], xaxis_labels=xaxis_labels, 
                 xaxis_ticks=xaxis_ticks, leg_labels=labels, 
                 output_path=get_path('data/results/eps/plots/tl_ensemble_vs_ibes_m1.pdf'))
    
    # plot hit rate for M2
    plot_results(x_vals=xaxis, y_vals_1=tl_results_m2[:,0], y_vals_2=dr_results_m2[:,0], 
                 y_vals_3=tl_results_m2[:,2], xaxis_labels=xaxis_labels, 
                 xaxis_ticks=xaxis_ticks, leg_labels=labels, 
                 output_path=get_path('data/results/eps/plots/tl_ensemble_vs_ibes_m2.pdf'))
    
    # plot hit rate for M3
    plot_results(x_vals=xaxis, y_vals_1=tl_results_m3[:,0], y_vals_2=dr_results_m3[:,0], 
                 y_vals_3=tl_results_m3[:,2], xaxis_labels=xaxis_labels, 
                 xaxis_ticks=xaxis_ticks, leg_labels=labels, 
                 output_path=get_path('data/results/eps/plots/tl_ensemble_vs_ibes_m3.pdf'))
    
    
    
    
    
    
    ### Find summary statistics
    
    
    ## Hit rate
    
    # Hit rate average for TL method
    hit_avg_tl_m1=np.mean(tl_results_m1, axis=1)
    hit_avg_tl_m2=np.mean(tl_results_m2, axis=1)
    hit_avg_tl_m3=np.mean(tl_results_m3, axis=1)
    
    # Hit rate std for TL method
    hit_std_tl_m1=np.std(tl_results_m1, axis=1)
    hit_std_tl_m2=np.std(tl_results_m2, axis=1)
    hit_std_tl_m3=np.std(tl_results_m3, axis=1)
    
    
    # Hit rate average for RE method for TL targets
    hit_avg_dr_m1=np.mean(dr_results_m1, axis=1)
    hit_avg_dr_m2=np.mean(dr_results_m2, axis=1)
    hit_avg_dr_m3=np.mean(dr_results_m3, axis=1)
    
    # Hit rate std for RE method for TL targets
    hit_std_dr_m1=np.std(dr_results_m1, axis=1)
    hit_std_dr_m2=np.std(dr_results_m2, axis=1)
    hit_std_dr_m3=np.std(dr_results_m3, axis=1)
       
    
    
    
    
    
    
    ## Error size
    
    # Error size average of TL method
    err_avg_tl_m1=np.mean(tl_size_results, axis=1)
    err_avg_tl_m2=np.mean(tl_size_results, axis=1)
    err_avg_tl_m3=np.mean(tl_size_results, axis=1)
    
    # Error size std of TL method
    err_std_tl_m1=np.std(tl_size_results, axis=1)
    err_std_tl_m2=np.std(tl_size_results, axis=1)
    err_std_tl_m3=np.std(tl_size_results, axis=1)
    
    
    # Error size average of RE method for TL targets
    err_avg_dr_m1=np.mean(dr_size_results, axis=1)
    err_avg_dr_m2=np.mean(dr_size_results, axis=1)
    err_avg_dr_m3=np.mean(dr_size_results, axis=1)
    
    # Error size std of RE method for TL targets
    err_std_dr_m1=np.std(dr_size_results, axis=1)
    err_std_dr_m2=np.std(dr_size_results, axis=1)
    err_std_dr_m3=np.std(dr_size_results, axis=1)
    
    
    
    ## Number of stocks
    
    # Average number of total stock predictions
    avg_num_stocks_pred=np.mean(tl_num_stocks[:,0])
    avg_num_stocks_pred=np.std(tl_num_stocks[:,0])
    
    
    
    
    
    
    ### Load file names and extract dates from file names 
    
    files=os.listdir(get_path('data/results/eps/raw_outputs_v2'))
    dates=np.array([np.datetime64(x[7:17]) for x in files])
    dates=np.sort(dates)
    dates=[str(dates[x]) for x in range(len(dates))]
    
    dates_np=np.array(dates)
    files_np=np.array(files)
   
    
   
    
    ### Calculate the hit rate, size of the error and number of stocks predicted
    
    # Calculate hit rate
    lt=list()
    for i in range(len(files)):
        loc=int(np.where(np.array([dates[i] in files[x] for x in range(len(files))]))[0])
        lt.append(agg_results(file_name=files[loc], 
                                 file_path='data/results/eps/ensemble_model_v4', 
                                 calc_type = "error_prop"))
    
    # Calculate error size    
    size_lt=list()
    for i in range(len(files)):
        loc=int(np.where(np.array([dates[i] in files[x] for x in range(len(files))]))[0])
        size_lt.append(agg_results(file_name=files[loc], 
                                 file_path='data/results/eps/ensemble_model_v4', 
                                 calc_type = "error_size"))
        
    # Find the total number of stocks and the number we have predictions for
    num_lt=list()
    for i in range(len(files)):
        loc=int(np.where(np.array([dates[i] in files[x] for x in range(len(files))]))[0])
        num_lt.append(avg_num_stocks(file_name=files[loc], 
                                        file_path='data/results/eps/ensemble_model_v4'))
    
        
    
    # Convert the number of stocks and error sizes to numpy array
    num_stocks=np.vstack(num_lt)
    size_results=np.vstack(size_lt)
   
   
    # Convert the hit rate to numpy arrays
    lt_m1=list()
    lt_m2=list()
    lt_m3=list()
    for i in range(len(lt)):
        temp=lt[i]
        lt_m1.append(temp[[0],:])
        lt_m2.append(temp[[1],:])
        lt_m3.append(temp[[2],:])
    
    results_m1=np.vstack(lt_m1)
    results_m2=np.vstack(lt_m2)
    results_m3=np.vstack(lt_m3)
    
    
    
    
    ### Plot results
    
    
    ## Hit rate over persistent baseline
    
    # Construct x axis labels and series labels
    xaxis=np.arange(0,41,1)
    xaxis_labels=['11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    labels=['RE Method', 'Single Best Model']
    xaxis_ticks=np.arange(0,41,4.5)
    
    
    # plot hit rate for M1
    plot_results(x_vals=xaxis, y_vals_1=results_m1[:,1], y_vals_2=results_m1[:,3], 
                 xaxis_labels=xaxis_labels, xaxis_ticks=xaxis_ticks, leg_labels=labels, 
                 output_path=get_path('data/results/eps/plots/ensemble_vs_baseline_m1.pdf'))
    
    # plot hit rate for M2
    plot_results(x_vals=xaxis, y_vals_1=results_m2[:,1], y_vals_2=results_m2[:,3], 
                 xaxis_labels=xaxis_labels, xaxis_ticks=xaxis_ticks, leg_labels=labels, 
                 output_path=get_path('data/results/eps/plots/ensemble_vs_baseline_m2.pdf'))
    
    # plot hit rate for M3
    plot_results(x_vals=xaxis, y_vals_1=results_m3[:,1], y_vals_2=results_m3[:,3], 
                 xaxis_labels=xaxis_labels, xaxis_ticks=xaxis_ticks, leg_labels=labels, 
                 output_path=get_path('data/results/eps/plots/ensemble_vs_baseline_m3.pdf'))
    
    
   
    
    
    ## Hit rate over IBES
    
    # plot hit rate for M1
    plot_results(x_vals=xaxis, y_vals_1=results_m1[:,0], y_vals_2=results_m1[:,2], 
                 xaxis_labels=xaxis_labels, xaxis_ticks=xaxis_ticks, leg_labels=labels, 
                 output_path=get_path('data/results/eps/plots/ensemble_vs_ibes_m1.pdf'))
    
    # plot hit rate for M2
    plot_results(x_vals=xaxis, y_vals_1=results_m2[:,0], y_vals_2=results_m2[:,2], 
                 xaxis_labels=xaxis_labels, xaxis_ticks=xaxis_ticks, leg_labels=labels, 
                 output_path=get_path('data/results/eps/plots/ensemble_vs_ibes_m2.pdf'))
    
    # plot hit rate for M3
    plot_results(x_vals=xaxis, y_vals_1=results_m3[:,0], y_vals_2=results_m3[:,2], 
                 xaxis_labels=xaxis_labels, xaxis_ticks=xaxis_ticks, leg_labels=labels, 
                 output_path=get_path('data/results/eps/plots/ensemble_vs_ibes_m3.pdf'))
    
    
    
    
    
    
    
    
    ### Find summary statistics
    
    
    ## Hit rate
    
    # Hit rate average for TL method
    hit_avg_re_m1=np.mean(results_m1, axis=1)
    hit_avg_re_m2=np.mean(results_m2, axis=1)
    hit_avg_re_m3=np.mean(results_m3, axis=1)
    
    # Hit rate std for TL method
    hit_std_re_m1=np.std(results_m1, axis=1)
    hit_std_re_m2=np.std(results_m2, axis=1)
    hit_std_re_m3=np.std(results_m3, axis=1)
    
        
    
      
    
    ## Error size
    
    # Error size average of TL method
    err_avg_re_m1=np.mean(size_results, axis=1)
    err_avg_re_m2=np.mean(size_results, axis=1)
    err_avg_re_m3=np.mean(size_results, axis=1)
    
    # Error size std of TL method
    err_std_re_m1=np.std(size_results, axis=1)
    err_std_re_m2=np.std(size_results, axis=1)
    err_std_re_m3=np.std(size_results, axis=1)
    
    
    
    ## Number of stocks
    
    # Average number of total stock predictions
    avg_num_stocks_pred=np.mean(num_stocks[:,0])
    avg_num_stocks_pred=np.std(num_stocks[:,0])
    
   