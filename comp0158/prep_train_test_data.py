

import numpy as np
from scipy.stats import spearmanr,rankdata






def spearman_rank(y,X):
    """Function to find the Spearman Rank correlation between a given target y 
    and each colum series in X.
    
    Parameters
    - y: An nx1 numpy array
    - X: An nxm numpy array
    
    Return
    - A numpy array containing the correlation values"""
    
    # Rank the data and find the squared difference
    rank_X=rankdata(X,axis=0)
    rank_y=rankdata(y,axis=0)
    d_sqrd=(rank_y-rank_X)**2
    
    # Calculate the correlation
    corr=1-(6*np.sum(d_sqrd,axis=0))/(len(y)*(len(y)**2-1))
    
    return corr






def find_cross_corr(y, X, labs):
    """Function to find the correlation between y and each column in X after
    applying different lags to the data of X. The features with the highest
    absolute correlation are then returned.
    
    Parameters
    - y: An nx1 numpy array
    - X: An nxm numpy array
    - labs: A numpy vector array of length m containing the column names of X
    
    Return
    - best_var_loc: The column index location of the features with the strongest
                    correlation with the target y
    - best_var: The column names of the features with the strongest correlation
                with the target y
    - best_corr: The correlation values of the features with the strongest correlation
                 with the target y
    - best_lag: The optimal number of lags to be applied to each feature"""
    
    # Find the correlation between the target and each feature after lagging the feature data
    corr_lag=list()
    for i in np.arange(0,13):
        X_lag=np.roll(X,shift=i, axis=0)[i:,:]
        temp_corr=spearman_rank(y[i:,:],X_lag)
        temp_corr=np.reshape(temp_corr, (1,len(temp_corr)))
        corr_lag.append(temp_corr)
    
    # Convert correlation values into an array and find the best correlation and lag
    corr_df=np.concatenate(corr_lag, axis=0)
    best_corr=np.max(np.abs(corr_df), axis=0)
    best_lag=np.argmax(np.abs(corr_df), axis=0)
    
    # Find the feature locations of the features with the strongest correlation, depending on correlation level
    if np.sum(best_corr > 0.6) > 15: best_var_loc=np.where(np.abs(best_corr) > 0.6)[0]
    elif np.sum(best_corr > 0.55) > 15: best_var_loc=np.where(np.abs(best_corr) > 0.55)[0]
    elif np.sum(best_corr > 0.5) > 15: best_var_loc=np.where(np.abs(best_corr) > 0.5)[0]
    elif np.sum(best_corr > 0.45) > 15: best_var_loc=np.where(np.abs(best_corr) > 0.45)[0]
    elif np.sum(best_corr > 0.4) > 15: best_var_loc=np.where(np.abs(best_corr) > 0.4)[0]
    else: best_var_loc=np.where(np.abs(best_corr) >= 0.4)[0]
        
    # Of the features with the strongest correlation, find their names, correlation values and optimal lags
    best_var=labs[best_var_loc]
    best_corr=best_corr[best_var_loc]
    best_lag=best_lag[best_var_loc]
    
    return best_var_loc,best_var,best_corr,best_lag






def rm_colinear(y, X, corr_var):
    """Given a subset of features with a relatively high level of correlation with
    the target variable, this function will remove the colinear features. Of those
    that are colinear, only the one with the highest correlation with the target
    variable is kept.
    
    Parameters
    - y: An nx1 numpy array
    - X: An nxm numpy array
    - corr_var: A numpy vector array containing the names of the features with the 
                strongest correlation to the target variable
                
    Return
    - opt_vars: A numpy vector array containing the names of the features with
                the strongest correlation to the target variable, with the 
                colinear variables removed"""
    
    # Determine the correlation matrix of the feature matrix
    corr_mx=spearmanr(X)[0]
    
    # Starting with the feature with the strongest correlation, remove colinear
    # features with a weaker correlation
    opt_vars=list()
    temp_corr_var=corr_var
    while corr_mx.shape[1] > 0:
        
        # Determine if there are any colinear variables
        high_corr=np.where(np.abs(corr_mx[:,0]) >= 0.8)[0]
        
        # If there are no colinear variables, other featues are not removed
        if len(high_corr)==1:
            opt_vars.append(temp_corr_var[0])
            corr_mx=corr_mx[1:,1:]
            temp_corr_var=temp_corr_var[1:]
        
        # Else we remove the colinear features with a weaker correlation with the target
        else:
            temp_corr=list()
            wanted_vars=np.where(np.in1d(corr_var,temp_corr_var[high_corr]))[0]
            for j in range(len(wanted_vars)):
                temp_corr.append(spearmanr(X[:,wanted_vars[j]],y)[0])
                
            var_to_kp=temp_corr_var[high_corr[np.argmax(np.abs(temp_corr))]]
            opt_vars.append(var_to_kp)
            
            temp_corr_var=np.delete(temp_corr_var,high_corr)
            corr_mx=np.delete(corr_mx,high_corr,axis=0)
            corr_mx=np.delete(corr_mx,high_corr,axis=1)
        
    # Find the names of the non-colinear features
    opt_vars=corr_var[np.where(np.in1d(corr_var,opt_vars))[0]]    
    
    return opt_vars    

    



def split_and_norm(stock_act,inputs,feats,wanted_dates,train_size=0.8):
    """Function to standardise the target and feature data before splitting both
    into training, validation, and testing data. 
    
    Note, care is taken when split is made to ensure that the split occurs at 
    the end of a quarter to avoid data leakage.
    
    Parameters
    - stock_act: A numpy array containing the target data
    - inputs: A numpy array containing the feature data
    - wanted_dates: A numpy vector containing the dates of the stock_act and inputs
                    numpy arrays.
    - train_size: A single numeric value between [0,1] that gives the fraction
                  of the split between training and validation data. The default
                  is set to 0.8.
        
    Return
    - data_sep: A dictionary containing the standardised training, validation and 
                testing data
    - dates_sep: A dictionary containing the standardised training, validation and 
                 testing dates
    - norm_meas: A dictionary containing the training averages and standard deviations
                 used to standardise the data"""

    # Split off the input testing data
    test_X=inputs[len(inputs)-3:,:]
    inputs_tt=inputs[:len(inputs)-3,:]
    
    # Split off the target testing data
    test_y=stock_act[len(stock_act)-3:,:]
    tt_y=stock_act[:len(stock_act)-3,:]
        
    # Split off the testing data dates
    test_dates=wanted_dates[len(wanted_dates)-3:]
    tt_dates=wanted_dates[:len(wanted_dates)-3]
    
    # Find training and validation cutoff
    train_cutoff=int((len(inputs_tt)/3)*train_size)*3
        
    # Split off training data
    train_X=inputs_tt[:train_cutoff,:]
    train_y=tt_y[:train_cutoff,:]
    
    # Split off validation data
    val_X=inputs_tt[train_cutoff:,:]
    val_y=tt_y[train_cutoff:,:]
    
    # Split off training and validation dates
    train_dates=tt_dates[:train_cutoff]
    val_dates=tt_dates[train_cutoff:]
    
    # Find the training average and standard deviation
    train_X_avg=np.nanmean(train_X, axis=0)
    train_X_std=np.nanstd(train_X, axis=0)
    
    train_y_avg=np.nanmean(train_y, axis=0)
    train_y_std=np.nanstd(train_y, axis=0)
    
    # Standardise data
    train_X=(train_X-train_X_avg)/train_X_std
    val_X=(val_X-train_X_avg)/train_X_std
    
    train_y=(train_y-train_y_avg)/train_y_std
    val_y=(val_y-train_y_avg)/train_y_std
    
    test_X=(test_X-train_X_avg)/train_X_std
    test_y=(test_y-train_y_avg)/train_y_std
    
    # Organise data, dates and standardising metrics into separate dictionaries
    data_sep={'train_X': train_X,
              'train_y': train_y,
              'test_X': test_X,
              'test_y': test_y,
              'val_X': val_X,
              'val_y': val_y}
    
    dates_sep={'train_dates': train_dates,
               'test_dates': test_dates,
               'val_dates': val_dates}
    
    norm_meas={'train_X_avg': train_X_avg,
               'train_y_avg': train_y_avg,
               'train_X_std': train_X_std,
               'train_y_std': train_y_std}
    
    return data_sep, dates_sep, norm_meas
    




def find_feats_lags(y,X,labs):
    """Function to screen a high-dimensional feature space and find the subset
    of independent features that hold the strongest relationship with the target. 
    
    Parameters
    - y: A numpy array containing the target data
    - X: A numpy array containing the feature space
    - labs: A numpy vector array containing the feature names
    
    Return
    - opt_vars: A numpy vector array containing the names of the features
                that form the screened feature space
    - best_lag: A numpy vector array containing the best lags for the features
                that form the screened feature space
    - best_corr: A numpy vector array containing the correlation values with the
                 target that form the screened feature space"""
    
    # Find the correlation and corresponding best lag of the features with the strongest relationship
    best_var_loc,best_var,best_corr,best_lag=find_cross_corr(y=y, X=X, labs=labs)
    
    # If less than 2 features hold a significant relationship with the target, return -1
    if len(best_var)<2:
        return -1, -1, -1
    
    # If at least 5 features hold a significant relationship, remove colinear
    if len(best_var) > 5:
        
        # Find subset of best features and apply specific lags
        X=X[:,best_var_loc]
        for i in range(len(best_lag)): X[:,i]=np.roll(X[:,i],best_lag[i],axis=0)
    
        # Remove the initial data points the same length as the largest lag
        if len(X)>24:
            lag_start_loc=int(np.ceil(np.max(best_lag)/3)*3)
            X=X[lag_start_loc:,:]
            y=y[lag_start_loc:]
        
        # Remove constant features, if present
        constant_vars_loc=np.where(np.isclose(np.var(X, axis=0),0))[0]
        if len(constant_vars_loc) > 0:
            X=np.delete(X,constant_vars_loc, axis=1)
            best_lag=np.delete(best_lag,constant_vars_loc)
            best_var=np.delete(best_var,constant_vars_loc)
            best_var_loc=np.delete(best_var_loc,constant_vars_loc)
            best_corr=np.delete(best_corr,constant_vars_loc)
        
        # Remove colinear features
        opt_vars=rm_colinear(y=y, X=X, corr_var=best_var)
        
        # Find the independent features with the strongest relationship to the target
        opt_vars_loc=np.where(np.in1d(best_var,opt_vars))[0]
        best_corr=best_corr[opt_vars_loc]
        best_lag=best_lag[opt_vars_loc]
        
        # Order features from strongest to weakest
        opt_vars_order=np.argsort(best_corr)[::-1]
        
        best_lag=best_lag[opt_vars_order]
        opt_vars=opt_vars[opt_vars_order]
        best_corr=best_corr[opt_vars_order]
    
    else: opt_vars=best_var
        
    return opt_vars, best_lag, best_corr
        





def prep_macro(mac,mac_vars):
    """Function to prepare the macroeconomic input data by carrying out the feature
    engineering. For each feature, this involves finding the 12 month difference (ACH),
    the 12 month percentage change (PCH), before finding the 3 month moving average
    and then finding the ACH and PCH on this smoothed series, giving 4 new features.
    
    Parameter
    - mac: A numpy array containing the macroeconomic data
    - mac_vars: A numpy array containing the column names for the parameter mac
        
    Return
    - new_mac: A numpy array containing the feature engineered features
    - new_labs: A numpy array containing the column names of the new features of new_mac"""
    
    # Find the 12 month difference (ACH) and the 12 month percentage change (PCH)
    ach=mac[12:,:]-np.roll(mac,12,axis=0)[12:,:]
    pch=((mac[12:,:]-np.roll(mac,12,axis=0)[12:])/np.abs(np.roll(mac,12,axis=0)[12:]))*100
    
    # Calculate the rolling 3 month moving average
    mav_3m=np.apply_along_axis(func1d = lambda x: np.convolve(x, np.ones(3), 'valid')/3, 
                               axis=0, arr=mac)
    
    # Calculate the ACH and PCH of the 3 month moving average
    ach_mav=mav_3m[12:,:]-np.roll(mav_3m,12,axis=0)[12:,:]
    pch_mav=((mav_3m[12,:]-np.roll(mav_3m,12,axis=0)[12:])/np.abs(np.roll(mav_3m,12,axis=0)[12:]))*100
    
    # Construct the new column names
    ach_lab=np.array([i+'_ach' for i in mac_vars])
    pch_lab=np.array([i+'_pch' for i in mac_vars])
    ach_mav_lab=np.array([i+'_ach_mav' for i in mac_vars])
    pch_mav_lab=np.array([i+'_pch_mav' for i in mac_vars])

    # Form the final array containing the new features
    new_mac=np.concatenate([ach[3:,:],pch[3:,:],ach_mav[1:,:],pch_mav[1:,:]], axis=1)
    
    # Form the final vector array containing the new feature names
    new_labs=np.append(ach_lab,pch_lab)
    new_labs=np.append(new_labs,ach_mav_lab)
    new_labs=np.append(new_labs,pch_mav_lab)
    
    return new_mac, new_labs






def rm_gaps(df):
    """Funtion to remove gaps 
    
    Parameters
    - df: A numpy array
    
    Return
    - df: A numpy array with gaps removed"""
    
    for i in range(df.shape[1]):
        non_nas=np.where(~np.isnan(df[:,i]))[0]
        non_nas_dif=np.diff(non_nas)
        gaps=np.where(non_nas_dif > 1)[0]
        if len(gaps) > 0:
            conv_nas=gaps[len(gaps)-1]+1
            df[non_nas[:conv_nas],i]=np.nan
    
    return df





def stock_data_prep(stock_act,full_prc_pch,full_prc_ach,full_est_pch,full_est_ach,
                    mac,mac_vars,stock_isin,full_isins,
                    wanted_dates,keep_est,train_size=0.8):
    """Parent function to prepare data for direct modelling. This involves
    feature engineering to create multiple different features through smoothing,
    differencing and percentage change. Further, the feature screening method is
    applied to screen features leaving just the independent features with a strong
    relationship with the target. The data is then standardised, before being
    split between training, validation and testing data.
    
    Parameters
    - stock_act: A numpy array containing the reported values
    - full_prc_pch: A numpy array containing the stock price data with PCH applied
    - full_prc_ach: A numpy array containing the stock price data with ACH applied
    - full_est_pch: A numpy array containing the estimated values with PCH applied
    - full_est_ach: A numpy array containing the estimated values with ACH applied
    - mac: A numpy array containing the macroeconomic data
    - mac_vars: A numpy vector array containing the column names of the macro data
    - stock_isin: A single character value giving the stock ISIN that we are interested in
    - full_isins: A numpy vector array containing the stock ISINs of the current stock universe
    - wanted_dates: A numpy vector array containing the dates corresponding to the time series data
    - keep_est: A boolean parameter that is True if we want to keep the estimates
                for the stock in question. False otherwise.
    - train_size: A single numeric value between [0,1] that gives the fraction
                  of data to be split between training and validation data
       
    
    Return
    - data_sep: A dictionary containing the standardised training, validation and 
                testing data
    - dates_sep: A dictionary containing the standardised training, validation and 
                 testing dates
    - norm_meas: A dictionary containing the training averages and standard deviations
                 used to standardise the data
    - feats_selected: A numpy vector array containing the names of the features selected
    - num_lags: A numpy vector array containing the optimal lag value for each feature
    - feats_corr: A numpy vector array containing the correlation values with the
                  target for each feature selected"""
    
    # If insufficient data, return -1
    if np.sum(~np.isnan(stock_act)) < 12:
        return -1, -1, -1, -1, -1, -1
        
    # Apply feature engineering to macro data
    mac, mac_vars=prep_macro(mac=mac, mac_vars=mac_vars)
    
    # Remove initial datapoints to ensure datasets have the same number of data points
    stock_act=stock_act[len(stock_act)-len(mac):,:]
    full_prc_pch=full_prc_pch[len(full_prc_pch)-len(mac):,:]
    full_prc_ach=full_prc_ach[len(full_prc_ach)-len(mac):,:]
    full_est_pch=full_est_pch[len(full_est_pch)-len(mac):,:]
    full_est_ach=full_est_ach[len(full_est_ach)-len(mac):,:]
    wanted_dates=wanted_dates[len(wanted_dates)-len(mac):]
    
    # Collate the features
    inputs=np.concatenate([full_est_pch,full_est_ach,full_prc_pch,full_prc_ach,mac], axis=1)
    
    # Construct the price and estimates feature names
    prc_pch_names=np.array(['prc_pch_'+i for i in full_isins])
    prc_ach_names=np.array(['prc_ach_'+i for i in full_isins])
    
    est_pch_names=np.array(['est_pch_'+i for i in full_isins])
    est_ach_names=np.array(['est_ach_'+i for i in full_isins])
        
    # Collate the feature names
    input_labels=np.append(est_pch_names,est_ach_names)
    input_labels=np.append(input_labels,prc_pch_names)
    input_labels=np.append(input_labels,prc_ach_names)
    input_labels=np.append(input_labels,mac_vars)
        
    # Remove empty data points, ensuring that we do not split quarters
    non_nas_loc=np.where(~np.isnan(stock_act))[0]
    if len(non_nas_loc)>0:
        
        # Ensure the start month is also the start of a quarter
        start_month=int(str(wanted_dates[non_nas_loc[0]])[5:7])
        start_month_dif=np.array([1,4,7,10])-start_month
        if 0 not in start_month_dif:
            if -1 in start_month_dif: non_nas_loc=non_nas_loc[2:]
            if -2 in start_month_dif: non_nas_loc=non_nas_loc[1:]
            
        # Ensure that the end month is also the end of a quarter
        end_month=int(str(wanted_dates[non_nas_loc[-1]])[5:7])
        end_month_dif=np.array([3,6,9,12])-end_month
        if 0 not in end_month_dif:
            if -1 in end_month_dif: non_nas_loc=non_nas_loc[:-1]
            if -2 in end_month_dif: non_nas_loc=non_nas_loc[:-2]   
            
    # Remove empty data points
    if non_nas_loc[0] > 0:
        stock_act=stock_act[non_nas_loc[0]:,:]
        inputs=inputs[non_nas_loc[0]:,:]
        wanted_dates=wanted_dates[non_nas_loc[0]:]
    
    # Ensure that only the latest 20 years of data are used, keeping an extra 2 years
    # as padding for when lagging and applying other applications
    if len(stock_act) > 264:
        stock_act=stock_act[(len(stock_act)-264):,:]
        inputs=inputs[(len(inputs)-264):,:]
        wanted_dates=wanted_dates[(len(wanted_dates)-264):]
    
    # Keep only inputs that have a full set of data, with no nas
    full_inputs_loc=np.where(np.sum(np.isnan(inputs),axis=0)==0)[0]
    inputs=inputs[:,full_inputs_loc]
    input_labels=input_labels[full_inputs_loc] 
    
    # Calculate training-validation cutoff
    train_cutoff=int(((len(inputs)-3)/3)*train_size)*3
        
    # Screen the features
    feats_selected,num_lags,feats_corr=find_feats_lags(y=stock_act[:train_cutoff,:],
                                            X=inputs[:train_cutoff,:],
                                            labs=input_labels)
    
    # If no features left after screening, return -1
    if feats_selected ==-1: return -1, -1, -1, -1, -1, -1
    if len(feats_corr) == 1: return -1, -1, -1, -1, -1, -1
    
    # Find screened features
    feat_loc=[int(np.where(feats_selected[i]==input_labels)[0]) for i in range(len(feats_selected))]
    inputs=inputs[:,feat_loc]
    
    # Apply optimal lags to screened features
    for i in range(len(num_lags)): inputs[:,i]=np.roll(inputs[:,i],num_lags[i])
    
    # Remove initial data points due to lag, ensuring that the data starts from the
    # first whole quarter
    qrtly_lag=int(np.ceil(np.max(num_lags)/3)*3)
    inputs=inputs[qrtly_lag:,:]
    stock_act=stock_act[qrtly_lag:,:]
    wanted_dates=wanted_dates[qrtly_lag:]
    
    # Standardise and split data into training, validation and testing data
    data_sep,dates_sep,norm_meas=split_and_norm(stock_act=stock_act,inputs=inputs,
                                                feats=feats_selected,wanted_dates=wanted_dates,
                                                train_size=train_size)
    
    return data_sep,dates_sep,norm_meas,feats_selected,num_lags,feats_corr





def stock_data_prep_trans(stock_act,full_prc_pch,full_prc_ach,full_est_pch,full_est_ach,mac,
                          mac_vars,stock_isin,full_isins,wanted_dates,keep_est,wanted_feats,
                          wanted_lags,num_qrters,train_size=0.8):
    
    """Parent function to prepare data for transfer learning. This involves
    feature engineering to create multiple different features through smoothing,
    differencing and percentage change. The transfer learning is homogeneous, so
    the wanted features used are provided as an input of the function. The data is 
    then standardised, before being split between training, validation and testing data.
          
    Parameters
    - stock_act: A numpy array containing the reported values
    - full_prc_pch: A numpy array containing the stock price data with PCH applied
    - full_prc_ach: A numpy array containing the stock price data with ACH applied
    - full_est_pch: A numpy array containing the estimated values with PCH applied
    - full_est_ach: A numpy array containing the estimated values with ACH applied
    - mac: A numpy array containing the macroeconomic data
    - mac_vars: A numpy vector array containing the column names of the macro data
    - stock_isin: A single character value giving the stock ISIN that we are interested in
    - full_isins: A numpy vector array containing the stock ISINs of the current stock universe
    - wanted_dates: A numpy vector array containing the dates corresponding to the time series data
    - keep_est: A boolean parameter that is True if we want to keep the estimates
                for the stock in question. False otherwise.
    - wanted_feats: The names of the wanted features
    - wanted_lags: The lags to be applied to the wanted features
    - train_size: A single numeric value between [0,1] that gives the fraction
                  of data to be split between training and validation data
             
            
    Return
    - data_sep: A dictionary containing the standardised training, validation and 
                testing data
    - dates_sep: A dictionary containing the standardised training, validation and 
                 testing dates
    - norm_meas: A dictionary containing the training averages and standard deviations
                 used to standardise the data
    - input_labels: A numpy vector array containing the names of the features selected"""
                        
    # Apply feature engineering to macro data
    mac, mac_vars=prep_macro(mac=mac, mac_vars=mac_vars)
    
    # Remove initial datapoints to ensure datasets have the same number of data points
    stock_act=stock_act[len(stock_act)-len(mac):,:]
    full_prc_pch=full_prc_pch[len(full_prc_pch)-len(mac):,:]
    full_prc_ach=full_prc_ach[len(full_prc_ach)-len(mac):,:]
    full_est_pch=full_est_pch[len(full_est_pch)-len(mac):,:]
    full_est_ach=full_est_ach[len(full_est_ach)-len(mac):,:]
    wanted_dates=wanted_dates[len(wanted_dates)-len(mac):]
    
    # Collate the features
    inputs=np.concatenate([full_est_pch,full_est_ach,full_prc_pch,full_prc_ach,mac], axis=1)
    
    # Construct the price and estimates feature names
    prc_pch_names=np.array(['prc_pch_'+i for i in full_isins])
    prc_ach_names=np.array(['prc_ach_'+i for i in full_isins])
    
    est_pch_names=np.array(['est_pch_'+i for i in full_isins])
    est_ach_names=np.array(['est_ach_'+i for i in full_isins])
    
    # Collate the feature names
    input_labels=np.append(est_pch_names,est_ach_names)
    input_labels=np.append(input_labels,prc_pch_names)
    input_labels=np.append(input_labels,prc_ach_names)
    input_labels=np.append(input_labels,mac_vars)

    # Ensure that only the latest 20 years of data are used, keeping an extra 2 years
    # as padding for when lagging and applying other applications
    if len(stock_act) > 264:
        stock_act=stock_act[(len(stock_act)-264):,:]
        inputs=inputs[(len(inputs)-264):,:]
        wanted_dates=wanted_dates[(len(wanted_dates)-264):]
    
    # Find screened features
    feats_loc=np.array([int(np.where(wanted_feats[i]==input_labels)[0]) for i in range(len(wanted_feats)) if wanted_feats[i] in input_labels])
    input_labels=input_labels[feats_loc]
    inputs=inputs[:,feats_loc]
        
    # Apply optimal lags to screened features
    for i in range(len(wanted_lags)): inputs[:,i]=np.roll(inputs[:,i],wanted_lags[i],axis=0)
    
    # Remove initial data points due to lag, ensuring that the data starts from the
    # first whole quarter
    lag_start_loc=int(np.ceil(np.max(wanted_lags)/3)*3)
    inputs=inputs[lag_start_loc:,:]
    stock_act=stock_act[lag_start_loc:,:]
    wanted_dates=wanted_dates[lag_start_loc:]
    
    # Remove empty data points, ensuring that we do not split quarters
    non_nas=np.where(~np.isnan(stock_act))[0]
    if len(non_nas)>0:
        
        # Ensure the start month is also the start of a quarter
        start_month=int(str(wanted_dates[non_nas[0]])[5:7])
        if start_month % 3 != 1:
            if start_month % 3 == 2: non_nas=non_nas[2:]
            if start_month % 3 == 0: non_nas=non_nas[1:]
        
        # Ensure that the end month is also the end of a quarter
        end_month=int(str(wanted_dates[non_nas[-1]])[5:7])
        if end_month % 3 != 0:
            if end_month % 3 == 1: non_nas=non_nas[:-1]
            if end_month % 3 == 0: non_nas=non_nas[:-2]
        
    # Standardise and split data into training, validation and testing data
    data_sep,dates_sep,norm_meas=split_and_norm(stock_act=stock_act[non_nas,:],inputs=inputs[non_nas,:],
                                                feats=mac_vars,wanted_dates=wanted_dates[non_nas],
                                                train_size=train_size,num_qrters=num_qrters)
    
    return data_sep,dates_sep,norm_meas,input_labels



    
    