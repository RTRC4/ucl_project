

import numpy as np
import pandas as pd
import pyreadr
import sys
import os
from statsmodels.tsa.stattools import adfuller


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





def rm_empty_cols(df, isins):
    """Function to remove empty columns or columns with insufficient data.
    
    Parameters
    - df: The numpy array that we want to remove the empty columns
    - isins: A numpy array containing the stock ISINs (column names)
    
    Returns
    - df: The numpy array containing the data initially passed, only with the
          empty columns removed.
    - isins: The numpy array containing the initial ISINs, only with the empty
             columns removed."""
            
    # Find the column positions of non empty columns with at least 12 data points
    num_nas=np.sum(np.isnan(df), axis=0)
    non_empty_cols=np.where(num_nas < (len(df)-12))[0]
    
    # Reduce isins and dataframe down to the previously found column positions
    isins=isins[non_empty_cols]
    df=df[:,non_empty_cols]
    
    return df, isins



def find_common_isins(act,est,prc,act_isins,est_isins,prc_isins,common_isins=None):
    """Function to find the common stock ISINs (column names) across multiple
    dataframes.
    
    
    Parameters
    - act: A numpy array containing the reported values. The columns are by stock, while
           the rows are by monthly date.
    - est: A numpy array containing the IBES consensus estimated values. The columns are 
           by stock, while the rows are by monthly date.
    - prc: A numpy array containing the stock price values. The columns are by stock, while
           while the rows are by monthly date.
    - act_isins: A numpy array containing the stock ISINs for the reported values
    - est_isins: A numpy array containing the stock ISINs for the estimated values
    - prc_isins: A numpy array containing the stock ISINs for the stock price values
    - common_isins: An optional parameter that gives a numpy array of existing 
                    common isins.
    
    
    Returns
    - act: A numpy array containing the reported values. The columns are reduced by
           common isins.
    - est: A numpy array containing the IBES consensus estimated values. The columns are 
           reduced by common isins.
    - prc: A numpy array containing the stock price values. The columns are reduced
           by common isins.
    - common_isins: A numpy array containing the common isins."""
    
    # Find common isins
    if common_isins is None:
        common_isins=act_isins[np.in1d(act_isins,est_isins)]
        common_isins=common_isins[np.in1d(common_isins,price_isins)]
        
    # Find location of common isins in different arrays
    act_isins_loc=np.array([int(np.where(common_isins[i]==act_isins)[0][0]) for i in range(len(common_isins)) if common_isins[i] in act_isins])
    est_isins_loc=np.array([int(np.where(common_isins[i]==est_isins)[0][0]) for i in range(len(common_isins)) if common_isins[i] in est_isins])
    prc_isins_loc=np.array([int(np.where(common_isins[i]==prc_isins)[0][0]) for i in range(len(common_isins)) if common_isins[i] in prc_isins])
    
    # Reduce each ISIN list down to their common ISINs (different orders)
    act_com_isins=act_isins[act_isins_loc]
    est_com_isins=est_isins[est_isins_loc]
    prc_com_isins=prc_isins[prc_isins_loc]
    
    # Keep only the common ISINs in each array (different orders)
    act=act[:,act_isins_loc]
    est=est[:,est_isins_loc]
    prc=prc[:,prc_isins_loc]
    
    # Sort each array so that the columns are in the same order
    act=act[:,np.argsort(act_com_isins)]
    est=est[:,np.argsort(est_com_isins)]
    prc=prc[:,np.argsort(prc_com_isins)]
    common_isins=common_isins[np.argsort(common_isins)]
    
    return act, est, prc, common_isins



def find_stationary(ts):
    """Function to determine if a given univariate time series is stationary
    using the Augmented Dickey-Fuller test. The changes applied to the series to
    produce a stationary series are the 12 month difference and the 12 month
    percentage change. Both are applied separately before being tested for stationarity.
    If both are stationary, then the change that leads to the lowest volatility is
    chosen.
    
    Parameter
    - ts:
    
    Return
    - A single character value giving the results of the stationarity test. If
      'insufficient_data', then there was insufficient data to carry out the
      test. If 'non-stationary', then the series is non stationary. If 'ach',
      then the series is stationary with the 12 month difference change. If 'pch',
      then the series is stationary with the 12 month percentage change."""
    
    # Calculate the 12 month percentage change and remove empty rows
    pch_ts=ts.copy()
    pch_ts[np.where(ts==0)]=0.0001
    pch_ts=(pch_ts[12:]-np.roll(pch_ts,12,axis=0)[12:])/np.abs(np.roll(pch_ts,12,axis=0))[12:]
    pch_clean=pch_ts[np.where(~np.isnan(pch_ts))]
    
    # If infinite values present, remove infinite rows. Calculate volatility as well
    if np.inf in np.abs(pch_clean): 
        pch_vol=np.inf
        non_inf=np.where(~np.isinf(pch_clean))
    else: 
        if pch_clean.std()==0: pch_vol=0
        else: pch_vol=max(np.abs((pch_clean-pch_clean.mean())/pch_clean.std()))
    
    # Calculate the 12 month difference, remove empty rows, calculate the volatility
    ach=(ts[12:]-np.roll(ts,12,axis=0)[12:])
    ach_clean=ach[np.where(~np.isnan(ach))]
    if ach_clean.std()==0: ach_vol=0
    else: ach_vol=max(np.abs((ach_clean-ach_clean.mean())/ach_clean.std()))
    
    # If inifinite value present in PCH, remove rows from both PCH and ACH
    if np.inf in np.abs(pch_clean):
        pch_clean=pch_clean[non_inf]
        ach_clean=ach_clean[non_inf]
    
    # Not enough data present, retun 'insufficient_data'
    if len(ach_clean) < 12:
        return 'insufficient_data'
    
    # Calculate the ADF test on both ACH and PCH series
    df_test_pch=adfuller(pch_clean)
    df_test_ach=adfuller(ach_clean)
    
    # Determine which ADF tests are statistically significant
    if df_test_pch[1] < 0.05: pch_stat_sig=True
    else: pch_stat_sig=False
    
    if df_test_ach[1] < 0.05: ach_stat_sig=True
    else: ach_stat_sig=False
    
    # If neither ACH or PCH series' ADF tests are statistically significant, return 'non-stationary'
    if pch_stat_sig is False and ach_stat_sig is False:
        return 'non-stationary'
        
    # Determine the statistical significant level for the PCH...
    if df_test_pch[0] < df_test_pch[4]['1%']: pch_t_val=1
    elif df_test_pch[0] < df_test_pch[4]['5%']: pch_t_val=5
    elif df_test_pch[0] < df_test_pch[4]['10%']: pch_t_val=10
    else: pch_t_val=100
    
    # ...and ACH series
    if df_test_ach[0] < df_test_ach[4]['1%']: ach_t_val=1
    elif df_test_ach[0] < df_test_ach[4]['5%']: ach_t_val=5
    elif df_test_ach[0] < df_test_ach[4]['10%']: ach_t_val=10
    else: ach_t_val=100
    
    # If neither PCH or ACH are statistically significant at least to the 10% level, return 'non-stationary'
    if pch_t_val > 10 and ach_t_val > 10:
        return 'non-stationary'
    
    # If only one series is statistically significant, return the change of that series
    if pch_stat_sig is True and ach_stat_sig is False:
        if pch_t_val <=10:
            return 'pch'
        else:
            return 'non-stationary'
    
    if pch_stat_sig is False and ach_stat_sig is True:
        if ach_t_val <=10:
            return 'ach'
        else:
            return 'non-stationary'
        
    # If both PCH and ACH series are statistically significant at the same level, return the least volatile
    if ach_t_val <= 10 and ach_t_val==pch_t_val:
        if pch_vol < ach_vol: 
            return 'pch'
        else: 
            return 'ach'
        
    # Return the least volatile change
    if min(pch_vol,ach_vol) == ach_vol:
        return 'ach'
    else:
        return 'pch'
        
        
    
 





def find_stat_change(df,isins):
    """Function to test for stationarity across multiple time series. For each
    column, stationarity is tested using the function find_stationarity.
    
    Parameters
    - df: A numpy array containing the time series to be tested for stationarity
    - isins: A numpy array containing the column names of the parameter df
        
    Return
    - change: A numpy array containing the change that is stationary. This can 
              be either ACH or PCH
    - new_isins: The column names (stock ISINs) of the columns that are stationary"""
    
    change_lt=list()
    new_isins=list()
    for i in range(len(isins)):
        change=find_stationary(ts=df[:,i].copy())
        if change in ['ach', 'pch']:
            change_lt.append(change)
            new_isins.append(isins[i])
    
    new_isins=np.array(new_isins)
    change=np.array(change_lt)
    
    return change, new_isins





def apply_grwth_change(act,est,prc,act_dates,est_dates,prc_dates,change):
    """Function to transform multiple dataframes by applying either the 12 month
    percentage change or the 12 month difference.
    
    
    Parameters
    - act: A numpy array containing the reported values. The columns are by stock, while
           the rows are by monthly date.
    - est: A numpy array containing the IBES consensus estimated values. The columns are 
           by stock, while the rows are by monthly date.
    - prc: A numpy array containing the stock price values. The columns are by stock, while
           while the rows are by monthly date.
    - act_dates: A numpy array containing the monthly dates for the reported values
    - est_dates: A numpy array containing the monthly dates for the estimated values
    - prc_dates: A numpy array containing the monthly dates for the stock price values
    - change: A numpy array of the same length as the number of columns as the act
              dataframe. Each element is either 'ACH' or 'PCH', depending on the
              change applied to the corresponding column in act.
    
    
    Returns
    - act_grwth_pd: A pandas dataframe containing the reported values, converted into 
                    growth values
    - act_lev_pd: A pandas dataframe containing the reported values, as level values
    - est_pch_pd: A pandas dataframe containing the estimated values, converted into
                  PCH growth values
    - est_ach_pd: A pandas dataframe containing the estimated values, converted into
                  ACH growth values
    - prc_pch_pd: A pandas dataframe containing the estimated values, converted into
                  PCH growth values
    - prc_ach_pd: A pandas dataframe containing the stock price values, converted into
                  ACH growth values
    - mac_pd: A pandas dataframe containing the cleaned macroeconomic data
    - change: A numpy array the same length as the number of columns in the act, est, prc
              dataframes. Each value is either 'ACH' or 'PCH', indicating if the stock
              change to the reported values is ACH or PCH.
              
    - act_grwth: A numpy array containing the reported values, with either an ACH or PCH
                 change applied to each column, depending on the parameter 'change'
    - est_grwth_pch: A numpy array containing the estimated values, with each column
                     transformed by taking the 12 month percentage change
    - est_grwth_ach: A numpy array containing the estimated values, with each column
                     transformed by taking the 12 difference
    - prc_grwth_pch: A numpy array containing the price values, with each column
                     transformed by taking the 12 month percentage change
    - prc_grwth_ach: A numpy array containing the price values, with each column
                     transformed by taking the 12 difference change
    - act_dates: A numpy array containing the monthly dates for the reported values
    - est_dates: A numpy array containing the monthly dates for the estimated values
    - prc_dates: A numpy array containing the monthly dates for the stock price values"""

    # Create empty lists to contain the growth values
    act_grwth_lt=list()
    prc_grwth_pch_lt=list()
    prc_grwth_ach_lt=list()
    est_grwth_pch_lt=list()
    est_grwth_ach_lt=list()
    
    # Build growth/difference values by column
    for i in range(act.shape[1]):
        
        # Calculate the reported 12 month percentage change
        if change[i]=='pch':
            act[np.where(act[:,i]==0),i]=0.0001
            temp_act=((act[12:,i]-np.roll(act[:,i],12,axis=0)[12:])/np.abs(np.roll(act[:,i],12,axis=0)[12:]))*100
            
        # Calculate the reported 12 month difference
        if change[i]=='ach':
            temp_act=act[12:,i]-np.roll(act[:,i],12,axis=0)[12:]
            
        # Find the change values for the estimates
        est[np.where(est[:,i]==0),i]=0.0001
        temp_est_pch=((est[12:,i]-np.roll(act[:,i],12,axis=0)[12:])/np.abs(np.roll(act[:,i],12,axis=0)[12:]))*100
        temp_est_ach=est[12:,i]-np.roll(act[:,i],12,axis=0)[12:]
        
        # Find the change values for the price values
        temp_prc_pch=((prc[12:,i]-np.roll(prc[:,i],12,axis=0)[12:])/np.abs(np.roll(prc[:,i],12,axis=0)[12:]))*100
        temp_prc_ach=prc[12:,i]-np.roll(prc[:,i],12,axis=0)[12:]
            
        # Add change values to corresponding lists
        act_grwth_lt.append(np.reshape(temp_act,(len(temp_act),1)))
        est_grwth_pch_lt.append(np.reshape(temp_est_pch,(len(temp_est_pch),1)))
        est_grwth_ach_lt.append(np.reshape(temp_est_ach,(len(temp_est_ach),1)))
        
        prc_grwth_pch_lt.append(np.reshape(temp_prc_pch,(len(temp_prc_pch),1)))
        prc_grwth_ach_lt.append(np.reshape(temp_prc_ach,(len(temp_prc_ach),1)))
        
    # Concatenate the change values by column
    act_grwth=np.concatenate(act_grwth_lt, axis=1)
    est_grwth_pch=np.concatenate(est_grwth_pch_lt, axis=1)
    est_grwth_ach=np.concatenate(est_grwth_ach_lt, axis=1)
    
    prc_grwth_pch=np.concatenate(prc_grwth_pch_lt, axis=1)
    prc_grwth_ach=np.concatenate(prc_grwth_ach_lt, axis=1)
    
    # Remove the first 12 months of dates
    act_dates=act_dates[12:]
    est_dates=est_dates[12:]
    prc_dates=prc_dates[12:]
    
    return act_grwth, est_grwth_pch, est_grwth_ach, prc_grwth_pch, prc_grwth_ach, act_dates, est_dates, prc_dates






def find_common_dates(act, est, prc, mac, act_dates, est_dates, prc_dates, mac_dates,
                      rm_lead_trail_nas=True, act_lev=None, act_lev_dates=None,
                      est_2=None, prc_2=None):
    """Function to find the common dates across multiple dataframes and reduce each
    dataframe down to the common dates.
    
    
    Parameters
    - act: A numpy array containing the reported values. The columns are by stock, while
           the rows are by monthly date.
    - est: A numpy array containing the IBES consensus estimated values. The columns are 
           by stock, while the rows are by monthly date.
    - mac: A numpy array containing the macroeconomic data. The columns are by macro indicator
           while the rows are by monthly date.
    - prc: A numpy array containing the stock price values. The columns are by stock, while
           while the rows are by monthly date.
    - act_dates: A numpy array containing the monthly dates for the reported values
    - est_dates: A numpy array containing the monthly dates for the estimated values
    - mac_dates: A numpy array containing the monthly dates for the macroeconomic values
    - prc_dates: A numpy array containing the monthly dates for the stock price values
    - rm_lead_trail_nas: A boolean parameter that is True if we are removing the leading
                         and trailing na values
    - act_lev: An optional parameter that gives a numpy array containing the reported
               values without any changes. That is, the level values. If None, no array
               is passed to this function. The default is None.
    - act_lev_dates: An optional parameter that gives a numpy array containing the monthly
                     dates for the level values of the reported values. If None, no array
                     is passed to this function. The default is None.
    - est_2: An optional parameter that gives a second numpy array containing the estimated
             values. This is used for the estimated values with different changes. For 
             instance, est can be defined for the PCH values while est_2 can be defined
             for the ACH values. If None, no array is passed to this function. The default 
             is None.
    - prc_2: An optional parameter that gives a second numpy array containing the price
             values. This is used for the estimated values with different changes. For 
             instance, est can be defined for the PCH values while est_2 can be defined
             for the ACH values. If None, no array is passed to this function. The default 
             is None.
    
    
    Returns
    - act: A numpy array containing the reported values, with the common dates
    - est: A numpy array containing the estimated values, with the common dates
    - prc: A numpy array containing the price values, with the common dates
    - mac: A numpy array containing the macroeconomic values, with the common dates
    - common_dates: A numpy array containing the common monthly dates
    - est_2: A numpy array containing the estimated values for the second change, with 
             the common dates.
    - prc_2: A numpy array containing the price values for the second change, with 
             the common dates.
    - act_lev: A numpy array containing the reported, level, values with no change, with 
             the common dates."""
             
    # Remove leading and trailing NaNs
    if rm_lead_trail_nas==True:
        row_nas=np.sum(~np.isnan(act), axis=1)
        row_non_nas_loc=np.where(row_nas>0)[0]
        act=act[row_non_nas_loc,:]
        act_dates=act_dates[row_non_nas_loc]
        
    # Find the common dates across dataframes
    common_dates=act_dates[np.in1d(act_dates,est_dates)]
    common_dates=common_dates[np.in1d(common_dates,prc_dates)]
    common_dates=common_dates[np.in1d(common_dates,mac_dates)]
    if act_lev is not None: common_dates=common_dates[np.in1d(common_dates,act_lev_dates)]
    
    # Find the row index location of the common dates
    act_dates_loc=np.array([int(np.where(common_dates[i]==act_dates)[0]) for i in range(len(common_dates)) if common_dates[i] in act_dates])
    est_dates_loc=np.array([int(np.where(common_dates[i]==est_dates)[0]) for i in range(len(common_dates)) if common_dates[i] in est_dates])
    prc_dates_loc=np.array([int(np.where(common_dates[i]==prc_dates)[0]) for i in range(len(common_dates)) if common_dates[i] in prc_dates])
    mac_dates_loc=np.array([int(np.where(common_dates[i]==mac_dates)[0]) for i in range(len(common_dates)) if common_dates[i] in mac_dates])
    if act_lev is not None:
        act_lev_dates_loc=np.array([int(np.where(common_dates[i]==act_lev_dates)[0]) for i in range(len(common_dates)) if common_dates[i] in act_lev_dates])
       
    # Reduce the dataframes down to the rows corresponding to the common dates
    act=act[act_dates_loc,:]
    est=est[est_dates_loc,:]
    prc=prc[prc_dates_loc,:]
    mac=mac[mac_dates_loc,:]
    if act_lev is not None: act_lev=act_lev[act_lev_dates_loc,:]
    if est_2 is not None: est_2=est_2[est_dates_loc,:]
    if prc_2 is not None: prc_2=prc_2[prc_dates_loc,:]
        
    if act_lev is None:
        return act,est,prc,mac,common_dates
    if act_lev is not None:
        return act,est,est_2,prc,prc_2,mac,act_lev,common_dates
    
    



def clean_data(act,est,mac,prc,act_dates,est_dates,mac_dates,prc_dates,
               act_isins,est_isins,prc_isins,mac_var):
    """Main parent function to clean data. This function will take in numerous dataframes
    and clean the data by finding common dates and for act, est and prc, find the common
    companies by ISIN. The non-stationary series in the reported dataframe are removed.
    The 12 month percentage change and 12 month difference is calculated for the est
    and prc dataframes. For stationary reported series, either the ACH or PCH is calculated,
    depending on which one is stationary and if both are stationary, which is least
    volatile.
    
    
    Parameters
    - act: A numpy array containing the reported values. The columns are by stock, while
           the rows are by monthly date.
    - est: A numpy array containing the IBES consensus estimated values. The columns are 
           by stock, while the rows are by monthly date.
    - mac: A numpy array containing the macroeconomic data. The columns are by macro indicator
           while the rows are by monthly date.
    - prc: A numpy array containing the stock price values. The columns are by stock, while
           while the rows are by monthly date.
    - act_dates: A numpy array containing the monthly dates for the reported values
    - est_dates: A numpy array containing the monthly dates for the estimated values
    - mac_dates: A numpy array containing the monthly dates for the macroeconomic values
    - prc_dates: A numpy array containing the monthly dates for the stock price values
    - act_isins: A numpy array containing the stock ISINs for the reported values
    - est_isins: A numpy array containing the stock ISINs for the estimated values
    - prc_isins: A numpy array containing the stock ISINs for the stock price values
    - mac_var: A numpy array containing the column names of the macroeconomic indicators
    
    
    Returns
    - act_grwth_pd: A pandas dataframe containing the reported values, converted into 
                    growth values
    - act_lev_pd: A pandas dataframe containing the reported values, as level values
    - est_pch_pd: A pandas dataframe containing the estimated values, converted into
                  PCH growth values
    - est_ach_pd: A pandas dataframe containing the estimated values, converted into
                  ACH growth values
    - prc_pch_pd: A pandas dataframe containing the estimated values, converted into
                  PCH growth values
    - prc_ach_pd: A pandas dataframe containing the stock price values, converted into
                  ACH growth values
    - mac_pd: A pandas dataframe containing the cleaned macroeconomic data
    - change: A numpy array the same length as the number of columns in the act, est, prc
              dataframes. Each value is either 'ACH' or 'PCH', indicating if the stock
              change to the reported values is ACH or PCH."""
    
    ### Subset data by common dates and columns, apply transformations
    
    # Remove empty columns
    act,act_dates,act_isins=rm_empty_cols(df=act, isins=act_isins)
    
    # Find subset of common columns by ISIN column names across dataframes
    act,est,prc,common_isins=find_common_isins(act,est,prc,act_isins,est_isins,prc_isins)
    
    # Find the stationary series and remove the non-stationary
    change, new_isins=find_stat_change(df=act, isins=common_isins)
    act,est,prc,common_isins=find_common_isins(act,est,prc,common_isins,common_isins,common_isins,new_isins)
    
    # Find subset of common dates across dataframes
    act,est,prc,mac,common_dates=find_common_dates(act,est,prc,mac,
                                                   act_dates,est_dates,prc_dates,
                                                   mac_dates,rm_lead_trail_nas=False)
    
    # Calculate growth
    act_grwth,est_pch,est_ach,prc_pch,prc_ach,act_dates,est_dates,prc_dates=apply_grwth_change(act,est,prc,common_dates,
                                                                                               common_dates,common_dates,change)

    # Find subset of common dates across dataframes
    act_grwth,est_pch,est_ach,prc_pch,prc_ach,mac,act_lev,common_dates=find_common_dates(act=act_grwth,
                                                                                         est=est_pch,
                                                                                         est_2=est_ach,
                                                                                         prc=prc_pch,
                                                                                         prc_2=prc_ach,
                                                                                         mac=mac, 
                                                                                         act_lev=act,
                                                                                         act_dates=act_dates,
                                                                 est_dates=est_dates,prc_dates=prc_dates,
                                                                 mac_dates=common_dates,act_lev_dates=common_dates,
                                                                 rm_lead_trail_nas=True)
        
    # Back fill the initial row if NaN present
    initial_nan=np.where(np.isnan(mac[0,:]))
    mac[0,initial_nan]=mac[1,initial_nan]
    
    
    
    
    ### Convert dataframes to pandas
    
    # Find pandas date series
    common_dates_pd=pd.date_range(str(common_dates[0,0]),periods=len(common_dates), freq='MS')
    
    # Convert actual values growth into a pandas dataframe
    act_grwth_pd=pd.DataFrame(act_grwth, index=common_dates_pd)
    act_grwth_pd.index=act_grwth_pd.index.date
    act_grwth_pd.columns=common_isins
    
    # Convert actual levels growth into a pandas dataframe
    act_lev_pd=pd.DataFrame(act_lev, index=common_dates_pd)
    act_lev_pd.index=act_lev_pd.index.date
    act_lev_pd.columns=common_isins
    
    # Convert estimates PCH growth into a pandas dataframe
    est_pch_pd=pd.DataFrame(est_pch, index=common_dates_pd)
    est_pch_pd.index=est_pch_pd.index.date
    est_pch_pd.columns=common_isins
    
    # Convert estimates ACH growth into a pandas dataframe
    est_ach_pd=pd.DataFrame(est_ach, index=common_dates_pd)
    est_ach_pd.index=est_ach_pd.index.date
    est_ach_pd.columns=common_isins
    
    # Convert price PCH growth into a pandas dataframe
    prc_pch_pd=pd.DataFrame(prc_pch, index=common_dates_pd)
    prc_pch_pd.index=prc_pch_pd.index.date
    prc_pch_pd.columns=common_isins
    
    # Convert price ACH growth into a pandas dataframe
    prc_ach_pd=pd.DataFrame(prc_ach, index=common_dates_pd)
    prc_ach_pd.index=prc_ach_pd.index.date
    prc_ach_pd.columns=common_isins
    
    # Convert macro dataframe into a pandas dataframe
    mac_pd=pd.DataFrame(mac, index=common_dates_pd)
    mac_pd.index=mac_pd.index.date
    mac_pd.columns=mac_var
    
    return act_grwth_pd,act_lev_pd,est_pch_pd,est_ach_pd,prc_pch_pd,prc_ach_pd,mac_pd,change
    
    
    
    
    




if __name__=='__main__':
    
    ## EPS Reported Data
    # Read in data and extract ISINs
    eps_act_path=os.path.join(get_path('data/raw_data_v2'), 'US500_EPS_Actuals_v2.rds')
    eps_act=pyreadr.read_r(eps_act_path)
    eps_act=eps_act[None]
    act_isins=np.array(eps_act.columns)
    eps_act=np.array(eps_act)
    
    
    
    
    # Construct EPS dates
    act_dates=np.array([i.strftime("%Y-%b") for i in pd.date_range(start='1991-01-01', end='2022-06-01', freq='MS')])
    act_dates=np.reshape(act_dates, (len(act_dates),1))
    
    
    
    
    
    
    ## Macro Data
    # Read in the macro data, convert to np and extract variable names
    macro_data_path=os.path.join(get_path('data/raw_data_v2'), 'US500_MacroData_v2.rds')
    macro_data=pyreadr.read_r(macro_data_path)
    macro_data=macro_data[None]
    macro_var=np.array(macro_data.columns)
    macro_data=np.array(macro_data)
    
    
    
    # Construct macro data dates
    mac_dates=np.array([i.strftime("%Y-%b") for i in pd.date_range(start='1991-01-01', end='2022-06-01', freq='MS')])
    mac_dates=np.reshape(mac_dates, (len(mac_dates),1))
    
    
    
    
    
    ## EPS Estimates Data
    # Read in the EPS estimates data
    eps_est_path=os.path.join(get_path('data/raw_data_v2'), 'US500_EPS_Estimates_v2.rds')
    eps_est=pyreadr.read_r(eps_est_path)
    eps_est=eps_est[None]

    # Creat the dates that correspond to the estimates data, extract the ISIN names and convert to numpy
    est_dates=np.array([i.strftime("%Y-%b") for i in pd.date_range(start='1991-01-01', end='2022-05-01', freq='MS')])
    est_isins=np.array(eps_est.columns)
    eps_est=np.array(eps_est)
    
    
    
    
    
    
    ## Equity Price Data
    # Read in the equity price data
    price_est_path=os.path.join(get_path('data/raw_data_v2'), 'US500_EPS_price_react_v2.csv')
    price=pd.read_csv(price_est_path, index_col=0)
    
    
    # Extract ISINs, create dates and convert to numpy
    price_isins=np.array(price.columns)
    price_dates=np.array([i.strftime("%Y-%b") for i in pd.date_range(start='1991-01-01', end='2022-06-01', freq='MS')])
    price=np.array(price)
    
    
    
    
    
    ## Refine and Align Data
    # Find the common stocks and dates between the reported and estimated EPS and apply PCH/ACH change to estimates
    act_grwth,act_lev,est_pch,est_ach,prc_pch,prc_ach,mac,change=clean_data(act=eps_act,est=eps_est,mac=macro_data,prc=price,
                                      act_dates=act_dates,est_dates=est_dates,mac_dates=mac_dates,
                                      prc_dates=price_dates,act_isins=act_isins,est_isins=est_isins,
                                      prc_isins=price_isins,mac_var=macro_var)
    
    # Save refined data
    act_grwth.to_pickle(os.path.join(get_path('data/refined_data_v2'), 'eps_reported_refined.pkl'))
    act_lev.to_pickle(os.path.join(get_path('data/refined_data_v2'), 'eps_reported_level_refined.pkl'))
    est_pch.to_pickle(os.path.join(get_path('data/refined_data_v2'), 'eps_estimated_pch.pkl'))
    est_ach.to_pickle(os.path.join(get_path('data/refined_data_v2'), 'eps_estimated_ach.pkl'))
    prc_pch.to_pickle(os.path.join(get_path('data/refined_data_v2'), 'price_pch.pkl'))
    prc_ach.to_pickle(os.path.join(get_path('data/refined_data_v2'), 'price_ach.pkl'))
    mac.to_pickle(os.path.join(get_path('data/refined_data_v2'), 'macro_eps_refined.pkl'))
    np.save(os.path.join(get_path('data/refined_data_v2'), 'eps_changes.npy'), change)
    
    
    
    
    
    
    ## Load and refigure S&P 500 proxy membership
    # Read in data
    sp_proxy_path=os.path.join(get_path('data/raw_data_v2'), 'US500_SP500_Proxy_v2.rds')
    sp_proxy=pyreadr.read_r(sp_proxy_path)
    sp_proxy=sp_proxy[None]
    
    # Construct dates and add to pandas dataframe as index
    sp_proxy_dates=np.array([i.strftime("%Y-%b") for i in pd.date_range(start='2010-01-01', end='2022-06-01', freq='MS')])
    sp_proxy_dates_pd=pd.date_range(str(sp_proxy_dates[0]),periods=len(sp_proxy_dates), freq='MS')
    sp_proxy.index=sp_proxy_dates_pd
    sp_proxy.index=sp_proxy.index.date
    
    # Save S&P 500 proxy membership
    sp_proxy.to_pickle(os.path.join(get_path('data/refined_data_v2'), 'sp500_proxy.pkl'))
    
    
    