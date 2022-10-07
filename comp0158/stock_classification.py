
import numpy as np




def assign_multi_clusters(parents_df, children_df, parents_isins, children_isins, 
                          village_num=5):
    """For a given set of targets and a set of possible source domain series, this
    function identifies the series to use as the source domain for each target. That
    is, for each target, the Euclidean distance is calculated between each possible
    source series with the N series that have the smallest distance used as the
    source domain.
    
    Parameters
    - parents_df: A numpy array containing the possible source domain series
    - children_df: A numpy array containing all target series
    - parents_isins: A numpy vector array containing the stock ISINs of each possible 
                     source series
    - children_isins: A numpy vector array containing the stock ISINs of each
                      target series
    - village_num: A single numeric value giving the number of series to be used
                   as the source domain
    
    Return
    - clust: A dictionary containing the source domain series for each target"""
    
    # Standardise the possible source domain series
    parents_avg=np.nanmean(parents_df, axis=0)
    parents_std=np.nanstd(parents_df, axis=0)
    parents_df=(parents_df-parents_avg)/parents_std
    
    # Standardise the target series
    children_avg=np.nanmean(children_df, axis=0)
    children_std=np.nanstd(children_df, axis=0)
    children_df=(children_df-children_avg)/children_std
    
    # For each target, find the series with the smallest distance to use as the source domain
    clusters=list()
    for i in range(len(children_isins)):
        non_nas=np.where(~np.isnan(children_df[:,[i]]))[0]
        temp_dist=np.mean((parents_df[non_nas,:]-children_df[non_nas,[i]].reshape(len(non_nas),1))**2,axis=0)
        clusters.append(parents_isins[np.argsort(temp_dist)[:village_num]])
       
    clust=dict(zip(children_isins,clusters)) 
           
    return clust


