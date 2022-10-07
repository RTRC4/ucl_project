
import numpy as np



def forward_stepwise_selection_inner(rand_preds,unnorm_val_y,rand_mae,
                                     subset_size=20,start_num=3):
    """Function to carry out forward stepwise selection of multiple ensemble
    models
    
    Parameters
    - rand_preds: A numpy array containing the randomly selected validation predictions
    - unnorm_val_y: A numpy array containing the validation target data
    - rand_mae: A numpy vector array containing the MAE of the validation predictions
    - subset_size: A single numeric value giving the number of models to be added
                   to the total ensemble group
    - start_num: A single numeric value giving the number of models to be used as
                 the starting group of models
        
    Return
    - ensemble_loc: A numpy vector array containing the column index locations of
                    the models selected"""
                
    # Set up the starting models and the possible index location array
    pos_loc=np.arange(0,len(rand_mae))
    initial_preds=np.argsort(rand_mae)[:start_num]
    ensemble_pred=rand_preds[:,initial_preds]    
    ensemble_loc=pos_loc[initial_preds]
    
    # Iterate through and add the models in a stepwise fashion
    k=0
    while k<(subset_size-start_num):
        
        # Stack the validation predictions
        rand_preds_lt=list(rand_preds.T)
        rand_preds_lt=[i.reshape(len(i),1) for i in rand_preds_lt]
        rand_preds_stack=np.stack(rand_preds_lt)
                
        # Calculate the average ensemble between the selected models and each
        # of the validation predictions
        ensemble_sum=np.sum(ensemble_pred,axis=1, keepdims=True)
        rand_preds_stack=rand_preds_stack+ensemble_sum
        rand_preds_stack=rand_preds_stack/(len(ensemble_loc)+1)
        
        # Calculate the MAE for each possible addition to the ensemble group
        new_mae=np.mean(np.abs(rand_preds_stack-unnorm_val_y),axis=1)
        
        # Add the the model to the set of ensembles that leads to the lowest MAE
        ensemble_loc=np.append(ensemble_loc, pos_loc[np.argmin(new_mae)])
        ensemble_pred=rand_preds[:,ensemble_loc]
                            
        k+=1
            
    return pos_loc[ensemble_loc]






def forward_stepwise_selection(val_preds,unnorm_val_y,subset_size=100,
                               start_num=10,ensemble_num=20,p=0.8):
    """Function to carry out forward stepwise selection of the ensemble models.
    This function will create an ensemble of the ensembles selected which are then
    averaged. This is based on the forward stepwise selection technique for ensemble
    model selection as proposed by 
    https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
    
    Parameters
    - val_preds: A numpy array containing the validation predictions
    - unnorm_val_y: A numpy array containing the validation target data
    - subset_size: A single numeric value giving the number of models to be added
                   to the total ensemble group
    - start_num: A single numeric value giving the number of models to be used as
                 the starting group of models
    - ensemble_num: A single numeric value giving the number of ensembles to be
                    created through forward stepwise selection
    - p: A single numeric value between [0,1] giving the fraction of the validation
         predictions to be used when applying the forward stepwise selection
    
    Return
    - ensemble_preds_lt: A list containing the ensembles"""
    
    # Calculate the MAE values for the validation predictions
    val_mae=np.mean(np.abs(val_preds-unnorm_val_y),axis=0)
    
    # Remove empty columns from the validation predictions array
    pos_loc=np.arange(0,len(val_mae))
    non_empty_cols=np.where(np.sum(np.isnan(val_preds),axis=0)==0)[0]
    if len(non_empty_cols) < val_preds.shape[1]:
        pos_loc=pos_loc[non_empty_cols]
        val_preds=val_preds[:,non_empty_cols]
        val_mae=val_mae[non_empty_cols]
    
    # If too few predictions left, return all predictions
    if int(p*val_preds.shape[1])==0:
        return pos_loc
    
    # Apply forward stepwise selection over multiple runs with different randomly
    # selected validation predictions
    ensemble_preds_lt=list()
    for i in range(ensemble_num):
        
        # Randomly select the validation predictions
        rand_preds_loc=np.random.choice(val_preds.shape[1],int(p*val_preds.shape[1]),
                                        replace=False)
        
        # Carry out forward stepwise selection and add index locations to list
        temp_loc=forward_stepwise_selection_inner(rand_preds=val_preds[:,rand_preds_loc],
                                                  rand_mae=val_mae[rand_preds_loc],
                                                  unnorm_val_y=unnorm_val_y,
                                                  subset_size=subset_size,
                                                  start_num=start_num)
        ensemble_preds_lt.append(pos_loc[rand_preds_loc[temp_loc]])
        
    return ensemble_preds_lt





def forward_stepwise_selection_ens(preds, combo_loc):
    """Given a collection of predictions constructed through the forward stepwise
    selection technique, this function finds the average predictions
    
    Parameters
    - preds: A numpy array containing the predictions
    - combo_loc: A list of numpy arrays containing the column locations of the
                 selected models
    
    Return
    - ensemble_test: A numpy array containing the average predictions of the
                     ensemble of forward stepwise selection predictions"""
    
    # Find the ensemble prediction and its MAE
    lt=list()
    for i in range(len(combo_loc)):
        temp=preds[:,combo_loc[i]]
        lt.append(np.mean(temp,axis=1,keepdims=True))
    ensemble_val=np.mean(np.hstack(lt), axis=1, keepdims=True)
    
    return ensemble_val
