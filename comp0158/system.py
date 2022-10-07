
import numpy as np
import os
import sys
import tensorflow as tf



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
    
from comp0158 import neural_networks as nn
from comp0158 import bagging as bg
from comp0158 import prep_train_test_data as prep
from comp0158 import fwd_stepwise_selection as fss








class single_model():
    """Class to define the object to hold the model and results of a single model
    for a given ensemble model."""
    
    def __init__(self, model, feats, rel_mae, preds, sampled_idx):
        self.model = model
        self.feats = feats
        self.rel_mae = rel_mae
        self.preds=preds
        self.sampled_idx=sampled_idx



class ensemble_model():
    """Class to define the object holding the ensemble model results for a given
    target and a given date. This includes all models for the ensemble model,
    as well as the standardising values and any parameter values."""
    
    def __init__(self, stock_isin, stock_change, train_X_avg, train_X_std, train_y_avg, 
                 train_y_std, train_dates, test_dates, val_dates, var_names, 
                 num_lags, bootstrap_num=20, subset_feat=0.5):
        
        self.stock_isin=stock_isin
        self.train_X_avg=train_X_avg
        self.train_X_std=train_X_std
        self.train_y_avg=train_y_avg
        self.train_y_std=train_y_std
        
        self.train_dates=train_dates
        self.test_dates=test_dates
        self.val_dates=val_dates
        
        self.stock_change=stock_change
        self.bootstrap_num=bootstrap_num
        self.subset_feat=subset_feat
        
        self.var_names=var_names
        self.num_lags=num_lags
        
        
        
    def build_model(self, train_X, train_y, val_X, val_y, ibes_est, opt_lr=None, 
                    opt_num_neurons=None, vary_neurons=False):
        """Function to build the randomised ensemble model. The results are
        saved in the class object, with nothing returned."""
        
        # Unnormalise the validation data
        unnorm_val_y=(val_y*self.train_y_std)+self.train_y_avg
        
        feats=self.var_names
        
        # Tune for optimal learning rate, if required
        if opt_lr is None: opt_lr=nn.tune_lr(train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y)
        
        # Set up the possible values for the number of neurons, number of layers, learning rate and epsilon
        pos_num_neurons=np.array([4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,96,128])
        pos_num_neurons=pos_num_neurons[np.where(pos_num_neurons<= max(4,int(self.subset_feat*train_X.shape[1])))]
        pos_num_layers=np.array([1,2])  
        
        pos_lr=np.array([1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4])
        pos_eps=np.array([1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8])
        
        
        # Convert data to tensorflow
        train_X=tf.convert_to_tensor(train_X)
        train_y=tf.convert_to_tensor(train_y)
        val_X=tf.convert_to_tensor(val_X)
        val_y=tf.convert_to_tensor(val_y)
        
        # Apply block bootstrap aggregation and create ensemble predictions
        pred_lt=list()
        pred_mae_lt=list()
        for i in range(self.bootstrap_num):
            
            # Create block bootstrap indices and sample
            if len(train_X) >=36:
                sampled_idx=bg.block_bootstrapping(len(train_X))
                sampled_train_X=tf.gather(train_X,sampled_idx,axis=0)
                sampled_train_y=tf.gather(train_y,sampled_idx,axis=0)
            else:
                sampled_idx=np.arange(0,len(train_X),1)
                sampled_train_X=train_X
                sampled_train_y=train_y
            
            # Randomly select the neural network parameters
            num_neurons=pos_num_neurons[np.random.randint(0,len(pos_num_neurons))]
            num_layers=pos_num_layers[np.random.randint(0,2)]
            num_layers=1
            opt_lr=pos_lr[np.random.randint(0,len(pos_lr))]
            eps=pos_eps[np.random.randint(0,len(pos_eps))]
            
            #  Build network with randomly subsetting the features
            if self.subset_feat is not None: 
                
                # Randomly subset features
                temp_feat=np.random.choice(train_X.shape[1],int(train_X.shape[1]*self.subset_feat), replace=False)
                sampled_train_X=tf.gather(sampled_train_X,temp_feat,axis=1)
                sampled_val_X=tf.gather(val_X,temp_feat,axis=1)
            
                # Build network
                net=nn.network_varied_parameters(lr=opt_lr, epsilon=eps, num_neurons=num_neurons, num_layers=num_layers,
                                                 input_shape=sampled_train_X.shape[1], epochs=100)
                temp_pred, temp_mod, temp_loss=net.train_model(sampled_train_X, sampled_train_y, sampled_val_X, val_y)
                    
                # Find the MAE of prediction and combine results into an object    
                temp_pred_m1=bg.quarterly_average(df=temp_pred,num_months=1)
                temp_mae=np.mean(np.abs(unnorm_val_y-((temp_pred_m1*self.train_y_std)+self.train_y_avg)))
                pred_mae_lt.append(temp_mae)
                model=single_model(model=temp_mod, feats=feats[temp_feat],
                                   rel_mae=temp_mae, preds=temp_pred, sampled_idx=sampled_idx)
            
            # Build network without randomly subsetting the features
            if self.subset_feat is None:
                
                # Build network
                net=nn.network_varied_parameters(lr=opt_lr, epsilon=eps, num_neurons=num_neurons, num_layers=num_layers,
                                   input_shape=sampled_train_X.shape[1], epochs=100)
                temp_pred, temp_mod, temp_loss=net.train_model(sampled_train_X, sampled_train_y, val_X, val_y)
                    
                # Find the MAE of prediction and combine results into an object    
                temp_pred_m1=bg.quarterly_average(df=temp_pred,num_months=1)
                temp_mae=np.mean(np.abs(unnorm_val_y-((temp_pred_m1*self.train_y_std)+self.train_y_avg)))
                pred_mae_lt.append(temp_mae)
                model=single_model(model=temp_mod, feats=feats,
                                   rel_mae=temp_mae, preds=temp_pred, sampled_idx=sampled_idx)
            
            # Add results to main object
            self.__dict__['model_'+str(i+1)]=model
            pred_lt.append(temp_pred)
            
        
        # Collate validation predictions and find the average prediction
        preds=np.concatenate(pred_lt, axis=1)
        avg_pred=np.mean(preds, axis=1)
        avg_pred=np.reshape(avg_pred,(len(avg_pred),1))
        
        # Find the average prediction at the first month of the quarter
        avg_pred_m1=bg.quarterly_average(df=avg_pred,num_months=1)
        avg_pred_m1=(avg_pred_m1*self.train_y_std)+self.train_y_avg
        avg_pred_m1_mae=np.mean(np.abs(unnorm_val_y-avg_pred_m1))
                
        # Find the IBES estimates at the first month of the quarter
        ibes_est_m1=bg.quarterly_average(df=ibes_est,num_months=1)
        ibes_est_m1_mae=np.mean(np.abs(unnorm_val_y-ibes_est_m1))
        
        # Find the average prediction at the second month of the quarter
        avg_pred_m2=bg.quarterly_average(df=avg_pred,num_months=2)
        avg_pred_m2=(avg_pred_m2*self.train_y_std)+self.train_y_avg
        avg_pred_m2_mae=np.mean(np.abs(unnorm_val_y-avg_pred_m2))
                
        # Find the IBES estimates at the second month of the quarter
        ibes_est_m2=bg.quarterly_average(df=ibes_est,num_months=2)
        ibes_est_m2_mae=np.mean(np.abs(unnorm_val_y-ibes_est_m2))
        
        # Find the average prediction at the third month of the quarter
        avg_pred_m3=bg.quarterly_average(df=avg_pred,num_months=3)
        avg_pred_m3=(avg_pred_m3*self.train_y_std)+self.train_y_avg
        avg_pred_m3_mae=np.mean(np.abs(unnorm_val_y-avg_pred_m3))
                
        # Find the IBES estimates at the third month of the quarter
        ibes_est_m3=bg.quarterly_average(df=ibes_est,num_months=3)
        ibes_est_m3_mae=np.mean(np.abs(unnorm_val_y-ibes_est_m3))
        
        # Add the average predictions and estimates to the main object
        self.avg_pred_m1_val=avg_pred_m1
        self.avg_pred_m2_val=avg_pred_m2
        self.avg_pred_m3_val=avg_pred_m3
        
        self.ibes_est_m1_val=ibes_est_m1
        self.ibes_est_m2_val=ibes_est_m2
        self.ibes_est_m3_val=ibes_est_m3
        
        # Add the MAE values to the main object
        self.avg_pred_m1_val_mae=avg_pred_m1_mae
        self.avg_pred_m2_val_mae=avg_pred_m2_mae
        self.avg_pred_m3_val_mae=avg_pred_m3_mae
        
        self.ibes_est_m1_val_mae=ibes_est_m1_mae
        self.ibes_est_m2_val_mae=ibes_est_m2_mae
        self.ibes_est_m3_val_mae=ibes_est_m3_mae
        
        
        
        
        
       
        
    def build_trans_model(self, train_X, train_y, val_X, val_y, ibes_est, parent_mod):
        """Function to build transfer learning model using RE method"""
        
        # Unnormalise testing data
        unnorm_val_y=(val_y*self.train_y_std)+self.train_y_avg #
        
        feats=parent_mod.var_names
        
        # Convert data to tensorflow
        train_X=tf.convert_to_tensor(train_X)
        train_y=tf.convert_to_tensor(train_y)
        val_X=tf.convert_to_tensor(val_X)
        val_y=tf.convert_to_tensor(val_y)
        
        # Fine-tune the source domain models on the target
        pred_lt=list()
        for i in range(self.bootstrap_num):
                        
            # Extract the model
            mod=parent_mod.__dict__['model_'+str(i+1)]
            
            # Extract the features used
            sub_feats=mod.feats
            feats_loc=np.array([int(np.where(sub_feats[i]==feats)[0]) for i in range(len(sub_feats)) if sub_feats[i] in feats])
            
            # Reduce the input space down to the selected features
            temp_train_X=tf.gather(train_X,feats_loc,axis=1)
            temp_val_X=tf.gather(val_X,feats_loc,axis=1) #
            
            # Tune model on target data
            model=mod.model        
            net=nn.network(lr=parent_mod.lr, num_neurons=parent_mod.num_neurons,
                           input_shape=temp_train_X.shape, epochs=100)
            temp_pred, temp_mod, temp_loss=net.train_model(train_X=temp_train_X, train_y=train_y, val_X=temp_val_X, 
                                                           val_y=val_y, model=model) #
            
            # Find the MAE and collate results into an object
            temp_pred_m1=bg.quarterly_average(df=temp_pred,num_months=1)
            temp_mae=np.mean(np.abs(unnorm_val_y-((temp_pred_m1*self.train_y_std)+self.train_y_avg))) #
            model=single_model(model=temp_mod, feats=feats[feats_loc],
                               rel_mae=temp_mae, preds=temp_pred, sampled_idx=np.arange(0,train_X.shape[1]))
                        
            # Add results to main object
            self.__dict__['model_'+str(i+1)]=model
            pred_lt.append(temp_pred)
            
            
            
        # Collate the test predictions and find the average predictions
        preds=np.concatenate(pred_lt, axis=1)
        avg_pred=np.mean(preds, axis=1)
        avg_pred=np.reshape(avg_pred,(len(avg_pred),1))
        
        # Find the average predictions for the first month of the quarter
        avg_pred_m1=bg.quarterly_average(df=avg_pred,num_months=1)
        avg_pred_m1=(avg_pred_m1*self.train_y_std)+self.train_y_avg
        avg_pred_m1_mae=np.nanmean(np.abs(unnorm_val_y-avg_pred_m1)) #
                
        # Find the IBES estimates for the first month of the quarter
        ibes_est_m1=bg.quarterly_average(df=ibes_est,num_months=1)
        ibes_est_m1_mae=np.nanmean(np.abs(unnorm_val_y-ibes_est_m1)) #
        
        # Find the average predictions for the second month of the quarter
        avg_pred_m2=bg.quarterly_average(df=avg_pred,num_months=2)
        avg_pred_m2=(avg_pred_m2*self.train_y_std)+self.train_y_avg
        avg_pred_m2_mae=np.nanmean(np.abs(unnorm_val_y-avg_pred_m2)) #
                
        # Find the IBES estimates for the second month of the quarter
        ibes_est_m2=bg.quarterly_average(df=ibes_est,num_months=2)
        ibes_est_m2_mae=np.nanmean(np.abs(unnorm_val_y-ibes_est_m2)) #
        
        # Find the average predictions for the third month of the quarter
        avg_pred_m3=bg.quarterly_average(df=avg_pred,num_months=3)
        avg_pred_m3=(avg_pred_m3*self.train_y_std)+self.train_y_avg
        avg_pred_m3_mae=np.nanmean(np.abs(unnorm_val_y-avg_pred_m3)) #
                
        # Find the IBES estimates for the third month of the quarter
        ibes_est_m3=bg.quarterly_average(df=ibes_est,num_months=3)
        ibes_est_m3_mae=np.nanmean(np.abs(unnorm_val_y-ibes_est_m3)) #
        
        # Add average predictions to the main object
        self.avg_pred_m1_val=avg_pred_m1
        self.avg_pred_m2_val=avg_pred_m2
        self.avg_pred_m3_val=avg_pred_m3
        
        # Add IBES estimates to the main object
        self.ibes_est_m1_val=ibes_est_m1
        self.ibes_est_m2_val=ibes_est_m2
        self.ibes_est_m3_val=ibes_est_m3
        
        # Add the prediction MAE values to the main object
        self.avg_pred_m1_val_mae=avg_pred_m1_mae
        self.avg_pred_m2_val_mae=avg_pred_m2_mae
        self.avg_pred_m3_val_mae=avg_pred_m3_mae
        
        # Add the IBES estimates MAE values to the main object
        self.ibes_est_m1_val_mae=ibes_est_m1_mae
        self.ibes_est_m2_val_mae=ibes_est_m2_mae
        self.ibes_est_m3_val_mae=ibes_est_m3_mae
        
        
        
        

    def test_pred(self,val_y,test_X,test_y,ibes_est_test):
        """Function to find the predictions on the out of sample test data"""
           
        unnorm_test_y=(test_y*self.train_y_std)+self.train_y_avg
        
        var_names=self.var_names
        test_preds_m1_lt=list()
        test_preds_m2_lt=list()
        test_preds_m3_lt=list()
        val_pred_lt=list()
        
        # Extract the test data at the end of the first and second month
        test_X_m1=test_X[0,:]
        test_X_m1=test_X_m1.reshape(1,len(test_X_m1))
        test_X_m2=test_X[:2,:]
        
        # Calculate predictions for each model and extract the predictions on the validation data
        for i in range(self.bootstrap_num):
            
            # Extract the features used as well as the model
            var_subset=self.__dict__['model_'+str(i+1)].feats
            var_subset_order=[int(np.where(var_subset[j]==var_names)[0]) for j in range(len(var_subset))]                        
            model=self.__dict__['model_'+str(i+1)].model
            
            # Find the out of sample prediction using just one month of data
            temp_test_pred_m1=model(test_X_m1[:,var_subset_order])
            temp_test_pred_m1=(temp_test_pred_m1*self.train_y_std)+self.train_y_avg
            test_preds_m1_lt.append(temp_test_pred_m1)
            
            # Find the out of sample prediction using just two months of data
            temp_test_pred_m2=model(test_X_m2[:,var_subset_order])
            temp_test_pred_m2=(temp_test_pred_m2*self.train_y_std)+self.train_y_avg
            test_preds_m2_lt.append(temp_test_pred_m2)
            
            # Find the out of sample prediction using just three months of data
            temp_test_pred_m3=model(test_X[:,var_subset_order])
            temp_test_pred_m3=(temp_test_pred_m3*self.train_y_std)+self.train_y_avg
            test_preds_m3_lt.append(temp_test_pred_m3)
            
            # Extract the in-sample test predictions and their respective mae
            val_pred_lt.append((self.__dict__['model_'+str(i+1)].preds*self.train_y_std)+self.train_y_avg)
        
        # Collate the test predictions
        test_preds_m1=np.concatenate(test_preds_m1_lt, axis=1)
        test_preds_m2=np.concatenate(test_preds_m2_lt, axis=1)
        test_preds_m3=np.concatenate(test_preds_m3_lt, axis=1)
        
        
        # Unnormalise the validation target data
        unnorm_val_y=(val_y*self.train_y_std)+self.train_y_avg
           
        # Find the quarterly average of the validation predictions
        val_preds=np.concatenate(val_pred_lt, axis=1)
        val_preds_m1=bg.quarterly_average(df=val_preds,num_months=1)
        val_preds_m2=bg.quarterly_average(df=val_preds,num_months=2)
        val_preds_m3=bg.quarterly_average(df=val_preds,num_months=3)
        
        
        
        ### Ensemble predictions with forward stepwise selection
        
        # Find the ensemble prediction at the end of the first month of the quarter
        start_num=3
        if np.sum(np.sum(np.isnan(val_preds_m1),axis=0)==0) >= start_num:
            
            
            # Carry out forward stepwise selection of the ensemble model
            combo_loc_m1=fss.forward_stepwise_selection(val_preds=val_preds_m1,unnorm_val_y=unnorm_val_y,
                                                    subset_size=200,start_num=5,ensemble_num=20,p=0.7)  
            ensemble_val_m1=fss.forward_stepwise_selection_ens(preds=val_preds_m1, combo_loc=combo_loc_m1)
            ensemble_val_m1_mae=np.mean(np.abs(unnorm_val_y-ensemble_val_m1))
        
            # Find the ensemble prediction on the test data and its MAE
            ensemble_test_m1=fss.forward_stepwise_selection_ens(preds=test_preds_m1, combo_loc=combo_loc_m1)
            ensemble_test_m1_mae=np.mean(np.abs(unnorm_test_y[0,:]-ensemble_test_m1))
            
        # If we can't produce any predictions, set predictions to nan
        if np.sum(np.sum(np.isnan(val_preds_m1),axis=0)==0) < start_num:
            ensemble_test_m1=np.empty((len(test_preds_m1),1))
            ensemble_test_m1[:]=np.nan
            
            ensemble_val_m1=np.empty((len(val_preds_m1),1))
            ensemble_val_m1[:]=np.nan
            
            ensemble_test_m1_mae=np.nan
            ensemble_val_m1_mae=np.nan
        
        
        
        # Find the ensemble prediction at the end of the second month of the quarter
        if np.sum(np.sum(np.isnan(val_preds_m2),axis=0)==0) >= start_num:
            
            
            # Carry out forward stepwise selection of the ensemble model
            combo_loc_m2=fss.forward_stepwise_selection(val_preds=val_preds_m2,unnorm_val_y=unnorm_val_y,
                                                    subset_size=200,start_num=5,ensemble_num=20,p=0.7)
            ensemble_val_m2=fss.forward_stepwise_selection_ens(preds=val_preds_m2, combo_loc=combo_loc_m2)
            ensemble_val_m2_mae=np.mean(np.abs(unnorm_val_y-ensemble_val_m2))
        
            # Find the ensemble prediction on the test data and its MAE
            ensemble_test_m2=fss.forward_stepwise_selection_ens(preds=test_preds_m2, combo_loc=combo_loc_m2)
            ensemble_test_m2_mae=np.mean(np.abs(unnorm_test_y[0,:]-ensemble_test_m2))
            
            
        # If we can't produce any predictions, set predictions to nan
        if np.sum(np.sum(np.isnan(val_preds_m2),axis=0)==0) < start_num:
            ensemble_test_m2=np.empty((len(test_preds_m2),1))
            ensemble_test_m2[:]=np.nan
            
            ensemble_val_m2=np.empty((len(val_preds_m2),1))
            ensemble_val_m2[:]=np.nan
            
            ensemble_test_m2_mae=np.nan
            ensemble_val_m2_mae=np.nan
        
        
        
        # Find the ensemble prediction at the end of the third month of the quarter
        if np.sum(np.sum(np.isnan(val_preds_m3),axis=0)==0) >= start_num:
            
            
            # Carry out forward stepwise selection of the ensemble model
            combo_loc_m3=fss.forward_stepwise_selection(val_preds=val_preds_m3,unnorm_val_y=unnorm_val_y,
                                                    subset_size=200,start_num=5,ensemble_num=20,p=0.7) 
            ensemble_val_m3=fss.forward_stepwise_selection_ens(preds=val_preds_m3, combo_loc=combo_loc_m3)
            ensemble_val_m3_mae=np.mean(np.abs(unnorm_val_y-ensemble_val_m3))
        
            # Find the ensemble prediction on the test data and its MAE
            ensemble_test_m3=fss.forward_stepwise_selection_ens(preds=test_preds_m3, combo_loc=combo_loc_m3)
            ensemble_test_m3_mae=np.mean(np.abs(unnorm_test_y[0,:]-ensemble_test_m3))
            
                        
        # If we can't produce any predictions, set predictions to nan
        if np.sum(np.sum(np.isnan(val_preds_m3),axis=0)==0) < start_num:
            ensemble_test_m3=np.empty((len(test_preds_m3),1))
            ensemble_test_m3[:]=np.nan
            
            ensemble_val_m3=np.empty((len(val_preds_m3),1))
            ensemble_val_m3[:]=np.nan
            
            ensemble_test_m3_mae=np.nan
            ensemble_val_m3_mae=np.nan
        
        
        
        # Collate test ensemble predictions and MAE
        ensemble_test=np.append(ensemble_test_m1,ensemble_test_m2[1])
        ensemble_test=np.append(ensemble_test,ensemble_test_m3[2])
        
        ensemble_test_mae=np.append(ensemble_test_m1_mae,ensemble_test_m2_mae)
        ensemble_test_mae=np.append(ensemble_test_mae,ensemble_test_m3_mae)
        
        # Collate validation ensemble MAE
        ensemble_val_mae=np.append(ensemble_val_m1_mae,ensemble_val_m2_mae)
        ensemble_val_mae=np.append(ensemble_val_mae,ensemble_val_m3_mae)
        
        # Add ensemble predictions and MAE for test and validation data to the main object
        self.ensemble_val_m1=ensemble_val_m1
        self.ensemble_val_m2=ensemble_val_m2
        self.ensemble_val_m3=ensemble_val_m3
        
        self.ensemble_test=ensemble_test
        self.ensemble_val_mae=ensemble_val_mae
        self.ensemble_test_mae=ensemble_test_mae
        
        
        
        
        ### Ensemble predictions with single best model
        
        # Find the validation MAE
        val_preds_m1_mae=np.mean(np.abs(val_preds_m1-unnorm_val_y),axis=0)
        val_preds_m2_mae=np.mean(np.abs(val_preds_m2-unnorm_val_y),axis=0)
        val_preds_m3_mae=np.mean(np.abs(val_preds_m3-unnorm_val_y),axis=0)
        
        # Find the ensemble predictions and MAE at the end of the first month using the single best model
        if np.sum(np.isnan(val_preds_m1_mae))<len(val_preds_m1_mae):
                    
            best_1_test_m1=test_preds_m1[:,np.nanargmin(val_preds_m1_mae)]
            best_1_test_m1_mae=np.nanmean(np.abs(unnorm_test_y[0,:]-best_1_test_m1))
            best_1_val_m1_mae=np.min(val_preds_m1_mae)
        
        # If no predictions, set values to nan
        if np.sum(np.isnan(val_preds_m1_mae))==len(val_preds_m1_mae):
            
            best_1_test_m1=np.empty((len(test_preds_m1),))
            best_1_test_m1[:]=np.nan
            best_1_test_m1_mae=np.nan
            best_1_val_m1_mae=np.nan
        
        
        # Find the ensemble predictions and MAE at the end of the second month using the single best model
        if np.sum(np.isnan(val_preds_m2_mae))<len(val_preds_m2_mae):
                    
            best_1_test_m2=test_preds_m2[:,np.nanargmin(val_preds_m2_mae)]
            best_1_test_m2_mae=np.nanmean(np.abs(unnorm_test_y[1,:]-best_1_test_m2[1]))
            best_1_val_m2_mae=np.min(val_preds_m2_mae)
        
        # If no predictions, set values to nan
        if np.sum(np.isnan(val_preds_m2_mae))==len(val_preds_m2_mae):
            
            best_1_test_m2=np.empty((len(test_preds_m2),))
            best_1_test_m2[:]=np.nan
            best_1_test_m2_mae=np.nan
            best_1_val_m2_mae=np.nan
            
            
         # Find the ensemble predictions and MAE at the end of the third month using the single best model   
        if np.sum(np.isnan(val_preds_m3_mae))<len(val_preds_m3_mae):
                        
            best_1_test_m3=test_preds_m3[:,np.nanargmin(val_preds_m3_mae)]
            best_1_test_m3_mae=np.nanmean(np.abs(unnorm_test_y[2,:]-best_1_test_m3[2]))
            best_1_val_m3_mae=np.min(val_preds_m3_mae)
            
        # If no predictions, set values to nan
        if np.sum(np.isnan(val_preds_m3_mae))==len(val_preds_m3_mae):
                
            best_1_test_m3=np.empty((len(test_preds_m3),))
            best_1_test_m3[:]=np.nan
            best_1_test_m3_mae=np.nan
            best_1_val_m3_mae=np.nan
        
        # Collate the test predictions and MAE
        best_1_test=np.append(best_1_test_m1,best_1_test_m2[1])
        best_1_test=np.append(best_1_test,best_1_test_m3[2])
        
        best_1_test_mae=np.append(best_1_test_m1_mae,best_1_test_m2_mae)
        best_1_test_mae=np.append(best_1_test_mae,best_1_test_m3_mae)
        
        # Collate the validation MAE
        best_1_val_mae=np.append(best_1_val_m1_mae,best_1_val_m2_mae)
        best_1_val_mae=np.append(best_1_val_mae,best_1_val_m3_mae)
        
        # Add single best model results to main object
        self.best_1_test=best_1_test
        self.best_1_test_mae=best_1_test_mae
        self.best_1_val_mae=best_1_val_mae
        





def rolling_average(df,n):
    """Function to calculate the rolling average over a numpy array
    
    Parameters
    - df: A numpy array
    - n: A single numeric value giving the size of the moving window
    
    Return
    - cummul: A numpy array containing the rolling average"""
    
    cummul=np.cumsum(df,axis=0,dtype=float)
    cummul[n:,:]=cummul[n:,:]-cummul[:-n,:]
    cummul=cummul[n-1:,:]/n
    cummul=cummul[np.arange(0,len(cummul),n)]
    
    return cummul





    
def find_lev_y(target_y,stock_act_lev,stock_change,test_dates,wanted_dates,
               train_y_std,train_y_avg, unnorm=True):
    """Function to reverse either differencing or percentage change to find
    the level values.
    
    Parameters
    - target_y: A numpy array containing the data we want to find the level values for
    - stock_act_lev:  A numpy array containing the level values for the equivalent data
    - stock_change: A single character value defining the type of change applied.
                    This can be either ACH for 12 month difference or PCH for 12
                    month percentage change.
    - val_dates: A numpy array containing the test dates
    - wanted_dates: A numpy array containing the wanted dates
    - train_y_std: A single numeric value giving the standard deviation for the 
                   series in question
    - train_y_avg: A single numeric value giving the average for the series in
                   question
    - unnorm: A boolean parameter that is True if we need to unnormalise the
              series in question and False otherwise. The default is True.
        
    Return
    - target_y: A numpy array containing the level data"""
    
    # Unnormalise data if required
    if unnorm is True: target_y=(target_y*train_y_std)+train_y_avg
    
    # Find the previous change dates
    prev_change_dates=np.where(np.in1d(wanted_dates,test_dates))[0]-12
    
    # Reverse the detrending transformation applied
    if stock_change=='ach': target_y=target_y+stock_act_lev[prev_change_dates]
    if stock_change=='pch': 
        target_y=((target_y/100)*np.abs(stock_act_lev[prev_change_dates]))+stock_act_lev[prev_change_dates]
    
    return target_y
    
    





    

def build_and_pred(stock_act,stock_act_lev,full_prc_pch,full_prc_ach,full_est_pch,
                   full_est_ach,mac,mac_vars,stock_isin,full_isins,wanted_dates,
                   stock_change, bootstrap_num=20,subset_feat=0.5,
                   keep_est=True):
    """For a given target, this function will carry out the feature screening,
    standardise the data, build the direct RE models and make the predictions.
    
    Parameters
    - stock_act: A numpy array containing the reported values for the series in 
                 question (de-trended)
    - stock_act_lev: A numpy array containing the reported values for the series
                     in question (level values)
    - full_prc_pch: A numpy array containing the stock price data (percentage change)
    - full_prc_ach: A numpy array containing the stock price data (difference)
    - full_est_pch: A numpy array containing the IBES estimated data (percentage change)
    - full_est_ach: A numpy array containing the IBES estimated data (difference)
    - mac: A numpy array containing the macroeconomic data
    - mac_vars: A numpy vector array containing the column names of the macro data
    - stock_isin: A single character value giving the stock ISIN for the series in question
    - full_isins: A numpy vector array containing the stock ISINs for all stocks in
                  the current stock universe
    - wanted_dates: A numpy vector array containing the dates of interest
    - stock_change: A single character value giving the type of change applied to
                    the reported values. Can be either 'ACH' or 'PCH'
    - bootstrap_num: A single numeric value giving the number of bootstrap samples
                     to form.
    - subset_feat: A single numeric value between [0,1] giving the fraction of
                   features that we are subsetting. The default is 0.5
    - keep_est: A boolean parameter that is True if we are keeping the estimated
                values for the given series when selecting features and False
                otherwise. The default is True.
    
    
    Return
    - var_corr: A numpy vector array containing the correlation values for the
                screened features
    - mae_res: A dictionary containing the prediction results
    - stock_ens_mod: An object containing the ensemble ANN models"""

    # Screen features, standardise data, and split data between train-valid-test
    data_sep,dates_sep,norm_meas,var_names,num_lags,var_corr=prep.stock_data_prep(stock_act=stock_act,
                                                                                  full_prc_pch=full_prc_pch,
                                                                                  full_prc_ach=full_prc_ach,
                                                                                  full_est_pch=full_est_pch,
                                                                                  full_est_ach=full_est_ach,
                                                                                  mac=mac,
                                                                                  mac_vars=mac_vars,
                                                                                  stock_isin=stock_isin,
                                                                                  full_isins=full_isins,
                                                                                  wanted_dates=wanted_dates,
                                                                                  keep_est=keep_est)

    # If no features hold a significant relationship with the target, return -1
    if data_sep ==-1:
        mae_res={'avg_pred_test':-1,
                 'ibes_test':-1,
                 'avg_pred_val':-1,
                 'ibes_val':-1,
                 'top_10_val':-1,
                 'top_20_val':-1,
                 'top_30_val':-1,
                 'top_40_val':-1,
                 'top_50_val':-1}
        return -1, mae_res, -1



    # Expand the data
    train_X=data_sep['train_X']
    train_y=data_sep['train_y']
    
    test_X=data_sep['test_X']
    test_y=data_sep['test_y']
    
    val_X=data_sep['val_X']
    val_y=data_sep['val_y']

    train_X_avg=norm_meas['train_X_avg']
    train_X_std=norm_meas['train_X_std']
    train_y_avg=norm_meas['train_y_avg']
    train_y_std=norm_meas['train_y_std']

    train_dates=dates_sep['train_dates']
    test_dates=dates_sep['test_dates']
    val_dates=dates_sep['val_dates']




    # Extract the IBES estimates
    if stock_change=='pch': ibes_est=full_est_pch[:,np.where(full_isins==stock_isin)[0]]
    if stock_change=='ach': ibes_est=full_est_ach[:,np.where(full_isins==stock_isin)[0]]

    # Find the validation and testing IBES estimates
    ibes_est_test=ibes_est[np.where(np.in1d(wanted_dates,test_dates))[0]]
    ibes_est_val=ibes_est[np.where(np.in1d(wanted_dates,val_dates))[0]]
    
    # Construct the persistent baseline series
    baseline_val=np.roll(stock_act,3,axis=0)[np.where(np.in1d(wanted_dates,val_dates))[0]]
    baseline_val_mae=np.nanmean(np.abs(baseline_val-stock_act[np.where(np.in1d(wanted_dates,val_dates))]))
    baseline_test_lev=np.roll(stock_act_lev,3,axis=0)[np.where(np.in1d(wanted_dates,test_dates))[0]]
    
    
    # Build the ensemble model
    stock_ens_mod=ensemble_model(stock_isin=stock_isin, stock_change=stock_change, train_X_avg=train_X_avg, 
                                 train_X_std=train_X_std, train_y_avg=train_y_avg, train_y_std=train_y_std, 
                                 train_dates=train_dates, test_dates=test_dates, val_dates=val_dates,
                                 var_names=var_names, num_lags=num_lags, bootstrap_num=bootstrap_num,
                                 subset_feat=subset_feat)
    stock_ens_mod.build_model(train_X, train_y, val_X, val_y, ibes_est=ibes_est_val, 
                              opt_lr=0.005, opt_num_neurons=None, vary_neurons=False)
    
    # Use ensemble models to predict the out of sample test data
    stock_ens_mod.test_pred(val_y, test_X, test_y, ibes_est_test=ibes_est_test)
    
    
    
    
    # Find the level values for the reported values
    test_y_lev=stock_act_lev[np.where(np.in1d(wanted_dates,test_dates))]
    
    # Find the level values for the test IBES estimates
    ibes_est_test_lev=find_lev_y(target_y=ibes_est_test,stock_act_lev=stock_act_lev,stock_change=stock_change,
                                test_dates=test_dates,wanted_dates=wanted_dates,train_y_std=train_y_std,
                                train_y_avg=train_y_avg, unnorm=False)
    
    # Find the level values for the test single best model
    best_1_pred_lev=find_lev_y(target_y=stock_ens_mod.best_1_test.reshape(len(stock_ens_mod.best_1_test),1),
                               stock_act_lev=stock_act_lev,stock_change=stock_change,
                               test_dates=test_dates,wanted_dates=wanted_dates,train_y_std=train_y_std,
                               train_y_avg=train_y_avg, unnorm=False)
        
    # Find the level valeus for the test RE method predictions
    ensemble_pred_lev=find_lev_y(target_y=stock_ens_mod.ensemble_test.reshape(len(stock_ens_mod.ensemble_test),1),
                                 stock_act_lev=stock_act_lev,stock_change=stock_change,
                                 test_dates=test_dates,wanted_dates=wanted_dates,train_y_std=train_y_std,
                                 train_y_avg=train_y_avg,unnorm=False)
        
    
    
    
    # Collate the results into a dictionary 
    mae_res={'ibes_val_mae':np.array([stock_ens_mod.ibes_est_m1_val_mae,
                                       stock_ens_mod.ibes_est_m2_val_mae,
                                       stock_ens_mod.ibes_est_m3_val_mae]),
             'best_1_val_mae':stock_ens_mod.best_1_val_mae,
             'ensemble_pred_val_mae':stock_ens_mod.ensemble_val_mae,
             'baseline_pred_val_mae':baseline_val_mae,
             
             'act_test_lev':test_y_lev,
             'ibes_test_lev':ibes_est_test_lev,
             'best_1_test_lev':best_1_pred_lev,
             'ensemble_test_lev':ensemble_pred_lev,
             'baseline_test_lev':baseline_test_lev}
    
    return var_corr, mae_res, stock_ens_mod








   



def transfer_learn(stock_ens_mod, child_act, child_act_lev, child_change, child_isin,
                   full_prc_pch, full_prc_ach, full_est_pch, full_est_ach, mac, mac_vars, 
                   full_isins, wanted_dates, keep_est=True):
    """For a given target, this function will apply the multi-source transfer learning
    technique to build the predictions.
    
    Parameters
    - stock_ens_mod: An object containing the ensemble ANN models of the source domain
    - child_act: A numpy array containing the reported values for the target series in 
                 question (de-trended)
    - child_act_lev: A numpy array containing the reported values for the target series
                     in question (level values)
    - full_prc_pch: A numpy array containing the stock price data (percentage change)
    - full_prc_ach: A numpy array containing the stock price data (difference)
    - full_est_pch: A numpy array containing the IBES estimated data (percentage change)
    - full_est_ach: A numpy array containing the IBES estimated data (difference)
    - mac: A numpy array containing the macroeconomic data
    - mac_vars: A numpy vector array containing the column names of the macro data
    - child_isin: A single character value giving the stock ISIN for the target 
                  series in question
    - full_isins: A numpy vector array containing the stock ISINs for all stocks in
                  the current stock universe
    - wanted_dates: A numpy vector array containing the dates of interest
    - child_change: A single character value giving the type of change applied to
                    the target reported values. Can be either 'ACH' or 'PCH'
    - keep_est: A boolean parameter that is True if we are keeping the estimated
                values for the given series when selecting features and False
                otherwise. The default is True.
    
    
    Return
    - var_corr: A numpy vector array containing the correlation values for the
                screened features
    - mae_res: A dictionary containing the prediction results
    - stock_ens_mod: An object containing the ensemble ANN models"""
    
    if stock_ens_mod==-1:
        return {'ibes_test_mae':-1,
                'best_1_test_mae':-1,
                'ensemble_pred_test_mae':-1,
                'baseline_pred_test_mae':-1,
                     
                'act_val_lev':-1,
                'ibes_val_lev':-1,
                'best_1_val_lev':-1,
                'ensemble_val_lev':-1,
                'baseline_val_lev':-1}
    
    # Find features used by source, standardise data, and split data between train-valid-test
    data_sep,dates_sep,norm_meas,var_names=prep.stock_data_prep_trans(stock_act=child_act,
                                                                     full_prc_pch=full_prc_pch,
                                                                     full_prc_ach=full_prc_ach,
                                                                     full_est_pch=full_est_pch,
                                                                     full_est_ach=full_est_ach,
                                                                     mac=mac,
                                                                     mac_vars=mac_vars,
                                                                     stock_isin=child_isin,
                                                                     full_isins=full_isins,
                                                                     wanted_dates=wanted_dates,
                                                                     keep_est=keep_est,
                                                                     wanted_feats=stock_ens_mod.var_names,
                                                                     wanted_lags=stock_ens_mod.num_lags)
    
    
    
    
    # Expand the data
    train_X=data_sep['train_X']
    train_y=data_sep['train_y']
    
    test_X=data_sep['test_X']
    test_y=data_sep['test_y']
    
    val_X=data_sep['val_X']
    val_y=data_sep['val_y']

    train_X_avg=norm_meas['train_X_avg']
    train_X_std=norm_meas['train_X_std']
    train_y_avg=norm_meas['train_y_avg']
    train_y_std=norm_meas['train_y_std']

    train_dates=dates_sep['train_dates']
    test_dates=dates_sep['test_dates']
    val_dates=dates_sep['val_dates']





    # Extract the IBES estimates
    if child_change=='pch': ibes_est=full_est_pch[:,np.where(full_isins==child_isin)[0]]
    if child_change=='ach': ibes_est=full_est_ach[:,np.where(full_isins==child_isin)[0]]

    # Find the validation and testing IBES estimates
    ibes_est_test=ibes_est[np.where(np.in1d(wanted_dates,test_dates))[0]]
    ibes_est_val=ibes_est[np.where(np.in1d(wanted_dates,val_dates))[0]]
    
    # Construct the persistent baseline series
    baseline_val=np.roll(child_act,3,axis=0)[np.where(np.in1d(wanted_dates,val_dates))[0]] #
    baseline_val_mae=np.nanmean(np.abs(baseline_val-child_act[np.where(np.in1d(wanted_dates,val_dates))])) #
    baseline_test_lev=np.roll(child_act_lev,3,axis=0)[np.where(np.in1d(wanted_dates,test_dates))[0]] #

    
    
    
    
    # Build the transfer learning model
    child_mod=ensemble_model(stock_isin=child_isin, stock_change=child_change, train_X_avg=train_X_avg, 
                                 train_X_std=train_X_std, train_y_avg=train_y_avg, train_y_std=train_y_std, 
                                 train_dates=train_dates, test_dates=test_dates, val_dates=val_dates,
                                 var_names=var_names, num_lags=stock_ens_mod.num_lags, 
                                 bootstrap_num=stock_ens_mod.bootstrap_num, subset_feat=stock_ens_mod.subset_feat)
    child_mod.build_trans_model(train_X, train_y, val_X, val_y, ibes_est=ibes_est_val, 
                              parent_mod=stock_ens_mod)
    
    # Use ensemble models to predict the out of sample test data
    child_mod.test_pred(val_y, test_X, test_y, ibes_est_test=ibes_est_test)
    
    
    
    
    # Find the level values for the target reported values
    test_y_lev=child_act_lev[np.where(np.in1d(wanted_dates,test_dates))]
    
    # Find the level values for the test IBES estimates
    ibes_est_test_lev=find_lev_y(target_y=ibes_est_test,stock_act_lev=child_act_lev,stock_change=child_change,
                                test_dates=test_dates,wanted_dates=wanted_dates,train_y_std=train_y_std,
                                train_y_avg=train_y_avg, unnorm=False)
    
    # Find the level values for the test single best model
    best_1_pred_lev=find_lev_y(target_y=child_mod.best_1_test,stock_act_lev=child_act_lev,stock_change=child_change,
                               test_dates=test_dates,wanted_dates=wanted_dates,train_y_std=train_y_std,
                               train_y_avg=train_y_avg, unnorm=False)
        
    # Find the level valeus for the test RE method predictions
    ensemble_pred_lev=find_lev_y(target_y=child_mod.ensemble_test,stock_act_lev=child_act_lev,stock_change=child_change,
                                 test_dates=test_dates,wanted_dates=wanted_dates,train_y_std=train_y_std,
                                 train_y_avg=train_y_avg, unnorm=False)
   
    
    
    
    # Collate the results into a dictionary 
    mae_res={'ibes_val_mae':np.array([child_mod.ibes_est_m1_val_mae,
                                       child_mod.ibes_est_m2_val_mae,
                                       child_mod.ibes_est_m3_val_mae]),
             'best_1_val_mae':child_mod.best_1_val_mae,
             'ensemble_pred_val_mae':child_mod.ensemble_val_mae,
             'baseline_pred_val_mae':baseline_val_mae,
             
             'act_test_lev':test_y_lev,
             'ibes_test_lev':ibes_est_test_lev,
             'best_1_test_lev':best_1_pred_lev,
             'ensemble_test_lev':ensemble_pred_lev,
             'baseline_test_lev':baseline_test_lev}
    
    return var_names, mae_res

    





def build_individual_stock(stock_isin, act, act_lev, full_est_pch, full_est_ach, full_prc_pch, full_prc_ach, mac,
                           mac_vars, wanted_change, full_isins, village_isins_np, children_isins, wanted_dates, 
                           bootstrap_num=100, subset_feat=0.5, keep_est=True, train_size=0.8):
    
    """Function for building predictions for an individual target series. The
    predictions are built using the RE method. If the individual series is also
    a source domain series for a target for transfer learning, then the model is
    built before being fine-tuned on the target series.
    
    Parameters
    - stock_isin: A single character value giving the stock ISIN of the target
    - act: A numpy array containing the reported values (de-trended)
    - act_lev: A numpy array containing the reported values (level values)
    - full_prc_pch: A numpy array containing the stock price data (percentage change)
    - full_prc_ach: A numpy array containing the stock price data (difference)
    - full_est_pch: A numpy array containing the IBES estimated data (percentage change)
    - full_est_ach: A numpy array containing the IBES estimated data (difference)
    - mac: A numpy array containing the macroeconomic data
    - mac_vars: A numpy vector array containing the column names of the macro data
    - wanted_change: A numpy vector array containing the changes applied to each
                     target series
    - full_isins: A numpy vector array containing the stock ISINs for all stocks in
                  the current stock universe
    - village_isins_np: A numpy array containing the source domains for each target
                        to be modelled using transfer learning. The data are
                        character values giving the ISINs of the source domain series.
                        The columns correspond to the source domains for each target (row)
    - children_isins: A numpy vector array containing the ISINs of the transfer 
                      learning target series
    - wanted_dates: A numpy vector array containing the dates of interest
    - bootstrap_num: A single numeric value giving the number of bootstrap samples
                     to form.
    - subset_feat: A single numeric value between [0,1] giving the fraction of
                   features that we are subsetting. The default is 0.5
    - keep_est: A boolean parameter that is True if we are keeping the estimated
                values for the given series when selecting features and False
                otherwise. The default is True.
    - train_size: A single numeric value between [0,1] giving the fraction that
                  we are to separate the training and validation data. The default
                  is 0.8."""
    
    # Find the location of the series in question and build the RE method prediction
    i=np.where(full_isins==stock_isin)[0][0]
    var_corr, mae_res, stock_ens_mod=build_and_pred(stock_act=act[:,[i]],
                                                    stock_act_lev=act_lev[:,[i]],
                                                    full_est_pch=full_est_pch,
                                                    full_est_ach=full_est_ach,
                                                    full_prc_pch=full_prc_pch,
                                                    full_prc_ach=full_prc_ach,
                                                    mac=mac,
                                                    mac_vars=mac_vars,
                                                    stock_change=wanted_change[i],
                                                    stock_isin=stock_isin,
                                                    full_isins=full_isins,
                                                    wanted_dates=wanted_dates,
                                                    bootstrap_num=bootstrap_num,
                                                    subset_feat=subset_feat,
                                                    keep_est=keep_est)
    print(i, flush=True)
    results={stock_isin: mae_res}    
        
    # If the target series is used as a source domain series for any transfer learning
    # target series, build model and fine tune on TL target for each TL target that
    # the series in questions acts as a source domain
    if (stock_isin in village_isins_np) and (stock_ens_mod != -1):
                
        clust_members=np.where(village_isins_np==stock_isin)[0]
        for j in clust_members:
            temp_child_isin=children_isins[j]
            temp_child_isin_loc=np.where(full_isins==temp_child_isin)[0][0]
                
            child_var_names, child_mae_res=transfer_learn(stock_ens_mod=stock_ens_mod,
                                                          child_act=act[:,[temp_child_isin_loc]],
                                                          child_act_lev=act_lev[:,[temp_child_isin_loc]],
                                                          child_change=wanted_change[temp_child_isin_loc],
                                                          child_isin=temp_child_isin,
                                                          full_prc_pch=full_prc_pch,
                                                          full_prc_ach=full_prc_ach,
                                                          full_est_pch=full_est_pch,
                                                          full_est_ach=full_est_ach,
                                                          mac=mac,
                                                          mac_vars=mac_vars,
                                                          full_isins=full_isins,
                                                          wanted_dates=wanted_dates,
                                                          keep_est=keep_est)
            results['child_'+temp_child_isin]=child_mae_res
                
    return results
    


