## MSc Thesis

### Requirements
Results were generated using python 3.8.13
with packages specified in `requirements.txt`

It is recommended to use a virtual environment with required packages installed.

To install required packages, to current (virtual) environment, use
`pip install -r requirements.txt`

### Overview

Objective
This package is used to build a predictive system for nowcasting the quarterly
reported company EPS. 

Target data
The target data is the quarterly reported EPS for the stocks of the S&P 500. For 
each target when building predictions, the stocks are first tested for stationarity. 
The 12 month difference is calculated, as well as the 12 month percentage change. 
Both changes are tested for stationarity using the Augmented Dickey-Fuller test. 
If neither changes are stationary, the target is discarded. If only one change is 
stationary, the stationary change is used when modelling. If both changes are 
stationary, the least volatile change is used when modelling.

Features
A high-dimensional feature space consisting of IBES consensus estimates, stock
prices, and various economic and financial indices. Each raw time series is used
to create 4 series by finding the 12 month difference, 12 month percentage change,
12 month difference of the 3 month moving average, and the 12 month percentage
change of the 3 month moving average. The original raw data is removed while the
transformed series are kept as the feature space.

Frequency of data and transformations
All data is converted to monthly series, if not monthly already. When modelling,
the target and feature data is standardised.

Training period and moving window
For each quarter, a twenty year moving window is used to build the models and make 
the predictions. Predictions are also built for each month of the quarter. The
predictions for the first, second and third month of the quarter are defined as the
M1, M2, and M3 predictions respectively. For instance, suppose you wish to nowcast 
the quarterly reported EPS for the stocks of the S&P 500 for Q1 2020. The twenty 
years worth of data preceding January 2020 would be used as the training data while 
the data for January, February, and March 2020 would be used as the testing data. 
The training data would be split with the first 80% used as training data while the 
final 20% would be used as validation. Predictions would then be made at the end of
January (M1), end of February (M2), and end of March (M3).

Feature screening
Given the high dimensionality of the feature space, an efficient feature screening
method is employed whenever modelling a given target. The Spearmans rank correlation
is calculated between the target series and feature. Those with an absolute correlation
of less than 0.4 are removed. For the remaining series, the colinear series are
removed, leaving just the features that are relatively independent among themselves
and show some relationship with the target. 

Modelling the stocks
When making predictions, the stocks are first split between those with a long enough
time series history to be able to model directly and those that are too short to
model directly. Those deemed too short are the ones with only ten years worth of
data. 

RE method
Those with more than ten years worth of data are modelled using a neural network
ensemble modelling technique proposed in this thesis called the Randomised Ensemble
(RE) method. Block bootstrapping is applied to build 500 permutations of the data.
For each permutation, a neural network model is built using a random subset of the
features and a random selection of the hyperparameter values. Once all 500 models
are then built, a modified version of forward stepwise selection is used to combine
the best models into the final ensemble. 

MSETS-TL method
Stocks with less than ten years worth of data are modelled using a time-dynamic,
multi-source transfer learning technique proposed in this thesis called the
Multi-Source Ensemble Time Series Transfer Learning (MSETS-TL) method. For this 
method, the stocks with a full twenty year history are taken as the potential source
domain. For a given target, the Euclidean distance is calculated between the target
and each of the potential source series for the dates in common. The 5 source series
with the smallest distance to the target are then used as the source domain. An
RE method is then built on each source series before being fine-tuned on the target.
Predictions are then made using each model before being averaged. This is then
repeated at each interval that we are making predictions for the given target until
it has a large enough history to be modelled directly.

### Run scripts to produce results

To fully run this package from raw data to final results, the process goes in
a number of stages

### 1. Raw data to refined data

The script data_prep.py takes the raw data saved in data/raw_data, which are in 
either CSV or RDS file format, and converts it into the refined data to be used 
when building the models. This process includes applying the data transformations, 
finding the stationary targets and ensuring the dates are common across dataframes. 
The final results are saved as a series of pandas dataframes that are saved in 
data/refined_data.

### 2. Build RE and MSETS-TL models

The script system.py takes the refined data, standardises it and splits it into
training, validation and testing data. It then uses this data to build the RE and
MSETS-TL models before applying them to the test data. The results of the test
data predictions and validation MAE are then saved as a series of pandas dataframes
in data/results/raw_outputs.

### 3. Summarise the test data predictions

The model predictions are then summarised into a single pandas dataframe for
each quarter and each M1, M2, and M3 predictions. These summary dataframes include 
the predictions, the IBES consensus estimates as well as the actual values. These 
dataframes are built the script collate_results.py and the summary dataframes are 
saved in data/results/ensemble_model or data/results/ensemble_model. 

### 4. Aggregate results for RE and MSETS-TL predictions separately

The prediction results are then evaluated by finding their aggregate hit rate,
error size and number of stocks predicted. This is carried out using agg_results.py
The plots are saved in data/results/plots. 






