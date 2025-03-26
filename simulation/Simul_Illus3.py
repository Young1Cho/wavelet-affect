# %% [markdown]
# ## Illustrative Code for "Interpreting Feature Importance in Wavelet-Based Deep Learning for Multi-Time Scale Affect Forecasting"
# 
# This code produces the plots and estimation results for illustration III reported in:
# 
# Chow, Cho, Xiong, Li, Shen, Das, Ji, & Kumara (2025, submitted). Interpreting Feature Importance in Wavelet-Based Deep Learning for Multi-Time Scale Affect Forecasting. Proceedings of the International Meeting of the Psychometric Society.
# 
# In this illustration, we generated time series data for 15 hypothetical participants contaminated with Gaussian noise, as dependent on three (features 4-6) out of 6 possible features that comprised structured sinusoidal signals during specific time spans. We tested the proposed procedures of splitting of the 15 participants into a training set and a test set, and optimization of the hyperparameters through Hyperopt over 15 trials with 30 epochs each. Following the estimation procedures, we plotted the weighted activation map to depict the importance of the 6 features, 3 of which were spurious.
# 
# 
# First we load some packages and define some functions that we are going to use.

# %%
import os
import sys
import gc
import random
#import pandas as pd

import matplotlib.pyplot as plt
import importlib
import seaborn as sns
import numpy as np
import shap

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK,space_eval
from hyperopt.pyll.base import scope

#from sklearn.model_selection import GroupKFold
#from sklearn.model_selection import KFold
#from sklearn.model_selection import train_test_split


from sklearn.model_selection import train_test_split
from tensorflow.keras.mixed_precision import set_global_policy

import myTVFunctions
importlib.reload(myTVFunctions)

from myTVFunctions import *


# Verify TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
print("Python version:", sys.version)
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())
tf.debugging.set_log_device_placement(False)

print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
print(tf.sysconfig.get_build_info())
tf.keras.backend.clear_session()
gc.collect()

fig_path = "C:/Users/symii/Dropbox/MachineLearning/Kymatio/ADID_Lab/IMPS_Proceedings2024/Manuscript/ScatterT_AffectForecast_IMPSProc2025/Figures/"

# %%
seednum=12345
set_seeds(seednum)
# Define the hyperparameter search space

activation_options = ['relu', 'elu']
space_scatter = {
    #'J': scope.int(hp.quniform('J', 2, 6, 1)),
    #'Q': scope.int(hp.quniform('Q', 1, 8, 1)),
    'hidden_dim': scope.int(hp.quniform('hidden_dim', 5, 32, 1)),
    'num_layers': scope.int(hp.quniform('num_layers', 3, 5, 1)), #By default includes at least two layers
    'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.7),
    'activation': hp.choice('activation', range(len(activation_options))),
    'lr': hp.loguniform('lr', -3, -1),
    'l2_reg': hp.loguniform('l2_reg', -3, -1)
    #'lr': hp.loguniform('lr', -5, -3),
    #'l2_reg': hp.loguniform('l2_reg', np.log(1e-5), np.log(2e-1))
}

space_scatter_1layer = {
    #'J': scope.int(hp.quniform('J', 2, 6, 1)),
    #'Q': scope.int(hp.quniform('Q', 1, 8, 1)),
    'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.7),
    'activation': hp.choice('activation', range(len(activation_options))),
    'lr': hp.loguniform('lr', -3, -1),
    'l2_reg': hp.loguniform('l2_reg', -3, -1)
    #'lr': hp.loguniform('lr', -5, -3),
    #'l2_reg': hp.loguniform('l2_reg', np.log(1e-5), np.log(2e-1))
}

def monte_carlo_simulation(num_MC, num_samples, num_features, num_T, num_epochs, frequencies, noise_std, J, Q, max_evals, test_ratio=.3,
                           is_Debug = False, use_GPU = False, batch_size=100, epochs=100, activation_options=['relu', 'elu'],
                           K=2, isPlot=False, sample_toPlot=0, feature_indices= [0,1,2], model_spec = "flex"):


    stuff = pd.DataFrame([{'num_samples': num_samples,
            'num_T': num_T,
            'k_folds': K,
            'test_ratio':test_ratio,
            'epochs': epochs,
            'batch_size': batch_size}])
    FlexScatterFile = f"../../Results/TVFreq/Scatter_n{num_samples}_T{num_T}_K{K}_test_ratio_{test_ratio}.csv"
    
    for i in range(0,num_MC):
        print("Good day!, this is run: ", i)
        # [1] Generate data
        run = pd.DataFrame([{'run': i}])
        Xsim, ysim = generateTVFreq_Data(num_samples, num_features, num_T, frequencies, amplitude, noise_std)

        ysim = ysim.reshape(num_samples, 1, num_T)
        print("Xsim shape:", Xsim.shape)  #Xsim shape: (num_samples, num_features, num_T)
        print("ysim shape:", ysim.shape)  #ysim shape: (num_samples, output_dim, num_T)

        LossesbyFold = dict()
        R2byFold = dict()
        bestParams = dict()
        
        if (isPlot):
            
        # Create the time series plot
            num_features_to_plot=len(feature_indices)
            fig, axes = plt.subplots(num_features_to_plot + 1, 1, figsize=(12, 2 * num_features_to_plot), sharex=True)

            for i, feature_idx in enumerate(feature_indices):
                ax = axes[i] if num_features_to_plot > 1 else axes  # Handle single feature case
                ax.plot(range(Xsim.shape[2]), Xsim[sample_toPlot, feature_idx, :], 
                        label=f"Feature {feature_idx+1}", color='b')
                ax.set_ylabel(f"Feature {feature_idx+1}")
                ax.legend(loc="upper right")

            axes[-1].set_xlabel("Time")  # Set x-axis label only on the last subplot
            plt.suptitle(f"Last {num_features_to_plot} Features for Sample {sample_toPlot}")
            
            ax = axes[num_features]
            ax.plot(range(ysim.shape[1]), ysim[sample_toPlot,:].reshape(-1), 
                label="y data", color='b')
                
            plt.tight_layout()
            plt.show()

    
        X_in_Train, X_in_Test, Y_data, Y_data2 = train_test_split(Xsim, ysim, test_size=test_ratio, random_state=seednum)
        print("Shape of X_in_Train: ", X_in_Train.shape)
        print("Shape of Y_data: ", Y_data.shape)
        
        #@@@@
        output_dim = Y_data.shape[1] 
        
        #------  Flexible embedding + scattering transform with CV ------#
        trials_flex_scatter = Trials()
        
        if (model_spec == "flex"):
            best_params_cv = fmin(
                fn=lambda params: objective_Flex_Scatter(params, X_in_Train, Y_data, 
                Y_data.shape[1],batch_size, epochs, K, num_T, J, Q,
                is_Debug = is_Debug, use_GPU=use_GPU, activation_options=activation_options, model_spec = model_spec),
                space=space_scatter,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials_flex_scatter
        )
        elif (model_spec == "1layer"):
            best_params_cv = fmin(
                fn=lambda params: objective_Flex_Scatter(params, X_in_Train, Y_data, 
                Y_data.shape[1],batch_size, epochs, K, num_T, J, Q,
                is_Debug = is_Debug, use_GPU=use_GPU, activation_options=activation_options, model_spec = model_spec),
                space=space_scatter_1layer,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials_flex_scatter
        )
        
        best_params_flex_embed_scatter = best_params_cv
        print("best_params_cv: ", best_params_cv)

        if (model_spec == "flex"):
            #scope.int made the returned num_layers a float, as opposed to int. Converting below.
            best_params_flex_embed_scatter['num_layers'] = int(best_params_cv['num_layers'])
            best_params_flex_embed_scatter['hidden_dim'] = int(best_params_cv['hidden_dim'])

            #Create a scattering model
            best_scatterModel = create_scattering_nn(params = best_params_flex_embed_scatter, 
                           num_features = X_in_Train.shape[1],
                           output_dim = Y_data.shape[1], 
                           num_T = num_T, J = J, Q = Q, is_Debug = is_Debug, use_GPU = use_GPU, activation_options=activation_options, is_Plot=False)
        
        elif (model_spec == "1layer"):
            #Create a scattering model
            best_scatterModel = create_scattering_nn_1layer(params = best_params_flex_embed_scatter, 
                           num_features = X_in_Train.shape[1],
                           output_dim = Y_data.shape[1], 
                           num_T = num_T, J = J, Q = Q, is_Debug = is_Debug, use_GPU = use_GPU, activation_options=activation_options, is_Plot=False)
        

           
        saveModelFile = f"../bestModels/TVFreq/Scatter_n{num_samples}_T{num_T}_K{K}_test_ratio_{test_ratio}run_{i+1}.h5"
        
        results, _, _, _, _, Y_data, Y_data2 = evaluate_model(best_scatterModel, 
                X_in_Train, X_in_Test, Y_data, Y_data2, 
                withID = False, batch_size=batch_size, 
                epochs=epochs, is_Debug=is_Debug)

        
        print("Best flex embed with scattering hyperparameters: ", best_params_cv)
        
        #ADD FOLD-LEVEL LOSSES
        # Extract the fold_losses from the trial with the best result
        LossesbyFold['FlexScatter'] = trials_flex_scatter.trials[trials_flex_scatter.best_trial['tid']]['result']['fold_losses']
        R2byFold['FlexScatter'] = trials_flex_scatter.trials[trials_flex_scatter.best_trial['tid']]['result']['R2byFold']
        
        print("Fold losses from the best trial:", LossesbyFold['FlexScatter'])
        print("Fold R2 from the best trial:", R2byFold['FlexScatter'])
        #ADD BEST HYPERPARAMETERS
        bestParams['FlexScatter'] =  best_params_cv
            
        # Extract the fold_losses from the last trial
        #LossesbyFold['lastLossesFlexEmbedScatter'] = trials_flex_scatterEmbed.trials[-1]['result']['fold_losses']
        #print("Fold losses from the last trial:", LossesbyFold['lastLossesFlexEmbedScatter'])


        results_pd4 = pd.concat([run, stuff, dictoDataFrame(results), 
                                rename_and_convert_array(LossesbyFold['FlexScatter'], 
                                                          output_dim , K, "Losses"), 
                                rename_and_convert_array(R2byFold['FlexScatter'],
                                                          output_dim , K, "R2"),  
                                dictoDataFrame(bestParams['FlexScatter']), pd.DataFrame([{'J': J, 'Q':Q}])], axis=1)
        if (is_Debug==True):
                print(results_pd4.head())
                print("Best flex scattering model config: ",best_params_cv)
        best_scatterModel.save(saveModelFile)
        results_pd4.to_csv(FlexScatterFile, mode='a', header=not os.path.exists(FlexScatterFile), index=False, na_rep="NaN")

    
        
    return LossesbyFold, R2byFold, bestParams, best_params_cv, best_scatterModel, X_in_Train, X_in_Test, Y_data, Y_data2


# %% [markdown]
# Expected outputs from final model to be used/produced in the creation of the activation map
# ### **🔍 Expected Output Shapes**
# | Variable                  | Shape                                   | Description |
# |---------------------------|----------------------------------------|-------------|
# | `scattering_activations`  | `(None, num_features * num_ScatteringCoefficients * num_TimeWindows)` | Raw activations |
# | `reshaped_scattering_activations` | `(None, num_features, num_ScatteringCoefficients, num_TimeWindows)` | Activations reshaped for element-wise multiplication |
# | `importance`              | `(num_features*num_ScatteringCoefficients*num_TimeWindows, output_dim)`                        | First layer weights|
# | `importance`              | `(num_features*num_ScatteringCoefficients*num_TimeWindows, output_dim*num_T)`                        | Backpropagated weights |
# | `Scattering activations (X_MLP)`| `(None, num_features*num_ScatteringCoefficients*num_TimeWindows, output_dim)` | Scattering activations |
# | `Flattened_scattering_activations`         | `(None, num_features*num_ScatteringCoefficients, num_TimeWindows)`  | Scattering activations by frequency band over time windows|
# | ` heat_np`   | `(None, num_features, num_T or num_TimeWindows, output_dim)`               | Feature importance (output-weighted scattering activations) for each individual |
# | ` heatmap_plot_AllSamples1`   | `(num_features, num_T or num_TimeWindows, output_dim)`               | Feature importance averaged across samples |
# 
# ---
# Example:
# * Shape of importance (weight matrix in first layer):  (1152, 1)
# * Replicated importance shape: (None, 1152, 1)
# * Reshaped replicated importance shape:  (None, 6, 3, 64, 1)
# * Shape of scattering_activations:  (None, 1152)
# * Shape of reshaped scattering_activations:  (None, 6, 3, 64)
# * Num_scattering_coefs: 3; num_TimeWindows: 64
# * Shape of weightedActivations:  (None, 6, 64, 1)

# %%
set_global_policy('mixed_float16')  # Reduce memory footprint


# Initialize parameters
gpus = tf.config.list_physical_devices('GPU')
use_GPU = len(gpus) > 0
print('use_GPU:', use_GPU)
#if (use_GPU):
#    for gpu in gpus:
#        tf.config.experimental.set_memory_growth(gpu, False) #disable memory growth
#        tf.config.experimental.set_virtual_device_configuration(gpu,[]) #clear memory allocation
num_MC=1
num_samples =  15
num_features = 6
num_T = 2**9 #
# Frequencies for the sine waves (one for each important feature)
frequencies = [.05, .2, .12] #[0.5, 1.0, 2.0, 3.0, 4.0]
amplitude = 1 #Amplitude of simulated data
noise_std = 1 # Set the standard deviation for the noise
log_eps = 1e-7 #Small positive constant to use as the min in calling the log function
max_evals = 15 #Max number of evaluations for hyperopt optimization
num_epochs = 30 #Number of epochs for estimating training model coefficients
batch_size = 50
k_folds = 5
J = 3
Q = 3
test_ratio = .5
is_Debug = False

Xsim, ysim = generateTVFreq_Data(num_samples, num_features, num_T, frequencies, amplitude, noise_std)

feature_indices = [0,3,4,5] #range(num_features)
sample_toPlot = 0

# Create the time series plot
num_features_to_plot=len(feature_indices)
fig, axes = plt.subplots(num_features_to_plot + 1, 1, figsize=(12, 2 * num_features_to_plot), sharex=True)

for i, feature_idx in enumerate(feature_indices):
    ax = axes[i] if num_features_to_plot > 1 else axes  # Handle single feature case
    ax.plot(range(Xsim.shape[2]), Xsim[sample_toPlot, feature_idx, :], 
        label=f"Feature {feature_idx+1}", color='b')
    ax.set_ylabel(f"Feature {feature_idx+1}")
    #ax.legend(loc="upper right")
    ax.set_ylim(-6.0, 6.0)  # Change values as needed

axes[-1].set_xlabel("Time")  # Set x-axis label only on the last subplot
plt.suptitle(f"Last {num_features_to_plot} Features for Sample {sample_toPlot}")
            
ax = axes[len(feature_indices)]
ax.plot(range(ysim.shape[1]), ysim[sample_toPlot,:].reshape(-1), label="y data", color='b')
ax.set_ylim(-7.5, 9)
ax.set_ylabel("Output data")
                
plt.tight_layout()
plt.savefig(f"{fig_path}Sim3.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()

# %%


# %%
model_spec = "1layer"
LossesbyFold, R2byFold, bestParams, best_params_cv, best_scatterModel, X_in_Train, X_in_Test, Y_data, Y_data2 = monte_carlo_simulation(num_MC, num_samples, num_features, num_T, num_epochs, frequencies, noise_std, J, Q, max_evals, test_ratio,
                       is_Debug = is_Debug, use_GPU = False, batch_size=batch_size, epochs=num_epochs, activation_options=['relu', 'elu'],
                       K=k_folds, isPlot=False, feature_indices= range(0, num_features), model_spec=model_spec)


# %%
#Create the final scattering model
firstLayer = True #Whether to use the first layer of weights only
if model_spec == "flex":
    best_scatterModel, scattering_activations, num_ScatteringCoefficients, num_TimeWindows, heat = create_scattering_nn(params = best_params_cv, 
                           num_features = X_in_Train.shape[1],
                           output_dim = Y_data.shape[1], 
                           num_T = num_T, J = J, Q = Q, is_Debug = is_Debug, use_GPU = use_GPU, activation_options=activation_options, 
                           is_Plot=True, firstLayer = firstLayer)
elif model_spec == "1layer":
    best_scatterModel, scattering_activations, num_ScatteringCoefficients, num_TimeWindows, heat = create_scattering_nn_1layer(params = best_params_cv, 
                           num_features = X_in_Train.shape[1],
                           output_dim = Y_data.shape[1], 
                           num_T = num_T, J = J, Q = Q, is_Debug = is_Debug, use_GPU = use_GPU, activation_options=activation_options, 
                           is_Plot=True, firstLayer = firstLayer)

best_scatterModel.fit(
            X_in_Train, Y_data, 
            batch_size=batch_size, verbose=0,  
            epochs=num_epochs
            )



# %%
print("Scattering nn input:", best_scatterModel.input)

scatteringOutput_model1 = tf.keras.Model(inputs=best_scatterModel.input, outputs=scattering_activations)
#Compute scattering activations using model.predict()
scattering_activations_np = scatteringOutput_model1.predict(X_in_Train)

#output_dim = Y_data.shape[1]

heat_model1 = tf.keras.Model(inputs=best_scatterModel.input, outputs=heat)
#Compute weighted activations using model.predict()
heat_np = heat_model1.predict(X_in_Train)

print("Shape of heat: ", heat_np.shape)

# Compute mean after converting to NumPy
#if (firstLayer == True):
heatmap_plot_AllSamples1 = np.mean(heat_np[:,:,:,0], axis=0)  # Shape: (num_features, num_T or num_TimeWindows)
#elif(firstLayer == False):
#    heatmap_plot_AllSamples1 = np.mean(heat_np[:,:,0,:], axis=0)  # Shape: (num_features, num_T)
print("Shape of heatmap_feature 1: ", heat_np.shape)
print("Shape of heatmap_plot1 across all samples: ", heatmap_plot_AllSamples1.shape)

#Compute scattering activations using model.predict()
scattering_activations = scatteringOutput_model1.predict(X_in_Train)
print("Predicted scattering activations shape:", scattering_activations.shape)

Reshaped_scattering_activations = scattering_activations.reshape(X_in_Train.shape[0], num_features, num_ScatteringCoefficients, num_TimeWindows)
print("Reshaped scattering activations shape:", Reshaped_scattering_activations.shape)

Flattened_scattering_activations = np.mean(scattering_activations.reshape(X_in_Train.shape[0], 
                                           num_features*num_ScatteringCoefficients, num_TimeWindows), axis=0)

Max_scattering_activations = np.mean(np.max(Reshaped_scattering_activations, axis=2), axis=0)


# %%
#Call function in myTVFunctions.py

#This is now identical to Max_Scattering_activaitions since the max operation was already taken inside the scattering model
create_and_save_activationMap(Flattened_scattering_activations, num_features*num_ScatteringCoefficients,
                         colors="Greens", barLabel = "Scattering activations",
                         ylabels= [f"Feature {f+1} band {b+1}" for f in range(num_features) for b in range(3)],
                         title="(A) Scattering activations by frequency band", saveFile = f"{fig_path}Sim3A.pdf")

create_and_save_activationMap(Max_scattering_activations, num_features,
                         ylabels=[f"Feature {i+1}" for i in range(0,num_features)],
                         colors="Greens", barLabel = "Max (scattering activations)",
                         title="(B) Max scattering activations across frequency bands by feature", saveFile = f"{fig_path}Sim3B.pdf")

create_and_save_activationMap(heatmap_plot_AllSamples1, num_features,
                        ylabels=[f"Feature {i+1}" for i in range(0,num_features)],
                         colors="Greens", barLabel = "Weighted scattering activations",
                         title="(C) Sample Feature Importance", saveFile = f"{fig_path}Sim3C.pdf")


