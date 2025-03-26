import os
import random
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers, optimizers, mixed_precision
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, BatchNormalization, Flatten, ReLU, ELU, Concatenate, Lambda, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow.keras.regularizers import l2

from kymatio.keras import Scattering1D
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK,space_eval
from sklearn.model_selection import KFold, GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns


tf.get_logger().setLevel('ERROR')  # Suppress INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
# Set the policy to mixed precision
mixed_precision.set_global_policy('mixed_float16')


import tensorflow as tf


def compute_frequency_bounds(J, Q, f_s):
    """
    Computes the corrected central, lower, and upper frequencies of scattering wavelets.
    
    Args:
        J (int): Number of octaves (scales).
        Q (int): Number of wavelets per octave.
        f_s (float): Sampling frequency.

    Returns:
        central_freqs (numpy.ndarray): Central frequencies of wavelets.
        lower_bounds (numpy.ndarray): Lower frequency bounds.
        upper_bounds (numpy.ndarray): Upper frequency bounds.
    """
    central_freqs = np.array([f_s / (2 ** (J - j + q / Q + 1)) for j in range(J) for q in range(Q)])
    
    # Compute lower and upper frequency bounds
    lower_bounds = central_freqs * 2**(-1/(2*Q))
    upper_bounds = central_freqs * 2**(1/(2*Q))

    # ✅ Sort frequencies in descending order
    sort_idx = np.argsort(-central_freqs)
    central_freqs = central_freqs[sort_idx]
    lower_bounds = lower_bounds[sort_idx]
    upper_bounds = upper_bounds[sort_idx]
    
    return central_freqs, lower_bounds, upper_bounds

#Use seaborn heatmap
def create_and_save_activationMap(heatmap, num_features,ylabels,
                         colors="Greens", barLabel = "Feature importance",
                         title="Multivariate summary", saveFile=None):
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(np.log1p(np.abs(heatmap)), cmap=colors, cbar=True, xticklabels=True, yticklabels=False)

# Format the Colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label(barLabel, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # 7Set Axis Labels & Titles
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Features", fontsize=16)
    plt.title(title, fontsize=16)

    # Add Time Window Labels (X-axis)
    num_T = heatmap.shape[-1]
    xtick_positions = np.linspace(0, num_T, min(20, num_T), dtype=int)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_positions, fontsize=12)

    # Set Y-axis Tick Labels with Frequency Info
    yticks = np.arange(.5,num_features+.5)  
    y_labels = ylabels
    ax.set_yticks(ticks=yticks)
    ax.set_yticklabels(y_labels, fontsize=12)

    # Set Background to White
    ax.set_facecolor("white")
    
    if (saveFile != None):
        plt.savefig(saveFile, format="pdf", bbox_inches="tight", dpi=300)
        # Show the heatmap
    plt.show()


def create_activationMapbyFreq(scattering_activations, 
                               num_features, 
                               num_T, central_freqs, lower_bounds, upper_bounds, 
                               batch_idx=-999, colors="Greens", 
                               barLabel="Magnitudes of Activated Scattering Coefficients",
                               title="Scattering Coefficients Heatmap"):
    """
    Plots a heatmap of scattering coefficients over time, grouped by feature and frequency band.

    Args:
        scattering_activations: Tensor of shape (batch_size, num_features * num_scattering_coefficients, num_T)
        num_features: Number of input features
        num_T: Number of time steps
        central_freqs: Central frequencies of wavelets
        lower_bounds: Lower frequency bounds
        upper_bounds: Upper frequency bounds
        batch_idx: Index of the batch to visualize
        colors: Colormap for the heatmap
        barLabel: Label for the colorbar
        title: Title of the heatmap
    """

    # Extract activations for selected batch
    if (batch_idx != -999):
        activations_flattened = scattering_activations[batch_idx].numpy()  # Convert to NumPy for plotting

    # Generate Y-axis labels for features and frequency bands
    y_labels = [
        f"Feature {f+1} - {cf:.2f} Hz\n({lb:.2f} - {ub:.2f} Hz)"
        for f in range(num_features)
        for cf, lb, ub in zip(central_freqs, lower_bounds, upper_bounds)]
    

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(np.log1p(np.abs(activations_flattened)), cmap=colors, cbar=True, xticklabels=True, yticklabels=False)

    # Format the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label(barLabel, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Set Axis Labels & Titles
    plt.xlabel("Time Steps", fontsize=16)
    plt.ylabel("Frequency Bands by Feature", fontsize=16)
    plt.title(title, fontsize=16)

    # Add X-axis Time Step Labels
    xtick_positions = np.linspace(0, num_T-1, min(10, num_T), dtype=int)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_positions, fontsize=12)

    # Set Y-axis Tick Labels with Frequency Info
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)

    # Show the heatmap
    plt.show()


def distribute_time_windows(scattering_activations, num_features, num_scattering_coefficients, num_time_windows, num_T):
    """
    Reshape `scattering_activations` to distribute time windows over `num_T`.

    Args:
        scattering_activations: Tensor of shape (batch_size, num_features * num_scattering_coefficients * num_time_windows)
        num_features: Number of features
        num_scattering_coefficients: Number of scattering coefficients per feature
        num_time_windows: Number of time windows in original scattering transform
        num_T: Desired number of time steps

    Returns:
        Tensor of shape (batch_size, num_features * num_scattering_coefficients, num_T)
    """

    # ✅ Step 1: Extract batch size dynamically (symbolic inside Keras models)
    batch_size = tf.shape(scattering_activations)[0]  

    # ✅ Step 2: Reshape to (batch_size, num_features, num_scattering_coefficients, num_time_windows)
    reshaped_activations = tf.reshape(
        scattering_activations, 
        [batch_size, num_features, num_scattering_coefficients, num_time_windows]
    )  

    # ✅ Step 3: Prepare for resizing → Swap axes (batch, features, time, coefficients)
    reshaped_activations = tf.transpose(reshaped_activations, perm=[0, 1, 3, 2])

    # ✅ Step 4: Merge batch and feature dimensions for resizing
    merged_activations = tf.reshape(reshaped_activations, [-1, num_time_windows, num_scattering_coefficients])

    # ✅ Step 5: Resize along time axis (num_time_windows → num_T)
    resized_activations = tf.image.resize(merged_activations, [num_T, num_scattering_coefficients], method="bilinear")

    # ✅ Step 6: Restore shape to (batch_size, num_features, num_T, num_scattering_coefficients)
    resized_activations = tf.reshape(resized_activations, [batch_size, num_features, num_T, num_scattering_coefficients])

    # ✅ Step 7: Swap axes back to (batch, features, coeffs, time)
    final_activations = tf.transpose(resized_activations, perm=[0, 1, 3, 2])  

    # ✅ Step 8: Reshape to (batch_size, num_features * num_scattering_coefficients, num_T)
    final_activations = tf.reshape(final_activations, [batch_size, num_features * num_scattering_coefficients, num_T])

    return final_activations

# Function to generate data
def generateTVFreq_Data(num_samples, num_features, num_T, frequencies, amplitude=5, noise_std=0.6):
    """
    Generate synthetic time-varying frequency data without using Torch.
    
    Parameters:
        num_samples (int): Number of samples (batch size)
        num_features (int): Number of features
        num_T (int): Number of time steps
        frequencies (list): List of frequencies for important features
        amplitude (float): Amplitude of sine waves
        noise_std (float): Standard deviation of Gaussian noise
    
    Returns:
        Xsim (np.array): Shape (num_samples, num_features, num_T) with generated feature data
        ysim (np.array): Shape (num_samples, num_T) with target output
    """

    
    # Initialize continuous output (zero matrix)
    y_data = np.zeros((num_samples, num_T))

    # Only 5 features are important during specific time spans
    important_features = {
        3: (0, round(num_T/2)),    
        4: (round(num_T/3)+1, round(2*(num_T/3))),  
        5: (round(2*(num_T/3))+1, round(3*(num_T/3)))
    }
    intercepts = [3,1,-3]
    amplitudes_all = amplitude*[1.5,2,3]
    
        
    # Generate random normal data for all features
    X_data = np.random.normal(loc=0, scale=noise_std, size=(num_samples, num_features, num_T))
    
    t = np.arange(num_T)
    # Add signals to the important features
    for feature_index, (start, end) in important_features.items():
        print("feature_index: ", feature_index)
        print(f"From t = {start} to {end}.")
        signal = intercepts[feature_index-3] + amplitudes_all[feature_index-3]*np.cos( 2* np.pi * frequencies[feature_index-3] * t)
        X_data[:, feature_index, start:end] = signal[start:end]  # Assign signal to corresponding time range
        
        # Influence y_data based on this feature
        y_data[:, start:end] += X_data[:, feature_index, start:end] # Simple sum of the signals for this example

    # Generate random normal noise
    noise = np.random.randn(*y_data.shape) * noise_std

    # Add the noise to y_data
    y_data_noisy = y_data + noise

    return X_data, y_data_noisy


def generateTVFreqMixed_Data(num_samples, num_features, num_T, frequencies, amplitude=5, noise_std=0.6):
    """
    Generate synthetic time-varying frequency data with mixed frequencies per sample, without using Torch.
    
    Parameters:
        num_samples (int): Number of samples (batch size)
        num_features (int): Number of features
        num_T (int): Number of time steps
        frequencies (list): List of frequencies for important features
        amplitude (float): Amplitude of sine waves
        noise_std (float): Standard deviation of Gaussian noise
    
    Returns:
        Xsim (np.array): Shape (num_samples, num_features, num_T) with generated feature data
        ysim (np.array): Shape (num_samples, num_T) with target output
    """
    
    # Initialize data arrays with zeros
    X_data = np.zeros((num_samples, num_features, num_T))
    y_data = np.zeros((num_samples, num_T))

    # Only 5 features are important during specific time spans
    important_features = {
        0: (0, round(num_T/5)),    
        1: (round(num_T/5)+1, round(2*(num_T/5))),  
        2: (round(2*(num_T/5))+1, round(3*(num_T/5))),  
        3: (round(3*(num_T/5))+1, round(4*(num_T/5))),  
        4: (round(4*(num_T/5))+1, round(5*(num_T/5)))   
    }

    for sample_index in range(num_samples):
        for feature_index, (start, end) in important_features.items():
            # Generate different frequencies per sample
            freq = np.random.uniform(0.1, 2.0)  # Random frequency for variability

            # Generate time vector and sine wave signal
            t = np.linspace(0, 2 * np.pi * freq, end - start)
            signal = np.sin(t) * amplitude

            # Assign signal to X_data
            X_data[sample_index, feature_index, start:end] = signal

            # Influence y_data based on this feature
            y_data[sample_index, start:end] += signal  

    # Generate random normal noise
    noise = np.random.randn(*y_data.shape) * noise_std

    # Add the noise to y_data
    y_data_noisy = y_data + noise

    return X_data, y_data_noisy



def custom_loss(y_true, y_pred):
    # Cast to float32 for numerical stability
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Compute Mean Squared Error
    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Add a small constant to stabilize the loss
    loss += 1e-7  # Add epsilon for numerical stability
    
    return loss



def rename_and_convert_array(data_array, output_dim, K_folds, colBase):
    """
    Renames the elements in fold_losses to the format 'loss_yp_foldk'
    and converts it into a DataFrame.
    """

    # Dynamically generate column names
    column_names = [f"loss_y{p}_fold{k}" 
                    for k in range(1, K_folds + 1) 
                    for p in range(1, output_dim + 1)]

    # Convert fold_losses to a DataFrame
    data_array_df = pd.DataFrame(data_array.reshape(1, -1), columns=column_names)

    return data_array_df

def r_squared(y_true, y_pred):
    """
    Calculate R-squared given the true and predicted values of a target variable.

    Parameters:
    y_true (array-like): True values of the target variable
    y_pred (array-like): Predicted values of the target variable

    Returns:
    float: R-squared value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r2 = 0
        
    if (len(y_true.shape) == 3):
        # Compute sum of squares of residuals (SSR) and total sum of squares (TSS) for each output dimension
        ss_res = np.sum((y_true - y_pred) ** 2, axis=(0, 1))
        ss_tot = np.sum((y_true - np.mean(y_true, axis=(0, 1), keepdims=True)) ** 2, axis=(0, 1))

        # Compute R² scores, with a safeguard for division by zero
        r2 = 1 - np.divide(ss_res, ss_tot, where=ss_tot != 0, out=np.full_like(ss_res, np.nan))

    else:
        ss_res = np.sum(np.square(y_true - y_pred))
        ss_tot = np.sum(np.square(y_true - np.mean(y_true)))

        r2 = 1 - (ss_res / ss_tot)
    
    return r2

def f1score(y_true, y_pred):
    """
    Calculate F1 score given the true and predicted values of a binary classification problem.

    Parameters:
    y_true (array-like): True binary labels (0 or 1)
    y_pred (array-like): Predicted probabilities of class 1

    Returns:
    float: F1 score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Convert predicted probabilities to binary predictions
    y_pred_binary = tf.where(y_pred >= 0.5, 1, 0)

    tp = np.sum(y_true * y_pred_binary)
    fp = np.sum((1 - y_true) * y_pred_binary)
    fn = np.sum(y_true * (1 - y_pred_binary))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1

    
def cast_to_float16(inputs):
    return tf.nest.map_structure(lambda x: tf.cast(x, tf.float16) if x.dtype == tf.float32 else x, inputs)
    

def dictoDataFrame(data):
    # Check if `data` is a dictionary
    if isinstance(data, dict):
        flat_data = {key: [value] for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        # If `data` is a NumPy array, convert directly to DataFrame
        flat_data = {f"Fold_{i+1}": [val] for i, val in enumerate(data)}
    else:
        raise TypeError("Unsupported data type: expected dict or numpy.ndarray")
    return pd.DataFrame(flat_data)


def nestedDictoDF(NDic, K_folds, output_dim, outName="y"):
    flat_data = []
    output_labels = [f"{outName}{i+1}" for i in range(output_dim)]
    for model_name, data in NDic.items():
        # Reshape losses: Remove `nan` and reshape into (num_folds, num_outputs)
        data_cleaned = np.array(data).reshape(K_folds, output_dim)
    
        # Create a dictionary for each model
        model_data = {f"{outName}{output + 1}_fold{fold + 1}": data_cleaned[fold, output]
                    for fold in range(K_folds)
                    for output in range(output_dim)}
        model_data['Model'] = model_name  # Add model name
        flat_data.append(model_data)

    # Convert to DataFrame
    df = pd.DataFrame(flat_data)

    return(df)


# Function to set all seeds for reproducibility
def set_seeds(seed=42):
    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Python's built-in random module
    random.seed(seed)

    # NumPy random number generator
    np.random.seed(seed)

    # TensorFlow random number generator
    tf.random.set_seed(seed)

    # Ensure TensorFlow operations run deterministically on GPU
    #os.environ['TF_DETERMINISTIC_OPS'] = '1' # Encountered error when using hyperopt to optimize embedding model
    os.environ['TF_DETERMINISTIC_OPS'] = '0'  # Allow non-deterministic operations

    # If using TensorFlow's GPU backend, set this to ensure deterministic behavior
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Ensure GPU 0 is visible (modify if needed)

    # Optionally, restrict TensorFlow to a specific GPU for more control
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to avoid dynamic memory allocation
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
set_seeds(12345)
    

def create_scattering_nn(params, num_features, output_dim, num_T, J, Q, is_Debug=False, use_GPU=False,activation_options=['relu','elu'], 
        is_Plot = False, firstLayer = True):
    """Creates a ScatteringNNFlex model with output shape (num_T, output_dim) and adaptive hidden dimensions."""
    
    device_to_use = "/GPU:0" if use_GPU else "/CPU:0"
    
    if not isinstance(params['activation'], str):
        params['activation'] = activation_options[params['activation']]
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'],clipnorm=1.0)
    original_policy = mixed_precision.global_policy()

    # Define Scattering Function
    def apply_scattering(x):
        def scatter_feature(feature_i):
            """Apply scattering and activation function to a single feature."""
            feature_i = tf.cast(feature_i, tf.float32)  # Ensure float32 input
            scattering_i = scattering_layer(feature_i)

            # Apply log1p(abs(...)) transformation
            #scattering_i = tf.math.log1p(tf.abs(scattering_i))

            # Apply activation
            if params['activation'] == "relu":
                scattering_i = ReLU()(scattering_i)
            elif params['activation'] == "elu":
                scattering_i = ELU()(scattering_i)
            
            # ✅ Get the max scattering coefficient per time window
            # Move axis=1 (scattering coefficients) to last position
            scattering_i_transposed = tf.transpose(scattering_i, perm=[0, 2, 1])  # Shape: (batch_size, num_time_windows, num_scattering_coeffs)

            # Now apply `top_k()`
            top_k_values, top_k_indices = tf.math.top_k(scattering_i_transposed, k=3, sorted=True)
            max_scattering_coeff  = tf.transpose(top_k_values, perm=[0, 2, 1])
            return max_scattering_coeff  # Returns max scattering coefficient for each time window


        if use_GPU:
            # ✅ GPU: No need to transpose since vectorized_map keeps batch first
            scattering_outputs = tf.vectorized_map(lambda feature: scatter_feature(feature), x)
        else:
            #@@@
            # ✅ CPU: Use `tf.while_loop` + `tf.TensorArray` (Graph-Compatible)
            scattering_outputs = tf.TensorArray(dtype=tf.float32, size=num_features)  # ✅ Explicitly set dtype=float32

            def body(i, scattering_outputs):
                result = tf.cast(scatter_feature(x[:, i, :]), tf.float32)  # ✅ Ensure float32 before writing
                scattering_outputs = scattering_outputs.write(i, result)
                return i + 1, scattering_outputs

            def condition(i, scattering_outputs):
                return tf.less(i, num_features)

            i = tf.constant(0, dtype=tf.int32)  # Initialize loop variable
            _, scattering_outputs = tf.while_loop(condition, body, [i, scattering_outputs])
            scattering_outputs = scattering_outputs.stack()

            # ✅ Fix shape order for CPU (batch_size first)
            #print("Shape of scattering_outputs: ", scattering_outputs.shape)
            scattering_outputs = tf.transpose(scattering_outputs, perm=[1, 0, 2, 3])

        return scattering_outputs

    # Temporarily disable mixed precision for the scattering operation
    mixed_precision.set_global_policy('float32')
    feature_input = Input(shape=(num_features, num_T), name="Feature_input")  # **Input Layer**: Shape (num_features, num_T)
    
    with tf.device(device_to_use):
        scattering_layer = Scattering1D(J=J, Q=Q)
        # Apply scattering transform to each feature independently 
        scattering_outputs = tf.keras.layers.Lambda(apply_scattering)(feature_input) #Scattering --> activation(.)
        #scattering_outputs = tf.keras.layers.BatchNormalization(name="Batch_normalization")(scattering_outputs)#Batch normalization
  
        S = scattering_outputs.shape[2] #Number of scattering coefficients per feature - should be 1 with max operation
        Tprime = scattering_outputs.shape[3] #Number of time windows after downsampling
        
        if (is_Debug == True): 
            print("Shape of scattering_outputs:", scattering_outputs.shape) # Shape[None, S, Tprime]
            print("S = ", S) 
            print("Tprime = ", Tprime)  
        
        # **Flatten for Fully Connected Layers**
        X_MLP = Flatten()(scattering_outputs)  # Output: Shape[None, num_features*S*Tprime]
        
        if (is_Debug == True): 
            print("Shape of X_MLP:", X_MLP.shape)
        
        hidden = Dropout(params['dropout_rate'], name='Dropout_MLP')(X_MLP)
        hidden = Dense(output_dim, activation=params['activation'],
                       kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']), name="Dense_0")(hidden)
        hidden = Dropout(params['dropout_rate'], name='Dropout_0')(hidden)#Layer 1

        # **Fully Connected Layers with Doubling and Halving Pattern**
        num_halfway = params['num_layers'] // 2  # Determine when to start halving hidden_dim
        current_hidden_dim = int(params['hidden_dim'])
    
        if (is_Debug == True): print("current_hidden_dim: ", current_hidden_dim)
        
        hidden = Dense(current_hidden_dim, activation=params['activation'],
                       kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']), name="Dense_1")(hidden)
        hidden = Dropout(params['dropout_rate'], name='Dropout_1')(hidden) #Layer 2

        if params['num_layers'] > 3:
            for i in range(2, params['num_layers']-1):
                #if i < num_halfway:
                #    current_hidden_dim *= 2  # Double the hidden dimension
                #else:
                #    current_hidden_dim //= 2  # Halve the hidden dimension
                #
                #current_hidden_dim = tf.maximum(current_hidden_dim, 1)

                hidden = Dense(int(current_hidden_dim), activation=params['activation'],
                            kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']), name=f"Dense_{i}")(hidden)
                hidden = Dropout(params['dropout_rate'], name=f"Dropout_{i}")(hidden)

        # **Final Output Layer (num_T, output_dim)**
        #print("num_T*out_dim = ", num_T*output_dim)
        output_layer = Dense(output_dim*num_T,
                       kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']), name=f"Dense_{params['num_layers']-1}")(hidden)  # Flattened output
        output_layer = Reshape((output_dim, num_T), name=f"Reshaped_Dense_Final")(output_layer)  # Reshape to (None, output_dim, num_T)

        # Restore the original mixed precision policy
        mixed_precision.set_global_policy(original_policy)    
        # **Model Definition**
        model = tf.keras.Model(inputs=feature_input, outputs=output_layer, name="ScatteringNNFlex")
        model.compile(
            optimizer=optimizer,
            loss=custom_loss,  # 'mean_squared_error'
            metrics=tf.keras.metrics.MeanSquaredError()#['mean_squared_error']
        )
    # if (is_Debug == True): plot_model(model)
    if (is_Plot):
        if (firstLayer == True):
        #Multiply with the first layer of weights only
            heat = create_heatmap0(model, X_MLP, num_features, params['num_layers'], S, Tprime, output_dim, num_T) 
        elif (firstLayer == False):
        #Backpropagation of all layers of weights
            heat = create_heatmap(model, X_MLP, num_features, params['num_layers'], S, Tprime, output_dim, num_T) 
        return model, X_MLP, S, Tprime, heat
    else: return model

def create_scattering_nn_1layer_intermediateOutputs(params, num_features, output_dim, num_T, J, Q, is_Debug=False, 
    use_GPU=False, activation_options=['relu', 'elu'], is_Plot=False, firstLayer=True):
    """Creates a ScatteringNNFlex model with output shape (num_T, output_dim) and adaptive hidden dimensions."""

    device_to_use = "/GPU:0" if use_GPU else "/CPU:0"

    if not isinstance(params['activation'], str):
        params['activation'] = activation_options[params['activation']]
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'], clipnorm=1.0)
    original_policy = mixed_precision.global_policy()

    # Temporarily disable mixed precision for the scattering operation
    mixed_precision.set_global_policy('float32')
    feature_input = Input(shape=(num_features, num_T), name="Feature_input")  # **Input Layer**
    
    with tf.device(device_to_use):
        scattering_layer = Scattering1D(J=J, Q=Q)

        # Apply scattering transform to each feature independently 
        scattering_outputs = tf.keras.layers.Lambda(lambda x: scattering_layer(x), name="Scattering1D")(feature_input)
        scattering_outputs = tf.keras.layers.ReLU(name="ReLU_Scattering")(scattering_outputs)

        S = scattering_outputs.shape[2]  # Number of scattering coefficients per feature
        Tprime = scattering_outputs.shape[3]  # Number of time windows after downsampling
        
        if is_Debug:
            print("Shape of scattering_outputs:", scattering_outputs.shape) 
            print("S =", S) 
            print("Tprime =", Tprime)  

        # **Flatten for Fully Connected Layers**
        X_out_after_Flattened_Scattering = Flatten(name="Flattened_Scattering")(scattering_outputs)  
        
        if is_Debug:
            print("Shape of X_out_after_Flattened_Scattering:", X_out_after_Flattened_Scattering.shape)

        ## Fully connected layers with explicit names for intermediate outputs
        X_out_after_Dropout_MLP = Dropout(params['dropout_rate'], name='Dropout_MLP')(X_out_after_Flattened_Scattering)
        X_out_after_Dense_0 = Dense(output_dim, activation=params['activation'],
                                kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']),
                                name="Dense_0")(X_out_after_Dropout_MLP)
        X_out_after_Dropout_0 = Dropout(params['dropout_rate'], name='Dropout_0')(X_out_after_Dense_0)

        # **Final Output Layer**
        X_out_after_Dense_1 = Dense(output_dim * num_T, kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']), 
                             name="Dense_1")(X_out_after_Dropout_0)  
        output_layer = Reshape((output_dim, num_T), name="Reshaped_Dense_Final")(X_out_after_Dense_1)  

        # Restore the original mixed precision policy
        mixed_precision.set_global_policy(original_policy)    

        # **Define Model (Only the Final Output is an Explicit Output)**
        model = Model(inputs=feature_input, 
                      outputs=output_layer, 
                      name="ScatteringNNFlex")

        model.compile(
            optimizer=optimizer,
            loss=custom_loss,
            metrics=[tf.keras.metrics.MeanSquaredError()]
        )

    # **Return Intermediate Outputs Only if is_Plot is True**
    if is_Plot:
        if firstLayer:
            heat = create_heatmap0(model, X_out_after_Flattened_Scattering, num_features, 2, S, Tprime, output_dim, num_T)
        else:
            heat = create_heatmap(model, X_out_after_Flattened_Scattering, num_features, 2, S, Tprime, output_dim, num_T)

        return model, X_out_after_Flattened_Scattering, S, Tprime, heat, X_out_after_Dropout_MLP, X_out_after_Dense_0, X_out_after_Dropout_0, X_out_after_Dense_1
    else:
        return model


def create_scattering_nn_1layer(params, num_features, output_dim, num_T, J, Q, is_Debug=False, 
    use_GPU=False,activation_options=['relu','elu'], is_Plot = False, firstLayer= True):
    """Creates a ScatteringNNFlex model with output shape (num_T, output_dim) and adaptive hidden dimensions."""
    
    device_to_use = "/GPU:0" if use_GPU else "/CPU:0"
    
    if not isinstance(params['activation'], str):
        params['activation'] = activation_options[params['activation']]
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'],clipnorm=1.0)
    original_policy = mixed_precision.global_policy()

    # Define Scattering Function
    def apply_scattering(x):
        def scatter_feature(feature_i):
            """Apply scattering and activation function to a single feature."""
            feature_i = tf.cast(feature_i, tf.float32)  # Ensure float32 input
            scattering_i = scattering_layer(feature_i)

            # Apply log1p(abs(...)) transformation
            #scattering_i = tf.math.log1p(tf.abs(scattering_i))

            # Apply activation
            if params['activation'] == "relu":
                scattering_i = ReLU()(scattering_i)
            elif params['activation'] == "elu":
                scattering_i = ELU()(scattering_i)
            
            # ✅ Get the max scattering coefficient per time window
            # Move axis=1 (scattering coefficients) to last position
            scattering_i_transposed = tf.transpose(scattering_i, perm=[0, 2, 1])  # Shape: (batch_size, num_time_windows, num_scattering_coeffs)

            # Now apply `top_k()`
            top_k_values, top_k_indices = tf.math.top_k(scattering_i_transposed, k=3, sorted=True)
            max_scattering_coeff  = tf.transpose(top_k_values, perm=[0, 2, 1])
            return max_scattering_coeff  # Returns max scattering coefficient for each time window


        if use_GPU:
            # ✅ GPU: No need to transpose since vectorized_map keeps batch first
            scattering_outputs = tf.vectorized_map(lambda feature: scatter_feature(feature), x)
        else:
            #@@@
            # ✅ CPU: Use `tf.while_loop` + `tf.TensorArray` (Graph-Compatible)
            scattering_outputs = tf.TensorArray(dtype=tf.float32, size=num_features)  # ✅ Explicitly set dtype=float32

            def body(i, scattering_outputs):
                result = tf.cast(scatter_feature(x[:, i, :]), tf.float32)  # ✅ Ensure float32 before writing
                scattering_outputs = scattering_outputs.write(i, result)
                return i + 1, scattering_outputs

            def condition(i, scattering_outputs):
                return tf.less(i, num_features)

            i = tf.constant(0, dtype=tf.int32)  # Initialize loop variable
            _, scattering_outputs = tf.while_loop(condition, body, [i, scattering_outputs])
            scattering_outputs = scattering_outputs.stack()

            # ✅ Fix shape order for CPU (batch_size first)
            #print("Shape of scattering_outputs: ", scattering_outputs.shape)
            scattering_outputs = tf.transpose(scattering_outputs, perm=[1, 0, 2, 3])

        return scattering_outputs

    # Temporarily disable mixed precision for the scattering operation
    mixed_precision.set_global_policy('float32')
    feature_input = Input(shape=(num_features, num_T), name="Feature_input")  # **Input Layer**: Shape (num_features, num_T)
    
    with tf.device(device_to_use):
        scattering_layer = Scattering1D(J=J, Q=Q)
        # Apply scattering transform to each feature independently 
        scattering_outputs = tf.keras.layers.Lambda(apply_scattering)(feature_input) #Scattering --> activation(.)
        #scattering_outputs = tf.keras.layers.BatchNormalization(name="Batch_normalization")(scattering_outputs)#Batch normalization

        # Compute the maximum scattering coefficient per time window (axis=2)
        #scattering_outputs = tf.reduce_max(scattering_outputs, axis=2, keepdims=True, name="MaxScatteringCoeff")  
  
        S = scattering_outputs.shape[2] #Number of scattering coefficients per feature - should be 1 with max operation
        Tprime = scattering_outputs.shape[3] #Number of time windows after downsampling
        
        if (is_Debug == True): 
            print("Shape of scattering_outputs:", scattering_outputs.shape) # Shape[None, S, Tprime]
            print("S = ", S) 
            print("Tprime = ", Tprime)  
        
        # **Flatten for Fully Connected Layers**
        X_MLP = Flatten()(scattering_outputs)  # Output: Shape[None, num_features*S*Tprime]
        
        if (is_Debug == True): 
            print("Shape of X_MLP:", X_MLP.shape)

        ## Fully connected layer
        #hidden = Dense(current_hidden_dim, activation=params['activation'],
        #               kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']), name="Dense_0")(X_MLP)
        #hidden = Dropout(params['dropout_rate'], name='Dropout_0')(hidden)

        #print("out_dim= ", output_dim)
        hidden = Dropout(params['dropout_rate'], name='Dropout_MLP')(X_MLP)
        hidden = Dense(output_dim, activation=params['activation'],
                       kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']), name="Dense_0")(hidden)
        hidden = Dropout(params['dropout_rate'], name='Dropout_0')(hidden)

        # **Final Output Layer (output_dim*num_T, )**
        #print("out_dim*num_T= ", output_dim*num_T)
        output_layer = Dense(output_dim* num_T,
                       kernel_regularizer=tf.keras.regularizers.l2(params['l2_reg']), name="Dense_1")(hidden)  # Flattened output
        output_layer = Reshape((output_dim, num_T), name=f"Reshaped_Dense_Final")(output_layer)  # Reshape to (output_dim, num_T)

        # Restore the original mixed precision policy
        mixed_precision.set_global_policy(original_policy)    
        # **Model Definition**
        model = tf.keras.Model(inputs=feature_input, outputs=output_layer, name="ScatteringNNFlex")
        model.compile(
            optimizer=optimizer,
            loss=custom_loss,  # 'mean_squared_error'
            metrics=tf.keras.metrics.MeanSquaredError()#['mean_squared_error']
        )
    # if (is_Debug == True): plot_model(model)
    if (is_Plot):
        if (firstLayer == True):
            #Multiply with the first layer of weights only
            heat = create_heatmap0(model, X_MLP, num_features, 2, S, Tprime, output_dim, num_T) #Hard coding in 2 layers (X_MLP -> S*T' -> output_dim*numT)
        elif (firstLayer==False):
            #Backpropagation of all layers of weights
            heat = create_heatmap(model, X_MLP, num_features, 2, S, Tprime, output_dim, num_T) #Hard coding in 2 layers (X_MLP -> S*T' -> output_dim*numT)

        return model, X_MLP, S, Tprime, heat
    else: return model

#Same as objective_Flex_Sequential_cv, except for addition of the `model_choice` argument to allow calling flexible scattering embedding model
def objective_Flex_Sequential_cv2(params, X_data, ID_data, Y_data, num_total_participants, output_dim, batch_size, 
                                  epochs, K, num_T, J = None, Q = None, model_choice="scatter", is_Debug=False, use_GPU=False,
                                  activation_options = ['relu', 'elu'], embedding_dim_options = [1, 2, 10, 50, 100]):
    
    # Initialize GroupKFold
    group_kf = GroupKFold(n_splits=K)
    val_losses = []
    #val_r2_1 = []
    val_r2 = []
    
    # Early stopping to avoid overfitting
    #early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    if  (is_Debug==True):
        print("Input_dim: ", X_data.shape)
        print("Ouput_dim: ", Y_data.shape)
        print("ID_data: ", ID_data.shape)
 
    for fold, (train_index, val_index) in enumerate(group_kf.split(X_data, Y_data, groups=ID_data)):
        if (is_Debug == True):
            print(f"\nFold {fold + 1}")
            print("train_index = ", train_index)
            print("val_index = ", val_index)
            #print(f"Train subjects: {np.unique(ID_data[train_index])}")
            #print(f"Validation subjects: {np.unique(ID_data[val_index])}")
            #print(f"Train size: {len(train_index)}, Validation size: {len(val_index)}")

        X_train_fold, X_val_fold = X_data[train_index], X_data[val_index]
        if (is_Debug == True): print("Training flexible embedding model...")
        
        Y_train_fold, Y_val_fold = Y_data[train_index], Y_data[val_index]
        ID_train_fold, ID_val_fold = ID_data[train_index], ID_data[val_index]

        #Y_train_fold = {key: value[train_index] for key, value in Y_data.items()}
        #Y_val_fold = {key: value[val_index] for key, value in Y_data.items()}
        
        if not isinstance(params['activation'], str):
            params['activation'] = activation_options[params['activation']]
            params['embedding_dim'] = int(embedding_dim_options[params['embedding_dim']])
            
        if  (is_Debug==True):
            print("Input_dim: ", X_train_fold.shape)
            print("Ouput_dim: ", Y_train_fold.shape)
            print("Y_val dim: ", Y_val_fold.shape)
            print("Activation: ", params['activation'])
            print("embedding dim: ", params['embedding_dim'])
    
         
        if (model_choice == 'scatter'):
                       
            # Reshape original data into np array of shape num_samples, num_features, num_T for input to keras api model
            num_samples_train = len(np.unique(ID_data[train_index])) #X_train_fold.shape[0]
            num_samples_val = len(np.unique(ID_data[val_index])) #X_train_fold.shape[0]
            num_features = X_train_fold.shape[1]
            #num_samples = num_samples_train + num_samples_val
            X_train_fold = X_train_fold.reshape(num_samples_train, num_T, num_features).transpose(0, 2, 1)
            X_val_fold = X_val_fold.reshape(num_samples_val, num_T, num_features).transpose(0, 2, 1)
            ID_train_fold = np.unique(ID_train_fold).reshape(-1, 1)
            ID_val_fold = np.unique(ID_val_fold).reshape(-1, 1)
            Y_train_fold = Y_train_fold.reshape(num_samples_train, num_T, output_dim)
            Y_val_fold = Y_val_fold.reshape(num_samples_val, num_T, output_dim)
            
            #model = create_vec_flex_scatterEmbed_model(params, num_features = num_features, output_dim = output_dim, 
            #                                num_total_participants = num_total_participants, num_T = num_T, J = J, Q = Q,
            #                                activation_options = activation_options,
            #                                 embedding_dim_options= embedding_dim_options)
            
            #model = create_flex_scatterEmbed_model(params, num_features = num_features, output_dim = output_dim, 
            #                                num_total_participants = num_total_participants, num_T = num_T, J = J, Q = Q,
            #                                activation_options = activation_options,
            #                                embedding_dim_options= embedding_dim_options)
            
            model = create_scatteringEmbed_nn(params, num_features = num_features, output_dim = output_dim, 
                                            num_total_participants = num_total_participants, num_T = num_T, J = J, Q = Q,
                                            is_Debug = is_Debug, use_GPU = use_GPU,
                                            activation_options = activation_options,
                                            embedding_dim_options= embedding_dim_options)       

        
        if  (is_Debug==True): print('Fitting!')
        
            
        history = model.fit(
            [X_train_fold, ID_train_fold], Y_train_fold, 
            batch_size=batch_size, verbose=0, #validation_split=0.4, 
            epochs=epochs
            )
        
        if  (is_Debug==True): print('Evaluating!')
        
        val_loss = model.evaluate([X_val_fold, ID_val_fold], Y_val_fold, verbose=0)
        
        threshold = 1e3  # Define the maximum allowed loss value
        capped_val_loss = [min(loss, threshold) if np.isfinite(loss) else threshold for loss in val_loss]

        #print("Original Losses:", val_loss)
        #print("Capped Losses:", capped_val_loss)
        
        stable_val_loss = [tf.cast(loss + 1e-7, tf.float32).numpy() if tf.math.is_finite(loss) else float('inf') for loss in capped_val_loss]

        val_losses.append(stable_val_loss)
        Y_val_pred_fold = model.predict([X_val_fold, ID_val_fold])
        

        # Check for extreme values in predictions
        #print("Max Prediction:", np.max(Y_val_pred_fold))
        #print("Min Prediction:", np.min(Y_val_pred_fold))

        
        if (is_Debug == True):
            print(f"Y_val_pred_fold shape: {Y_val_pred_fold.shape}")
            print(f"Y_val_fold shape: {Y_val_fold.shape}")

        # Calculate R-squared for each continuous output
        r2_fold = []
        for j in range(0,output_dim):
            if (model_choice == "embed"):
                r2_fold.append(r_squared(Y_val_fold[:,j], Y_val_pred_fold[:,j]))
        
            if (model_choice == "scatter"): 
                r2_fold.append(r_squared(Y_val_fold[:,j,:].reshape(-1,1), Y_val_pred_fold[:,j,:].reshape(-1,1)))
        val_r2.append(r2_fold)
        print(f"r2 fold {fold+1}: {r2_fold}")
        print(f"Loss fold {fold+1}: {stable_val_loss}")
    
    val_losses = np.where(np.isinf(val_losses), np.nan, val_losses)
    val_r2 = np.where(np.isinf(val_r2), np.nan, val_r2)
    mean_val_loss = np.nanmean(val_losses) 
    mean_r2 = np.nanmean(val_r2)
    
    print(f"mean_val_loss: {mean_val_loss}, mean_r2: {mean_r2}")
    
    results= {
        'loss': float(mean_val_loss),
        'mean_R2': float(mean_r2),
        'status': STATUS_OK,
        'fold_losses': np.array(val_losses),
        'R2byFold': np.array(val_r2)
    } 
    
    return results


#No embedding
def objective_Flex_Scatter(params, X_data, Y_data, output_dim, batch_size, 
                                  epochs, K, num_T, J = None, Q = None, is_Debug=False,use_GPU = False, 
                                  activation_options = ['relu', 'elu'], model_spec="flex"):
        
    # Initialize GroupKFold
    #group_kf = GroupKFold(n_splits=K)
    kf = KFold(n_splits=K, shuffle=True)
    val_losses = []
    val_r2 = []
    
    # Early stopping to avoid overfitting
    #early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    if  (is_Debug==True):
        print("Input_dim: ", X_data.shape)
        print("Ouput_dim: ", Y_data.shape)
 
    for fold, (train_index, val_index) in enumerate(kf.split(X_data),1):
        if (is_Debug == True):
            print(f"Train size: {len(train_index)}, Validation size: {len(val_index)}")
            print("Training flexible scattering model...")

        X_train_fold, X_val_fold = X_data[train_index], X_data[val_index]      
        Y_train_fold, Y_val_fold = Y_data[train_index], Y_data[val_index]
        
        if not isinstance(params['activation'], str):
            params['activation'] = activation_options[params['activation']]
             
        if  (is_Debug==True):
            print("output_dim passed in: ", output_dim)
            print("X_train_fold dim: ", X_train_fold.shape)
            print("X_val_fold dim: ", X_val_fold.shape)
            print("Y_train_fold dim: ", Y_train_fold.shape)
            print("Y_val_fold dim: ", Y_val_fold.shape)
            print("Activation: ", params['activation'])
                             
        num_features = X_train_fold.shape[1]

        if (model_spec == "flex"):
            model = create_scattering_nn(params, num_features = num_features, output_dim = output_dim, 
                        num_T = num_T, J = J, Q = Q, is_Debug=is_Debug, use_GPU=use_GPU,
                        activation_options = activation_options, is_Plot= False)
        elif (model_spec == "1layer"):
            model = create_scattering_nn_1layer(params, num_features = num_features, output_dim = output_dim, 
                    num_T = num_T, J = J, Q = Q, is_Debug=is_Debug, use_GPU=use_GPU,
                    activation_options = activation_options, is_Plot= False)

        
        if  (is_Debug==True): print('Fitting!')
            
        history = model.fit(
            X_train_fold, Y_train_fold, 
            batch_size=batch_size, verbose=0,  
            epochs=epochs
            )
        
        #print(history.history['loss']) #Returns one loss value
        
        if  (is_Debug==True): print('Evaluating!')
        val_loss = model.evaluate(X_val_fold, Y_val_fold, verbose=0, return_dict=True)
        val_loss = val_loss['loss']
        if (is_Debug): print("val_loss here: ", val_loss)
        
        threshold = 1e3  # Define the maximum allowed loss value
        new_val_loss = [min(val_loss, threshold) if np.isfinite(val_loss) else threshold]
        new_val_loss = [max(val_loss, 1e-7) if np.isfinite(new_val_loss) else 1e-7]
        if (is_Debug): print("new_val_loss:", new_val_loss)
 

        
        #stable_val_loss = [
        #tf.cast(loss + 1e-7, tf.float32).numpy() if tf.math.is_finite(loss) else float('inf') 
        #for loss in capped_val_loss]
        
        val_losses.append(new_val_loss)
        Y_val_pred_fold = model.predict(X_val_fold)
        

        # Check for extreme values in predictions
        #print("Max Prediction:", np.max(Y_val_pred_fold))
        #print("Min Prediction:", np.min(Y_val_pred_fold))

        
        if (is_Debug == True):
            print(f"Y_val_pred_fold shape: {Y_val_pred_fold.shape}")
            print(f"Y_val_fold shape: {Y_val_fold.shape}")

        # Calculate R-squared for each continuous output
        r2_fold = []
        for j in range(0,output_dim):
                r2_fold.append(r_squared(Y_val_fold[:,j,:].reshape(-1,1), Y_val_pred_fold[:,j,:].reshape(-1,1)))
        val_r2.append(r2_fold)
        print(f"r2 fold {fold}: {r2_fold}")
        print(f"Loss fold {fold}: {new_val_loss}")
    
    val_losses = np.where(np.isinf(val_losses), np.nan, val_losses)
    val_r2 = np.where(np.isinf(val_r2), np.nan, val_r2)
    mean_val_loss = np.nanmean(val_losses) 
    mean_r2 = np.nanmean(val_r2)
    
    print(f"mean_val_loss: {mean_val_loss}, mean_r2: {mean_r2}")
    
    results= {
        'loss': float(mean_val_loss),
        'mean_R2': float(mean_r2),
        'status': STATUS_OK,
        'fold_losses': np.array(val_losses),
        'R2byFold': np.array(val_r2)
    } 
    
    return results


def evaluate_model(model, X_data, X_data2, Y_data, Y_data2, 
                   withID = False, ID_data = [], ID_data2 = [],
                   batch_size=50, epochs=50, is_Debug=False):
    """
    Fits a model at the specified optimized hyperparameters and evaluate
    performance with training and test data sets.

    Returns:
    float: R-squared value
    """

    output_dim = Y_data.shape[1]
    
    ## Define the ModelCheckpoint callback
    #checkpoint = ModelCheckpoint(
    #filepath=saveModelFile,  # Filepath to save the model
    #monitor='val_loss',       # Metric to monitor
    #save_best_only=True,      # Save only the best model
    #mode='min',               # 'min' because we're monitoring loss
    #verbose=1                 # Print messages when saving
    #)
    
    if (withID==True):

        # Train the final model
        model.fit([X_data, ID_data], Y_data, 
        batch_size=batch_size, verbose=0, epochs=epochs)#, callbacks=[checkpoint])
    
        loss_train = model.evaluate([X_data, ID_data], Y_data, verbose=0)
        loss_test = model.evaluate([X_data2, ID_data2], Y_data2, verbose=0)
    
        predicted_train = model.predict([X_data, ID_data])
        predicted_test = model.predict([X_data2, ID_data2])
    else: 
        model.fit(X_data, Y_data, batch_size=batch_size, verbose=0, epochs=epochs)#, callbacks=[checkpoint])
    
        loss_train = model.evaluate(X_data, Y_data, verbose=0)
        loss_test = model.evaluate(X_data2, Y_data2, verbose=0)
    
        predicted_train = model.predict(X_data)
        predicted_test = model.predict(X_data2)
        
    r2_train = np.zeros(output_dim)
    r2_test = np.zeros(output_dim)
    
    for j in range(len(r2_train)):   
        if (len(np.unique(Y_data.shape))==3):
            dat1 = Y_data[:,j,:].reshape(-1,1)
            pred1 = predicted_train[:,j,:].reshape(-1,1)
            dat2 = Y_data2[:,j,:].reshape(-1,1)
            pred2 = predicted_test[:,j,:].reshape(-1,1)   
        else:
            dat1 = Y_data[:,j]
            dat2 = Y_data2[:,j]    
            pred1 = predicted_train[:,j]       
            pred2 = predicted_test[:,j]
        r2_train[j] = r_squared(dat1, pred1)
        r2_test[j] = r_squared(dat2, pred2)
        if (is_Debug == True):
            mask = ~np.isnan(dat1) & ~np.isnan(pred1)  # Ignore rows where either value is NaN
            print("Corr between true and predicted jth train element: ", np.corrcoef(dat1[mask], pred1[mask]))    
            mask = ~np.isnan(dat2) & ~np.isnan(pred2)  # Ignore rows where either value is NaN
            print("Corr between true and predicted jth test element: ", np.corrcoef(dat2[mask], pred2[mask]))    
    results= {
        'loss_train': loss_train,
        'loss_test': loss_test,
        'r2_train': r2_train,
        'r2_test': r2_test
    } 
    
    #print(results)
    
    return results, predicted_train, predicted_test, X_data, X_data2, Y_data, Y_data2


def create_heatmap(model, scattering_activations, num_features, num_fc_layers, num_ScatteringCoefficients, num_TimeWindows, output_dim, num_T):
    """
    Computes feature importance by backpropagating through fully connected (FC) layers
    and generates a heatmap of feature importance over time.

    Args:
        model (tf.keras.Model): The trained model.
        scattering_activations (list of tensors): Scattering activations for each feature.
        num_features (int): Number of features in the input data.
        num_fc_layers (int): Number of fully connected layers.
        title (str): Title of the heatmap.
        xlabel (str): Label for the x-axis (default: "Time Steps").
        ylabel (str): Label for the y-axis (default: "Features").
        cmap (str): Color map for visualization (default: "hot").

    Returns:
        heatmap (numpy.ndarray): Feature importance heatmap of shape (num_samples, num_features, num_T).
        
    # Example usage:
    # heatmap = create_heatmap(model, scattering_activations, num_features, num_fc_layers)
    """


    for layer in model.layers:
        print("Layer names: ", layer.name)


    # Extract FC layer weights
    Last_weights = model.get_layer(f"Dense_{num_fc_layers-1}").weights[0]
    fc_weights=[]
    if num_fc_layers > 1:
        fc_weights = [model.get_layer(f"Dense_{i}").weights[0] for i in range(num_fc_layers-1)] #Exclude 1 cuz last layer already extracted
    
    # Ensure weight shapes are correct for multiplication
    importance = Last_weights # Start with final layer weight
    print("Shape of importance (weight matrix in last layer): ", importance.shape)

    print("Length of fc_weights: ", len(fc_weights))
    if len(fc_weights) > 0:
        for i in reversed(range(len(fc_weights))):
            print(f"Layer {i}, shape of weight matrix: ", fc_weights[i].shape)
            importance = tf.matmul(fc_weights[i], importance)  # Backpropagate through FC layers
    print("Shape of backpropagated importance: ", importance.shape)

    ### 3️⃣ **Expand & Tile Importance Over Samples**
    num_samples = tf.shape(scattering_activations)[0] 
    print("num_samples: ", num_samples)
    importance = tf.expand_dims(importance, axis=0)  # Shape: (1, num_features*S*T', output_dim*num_T)
    
    importance = tf.tile(importance, [num_samples, 1, 1])  # Shape: (num_samples, num_features*S*T', output_dim*num_T)

    print("Replicated importance shape:", importance.shape)
    importance = tf.reshape(importance, (num_samples, num_features, num_ScatteringCoefficients*num_TimeWindows, output_dim, num_T))
    print("Reshaped replicated importance shape: ", importance.shape)

   ### 4️⃣ **Compute Weighted Activations Using Element-wise Multiplication**
    print ("Shape of scattering_activations: ", scattering_activations.shape)
    Reshaped_scattering_activations = tf.reshape(scattering_activations, (num_samples, num_features, num_ScatteringCoefficients*num_TimeWindows))
    print ("Shape of reshaped scattering_activations: ", Reshaped_scattering_activations.shape) 
        #(Shape: None, num_features, num_ScatteringCoefficients*num_TimeWindows)

    print(f"Num_scattering_coefs: {num_ScatteringCoefficients}; num_TimeWindows: {num_TimeWindows}")


    # Compute weighted activations
    weightedActivations = tf.zeros((num_samples, num_features, output_dim, num_T))
    #K = num_ScatteringCoefficients*num_TimeWindows
    #for j in range(output_dim):
    #    for k in range(num_features):
    #        for t in range(num_T):
    #            #Take all scattering coefficients for feature k and multiply with their weights in importance for time t, output j
    #            weightedActivations[:,k,j,t] =  tf.reduce_mean(tf.reshape(Reshaped_scattering_activations[:,k,:],(-1,K))*
    #                tf.reshape(importance[:,k,:,j,t],(-1,K)))
    

    # Compute weighted activations efficiently
    weightedActivations = tf.einsum('bfi, bfijt -> bftj', Reshaped_scattering_activations, importance)
    
    print("Shape of weightedActivations: ", weightedActivations.shape) # Shape: (num_samples, num_features, num_T, output_dim)

    heatmapFeat =  weightedActivations

    return heatmapFeat


def create_heatmap0(model, scattering_activations, num_features, num_fc_layers, num_ScatteringCoefficients, num_TimeWindows, output_dim, num_T):
    """
    Computes feature importance by using the second layer of weights (from X_MLP to the output from the first layer)
    and generates a heatmap of feature importance over time.

    Args:
        model (tf.keras.Model): The trained model.
        scattering_activations (list of tensors): Scattering activations for each feature.
        num_features (int): Number of features in the input data.
        num_fc_layers (int): Number of fully connected layers.
        title (str): Title of the heatmap.
        xlabel (str): Label for the x-axis (default: "Time Steps").
        ylabel (str): Label for the y-axis (default: "Features").
        cmap (str): Color map for visualization (default: "hot").

    Returns:
        heatmap (numpy.ndarray): Feature importance heatmap of shape (num_samples, num_features, num_T).
        
    # Example usage:
    # heatmap = create_heatmap(model, scattering_activations, num_features, num_fc_layers)
    """


    for layer in model.layers:
        print("Layer names: ", layer.name)


    ## Extract first layer of weights only
    importance = model.get_layer("Dense_0").weights[0]
    
    print("Shape of importance (weight matrix in first dense layer): ", importance.shape)

    ### 3️⃣ **Expand & Tile Importance Over Samples**
    num_samples = tf.shape(scattering_activations)[0] 
    print("num_samples: ", num_samples)
    importance = tf.expand_dims(importance, axis=0)  # Shape: (1, num_features*S*T', output_dim)
    
    importance = tf.tile(importance, [num_samples, 1, 1])  # Shape: (num_samples, num_features*S*T', output_dim)

    print("Replicated importance shape:", importance.shape)
    importance = tf.reshape(importance, (num_samples, num_features, num_ScatteringCoefficients,num_TimeWindows, output_dim))
    print("Reshaped replicated importance shape: ", importance.shape)

   ### 4️⃣ **Compute Weighted Activations Using Element-wise Multiplication**
    print ("Shape of scattering_activations: ", scattering_activations.shape)
    Reshaped_scattering_activations = tf.reshape(scattering_activations, (num_samples, num_features, num_ScatteringCoefficients,num_TimeWindows))
    print ("Shape of reshaped scattering_activations: ", Reshaped_scattering_activations.shape) 
        #(Shape: None, num_features, num_ScatteringCoefficients*num_TimeWindows)

    print(f"Num_scattering_coefs: {num_ScatteringCoefficients}; num_TimeWindows: {num_TimeWindows}")


    # Compute multiplication

    # Compute weighted activations efficiently
    weightedActivations = tf.einsum('bfsw, bfswo -> bfwo', Reshaped_scattering_activations, importance)

    
    print("Shape of weightedActivations: ", weightedActivations.shape) # Shape: (num_samples, num_features, num_timeWindows, output_dim)

    heatmapFeat =  weightedActivations

    return heatmapFeat