import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from flaml.automl.automl import AutoML
import pickle
import scipy
from sklearn import preprocessing as prep
from sklearn.model_selection import train_test_split


max_balanced = pd.read_csv('max_balanced.csv', index_col='Time,ms')
max_bal_diag = pd.read_csv('max_balanced_diagnoses.csv', index_col = 0)
def load_signal(sig):
    assert type(sig) is str, f"{type(sig)}"
    signal_no = np.array(max_balanced[sig])
    return signal_no
    
def compute_spectrogram(signal_no, nperseg, noverlap, window):
    time_step = 0.005
    sample_freq = 1 / time_step
    freqs, times, Sxx = scipy.signal.spectrogram(signal_no, fs = sample_freq, nperseg = nperseg, noverlap = noverlap, window = window)
    
    return freqs, times, Sxx
    
def compute_features(sig, nperseg, noverlap, window):
    freqs, times, Sxx = compute_spectrogram(sig, nperseg, noverlap, window)
    spectrum = Sxx.flatten()
    bmin = np.min(spectrum)
    bmedian = np.median(spectrum)
    bmax = np.max(spectrum)
    bmean = np.mean(spectrum)
    return bmin, bmedian, bmax, bmean
    

dfeatures = pd.DataFrame(columns = ['params', 'Bmin', 'Bmedian', 'Bmax', 'Bmean', 'pat_no'])

window, size, overlap = 'boxcar', 128, 64
for col in max_balanced.columns:

    signal_no = load_signal(col)
#     freqs, times, Sxx = compute_spectrogram(signal_no, nperseg = window_size, noverlap = overlap, window = window_type)
    bmin, bmedian, bmax, bmean = compute_features(signal_no, size, overlap, window)

    # Create a dictionary with the features
    features_dict = {
        'params': f"{window}{size}{overlap}",
        'Bmin': bmin,
        'Bmedian': bmedian,
        'Bmax': bmax,
        'Bmean': bmean,
        'pat_no': col
    }

    features = pd.DataFrame(features_dict, index = range(0, len(features_dict)))

    dfeatures = pd.concat([dfeatures, features], ignore_index = True)
    
dfeatures.duplicated()
dfeatures.drop_duplicates(inplace = True, ignore_index = True)
dfeatures.drop(['params', 'pat_no'], axis = 1, inplace = True)

features, targets = dfeatures, max_bal_diag['Diagnosis'].to_numpy()
encoder = prep.LabelEncoder()
encoder.fit(targets)
target_labels, encoded_targets = encoder.classes_, encoder.transform(targets)

random_state = 42
X_train, X_test, y_train, y_test = train_test_split(features, encoded_targets, test_size = 0.1, random_state = random_state)

automl = AutoML()

automl.fit(X_train, y_train,
          task = 'classification',
          metric = 'accuracy',
          time_budget = 10800,
          estimator_list = ['lgbm', 'xgboost', 'rf', 'histgb'],
          eval_method = 'cv',
          split_type = 'stratified',
          n_splits = 5)