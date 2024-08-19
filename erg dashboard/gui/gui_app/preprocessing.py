import pandas as pd
import numpy as np
import scipy
import scipy.signal as signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing as prep
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


def compute_spectrogram(sig_id, window, nfft, overlap):
    time_step = 0.005
    sample_freq = 1 / time_step
    sig = np.array(sig_id)
    freqs, time, sxx = signal.spectrogram(sig,
                                          fs=sample_freq,
                                          window=window,
                                          nperseg=nfft,
                                          noverlap=overlap)
    return sxx

def extract_features(sig, window, nfft, overlap):
    sxx = compute_spectrogram(sig, window, nfft, overlap)
    spectrum = sxx.flatten()
    bmin = np.min(spectrum)
    bmax = np.max(spectrum)
    bmedian = np.median(spectrum)
    bmean = np.mean(spectrum)

    features_dict = {
        'bmin': bmin,
        'bmax': bmax,
        'bmedian': bmedian,
        'bmean': bmean,
        'ids': sig
    }

    features_df = pd.DataFrame(features_dict,
                               range(0, len(features_dict)))
    features_df.drop_duplicates

def encode_targets(to_encode):
    to_encode = np.ravel(to_encode)
    encoder = prep.LabelEncoder()
    encoder.fit(to_encode)
    target_labels, encoded_targets = encoder.classes_, encoder.transform(to_encode)
    return target_labels, encoded_targets

def split(data, targets):
    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        test_size = 0.1,
        random_state = random_state,
        stratify = targets
    )
    return X_train, X_test, y_train, y_test