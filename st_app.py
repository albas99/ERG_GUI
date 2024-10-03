import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy

def load_signal(sig):
    assert type(sig) is str, f"{type(sig)}"
    signal = np.array(df[sig])
    return signal
    
def plot_signal(sig_no):
    fig, ax = plt.subplots()
    # sig = np.array(df[sig_no])
    sig = load_signal(sig_no)
    duration = np.array(df.index)
    ax.plot(duration, sig)
    st.pyplot(fig)
    
def plot_spectrogram(sig_no):
    fig, ax = plt.subplots()
    # sig = np.array(df[sig_no])
    sig = load_signal(sig_no)
    sxx, freqs, bins, im = ax.specgram(sig, Fs = 1 / 0.005, NFFT = 128, noverlap = 32, cmap = 'jet')
    st.pyplot(fig)
    
def compute_features(sig_no):
    # sig = np.array(df[sig_no])
    sig = load_signal(sig_no)
    sxx, freqs, bins= scipy.signal.spectrogram(sig, fs = 1 / 0.005, nperseg = 128, noverlap = 32)
    spectrum = sxx.flatten()
    bmin = np.min(spectrum)
    bmedian = np.median(spectrum)
    bmax = np.max(spectrum)
    bmean = np.mean(spectrum)
    features_dict = {
        'Bmin': bmin,
        'Bmax': bmax,
        'Bmedian': bmedian,
        'Bmean': bmean,
        # 'ids': sig
    }

    features_df = pd.DataFrame(features_dict, index = range(0, len(features_dict)))
    features_df.drop_duplicates(inplace=True)

    return features_df
    
with open('model.pkl', 'rb') as model:
    loaded_model = pickle.load(model)
    
with open('encoder.pkl', 'rb') as encoder:
    loaded_encoder = pickle.load(encoder)

col1, col2 = st.columns([3, 1])

with col2:
    st.button("Upload", key = "upload")
    
uploaded_file = st.file_uploader("Upload file:", type = "csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col = 'Time,ms')
    
    options = df.columns.tolist()
    
    selected_column = st.selectbox("Select Signal", options, key = "signal_select")
    
    if selected_column:
        sigplot, spectrogram = st.columns([1, 1])
        
        with sigplot:
            plot_signal(selected_column)
        with spectrogram:
            plot_spectrogram(selected_column)
        
        features = compute_features(selected_column)
        # st.write(features)
        
        if features is not None:    
            prediction = loaded_model.predict(features)
            prediction_probs = loaded_model.predict_proba(features)
            # st.write(prediction, prediction_probs)
            
            probs_plot, fi_plot = st.columns([1, 1])
                    
            with probs_plot:
                fig, ax = plt.subplots()
                ax.barh(loaded_encoder.classes_, prediction_probs[0])
                st.pyplot(fig)
            
            with fi_plot:
                fig, ax = plt.subplots()
                ax.barh(loaded_model.model.estimator.feature_name_, loaded_model.model.estimator.feature_importances_)
                st.pyplot(fig)
        
    # with sigplot:
    #     fig, ax = plt.subplots()
    #     sig = np.array(df[selected_column])
    #     duration = np.array(df.index)
    #     ax.plot(duration, sig)
    #     st.pyplot(fig)
    # with spectrogram:
    #     fig, ax = plt.subplots()
    #     sig = np.array(df[selected_column])
    #     sxx, freqs, bins, im = ax.specgram(sig, Fs = 1 / 0.005, NFFT = 128, noverlap = 32, cmap = 'jet')
    #     st.pyplot(fig)
    
    
    
    # if selected_column:
    #     sigplot, spectrogram = st.columns([1, 1])
    #     with sigplot:
    #         fig, ax = plt.subplots()
    #         sig = np.array(df[selected_column])
    #         duration = np.array(df.index)
    #         ax.plot(duration, sig)
    #         st.pyplot(fig)
    #     with spectrogram:
    #         fig, ax = plt.subplots()
    #         sig = np.array(df[selected_column])
    #         ax.specgram(sig, Fs = 1 / 0.005, NFFT = 128, noverlap = 32, cmap = 'jet')
    #         st.pyplot(fig)
            
    #     with open('model.pkl', 'rb') as model:
    #         loaded_model = pickle.load(model)
            
    #         prediction = loaded_model.predict(df[selected_column])
            
    #         prediction_probs = loaded_model.predict_proba(df[selected_column])
    #         target_labels = np.array('healthy', 'unhealthy')
            
    #         probs_plot, fi_plot = st.columns([1, 1])
            
    #         with probs_plot:
    #             fig, ax = plt.subplots()
    #             ax.barh(target_labels, prediction_probs)
            
    #         with fi_plot:
    #             fig, ax = plt.subplots()
    #             ax.barh(loaded_model.model.estimator.feature_name_, loaded_model.model.estimator.feature_importances_)