from pyarrow import duration
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# def display_navbar():
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.button("Home", key = "home")
    
#     with col2:
#         st.button("Upload", key = "upload")
        
# def upload_file():
#     uploaded_file = st.file_uploader("Choose a CSV file to upload", type = "csv")
    
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file, index_col='Time,ms')
        
#         options = df.columns.tolist()

#     return df, options
    
# def plot_signal():
#     df, options = upload_file()
    
#     selected_column = st.selectbox("Select Signal", options)
    
#     if selected_column:
#         st.line_chart(df[selected_column])
        
        
# def main():
#     display_navbar()
    
#     if st.session_state.get("upload"):
#         upload_file()
# 
col1, col2 = st.columns([3, 1])

with col2:
    st.button("Upload", key = "upload")
    
uploaded_file = st.file_uploader("Upload file:", type = "csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col = 'Time,ms')
    
    options = df.columns.tolist()
    
    selected_column = st.selectbox("Select Signal", options)
    
    if selected_column:
        sigplot, spectrogram = st.columns([1, 1])
        with sigplot:
            fig, ax = plt.subplots()
            sig = np.array(df[selected_column])
            duration = np.array(df.index)
            ax.plot(duration, sig)
            st.pyplot(fig)
        with spectrogram:
            fig, ax = plt.subplots()
            sig = np.array(df[selected_column])
            ax.specgram(sig, Fs = 1 / 0.005, NFFT = 128, noverlap = 32, cmap = 'jet')
            st.pyplot(fig)