# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 04:24:49 2024

@author: buing
"""
import os
import pandas as pd
import numpy as np
import librosa
import brain_signal_similarity
from scipy.fftpack import fft
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import euclidean, cosine

# min max normalization
def normalize_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    normalized_list = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_list

# Function to extract features from signals
def extract_features(fp1, fp2, fs=250, window_size=2, overlap=0.5, sub_window_size=1, sub_overlap=0.5):
    features = []
    start_index = int(20 * fs)  # Index corresponding to second 20
    end_index = int(70 * fs)    # Index corresponding to second 70
    n_samples = len(fp1)
    n_per_window = int(window_size * fs)
    step_size = int(n_per_window * (1 - overlap))
    sub_window_size_samples = int(sub_window_size * fs)
    sub_overlap_samples = int(sub_window_size_samples * sub_overlap)
    
    for i in range(start_index, end_index - n_per_window, step_size):
        sfp1 = fp1[i:i+n_per_window]
        sfp2 = fp2[i:i+n_per_window]
        sfp1_bands = np.array(extract_band_features(sfp1, fs, sub_window_size_samples, sub_overlap_samples))
        sfp2_bands = np.array(extract_band_features(sfp2, fs, sub_window_size_samples, sub_overlap_samples))
        euclidean_distances = [brain_signal_similarity.euclidean_distance(sfp1_bands[:, i], sfp2_bands[:, i]) for i in range(sfp1_bands.shape[1])]
        euclidean_distances=normalize_list(euclidean_distances)
        cosine_distances = [brain_signal_similarity.cosine_similarity(sfp1_bands[:, i], sfp2_bands[:, i]) for i in range(sfp1_bands.shape[1])]
        mfccs_sfp1 = librosa.feature.mfcc(y=sfp1, sr=fs, n_mfcc=13)
        mfccs_sfp1_flat=np.mean(mfccs_sfp1, axis=1)
        mfccs_sfp2 = librosa.feature.mfcc(y=sfp2, sr=fs, n_mfcc=13)
        mfccs_sfp2_flat=np.mean(mfccs_sfp2, axis=1)    
        features.append((euclidean_distances+cosine_distances+mfccs_sfp1_flat.tolist()+mfccs_sfp2_flat.tolist()))
        # features.append((euclidean_distances+cosine_distances))
   
    return features

# Function to extract band features using FFT
def extract_band_features(signal, fs, window_size, overlap):
    n_samples = len(signal)
    step_size = window_size - overlap
    n_windows = int((n_samples - window_size) / step_size) + 1
    
    delta = (0.5, 4)  # Delta waves (0.5 - 4 Hz)
    theta = (4, 8)    # Theta waves (4 - 8 Hz)
    alpha1 = (8, 10)  # Alpha 1 waves (8 - 10 Hz)
    alpha2 = (10, 13) # Alpha 2 waves (10 - 13 Hz)
    beta = (13, 30)   # Beta waves (13 - 30 Hz)
    gamma = (30, 100) # Gamma waves (30 - 100 Hz)
    
    band_features = []
    
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        
        segment = signal[start:end]
        freqs = np.fft.fftfreq(len(segment), 1/fs)
        fft_values = fft(segment)
        
        delta_indices = np.where((freqs >= delta[0]) & (freqs <= delta[1]))[0]
        theta_indices = np.where((freqs >= theta[0]) & (freqs <= theta[1]))[0]
        alpha1_indices = np.where((freqs >= alpha1[0]) & (freqs <= alpha1[1]))[0]
        alpha2_indices = np.where((freqs >= alpha2[0]) & (freqs <= alpha2[1]))[0]
        beta_indices = np.where((freqs >= beta[0]) & (freqs <= beta[1]))[0]
        gamma_indices = np.where((freqs >= gamma[0]) & (freqs <= gamma[1]))[0]
        
        delta_amplitude = np.sum(np.abs(fft_values[delta_indices]))
        theta_amplitude = np.sum(np.abs(fft_values[theta_indices]))
        alpha1_amplitude = np.sum(np.abs(fft_values[alpha1_indices]))
        alpha2_amplitude = np.sum(np.abs(fft_values[alpha2_indices]))
        beta_amplitude = np.sum(np.abs(fft_values[beta_indices]))
        gamma_amplitude = np.sum(np.abs(fft_values[gamma_indices]))
        
        band_features.append((delta_amplitude, theta_amplitude, alpha1_amplitude, alpha2_amplitude, beta_amplitude, gamma_amplitude))
    
    return band_features

# Function to process each file
def process_file(file_path):
    # directory, filename = os.path.split(file_path)
    dir_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    df = pd.read_csv(file_path)
    fp1 = df['FP1'].values
    fp2 = df['FP2'].values
    features = extract_features(fp1, fp2)
    label = None
    if '-salt' in file_path:
        label = 'salt'
    elif '-water' in file_path:
        label = 'water'
    elif '-lemon' in file_path:
        label = 'sour'
    elif '-melon' in file_path:
        label = 'bitter'
    elif '-sugar' in file_path:
        label = 'sweet'
    return [(feat, label, dir_name) for feat in features]

# Main function to iterate through folders and files
def main(root_folder, output_file):
    all_samples = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                samples = process_file(file_path)
                all_samples.extend(samples)
    
    df = pd.DataFrame(all_samples, columns=['features', 'label', 'directory'])
    # df[['cosine_similarity', 'euclidean_distance']] = df['features'].apply(lambda x: pd.Series(calculate_distances(x)))
    # df.drop(columns=['features'], inplace=True)
    df.to_csv(output_file, index=False)

# Function to calculate cosine and euclidean distances
def calculate_distances(features):
    cosine_dists = []
    euclidean_dists = []
    for sfp1_bands, sfp2_bands in features:
        cosine_dists.append(cosine_similarity([sfp1_bands], [sfp2_bands])[0][0])
        euclidean_dists.append(euclidean_distances([sfp1_bands], [sfp2_bands])[0][0])
    return (cosine_dists, euclidean_dists)

# Example usage:
root_folder = r'C:\Users\oler9\OneDrive\Documents\UC\24Spring\ELEC 3520 IoT\EUReCA - dietary eeg\Mixed Data'
output_file = 'mixed_data.csv'
main(root_folder,root_folder + output_file)
