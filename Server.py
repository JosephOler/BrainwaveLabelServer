from flask import Flask, request, redirect, render_template, send_from_directory
import numpy as np
import pandas as pd
import joblib
import librosa
import brain_signal_similarity
from scipy.fftpack import fft
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_selection import mutual_info_classif


ALLOWED_EXTENSIONS = {'csv'}
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'processed'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load the trained classifier
classifier = joblib.load('RF_classifier.joblib')

# min max normalization
def normalize_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    normalized_list = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_list

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
        euclidean_distances = normalize_list(euclidean_distances)
        cosine_distances = [brain_signal_similarity.cosine_similarity(sfp1_bands[:, i], sfp2_bands[:, i]) for i in range(sfp1_bands.shape[1])]
        mfccs_sfp1 = librosa.feature.mfcc(y=sfp1, sr=fs, n_mfcc=13)
        mfccs_sfp1_flat = np.mean(mfccs_sfp1, axis=1)
        mfccs_sfp2 = librosa.feature.mfcc(y=sfp2, sr=fs, n_mfcc=13)
        mfccs_sfp2_flat = np.mean(mfccs_sfp2, axis=1)    
        features.append((euclidean_distances + cosine_distances + mfccs_sfp1_flat.tolist() + mfccs_sfp2_flat.tolist()))
        # features.append((euclidean_distances+cosine_distances))
   
    return features

def process_file(file):
    df = pd.read_csv(file)
    fp1 = df['FP1'].values
    fp2 = df['FP2'].values
    features = extract_features(fp1, fp2)
    
    # Assume the classifier has a feature_importances_ attribute
    if hasattr(classifier, 'feature_importances_'):
        # Get feature importance scores from the trained classifier
        feature_importances = classifier.feature_importances_
        # Get indices of top k features
        k = 12  # Change k as needed
        top_k_indices = feature_importances.argsort()[-k:][::-1]
        # Select the top 12 features based on feature importance scores
        selected_features = np.array(features)[:, top_k_indices]
    else:
        # If the classifier doesn't have feature importances, select the first 12 features
        selected_features = np.array(features)[:, :12]
    
    return selected_features

# Define route to handle file upload
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # Process the uploaded file
        features = process_file(file)
        
        # Extract features for prediction
        X_pred = np.array(features)
        
        # Predict labels using the classifier
        predicted_labels = classifier.predict(X_pred)
        
        # Render the template with the predicted labels
        return render_template('template.html', filename=file.filename, predicted_labels=predicted_labels.tolist())
    else:
        return redirect(request.url)

# Define route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Define function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define route to render the template
@app.route('/')
def template():
    return render_template('template.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)