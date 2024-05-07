# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 23:32:48 2024

@author: buing
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

############### Random Forest ###################
# Initialize the Random Forest classifier with custom parameters
classifier = RandomForestClassifier(n_estimators=1000, random_state=42)

# Read the CSV file
file = r'C:\Users\oler9\OneDrive\Documents\UC\24Spring\ELEC 3520 IoT\EUReCA - dietary eeg\ML\solo_data.csv'
data = pd.read_csv(file)


# Extract features and labels
features = data['features'].apply(lambda x: np.array(eval(x)))
labels = data['label']
# Convert features to numpy array
features = np.vstack(features)


# Compute mutual information scores
mi_scores = mutual_info_classif(features, labels)
# Get indices of top k features
k = 12  # Change k as needed
top_k_indices = mi_scores.argsort()[-k:][::-1]
# Get names of top k features
top_k_features = [f'Feature {i+1}' for i in top_k_indices]
# Select the most discriminative features (if not done already)
selected_features = features[:, top_k_indices]  # Assuming top_k_indices is already defined

### Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(selected_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42)

# Train the classifier
classifier.fit(X_train, y_train)
# Predict on the data
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)


train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Plot side-by-side confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot confusion matrix for training data
disp_train = ConfusionMatrixDisplay(confusion_matrix=train_cm, display_labels=np.unique(y_train))
disp_train.plot(ax=axes[0])
axes[0].set_title(f"Training Confusion Matrix\nAccuracy: {accuracy_score(y_train, y_train_pred)*100:.2f}%")

# Plot confusion matrix for testing data
disp_test = ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels=np.unique(y_test))
disp_test.plot(ax=axes[1])
axes[1].set_title(f"Testing Confusion Matrix\nAccuracy: {accuracy_score(y_test, y_test_pred)*100:.2f}%")

plt.tight_layout()
plt.show()

# Save the trained classifier
joblib.dump(classifier, 'RF_classifier.joblib')