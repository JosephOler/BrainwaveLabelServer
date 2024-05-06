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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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


### Standardize features for PCA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(selected_features)

### Find best PCA
n_list = range(1, k)
best_accuracy = 0
best_n = 0
for n in n_list:
    pca = PCA(n_components=n)
    features_pca = pca.fit_transform(features_scaled)

    # Cross-validation to find optimal n
    scores = cross_val_score(classifier, features_pca, labels, cv=5)
    avg_accuracy = np.mean(scores)

    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_n = n

print(f"The best number of components is {best_n} with {best_accuracy*100:.2f}% accuracy")


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=0.3, random_state=42)

### Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

############### PCA ###################
# Apply PCA to reduce dimensionality
pca = PCA(n_components=best_n)  # You can choose the number of components based on your requirement
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# Train the classifier
classifier.fit(X_train_scaled, y_train)
# Predict on the test data
y_pred = classifier.predict(X_test_scaled)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix for Random Forest
disp_rf = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=np.unique(y_test))
fig_rf, ax_rf = plt.subplots(figsize=(8, 6))
disp_rf.plot(ax=ax_rf)
ax_rf.set_title("Random Forest Confusion Matrix for Test Data")
ax_rf.set_xlabel('Predicted')
ax_rf.set_ylabel('True')
plt.show()

# Save the trained classifier
joblib.dump(classifier, 'RF_classifier.joblib')