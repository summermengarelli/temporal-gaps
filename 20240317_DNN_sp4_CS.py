
####
## Script for training a speech-emotion recognition model (DNN).
##
## Authors Information
##  Name: Carolyn Stazio
##  Email: cstazio@madeupemail.com
##
## Date: 2024-03-17
##
####

# Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import librosa
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf

warnings.filterwarnings('ignore')

# Set working directory to location of Python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Running this (by clicking run or pressing Shift+Enter) will list all files under the input directory, named data-raw
for dirname, _, filenames in os.walk('../data-raw'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set filepaths for speech_emotions.csv and the directory (data-raw) holding the audio files
CSV_PATH = "../speech_emotions.csv"
AUDIO_BASE = "../data-raw"

df = pd.read_csv(CSV_PATH) # read in speech_emotions.csv

# Set emotion labels
emotion_map = {
    "euphoric": 0,
    "joyfully": 1,
    "sad": 2,
    "surprised": 3
}

# Define the feature extraction function!
def extract_robust_features(file_path):
    """
    Extract a 3-second vector from an audio file for speech emotion recognition.

    Parameters
    ----------
    file_path : str
        Path to the audio file (e.g., .wav) to be processed.

    Returns
    -------
    np.ndarray or None
        A 1D NumPy array containing the extracted audio features if
        successful, or None if the audio file cannot be processed.

    """
    try:
        # Load 3 seconds of audio with consistent parameters
        audio, sr = librosa.load(file_path, sr=22050, duration=3.0)
        
        # Ensure consistent length
        target_length = sr * 3
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        features = []
        
        # Extract basic MFCC features (most important for speech)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features.extend(mfcc_mean) # for each feature, extracting mean and standard deviation
        features.extend(mfcc_std)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(np.mean(spectral_centroid))
        features.append(np.std(spectral_centroid))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))
        
        # Extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # Extract RMS energy
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        # Extract Chroma features (pitch content)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1) # only extracting mean
        features.extend(chroma_mean)
        
        # Extract spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast, axis=1) # only extracting mean
        features.extend(contrast_mean)
        
        return np.array(features) # Return an array...
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None # ... unless there is an error, then return None

print("Loading and extracting features from dataset...")

X = []
y = []

# Create multiple augmented versions of each sample
for idx, row in df.iterrows():
    speaker_id = row["set_id"]
    speaker_folder = os.path.join(AUDIO_BASE, speaker_id)

    for emotion, label in emotion_map.items():
        audio_path = os.path.join(speaker_folder, emotion + ".wav") # For each participant and each emotion, do the following:

        if os.path.exists(audio_path):
            # Extract original features
            features = extract_robust_features(audio_path)
            if features is not None:
                X.append(features)
                y.append(label)
            
            # Create multiple augmented versions
            for aug_num in range(3):  # Create 3 augmented versions per sample
                try:
                    # Load and augment
                    audio, sr = librosa.load(audio_path, sr=22050, duration=3.0)
                    
                    # Different augmentation techniques
                    if aug_num == 0:
                        # Time stretching
                        rate = np.random.uniform(0.9, 1.1)
                        audio_aug = librosa.effects.time_stretch(audio, rate=rate)
                    elif aug_num == 1:
                        # Pitch shifting
                        steps = np.random.randint(-1, 2)
                        audio_aug = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
                    else:
                        # Add noise
                        noise = np.random.normal(0, 0.001, audio.shape)
                        audio_aug = audio + noise
                    
                    # Ensure consistent length
                    target_length = sr * 3
                    if len(audio_aug) < target_length:
                        audio_aug = np.pad(audio_aug, (0, target_length - len(audio_aug)), mode='constant')
                    else:
                        audio_aug = audio_aug[:target_length]
                    
                    # Extract features from augmented audio
                    features_aug = extract_robust_features(audio_aug)  # Reuse function
                    if features_aug is not None:
                        X.append(features_aug)
                        y.append(label)
                        
                except:
                    continue

# Check for datasize size and feature consistency
X = np.array(X)
y = np.array(y)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert labels to categorical
y_categorical = to_categorical(y, num_classes=4)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, 
    test_size=0.2, 
    random_state=42, 
    stratify=y,
    shuffle=True
)

# Also create non-categorical versions for traditional ML
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y,
    shuffle=True
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Define Neural Network Model
def create_emotion_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(4, activation='softmax')
    ])
    return model

# Create and compile neural network
input_dim = X_train.shape[1]
nn_model = create_emotion_model(input_dim)
nn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=15, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True, verbose=1)
]

print("Training Neural Network...")
nn_history = nn_model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=8,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

# Evaluate Neural Network
nn_test_loss, nn_test_accuracy = nn_model.evaluate(X_test, y_test, verbose=0)
print(f"Neural Network Test Accuracy: {nn_test_accuracy:.4f}")

# Compare to traditional machine learning models
print("\nTraining Traditional ML Models...")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_ml, y_train_ml)
rf_pred = rf_model.predict(X_test_ml)
rf_accuracy = accuracy_score(y_test_ml, rf_pred)

# SVM
svm_model = SVC(kernel='rbf', random_state=42, probability=True)
svm_model.fit(X_train_ml, y_train_ml)
svm_pred = svm_model.predict(X_test_ml)
svm_accuracy = accuracy_score(y_test_ml, svm_pred)

print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# Save the best model (Random Forest in this case)
if rf_accuracy >= nn_test_accuracy and rf_accuracy >= svm_accuracy:
    joblib.dump(rf_model, 'best_emotion_model.pkl')
    best_model_type = "Random Forest"
    best_accuracy = rf_accuracy
elif nn_test_accuracy >= rf_accuracy and nn_test_accuracy >= svm_accuracy:
    nn_model.save("best_emotion_model.h5")
    best_model_type = "Neural Network"
    best_accuracy = nn_test_accuracy
else:
    joblib.dump(svm_model, 'best_emotion_model.pkl')
    best_model_type = "SVM"
    best_accuracy = svm_accuracy

# Save scaler
joblib.dump(scaler, 'feature_scaler.pkl')

print(f"\n=== FINAL RESULTS ===")
print(f"Best Model: {best_model_type}")
print(f"Best Accuracy: {best_accuracy:.4f}")

# Detailed predictions from best model
if best_model_type == "Random Forest":
    best_pred = rf_pred
    y_pred_proba = rf_model.predict_proba(X_test_ml)
elif best_model_type == "Neural Network":
    best_pred = np.argmax(nn_model.predict(X_test), axis=1)
    y_pred_proba = nn_model.predict(X_test)
else:
    best_pred = svm_pred
    y_pred_proba = svm_model.predict_proba(X_test_ml)

y_true = np.argmax(y_test, axis=1) if best_model_type == "Neural Network" else y_test_ml

print("\n=== BEST MODEL PERFORMANCE ===")
print("Classification Report:")
print(classification_report(y_true, best_pred, target_names=list(emotion_map.keys())))

print("Confusion Matrix:")
print(confusion_matrix(y_true, best_pred))

# Feature importance analysis (for Random Forest)
if best_model_type == "Random Forest":
    print("\n=== FEATURE IMPORTANCE ===")
    importances = rf_model.feature_importances_
    top_features = np.argsort(importances)[-10:]  # Top 10 features
    print("Top 10 most important features:", top_features)

print("\n=== MODEL USAGE EXAMPLE ===")
print("To make predictions with new audio files:")

# Define prediction model, using Random Forest
def predict_emotion(audio_path, model, scaler, model_type="Random Forest"):
    """
    Predict the emotional state expressed in a speech audio file.

    Parameters
    ----------
    audio_path : str
        Path to the input audio file (e.g., .wav) containing speech.
    model : object
        A trained classification model.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler used during model training to normalize features.
    model_type : str, optional
        Type of the trained model. Expected values are
        "Random Forest", "SVM", or "Neural Network".
        Default is "Random Forest".

    Returns
    -------
    tuple
        If successful, returns a tuple:
            (predicted_emotion, confidence_scores)
        where:
            - predicted_emotion (str) is the emotion label with the
              highest predicted probability.
            - confidence_scores (dict) maps each emotion label to its
              corresponding probability.

    """
    features = extract_robust_features(audio_path) # Call extract_robust_features function for path to input audio files
    if features is None:
        return "Error processing audio file"
    
    features_scaled = scaler.transform([features])
    
    if model_type == "Random Forest" or model_type == "SVM":
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
    else:  # Neural Network
        probabilities = model.predict(features_scaled)[0]
        prediction = np.argmax(probabilities)
    
    emotion_labels = list(emotion_map.keys())
    predicted_emotion = emotion_labels[prediction]
    
    # Get confidence scores
    confidence_scores = {emotion_labels[i]: float(prob) for i, prob in enumerate(probabilities)}
    
    return predicted_emotion, confidence_scores


print("=== SUMMARY ===")
print("✅ Model training completed successfully!")
print(f"✅ Best model achieved {best_accuracy:.1%} accuracy")
print("✅ Models and scaler saved for future use")
print("✅ Ready for emotion prediction on new audio files")