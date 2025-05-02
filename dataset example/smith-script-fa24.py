import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

dataset_path = "/csmith/input/emotions-on-audio-dataset/files"
emotions = ["euphoric", "joyfully", "sad", "surprised"]
sample_rate = 22050
max_length = 128

def find_audio_files(dataset_dir, emotions):
    "Search for audio files inside dataset subdirectories"
    audio_files = {}
    
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                label = file.split(".")[0]  
                if label in emotions:
                    audio_files[label] = os.path.join(root, file)
    
    return audio_files

def plot_audios(audio_files):
    plt.figure(figsize=(12, 8))

    for i, (emotion, file_path) in enumerate(audio_files.items()):
        y, sr = librosa.load(file_path, sr=22050)

        # plota
        plt.subplot(4, 2, 2 * i + 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title(f"{emotion.capitalize()} - Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        # plotb
        plt.subplot(4, 2, 2 * i + 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"{emotion.capitalize()} - Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()

audio_files = find_audio_files(dataset_path, emotions)
plot_audios(audio_files)

emotion_features = {emotion: [] for emotion in emotions}

def extract_audio_features(file_path):
    """Extract MFCCs, Chroma, Spectral Contrast, and Zero-Crossing Rate."""
    y, sr = librosa.load(file_path, sr=22050)

    # MFCCs 
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Chroma 
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral 
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    return np.concatenate([mfccs_mean, chroma_mean, contrast_mean, [zcr_mean]])


for emotion in emotions:
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.startswith(emotion) and file.endswith(".wav"):
                file_path = os.path.join(root, file)
                feature_vector = extract_audio_features(file_path)
                emotion_features[emotion].append(feature_vector)

mean_emotion_features = {emotion: np.mean(np.array(features), axis=0) 
                         for emotion, features in emotion_features.items()}

for emotion, mean_vector in mean_emotion_features.items():
    print(f"Mean feature vector for {emotion}:")
    print(mean_vector)
    print("-" * 50)

mean_matrix = np.array(list(mean_emotion_features.values()))
feature_labels = [f"F{i}" for i in range(mean_matrix.shape[1])]  # Generic labels for features

plt.figure(figsize=(12, 6))

for idx, emotion in enumerate(emotions):
    plt.plot(feature_labels, mean_matrix[idx], label=emotion, marker='o')

plt.xlabel("Feature Index")
plt.ylabel("Mean Value")
plt.title("Mean Feature Distribution Across Emotions")
plt.legend()
plt.xticks(rotation=45)
plt.show()

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=sample_rate)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if mel_spec_db.shape[1] < max_length:
        pad_width = max_length - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_length]

    return mel_spec_db.T

def load_dataset():
    features, labels = [], []
    for emotion_idx, emotion in enumerate(emotions):
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.startswith(emotion) and file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(emotion_idx)

    return np.array(features), np.array(labels)

X, y = load_dataset()
y = to_categorical(y, num_classes=len(emotions))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(len(emotions), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")