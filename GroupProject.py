import librosa
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Core feature extractor function
def extract_audio_features(file_path_label):
    file_path, label = file_path_label
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Feature 1: Fundamental Frequency (Pitch)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        mean_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        std_pitch = np.std(pitch_values) if len(pitch_values) > 0 else 0

        # Feature 2: Jitter (variation in pitch)
        jitter = np.mean(np.abs(np.diff(pitch_values))) if len(pitch_values) > 1 else 0

        # Feature 3: Shimmer (variation in amplitude)
        frame_amplitudes = librosa.feature.rms(y=y)[0]
        shimmer = np.mean(np.abs(np.diff(frame_amplitudes)))

        # Feature 4: Harmonics-to-Noise Ratio (approximation)
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = np.mean(harmonic) / (np.mean(percussive) + 1e-6)

        # Feature 5: MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # Feature 6: Spectral Flux
        spectral_flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr))

        # Feature 7: Zero-Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        features = {
            'mean_pitch': mean_pitch,
            'std_pitch': std_pitch,
            'jitter': jitter,
            'shimmer': shimmer,
            'hnr': hnr,
            'spectral_flux': spectral_flux,
            'zero_crossing_rate': zcr
        }

        for i, (mean_val, std_val) in enumerate(zip(mfccs_mean, mfccs_std)):
            features[f'mfcc_{i+1}_mean'] = mean_val
            features[f'mfcc_{i+1}_std'] = std_val

        features['label'] = label
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Batch extractor for a folder with multiprocessing
def batch_extract(folder_path, label):
    file_paths = [(os.path.join(folder_path, file_name), label) for file_name in os.listdir(folder_path) if file_name.endswith((".wav", ".mp3"))]
    
    with Pool(cpu_count()) as pool:
        features_list = list(tqdm(pool.imap(extract_audio_features, file_paths), total=len(file_paths), desc=f"Extracting features for label {label}", dynamic_ncols=True))
    
    features_list = [f for f in features_list if f is not None]
    return pd.DataFrame(features_list)

# Example usage
if __name__ == "__main__":
    human_folder = "/Users/evanryser/Downloads/539-dataset/real/real_with_genres (2821)"
    ai_folder = "/Users/evanryser/Downloads/539-dataset/suno/suno_with_genres (2243)"

    print("Starting feature extraction...")
    human_df = batch_extract(human_folder, label=0)
    ai_df = batch_extract(ai_folder, label=1)

    full_df = pd.concat([human_df, ai_df], ignore_index=True)
    
    # Save features to CSV
    full_df.to_csv("extracted_features.csv", index=False)
    print("Extracted features saved to extracted_features.csv")

    X = full_df.drop(columns=['label'])
    y = full_df['label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Feature importance visualization
    feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title("Top 10 Important Features for AI vs Human Audio Classification")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # 2D PCA Scatterplot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', alpha=0.6)
    plt.title("2D PCA of Extracted Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Label', labels=['Human', 'AI'])
    plt.tight_layout()
    plt.show()
