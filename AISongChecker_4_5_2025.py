import librosa
import numpy as np
import pandas as pd
import os
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time

# Define base output folder
base_output_folder = "/outputfolder"
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M%p")
output_folder = os.path.join(base_output_folder, timestamp)
os.makedirs(output_folder, exist_ok=True)
print(f"Saving all outputs to: {output_folder}")

# Core feature extractor function
def extract_audio_features(file_path_label):
    file_path, label = file_path_label
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Fundamental Frequency (Pitch)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        mean_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        std_pitch = np.std(pitch_values) if len(pitch_values) > 0 else 0

        # Jitter
        jitter = np.mean(np.abs(np.diff(pitch_values))) if len(pitch_values) > 1 else 0

        # Shimmer
        frame_amplitudes = librosa.feature.rms(y=y)[0]
        shimmer = np.mean(np.abs(np.diff(frame_amplitudes)))

        # Harmonics-to-Noise Ratio
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = np.mean(harmonic) / (np.mean(percussive) + 1e-6)

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Zero-Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # Spectral Flux
        spectral_flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr))

        # Dynamic Range
        dynamic_range = np.max(frame_amplitudes) - np.min(frame_amplitudes)

        # Onset Rate
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        onset_rate = len(onsets) / (len(y)/sr)

        # Key + Mode Estimation
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_index = np.argmax(chroma_mean)
        key = key_index
        mode = 1 if chroma_mean[key_index] > np.median(chroma_mean) else 0

        features = {
            'mean_pitch': mean_pitch,
            'std_pitch': std_pitch,
            'jitter': jitter,
            'shimmer': shimmer,
            'hnr': hnr,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_contrast': spectral_contrast,
            'spectral_flatness': spectral_flatness,
            'tempo': tempo,
            'spectral_flux': spectral_flux,
            'zero_crossing_rate': zcr,
            'dynamic_range': dynamic_range,
            'onset_rate': onset_rate,
            'key': key,
            'mode': mode
        }

        for i, (mean_val, std_val) in enumerate(zip(mfccs_mean, mfccs_std)):
            features[f'mfcc_{i+1}_mean'] = mean_val
            features[f'mfcc_{i+1}_std'] = std_val

        features['label'] = label
        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Batch extractor
def batch_extract(folder_path, label):
    file_paths = [(os.path.join(folder_path, file_name), label) for file_name in os.listdir(folder_path) if file_name.endswith((".wav", ".mp3"))]
    
    start_time = time.time()
    print(f"Starting extraction for label {label} with {len(file_paths)} files...")

    with Pool(cpu_count()) as pool:
        results = []
        for i, res in enumerate(tqdm(pool.imap(extract_audio_features, file_paths), total=len(file_paths), desc=f"Extracting label {label}", dynamic_ncols=True)):
            if res is not None:
                results.append(res)

            if (i + 1) % 100 == 0 or (i + 1) == len(file_paths):
                elapsed = time.time() - start_time
                files_done = i + 1
                time_per_file = elapsed / files_done
                files_left = len(file_paths) - files_done
                est_remaining = files_left * time_per_file
                print(f"Processed {files_done}/{len(file_paths)} files. Est. time left: {est_remaining/60:.1f} min ({est_remaining:.0f} sec)")

    return pd.DataFrame(results)

if __name__ == "__main__":
    human_folder = "/pathtohumanfolder"
    ai_folder = "/pathtoaifolder"

    overall_start_time = time.time()
    runlog_lines = []
    runlog_lines.append(f"Run started at: {datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")

    print("Starting feature extraction...")
    human_df = batch_extract(human_folder, label=0)
    ai_df = batch_extract(ai_folder, label=1)

    full_df = pd.concat([human_df, ai_df], ignore_index=True)

    # Save features to CSV
    full_df.to_csv(os.path.join(output_folder, "extracted_features.csv"), index=False)
    print("Extracted features saved.")

    X = full_df.drop(columns=['label'])
    y = full_df['label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    X_test_with_preds = X_test.copy()
    X_test_with_preds['true_label'] = y_test.values
    X_test_with_preds['predicted_label'] = y_pred
    X_test_with_preds.to_csv(os.path.join(output_folder, 'test_predictions_with_labels.csv'), index=False)

    feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
    top_features = feature_importances.sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title("Top 10 Important Features")
    plt.xlabel("Relative Importance (unitless)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "feature_importance.png"))
    plt.show()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', alpha=0.6)
    plt.title("2D PCA of Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Label', labels=['Human', 'AI'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "pca_scatterplot.png"))
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Human', 'AI'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
    plt.show()

    top_feature_name = feature_importances.sort_values(ascending=False).index[0]
    print(f"Top important feature: {top_feature_name}")

    plt.figure(figsize=(10,6))
    sns.histplot(data=full_df, x=top_feature_name, hue='label', bins=30, kde=False, palette='coolwarm', alpha=0.7)
    plt.title(f"Histogram of {top_feature_name} (Human vs AI)")
    plt.xlabel(top_feature_name)
    plt.ylabel("Count")
    plt.legend(title='Label', labels=['Human', 'AI'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"histogram_{top_feature_name}.png"))
    plt.show()

    plt.figure(figsize=(10,6))
    sns.kdeplot(data=full_df, x=top_feature_name, hue='label', fill=True, common_norm=False, palette='coolwarm', alpha=0.5)
    plt.title(f"PDF of {top_feature_name} (Human vs AI)")
    plt.xlabel(top_feature_name)
    plt.ylabel("Density")
    plt.legend(title='Label', labels=['Human', 'AI'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"pdf_{top_feature_name}.png"))
    plt.show()

    plt.figure(figsize=(14,12))
    corr = X.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False, linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
    plt.show()

    y_proba = clf.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "roc_curve.png"))
    plt.show()

    # Save overall timing to runlog.txt
    overall_end_time = time.time()
    elapsed_total = overall_end_time - overall_start_time

    runlog_lines.append(f"Run ended at: {datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
    runlog_lines.append(f"Total runtime: {elapsed_total/60:.2f} minutes ({elapsed_total:.0f} seconds)")
    runlog_lines.append(f"Total files processed: {len(full_df)}")
    runlog_lines.append(f"Approx. avg time per file: {elapsed_total/len(full_df):.2f} seconds")

    with open(os.path.join(output_folder, "runlog.txt"), "w") as f:
        for line in runlog_lines:
            f.write(line + "\n")
