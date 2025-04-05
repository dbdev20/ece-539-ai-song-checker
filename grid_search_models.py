import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# -------------------------------
# Audio Feature Extraction
# -------------------------------
def extract_audio_features(file_path_label):
    file_path, label = file_path_label
    try:
        y, sr = librosa.load(file_path, sr=None)

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        mean_pitch = np.mean(pitch_values) if pitch_values.size > 0 else 0
        std_pitch = np.std(pitch_values) if pitch_values.size > 0 else 0
        jitter = np.mean(np.abs(np.diff(pitch_values))) if pitch_values.size > 1 else 0
        frame_amplitudes = librosa.feature.rms(y=y)[0]
        shimmer = np.mean(np.abs(np.diff(frame_amplitudes)))
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = np.mean(harmonic) / (np.mean(percussive) + 1e-6)
        spectral_flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

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

def batch_extract(folder_path, label):
    file_paths = [(os.path.join(folder_path, f), label)
                  for f in os.listdir(folder_path) if f.endswith((".wav", ".mp3"))]
    with Pool(cpu_count()) as pool:
        features_list = list(tqdm(pool.imap(extract_audio_features, file_paths), total=len(file_paths)))
    return pd.DataFrame([f for f in features_list if f is not None])

# -------------------------------
# Model Training and Evaluation
# -------------------------------
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier  # requires `pip install xgboost`

def train_models(X_train, y_train):
    models = {
        'RandomForest': (RandomForestClassifier(), {
            'n_estimators': [100, 200],
            'max_depth': [None, 10]
        }),
        'ExtraTrees': (ExtraTreesClassifier(), {
            'n_estimators': [100, 200],
            'max_depth': [None, 10]
        }),
        'GradientBoosting': (GradientBoostingClassifier(), {
            'n_estimators': [100],
            'learning_rate': [0.1, 0.05]
        }),
        'AdaBoost': (AdaBoostClassifier(), {
            'n_estimators': [50, 100]
        }),
        'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
            'n_estimators': [100],
            'max_depth': [3, 5],
            'learning_rate': [0.1]
        }),
        'DecisionTree': (DecisionTreeClassifier(), {
            'max_depth': [None, 10]
        }),
        'KNN': (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7]
        }),
        'SVM': (SVC(), {
            'C': [1, 10],
            'kernel': ['linear', 'rbf']
        }),
        'LogisticRegression': (LogisticRegression(max_iter=1000), {
            'C': [1.0, 10.0]
        }),
        'RidgeClassifier': (RidgeClassifier(), {
            'alpha': [1.0, 0.1]
        }),
        'NaiveBayes': (GaussianNB(), {
            # No hyperparameters to tune for GaussianNB
        }),
        'MLP': (MLPClassifier(max_iter=500), {
            'hidden_layer_sizes': [(100,), (50, 50)],
            'alpha': [0.0001, 0.001]
        }),
    }

    best_estimators = {}

    for name, (model, params) in models.items():
        print(f"\n🔍 Tuning {name}...")
        if params:
            grid = GridSearchCV(model, params, cv=3, n_jobs=-1, scoring='accuracy')
        else:
            # No params: just fit the base model
            model.fit(X_train, y_train)
            best_estimators[name] = model
            print(f"{name} has no hyperparameters to search.")
            continue
        grid.fit(X_train, y_train)
        print(f"✅ Best params for {name}: {grid.best_params_}")
        best_estimators[name] = grid.best_estimator_

    # Optional: Voting Classifier Ensemble
    voting = VotingClassifier(
        estimators=[(name, clf) for name, clf in best_estimators.items()],
        voting='soft'
    )
    voting.fit(X_train, y_train)
    best_estimators['VotingEnsemble'] = voting

    return best_estimators

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# -------------------------------
# Visualization
# -------------------------------
def plot_feature_importance(model, X):
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X.columns)
        top = importances.sort_values(ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top.values, y=top.index)
        plt.title("Top 10 Feature Importances")
        plt.tight_layout()
        plt.show()

def plot_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', alpha=0.6)
    plt.title("2D PCA Projection")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Label', labels=['Human', 'AI'])
    plt.tight_layout()
    plt.show()

# -------------------------------
# Main Pipeline
# -------------------------------
def main():
    human_folder = "../539-dataset/real/real_with_genres"
    ai_folder = "../539-dataset/suno/suno_with_genres"

    print("Extracting features...")
    human_df = batch_extract(human_folder, label=0)
    ai_df = batch_extract(ai_folder, label=1)
    full_df = pd.concat([human_df, ai_df], ignore_index=True)
    full_df.to_csv("extracted_features.csv", index=False)
    print("Features saved to extracted_features.csv")

    X = full_df.drop(columns=['label'])
    y = full_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_models = train_models(X_train, y_train)

    for name, model in best_models.items():
        print(f"\nEvaluating {name}")
        evaluate_model(model, X_test, y_test)
        if name == "RandomForest":
            plot_feature_importance(model, X)
    
    plot_pca(X, y)

if __name__ == "__main__":
    main()

