"""
============================================================
  Edge-AI Fire Hazard Detection System
  ML Training Script - Random Forest Classifier
  
  Author  : [Your Name]
  Project : Academic Minor Project
============================================================

  WHAT THIS SCRIPT DOES:
  1. Generates / loads sensor dataset
  2. Preprocesses and splits data
  3. Trains Random Forest model
  4. Evaluates with accuracy, F1, confusion matrix
  5. Saves trained model as .pkl for Flask API

  INSTALL DEPENDENCIES:
  pip install scikit-learn pandas numpy matplotlib seaborn joblib
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import (accuracy_score, classification_report,
                                      confusion_matrix, f1_score)

# ============================================================
#  CONFIG
# ============================================================
RANDOM_SEED   = 42
MODEL_PATH    = "model/fire_hazard_model.pkl"
SCALER_PATH   = "model/scaler.pkl"
ENCODER_PATH  = "model/label_encoder.pkl"
DATASET_PATH  = "data/sensor_data.csv"
N_SAMPLES     = 3000   # Synthetic samples to generate

os.makedirs("model", exist_ok=True)
os.makedirs("data",  exist_ok=True)

np.random.seed(RANDOM_SEED)

# ============================================================
#  STEP 1: GENERATE SYNTHETIC DATASET
#  (Replace with real sensor CSV if available)
# ============================================================
def generate_dataset(n=N_SAMPLES):
    """
    Simulates realistic IoT sensor readings for 3 fire risk classes.
    In real deployment, replace this with actual ESP32 sensor logs.
    """
    print("[DATA] Generating synthetic sensor dataset...")

    records = []

    # --- SAFE conditions (60% of data) ---
    n_safe = int(n * 0.60)
    records += [{
        "temperature" : np.random.uniform(15, 38),
        "humidity"    : np.random.uniform(45, 90),
        "gas_level"   : np.random.uniform(30, 150),
        "label"       : "SAFE"
    } for _ in range(n_safe)]

    # --- WARNING conditions (25% of data) ---
    n_warn = int(n * 0.25)
    records += [{
        "temperature" : np.random.uniform(38, 65),
        "humidity"    : np.random.uniform(20, 45),
        "gas_level"   : np.random.uniform(150, 450),
        "label"       : "WARNING"
    } for _ in range(n_warn)]

    # --- FIRE conditions (15% of data) ---
    n_fire = int(n * 0.15)
    records += [{
        "temperature" : np.random.uniform(65, 100),
        "humidity"    : np.random.uniform(5, 22),
        "gas_level"   : np.random.uniform(450, 1000),
        "label"       : "FIRE"
    } for _ in range(n_fire)]

    df = pd.DataFrame(records).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"[DATA] Dataset saved -> {DATASET_PATH}")
    print(f"[DATA] Shape: {df.shape}")
    print(f"[DATA] Class distribution:\n{df['label'].value_counts()}\n")
    return df


# ============================================================
#  STEP 2: FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    """
    Add derived features that improve model performance.
    These same features must be computed in Flask API at inference time.
    """
    df = df.copy()

    # Composite risk index (domain knowledge)
    df["temp_gas_index"]  = df["temperature"] * df["gas_level"] / 1000
    df["hum_inverse"]     = 100 - df["humidity"]
    df["combined_risk"]   = (df["temperature"] / 100) + (df["hum_inverse"] / 100) + (df["gas_level"] / 1000)

    return df


# ============================================================
#  STEP 3: PREPROCESS
# ============================================================
def preprocess(df):
    df = engineer_features(df)

    feature_cols = ["temperature", "humidity", "gas_level",
                    "temp_gas_index", "hum_inverse", "combined_risk"]

    X = df[feature_cols].values
    y = df["label"].values

    # Encode labels: SAFE=2, WARNING=1, FIRE=0
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_enc, le, scaler, feature_cols


# ============================================================
#  STEP 4: TRAIN MODEL
# ============================================================
def train_model(X_train, y_train):
    print("[MODEL] Training Random Forest Classifier...")

    model = RandomForestClassifier(
        n_estimators     = 100,
        max_depth        = 12,
        min_samples_leaf = 2,
        class_weight     = "balanced",  # handles class imbalance
        random_state     = RANDOM_SEED,
        n_jobs           = -1
    )
    model.fit(X_train, y_train)
    print("[MODEL] Training complete!\n")
    return model


# ============================================================
#  STEP 5: EVALUATE
# ============================================================
def evaluate(model, X_test, y_test, le):
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    cm   = confusion_matrix(y_test, y_pred)
    cr   = classification_report(y_test, y_pred, target_names=le.classes_)

    print("=" * 50)
    print(f"  ACCURACY  : {acc * 100:.2f}%")
    print(f"  F1 SCORE  : {f1 * 100:.2f}%")
    print("=" * 50)
    print("\n[REPORT]\n", cr)

    # Cross-validation
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring="accuracy")
    print(f"[CV] 5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%\n")

    return y_pred, cm


# ============================================================
#  STEP 6: VISUALIZATIONS
# ============================================================
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=classes, yticklabels=classes,
                linewidths=1, linecolor="white")
    plt.title("Confusion Matrix — Fire Hazard Detection", fontsize=14, pad=15)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("model/confusion_matrix.png", dpi=150)
    plt.show()
    print("[PLOT] Saved -> model/confusion_matrix.png")


def plot_feature_importance(model, feature_cols):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(feature_cols)),
            importances[indices],
            color=["#ff4444", "#ff8800", "#ffcc00", "#00cc88", "#00aaff", "#aa44ff"])
    plt.xticks(range(len(feature_cols)),
               [feature_cols[i] for i in indices],
               rotation=25, ha="right")
    plt.title("Feature Importance — Random Forest", fontsize=13)
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("model/feature_importance.png", dpi=150)
    plt.show()
    print("[PLOT] Saved -> model/feature_importance.png")


def plot_class_distribution(df):
    counts = df["label"].value_counts()
    colors = {"SAFE": "#00cc88", "WARNING": "#ffaa00", "FIRE": "#ff3333"}

    plt.figure(figsize=(5, 4))
    plt.bar(counts.index, counts.values,
            color=[colors[c] for c in counts.index], edgecolor="white")
    plt.title("Dataset Class Distribution")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig("model/class_distribution.png", dpi=150)
    plt.show()
    print("[PLOT] Saved -> model/class_distribution.png")


# ============================================================
#  STEP 7: SAVE MODEL
# ============================================================
def save_model(model, scaler, le):
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le,     ENCODER_PATH)
    print(f"\n[SAVE] Model   -> {MODEL_PATH}")
    print(f"[SAVE] Scaler  -> {SCALER_PATH}")
    print(f"[SAVE] Encoder -> {ENCODER_PATH}")


# ============================================================
#  STEP 8: SINGLE PREDICTION TEST
# ============================================================
def test_single_prediction(model, scaler, le):
    """Test the saved model with a few manual inputs."""
    print("\n[TEST] Running sample predictions...\n")

    samples = [
        {"temperature": 28, "humidity": 65, "gas_level": 100,  "expected": "SAFE"},
        {"temperature": 52, "humidity": 30, "gas_level": 300,  "expected": "WARNING"},
        {"temperature": 85, "humidity": 12, "gas_level": 720,  "expected": "FIRE"},
    ]

    for s in samples:
        t, h, g = s["temperature"], s["humidity"], s["gas_level"]
        tgi = t * g / 1000
        hi  = 100 - h
        cr  = (t / 100) + (hi / 100) + (g / 1000)

        features = np.array([[t, h, g, tgi, hi, cr]])
        features_scaled = scaler.transform(features)

        pred_enc   = model.predict(features_scaled)[0]
        pred_label = le.inverse_transform([pred_enc])[0]
        proba      = model.predict_proba(features_scaled)[0].max() * 100

        status = "✅" if pred_label == s["expected"] else "❌"
        print(f"  {status} Temp={t}°C | Hum={h}% | Gas={g}ppm "
              f"→ Predicted: {pred_label} ({proba:.1f}%) | Expected: {s['expected']}")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  EDGE-AI FIRE HAZARD DETECTION — ML TRAINING")
    print("=" * 55 + "\n")

    # 1. Load or generate dataset
    if os.path.exists(DATASET_PATH):
        print(f"[DATA] Loading existing dataset from {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
    else:
        df = generate_dataset()

    # 2. Visualize distribution
    plot_class_distribution(df)

    # 3. Preprocess
    X, y, le, scaler, feature_cols = preprocess(df)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"[SPLIT] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")

    # 5. Train
    model = train_model(X_train, y_train)

    # 6. Evaluate
    y_pred, cm = evaluate(model, X_test, y_test, le)

    # 7. Plots
    plot_confusion_matrix(cm, le.classes_)
    plot_feature_importance(model, feature_cols)

    # 8. Save
    save_model(model, scaler, le)

    # 9. Quick test
    test_single_prediction(model, scaler, le)

    print("\n[DONE] Training pipeline complete! [OK]")
    print("[NEXT] Run backend/app.py to start the Flask API.\n")
