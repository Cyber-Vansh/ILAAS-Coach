import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans


def build_performance_labels(df: pd.DataFrame, target_column: str):
    df = df.copy()
    scores = pd.to_numeric(df[target_column], errors="coerce")
    mask = ~scores.isna()
    df = df.loc[mask].copy()
    scores = scores.loc[mask]
    if scores.nunique() <= 2:
        low = scores.min()
        high = scores.max()
    else:
        low, high = scores.quantile([0.33, 0.66])

    def label(value: float) -> str:
        if value <= low:
            return "At-risk"
        if value <= high:
            return "Average"
        return "High-performing"

    df["performance_label"] = scores.apply(label)
    thresholds = {"low": float(low), "high": float(high)}
    return df, thresholds


def preprocess_for_classification(df: pd.DataFrame, feature_columns, label_column: str):
    numeric_features = df[feature_columns].select_dtypes(include=[np.number]).astype(float)
    y = df[label_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler


def train_classifier(X_train, y_train):
    model = LogisticRegression(max_iter=1000, multi_class="auto")
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return accuracy, report


def classify_students(model, scaler, df: pd.DataFrame, feature_columns, label_column: str = "predicted_label"):
    numeric_features = df[feature_columns].select_dtypes(include=[np.number]).astype(float)
    X_scaled = scaler.transform(numeric_features)
    predictions = model.predict(X_scaled)
    confidences = model.predict_proba(X_scaled).max(axis=1)
    out_df = df.copy()
    out_df[label_column] = predictions
    out_df["prediction_confidence"] = confidences
    return out_df


def recommendation_for_label(label: str) -> str:
    if label == "At-risk":
        return "Focus on weakest topics, attend support sessions, and complete targeted practice this week."
    if label == "Average":
        return "Maintain regular study, review mistakes carefully, and allocate extra time to below-average topics."
    if label == "High-performing":
        return "Continue current strategy, attempt advanced problems, and explore enrichment resources."
    return "Review your recent performance and set a focused short-term study goal."


def add_recommendations(df: pd.DataFrame, label_column: str = "predicted_label"):
    out_df = df.copy()
    out_df["recommendation"] = out_df[label_column].apply(recommendation_for_label)
    return out_df


def cluster_learners(df: pd.DataFrame, feature_columns, n_clusters: int = 3, cluster_column: str = "cluster"):
    numeric_features = df[feature_columns].select_dtypes(include=[np.number]).astype(float)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(numeric_features)
    out_df = df.copy()
    out_df[cluster_column] = clusters
    return out_df, model

