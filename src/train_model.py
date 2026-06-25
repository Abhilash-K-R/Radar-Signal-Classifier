import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


def load_dataset(path="../data/dataset.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y


def train_and_evaluate():
    X, y = load_dataset()

    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features — SVM and KNN are sensitive to feature magnitude differences
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        # Random Forest doesn't need scaling, but it doesn't hurt it either
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average="weighted")
        rec = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        }
        trained_models[name] = model

        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

    # Save the best model + scaler for later use in predictor.py / dashboard
    best_model_name = max(results, key=lambda name: results[name]["accuracy"])
    best_model = trained_models[best_model_name]

    os.makedirs("../results", exist_ok=True)
    joblib.dump(best_model, "../results/best_model.pkl")
    joblib.dump(scaler, "../results/scaler.pkl")

    print(f"\nBest model: {best_model_name} (saved to results/best_model.pkl)")

    return results, trained_models, X_test_scaled, y_test


if __name__ == "__main__":
    train_and_evaluate()