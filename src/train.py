import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             roc_auc_score,
                             f1_score,
                             ConfusionMatrixDisplay,
                             confusion_matrix,
                             classification_report)
import matplotlib.pyplot as plt
import pathlib, tempfile
import pandas as pd
import joblib
import os
import logging

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
logging.basicConfig(filename='logs/predictions.log', level=logging.INFO)


logging.info("Reading the dataset.")
df = pd.read_csv('data/iris.csv')
X = df.drop("target", axis=1)
y = df['target']
logging.info("Splitting the dataset.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "Iris-Classifier")
mlflow.set_experiment(EXPERIMENT_NAME)


def train_model(model, model_name):
    logging.info("Training the model.")
    model.fit(X_train, y_train)
    logging.info("Creating predictions.")
    y_pred = model.predict(X_test)
    logging.info("Calculating accuracy score.")
    accuracy = accuracy_score(y_test, y_pred)
    logging.info("Calculating precision score.")
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    logging.info("Calculating recall score.")
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr") if y_proba is not None else None
    logging.info("Calculating f1 score.")
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("Model_Type", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        if roc_auc is not None:
            mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(model, model_name)
        joblib.dump(model, f"models/{model_name}.pkl")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(values_format="d")
        pathlib.Path("artifacts").mkdir(exist_ok=True)
        cm_path = f"artifacts/{model_name}_cm.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cm_path)

        with open(f"artifacts/{model_name}_report.txt", "w") as f:
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact(f"artifacts/{model_name}_report.txt")

    logging.info(f"{model_name}: \nAccuracy: {accuracy} \nPrecision: {precision}, \nRecall: {recall}, "
                 f"\nROC AUC: {roc_auc}, \nF1 score: {f1}")


if __name__ == "__main__":
    train_model(LogisticRegression(max_iter=200), "LogisticRegression")
    train_model(RandomForestClassifier(n_estimators=200, random_state=42), "RandomForestClassifier")
    results = [("LogisticRegression", "models/LogisticRegression.pkl"),
               ("RandomForestClassifier", "models/RandomForestClassifier.pkl")]


    best_name, best_acc, best_path = None, -1, None
    for name, path in results:
        m = joblib.load(path)
        acc = accuracy_score(y_test, m.predict(X_test))
        if acc > best_acc:
            best_acc, best_name, best_path = acc, name, path

    joblib.dump(joblib.load(best_path), "models/best_model.pkl")
    logging.info(f"Best model: {best_name} (acc={best_acc:.4f}) -> models/best_model.pkl")

