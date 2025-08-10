import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import pandas as pd
import joblib
import os


df = pd.read_csv('data/iris.csv')
X = df.drop("target", axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "Iris-Classifier")
mlflow.set_experiment(EXPERIMENT_NAME)


def train_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    y_proba = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr") if y_proba is not None else None
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

    print(f"{model_name}: \nAccuracy: {accuracy}") # \nPrecision: {precision}, \nRecall: {recall}, \nROC AUC: {roc_auc}, \nF1 score: {f1}")


if __name__ == "__main__":
    train_model(LogisticRegression(max_iter=200), "LogisticRegression")
    train_model(RandomForestClassifier(n_estimators=200, random_state=42), "RandomForestClassifier")

