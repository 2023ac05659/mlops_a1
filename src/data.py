from sklearn.datasets import load_iris
import os


def load_and_save_dataset(dataset_path="data/iris.csv"):
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['target'] = iris.target
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    df.to_csv(dataset_path, index=False)
    return df


if __name__ == "__main__":
    df = load_and_save_dataset()
    print(df.head())
