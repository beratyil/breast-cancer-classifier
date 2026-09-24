import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_breast_cancer_dataset():
    # Load the breast cancer dataset
    data = load_breast_cancer()
    # Create a DataFrame from the dataset
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return df, data

def get_dataset_info(df, data):
    print("Breast Cancer Dataset Information:")
    print(df.head())
    print('Dataset description:')
    print(df.describe())
    print('Dataset info:')
    df.info()

    print("--------------------------------")
    print(data.target_names)
    print(data.target)
    print(data.feature_names)
    print(data.data.shape)
    print(data.target.shape)

    print("--------------------------------")
    print("Target variable distribution:")
    df['target'] = data.target
    print(df["target"].value_counts())
    print(df["target"].value_counts(normalize=True))
    print(df.shape)

    print("--------------------------------")
    print("Missing Values:")
    print(df.isnull().sum())
    print("NA Values:")
    print(df.isna().sum())

    print("--------------------------------")
    print(df.duplicated())
    print("Duplicated Rows:")
    print(df.duplicated().sum())


def split_train_val_test(x, y, train_size=0.7, test_size=0.15, random_state=42, stratify=None):

    stratify = y if stratify else None
    
    x_train, x_temp, y_train, y_temp = train_test_split(
    x,
    y,
    test_size=1-train_size,
    random_state=random_state,
    stratify=stratify
    )

    stratify = y_temp if stratify is not None else None

    # Geçici veriyi ikiye böl: %15 validation, %15 test
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    return x_train, x_test, x_val, y_train, y_test, y_val

def get_dataset_distribution(y_train, y_val, y_test):
    distribution = pd.DataFrame({
        "train": y_train.value_counts(normalize=True),
        "val": y_val.value_counts(normalize=True),
        "test": y_test.value_counts(normalize=True)
    })
    return distribution

def model_training(models, x_train, y_train, x_val, y_val):
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        results[name] = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
        }
    return results

def model_evaluation(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

def scale_features(x_train, x_val, x_test):
    # StandardScaler yalnızca train ile fit edilir; val/test'e sadece transform uygulanır (veri sızıntısını önlemek için).
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_val_scaled = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns, index=x_val.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
    return x_train_scaled, x_val_scaled, x_test_scaled

def compare_scaling_results(unscaled_results, scaled_results):
    rows = []
    for model_name in unscaled_results:
        for metric in unscaled_results[model_name]:
            rows.append({
                "model": model_name,
                "metric": metric,
                "unscaled": unscaled_results[model_name][metric],
                "scaled": scaled_results[model_name][metric],
                "diff": scaled_results[model_name][metric] - unscaled_results[model_name][metric],
            })
    return pd.DataFrame(rows)

def plot_scaling_comparison(unscaled_results, scaled_results, title, filename):
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = list(unscaled_results.keys())
    width = 0.35
    x = range(len(metrics))

    fig, axes = plt.subplots(1, len(model_names), figsize=(15, 5), sharey=True)
    for ax, name in zip(axes, model_names):
        unscaled_scores = [unscaled_results[name][m] for m in metrics]
        scaled_scores = [scaled_results[name][m] for m in metrics]
        ax.bar([p - width / 2 for p in x], unscaled_scores, width, label="Unscaled")
        ax.bar([p + width / 2 for p in x], scaled_scores, width, label="Scaled")
        ax.set_xticks(list(x))
        ax.set_xticklabels(metrics, rotation=45)
        ax.set_ylim(0, 1)
        ax.set_title(name)

    axes[0].set_ylabel("Score")
    axes[0].legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_model_comparison(results):
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results.keys())
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(model_names):
        scores = [results[name][m] for m in metrics]
        positions = [x + i * width for x in range(len(metrics))]
        ax.bar(positions, scores, width, label=name)

    ax.set_xticks([x + width * (len(model_names) - 1) / 2 for x in range(len(metrics))])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Validation Set Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()

if __name__ == "__main__":
    print("This script loads the breast cancer dataset and provides basic information about it.")
    
    df, data = load_breast_cancer_dataset()
    
    get_dataset_info(df, data)

    y_target = df['target']
    x_features = df.drop(columns=['target'])

    x_train, x_test, x_val, y_train, y_test, y_val = split_train_val_test(x_features, y_target)

    distribution = get_dataset_distribution(y_train, y_val, y_test)

    print("--------------------------------")
    print("Target variable distribution in train, validation, and test sets:")
    print(distribution)

    x_train, x_test, x_val, y_train, y_test, y_val = split_train_val_test(x_features, y_target, train_size=0.7, test_size=0.15, random_state=42, stratify=True)

    distribution = get_dataset_distribution(y_train, y_val, y_test)

    print("--------------------------------")
    print("Target variable distribution with stratification in train, validation, and test sets:")
    print(distribution)

    print("--------------------------------")
    print("Training and evaluating models on the validation set:")
    models = {
            "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(random_state=42),
        }
    results = model_training(models, x_train, y_train, x_val, y_val)
    for model_name, metrics in results.items():
        print(model_name, metrics)

    plot_model_comparison(results)

    print("--------------------------------")
    print("Evaluating the best model on the test set:")
    model_evaluation_results = model_evaluation(models["Random Forest"], x_test, y_test)
    print("Random Forest Test Set Evaluation:", model_evaluation_results)
    model_evaluation_results = model_evaluation(models["Logistic Regression"], x_test, y_test)
    print("Logistic Regression Test Set Evaluation:", model_evaluation_results)
    model_evaluation_results = model_evaluation(models["SVM"], x_test, y_test)
    print("SVM Test Set Evaluation:", model_evaluation_results)

    unscaled_test_results = {
        name: model_evaluation(model, x_test, y_test) for name, model in models.items()
    }

    print("--------------------------------")
    print("Scaling features with StandardScaler (fit on train only):")
    x_train_scaled, x_val_scaled, x_test_scaled = scale_features(x_train, x_val, x_test)

    scaled_models = {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(random_state=42),
    }

    print("--------------------------------")
    print("Training and evaluating scaled models on the validation set:")
    scaled_val_results = model_training(scaled_models, x_train_scaled, y_train, x_val_scaled, y_val)
    for model_name, metrics in scaled_val_results.items():
        print(model_name, metrics)

    print("--------------------------------")
    print("Evaluating scaled models on the test set:")
    scaled_test_results = {}
    for model_name, model in scaled_models.items():
        scaled_test_results[model_name] = model_evaluation(model, x_test_scaled, y_test)
        print(model_name, "Test Set Evaluation:", scaled_test_results[model_name])

    print("--------------------------------")
    print("Unscaled vs Scaled comparison (validation set):")
    val_comparison = compare_scaling_results(results, scaled_val_results)
    print(val_comparison)

    print("--------------------------------")
    print("Unscaled vs Scaled comparison (test set):")
    test_comparison = compare_scaling_results(unscaled_test_results, scaled_test_results)
    print(test_comparison)

    plot_scaling_comparison(
        results, scaled_val_results,
        "Unscaled vs Scaled Model Comparison (Validation Set)",
        "scaling_comparison_val.png",
    )
    plot_scaling_comparison(
        unscaled_test_results, scaled_test_results,
        "Unscaled vs Scaled Model Comparison (Test Set)",
        "scaling_comparison_test.png",
    )
