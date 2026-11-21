"""Compare six intrusion classifiers and classify new KDD connection records."""
import argparse
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_kddcup99
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from schema import CATEGORIES, DROP, FEATURES, LABELS

ROOT = Path(__file__).resolve().parent


def load_data(path):
    if path:
        frame = pd.read_csv(path, header=None)
        if frame.shape[1] != len(FEATURES) + 1:
            raise ValueError("Expected 41 connection features followed by one attack label.")
        frame.columns = FEATURES + ["label"]
        labels = frame.pop("label")
    else:
        data = fetch_kddcup99(data_home=str(ROOT / "data"), percent10=True)
        frame = pd.DataFrame(data.data, columns=FEATURES)
        for col in ("protocol_type", "service", "flag"):
            frame[col] = frame[col].str.decode("ascii")
        labels = pd.Series(data.target).str.decode("ascii")
    for col in FEATURES:
        if col not in ("protocol_type", "service", "flag"):
            frame[col] = pd.to_numeric(frame[col], errors="raise")
    labels = labels.str.strip().str.rstrip(".").map(LABELS)
    if labels.isna().any() or frame.isna().any().any():
        raise ValueError("Missing features or unrecognized KDD training labels.")
    return frame, labels


def chart_correlations(frame, path):
    corr = frame.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mesh = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    fig.colorbar(mesh, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train(args):
    frame, labels = load_data(args.data)
    if args.quick and len(frame) > 30000:
        frame, _, labels, _ = train_test_split(
            frame, labels, train_size=30000, stratify=labels, random_state=42)
    out = ROOT / ("results-quick" if args.quick else "results-full")
    out.mkdir(exist_ok=True)
    x, holdout, y, truth = train_test_split(frame, labels, test_size=0.33, random_state=42)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for series, ax in zip([frame.protocol_type, frame.logged_in, labels], axes):
        series.value_counts().plot.bar(ax=ax)
        ax.set_ylabel("Connections")
    for ax, title in zip(axes, ["Protocol", "Logged in", "Attack family"]):
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out / "data-overview.png")
    plt.close(fig)

    numeric = [c for c in FEATURES if c not in DROP and c not in CATEGORIES]
    encoder = OrdinalEncoder(categories=list(CATEGORIES.values()),
                             handle_unknown="use_encoded_value", unknown_value=-1)
    prepare = make_pipeline(ColumnTransformer([
        ("codes", encoder, list(CATEGORIES)), ("numeric", "passthrough", numeric)
    ]), MinMaxScaler())
    train_matrix = prepare.fit_transform(x)
    test_matrix = prepare.transform(holdout)
    before = x.drop(columns="service").copy()
    for col, values in CATEGORIES.items():
        before[col] = before[col].map({v: i for i, v in enumerate(values)})
    chart_correlations(before, out / "correlations-before.png")
    chart_correlations(pd.DataFrame(train_matrix, columns=list(CATEGORIES) + numeric),
                       out / "correlations-after.png")

    models = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=30, n_jobs=-1, random_state=42),
        "SVM": SVC(gamma="scale"),
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0),
    }
    scores = []
    for name, estimator in models.items():
        print(f"Training {name} ...", flush=True)
        start = perf_counter()
        estimator.fit(train_matrix, y)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        predicted = estimator.predict(test_matrix)
        predict_seconds = perf_counter() - start
        row = dict(model=name, train_accuracy=estimator.score(train_matrix, y),
                   test_accuracy=accuracy_score(truth, predicted),
                   macro_f1=f1_score(truth, predicted, average="macro", zero_division=0),
                   train_seconds=fit_seconds, test_seconds=predict_seconds)
        scores.append(row)
        print(f"  Holdout accuracy {row['test_accuracy']:.2%}, macro F1 {row['macro_f1']:.3f}")
        report = classification_report(truth, predicted, labels=sorted(set(LABELS.values())),
                                       output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(out / (name + ".csv"))
        if name == "Random Forest":
            joblib.dump(make_pipeline(prepare, estimator), out / "detector.joblib")
            example = holdout.iloc[:5].copy()
            example.to_csv(out / "example-connections.csv", index=False)
    pd.DataFrame(scores).to_csv(out / "comparison.csv", index=False)
    print(f"Trained on {len(x):,} connections. Evaluated on {len(truth):,}. Results in {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--quick", action="store_true", help="use up to 30,000 stratified records")
    fit.add_argument("--data", type=Path, help="headerless labeled KDD training CSV or CSV.gz")
    predict = commands.add_parser("predict")
    predict.add_argument("model", type=Path)
    predict.add_argument("csv", type=Path, help="headered CSV with all 41 KDD features")
    args = parser.parse_args()
    if args.action == "train":
        train(args)
    else:
        records = pd.read_csv(args.csv)
        missing = set(FEATURES).difference(records.columns)
        if missing:
            parser.error("Missing columns: " + ", ".join(sorted(missing)))
        predictions = joblib.load(args.model).predict(records[FEATURES])
        print(pd.DataFrame({"record": range(1, len(records)+1),
                            "prediction": predictions}).to_csv(index=False), end="")


if __name__ == "__main__":
    main()
