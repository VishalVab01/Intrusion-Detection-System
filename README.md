# Network Intrusion Detection System

Train six machine-learning models to classify recorded network connections as normal, denial of service, probe, remote-to-local, or user-to-root. Each input is a connection described by the 41 KDD Cup 99 fields.

## Run on macOS

Use Python 3.11 or later. Double-click **Run.command** in Finder or run `./Run.command` in Terminal from this folder. The first run installs the Python packages into a local virtual environment and downloads the KDD Cup 99 ten-percent training dataset through scikit-learn. Subsequent runs reuse the cached data. The launcher trains all six models on a reproducible sample of 30,000 connections and prints five example predictions.

To use a copy of the dataset you already downloaded, put the headerless file at `data/kddcup.data_10_percent.gz` before launching. An alternate filename can be passed with `--data`.

## Reproduce the full experiment

After setup, run the following in the project directory. Full training includes the slower RBF support vector machine and gradient boosting steps.

```sh
.venv/bin/python ids.py train
.venv/bin/python ids.py predict results-full/detector.joblib results-full/example-connections.csv
```

For a manually downloaded dataset, run `.venv/bin/python ids.py train --data data/kddcup.data_10_percent.gz`.

## Interpret the results

The full run uses a 67/33 random split with seed 42. The preprocessing pipeline encodes protocol and flag, removes the article's eleven excluded columns, and fits min-max scaling on the training partition. Six classifiers use the same split. The saved random forest can score additional headered CSVs with the same schema as example-connections.csv. Load only model files you created or trust.

Open `results-quick/comparison.csv` or `results-full/comparison.csv` to compare training accuracy, holdout accuracy, macro F1 and runtime. Each model also has a CSV with precision, recall, F1 and support by family. Macro F1 gives each family equal weight. Inspect the support for rare classes before interpreting their scores. The charts show dataset composition and training-feature correlations. Repeated KDD records can appear across a random split, so reported metrics describe this historical benchmark. The project operates on recorded connection features.

## Sources

- [GeeksforGeeks project and original notebook link](https://www.geeksforgeeks.org/machine-learning/intrusion-detection-system-using-machine-learning-algorithms/)
- [UCI KDD Cup 1999 data, field names and label definitions](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
