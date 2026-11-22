# Review Rating Prediction (Prediction Challenge 2026)

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Solution for a class prediction challenge: given the title and body of a **Portuguese product
review**, predict its star rating (1 to 5). Classical NLP only: sparse text features plus a
linear classifier, no pre-trained models. The challenge also **prohibited IDF weighting**,
which shaped most of the engineering decisions.

## Approach

```
title + text ─► normalization ─┬─► word n-grams (1–2)   ─┐
                               ├─► char n-grams (3–5)   ─┼─► ColumnTransformer ─► LinearSVC ─► rating
                               └─► hand-crafted features ┘
```

- **Text:** two `CountVectorizer`s in parallel: words (uni- and bigrams) and characters
  (`char_wb`, 3–5), which catch typos and spelling variants common in user reviews.
- **Hand-crafted features** ([`src/features.py`](src/features.py)): length and punctuation,
  positive/negative lexicon counts, e-commerce signals (defect, price, delivery), title/body
  ratios, product frequency, and **antithesis connectives** ("mas", "porém") that flip the
  polarity of a sentence.
- **Stopwords** that keep negations and antithesis terms instead of removing them.
- **Model:** multiclass `LinearSVC`, everything in one scikit-learn `Pipeline`, with optional
  stratified cross-validation.

## Results

Metric: macro F1 on the challenge leaderboard. Classes are balanced (~8.2k reviews each).

| Version | Change | Macro F1 |
|---|---|---|
| 1 | TF-IDF + logistic regression | ~0.52 (disqualified: IDF not allowed) |
| 2 | Raw counts, better preprocessing, antithesis-aware stopwords, richer features | 0.46 |
| 3 | Character n-grams alongside word n-grams, leaner numeric features | **0.4673** |

Local hold-out with the current defaults: macro F1 0.459, accuracy 0.463. Most errors are
between neighbouring ratings (2↔3, 3↔4), where reviews mix praise and criticism. The full
discussion is in the [notebook](notebooks/rating-prediction.ipynb) and in
[`docs/model-changes.md`](docs/model-changes.md).

## Getting started

```bash
pip install -r requirements.txt
python src/main.py                  # trains on data/train.csv, writes submission.csv
python src/main.py --cv-folds 5     # adds stratified cross-validation
```

Defaults reproduce version 3. Other options: `--model linear_svm|logreg|lightgbm`,
`--feature-mode full|minimal`, `--no-char-ngrams`, `--svm-c`,
`--svm-class-weight none|balanced` (see `python src/main.py --help`).

## Project structure

```
├── src/
│   ├── main.py            # CLI: training, validation, submission
│   ├── preprocessing.py   # loading, text cleaning, stopwords
│   ├── features.py        # hand-crafted numeric features
│   ├── model.py           # pipeline and estimators
│   └── utils.py           # metrics and logging
├── notebooks/
│   └── rating-prediction.ipynb   # full write-up: EDA, features, evolution, evaluation
├── data/                  # challenge data: 41k training and 10k test reviews
├── submissions/           # submitted predictions (version 3)
└── docs/                  # notes on each model change
```

The notebook and the notes are in Portuguese.

## License

[MIT](LICENSE)
