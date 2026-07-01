# Cyberbullying Detection using NLP & Machine Learning

A multi-class text classification project that detects and categorizes cyberbullying in tweets, using a TF-IDF NLP pipeline and three classic machine learning models: **Logistic Regression**, **Random Forest**, and **XGBoost**.

## Overview

Cyberbullying detection is typically framed as a text classification problem. This project builds an end-to-end pipeline that:

1. Loads and explores ~47,000 labeled tweets
2. Cleans and preprocesses raw tweet text
3. Extracts TF-IDF (unigram + bigram) features
4. Trains and evaluates three ML models
5. Compares model performance side-by-side
6. Saves the best-performing model for reuse
7. Demonstrates inference on new, unseen text

## Dataset

[**Cyberbullying Classification**](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification) (Kaggle) — ~47,000 tweets labeled into 6 balanced categories:

| Category | Description |
|---|---|
| `age` | Age-based bullying |
| `ethnicity` | Ethnicity/race-based bullying |
| `gender` | Gender-based bullying |
| `religion` | Religion-based bullying |
| `other_cyberbullying` | Bullying that doesn't fit the above categories |
| `not_cyberbullying` | Non-bullying tweets |

## Approach

- **Preprocessing:** lowercasing, URL/mention removal, punctuation/digit stripping, stopword removal
- **Feature extraction:** TF-IDF vectorization (unigrams + bigrams, top 10,000 features)
- **Models compared:**
  - Logistic Regression (`class_weight="balanced"`)
  - Random Forest (200 trees, `class_weight="balanced"`)
  - XGBoost (multi-class softmax objective)
- **Evaluation metrics:** accuracy, macro precision/recall/F1, confusion matrices
- **Model selection:** best model chosen by macro F1-score, saved with `joblib` alongside the fitted TF-IDF vectorizer

## Project Structure

```
.
├── Cyberbullying_Detection.ipynb   # Main notebook (EDA -> preprocessing -> modeling -> evaluation)
├── requirements.txt
├── README.md
└── artifacts/                      # Created after running the notebook
    ├── best_model_*.joblib
    ├── tfidf_vectorizer.joblib
    └── label_encoder.joblib
```

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/Ayeza-Irfan/cyberbullying-detection.git

pip install -r requirements.txt
```

### 2. Get the dataset

The notebook downloads the dataset automatically via [`kagglehub`](https://github.com/Kaggle/kagglehub):

- **Kaggle Notebooks:** works out of the box (dataset auto-mounted).
- **Colab / local:** you'll need a Kaggle account and API token (`kaggle.json`). See [Kaggle API docs](https://www.kaggle.com/docs/api).
- **Manual fallback:** download `cyberbullying_tweets.csv` from the [dataset page](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification) and place it in the project root — the notebook will detect and use it automatically.

### 3. Run the notebook

```bash
jupyter notebook Cyberbullying_Detection.ipynb
```

Run all cells top to bottom. Trained artifacts are saved to `artifacts/`.

## Results

Model performance (accuracy, macro precision/recall/F1) is summarized in a comparison table and bar chart in Section 6 of the notebook — see the notebook output for the exact numbers on a given run.

## Possible Extensions

- Hyperparameter tuning via `GridSearchCV` or `Optuna`
- Transformer-based embeddings (BERT/DistilBERT) as a stronger baseline
- Deploy the saved model behind a Flask/FastAPI inference endpoint

## License

This project is released under the MIT License. The dataset is subject to its own license on Kaggle.
