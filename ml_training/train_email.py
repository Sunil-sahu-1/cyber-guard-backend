"""Train a lightweight phishing-email classifier.

Input:
datasets/phishing/emails/emails.csv

Required columns:
- text/body/content
- label/class/target

The model uses TF-IDF + Logistic Regression. This is intentionally lightweight
and reproducible on the Cyber Guard development machine.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "phishing" / "emails"
MODELS = ROOT / "trained_models"
RESULTS = ROOT / "training_results"


def main() -> None:
    files = sorted(DATA.glob("*.csv"))
    if not files:
        raise SystemExit(
            "Add a labelled email CSV to datasets/phishing/emails. "
            "Expected columns: text/body/content and label/class/target."
        )

    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    text_col = next((c for c in df.columns if str(c).lower() in {"text", "body", "content"}), None)
    label_col = next((c for c in df.columns if str(c).lower() in {"label", "class", "target"}), None)

    if not text_col or not label_col:
        raise SystemExit("Email CSV must contain text/body/content and label/class/target")

    text = df[text_col].fillna("").astype(str)
    labels = df[label_col].astype(str).str.lower().str.strip()
    y = labels.map(lambda v: 1 if any(x in v for x in ["phish", "spam", "malicious", "1"]) else 0)

    x_train, x_test, y_train, y_test = train_test_split(
        text, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=200_000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / "phishing_email_tfidf_logreg.joblib")
    (RESULTS / "phishing_email_tfidf_logreg.json").write_text(
        __import__("json").dumps({
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "classification_report": classification_report(
                y_test, pred, output_dict=True, zero_division=0
            ),
        }, indent=2),
        encoding="utf-8",
    )
    print("[done] phishing_email_tfidf_logreg.joblib")


if __name__ == "__main__":
    main()
