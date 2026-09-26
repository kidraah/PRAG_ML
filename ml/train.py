"""
FeedLab — Train the Preference Model

Learns "is this reel informative?" (value score >= WATCH_THRESHOLD) from
labelled reels, so the agent can one day score reels in milliseconds
without an LLM. Two feature sets are compared, each with a class-balanced
logistic regression:

  tfidf      character 2-5-grams of caption + cover text + transcript
             (works across Hindi, English and Hinglish without a model)
  embedding  the multilingual sentence embedding used for story grouping

Evaluation is 5-fold cross-validation grouped by creator — every test
fold holds creators the model never saw — so the scores measure
generalisation to new accounts, not memorised caption styles. The quick
rules are scored on the same reels where they were logged.

    python -m ml.train [--scorer review] [--save]
"""

import argparse
import datetime
import os

import numpy as np

from config import EMBEDDING_MODEL, WATCH_THRESHOLD
from database.db import get_connection
from ml.dataset import Example, load_examples

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


def make_model(kind: str):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    classifier = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
    if kind == "tfidf":
        return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
                             classifier)
    return classifier  # "embedding": features are precomputed vectors


def features(kind: str, examples: list[Example], embed_fn=None):
    if kind == "tfidf":
        return np.array([e.text for e in examples], dtype=object)
    if embed_fn is None:
        from stories.grouping import embed as embed_fn
    return embed_fn([e.text or "(empty)" for e in examples])


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    predicted = scores >= threshold
    both = len(set(y_true)) == 2
    return {
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 3) if both else None,
        "avg_precision": round(float(average_precision_score(y_true, scores)), 3) if both else None,
    }


def cross_validate(kind: str, examples: list[Example], folds: int = 5, embed_fn=None, seed: int = 0) -> np.ndarray:
    """Out-of-fold probabilities, folds grouped by creator."""
    from sklearn.model_selection import StratifiedGroupKFold

    X = features(kind, examples, embed_fn)
    y = np.array([e.informative for e in examples])
    groups = np.array([e.creator or e.code for e in examples])
    out = np.zeros(len(examples))
    for train, test in StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y, groups):
        model = make_model(kind).fit(X[train], y[train])
        out[test] = model.predict_proba(X[test])[:, 1]
    return out


def evaluate(examples: list[Example], folds: int = 5, embed_fn=None) -> dict:
    y = np.array([e.informative for e in examples])
    results = {kind: {"scores": cross_validate(kind, examples, folds, embed_fn)} for kind in ("tfidf", "embedding")}
    for kind in results:
        results[kind]["all"] = metrics(y, results[kind]["scores"])
    ruled = np.array([e.rule_score is not None for e in examples])
    if ruled.any():
        rules = np.array([e.rule_score or 0.0 for e in examples])
        results["rules"] = {"subset": metrics(y[ruled], rules[ruled], threshold=WATCH_THRESHOLD)}
        for kind in ("tfidf", "embedding"):
            results[kind]["subset"] = metrics(y[ruled], results[kind]["scores"][ruled])
    return results


def train_and_save(kind: str, examples: list[Example], report: dict, embed_fn=None) -> str:
    import joblib

    y = np.array([e.informative for e in examples])
    model = make_model(kind).fit(features(kind, examples, embed_fn), y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"informative-{kind}.joblib")
    joblib.dump({
        "kind": kind, "model": model, "embedding_model": EMBEDDING_MODEL if kind == "embedding" else None,
        "threshold": 0.5, "trained_on": len(examples), "positives": int(y.sum()),
        "cv": report, "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the 'is this reel informative?' model")
    parser.add_argument("--scorer", default="review", help="Whose labels to learn from (default: review)")
    parser.add_argument("--save", action="store_true", help="Fit the better model on all reels and save it")
    args = parser.parse_args()

    conn = get_connection()
    try:
        examples = load_examples(conn, args.scorer)
    finally:
        conn.close()
    positives = sum(e.informative for e in examples)
    creators = len({e.creator for e in examples})
    print(f"📚  {len(examples)} labelled reels ({positives} informative) from {creators} creators")
    if positives < 5 or len(examples) - positives < 5:
        print("   Not enough of each class to train yet.")
        return

    results = evaluate(examples)
    print("\n  5-fold cross-validation, test folds hold unseen creators (threshold 0.5)")
    print(f"  {'model':10} {'precision':>9} {'recall':>7} {'F1':>6} {'ROC-AUC':>8} {'avg prec':>9}")
    for kind in ("tfidf", "embedding"):
        m = results[kind]["all"]
        print(f"  {kind:10} {m['precision']:9.2f} {m['recall']:7.2f} {m['f1']:6.2f} {m['roc_auc']:8.2f} {m['avg_precision']:9.2f}")
    if "rules" in results:
        n = sum(e.rule_score is not None for e in examples)
        print(f"\n  On the {n} reels where the quick rules were logged:")
        for kind in ("rules", "tfidf", "embedding"):
            m = results[kind]["subset"]
            auc = f"{m['roc_auc']:8.2f}" if m["roc_auc"] is not None else f"{'—':>8}"
            print(f"  {kind:10} {m['precision']:9.2f} {m['recall']:7.2f} {m['f1']:6.2f} {auc}")

    if args.save:
        best = max(("tfidf", "embedding"), key=lambda k: results[k]["all"]["avg_precision"] or 0)
        report = {k: v["all"] for k, v in results.items() if "all" in v}
        print(f"\n💾  Saved {best} model → {train_and_save(best, examples, report)}")


if __name__ == "__main__":
    main()
