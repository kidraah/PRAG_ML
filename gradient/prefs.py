"""
PRAG — the Gradient's preference models: one trained classifier per persona.

The task: given a reel's text, will this persona watch it to the end and
engage? (Jev's judgement, gradient/label.py: the binary label; its graded
score is never a feature.) Each persona gets its own model, trained only on
its own labels. Four candidates, the same splits for all:

  majority      always the commoner class: the floor
  quick rules   the persona's own watch words and skip words: the hand-written baseline
  tfidf         word 1-2 grams + character 2-5 grams, logistic regression
  embedding     multilingual MiniLM sentence vectors, logistic regression

Splits are fixed per group (store.split_for): a story's retellings, and one
generator brief's reels, share a partition. Regularisation and the decision
threshold are picked on validation; the test split is only scored. The first
run of each persona freezes its test items as the evaluation set, so every
later version is compared on exactly the same items.

Features read only the reel's text, with the persona-location words the
writer slipped in (the persona's city) and creator handles scrubbed: a
shortcut a model would otherwise learn instead of the content.

The second model, "whose feed", learns which account's feed a reel came
from: the algorithm's side of the story rather than the persona's.

    python -m gradient.prefs [--account mlproject01] [--feed]
"""

import argparse
import datetime
import json
import os
import re
import sqlite3

import numpy as np

from agent import bias as biases
from config import EMBEDDING_MODEL
from gradient import store, vectors

SCRUB = re.compile(r"(?i)@[\w.]+|#?\b(?:pune|puneri|indore|indori|punekar)\w*|इंदौर\w*|पुणे\w*")
C_GRID = (0.25, 1.0, 4.0)
WORD = r"(?u)[\w\u0900-\u097F]{2,}"  # a word, Devanagari vowel signs included (\w alone splits Hindi words apart)
MODEL_NAMES = {"majority": "Majority class", "rules": "Quick rules (watch / skip words)",
               "tfidf": "TF-IDF + logistic regression", "embedding": "Multilingual embeddings + logistic regression"}


def scrub(text: str) -> str:
    return SCRUB.sub(" ", text or "")


# ── Data ──────────────────────────────────────────────────────────────

def dataset(conn: sqlite3.Connection, account: str) -> list[dict]:
    """Every item with this persona's current Jev label (the newest, if judged twice)."""
    fingerprint = biases.load(account).fingerprint
    rows = conn.execute("SELECT i.*, l.score AS teacher_score, l.label AS y, l.created_at AS labelled_at "
                        "FROM items i JOIN labels l ON l.item_id = i.id WHERE l.account = ? AND l.teacher = 'jev' "
                        "AND l.persona_version = ? ORDER BY l.created_at", (account, fingerprint))
    return list({row["id"]: dict(row) for row in rows}.values())


def feed_dataset(conn: sqlite3.Connection) -> list[dict]:
    """Items in exactly one account's feed; y = 1 for the first account's."""
    rows = conn.execute("SELECT i.*, MIN(f.account) AS account FROM items i JOIN feed f ON f.item_id = i.id "
                        "GROUP BY i.id HAVING COUNT(DISTINCT f.account) = 1")
    first = store.ACCOUNTS[0]
    return [{**dict(row), "y": int(row["account"] == first)} for row in rows]


def splits(rows: list[dict]) -> dict[str, list[dict]]:
    return {name: [r for r in rows if r["split"] == name] for name in ("train", "val", "test")}


# ── Models ────────────────────────────────────────────────────────────

def tfidf_model(c: float):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, make_pipeline

    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True,
                                  token_pattern=WORD)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=120000,
                                  sublinear_tf=True)),
    ])
    return make_pipeline(features, LogisticRegression(C=c, class_weight="balanced", max_iter=3000))


def embedding_model(c: float):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=c, class_weight="balanced", max_iter=3000)


def rule_scores(rows: list[dict], bias) -> np.ndarray:
    watch = [w.lower() for w in bias.watch_words]
    skip = [w.lower() for w in bias.skip_words]
    out = []
    for row in rows:
        text = (row["text"] or "").lower()
        hits = sum(w in text for w in watch) - 2 * sum(w in text for w in skip)
        out.append(min(1.0, max(0.0, 0.3 + 0.2 * hits)))
    return np.array(out)


def inputs(kind: str, rows: list[dict], embed_fn=None):
    if kind == "tfidf":
        return np.array([scrub(r["text"]) for r in rows], dtype=object)
    scrubbed = [{"id": f"s:{r['id']}", "text": scrub(r["text"])} for r in rows]
    return vectors.for_items(scrubbed, embed_fn=embed_fn)


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    both = len(set(y.tolist())) == 2
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    top = np.argsort(-p)[:50]
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "precision": round(float(precision_score(y, predicted, zero_division=0)), 3),
        "recall": round(float(recall_score(y, predicted, zero_division=0)), 3),
        "f1": round(float(f1_score(y, predicted, zero_division=0)), 3),
        "pr_auc": round(float(average_precision_score(y, p)), 3) if both else None,
        "roc_auc": round(float(roc_auc_score(y, p)), 3) if both else None,
        "brier": round(float(brier_score_loss(y, np.clip(p, 0, 1))), 3),
        "precision_at_50": round(float(y[top].mean()), 3) if len(y) else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return round(float(grid[int(np.argmax(scores))]), 3)


def fit_candidates(parts: dict, rules_fn=None, embed_fn=None, progress=print) -> dict:
    """Every candidate, fitted on train, tuned on val: {kind: {model, threshold, val, val_p, test_p}}."""
    from sklearn.metrics import average_precision_score

    y = {name: np.array([r["y"] for r in rows]) for name, rows in parts.items()}
    out = {}
    majority = float(y["train"].mean() >= 0.5)
    out["majority"] = {"model": None, "threshold": 0.5,
                       "val_p": np.full(len(y["val"]), majority), "test_p": np.full(len(y["test"]), majority)}
    if rules_fn:
        val_p, test_p = rules_fn(parts["val"]), rules_fn(parts["test"])
        out["rules"] = {"model": None, "threshold": best_threshold(y["val"], val_p), "val_p": val_p, "test_p": test_p}
    for kind, make in (("tfidf", tfidf_model), ("embedding", embedding_model)):
        progress(f"  fitting {MODEL_NAMES[kind]} …")
        X = {name: inputs(kind, rows, embed_fn) for name, rows in parts.items()}
        best = None
        for c in C_GRID:
            model = make(c).fit(X["train"], y["train"])
            val_p = model.predict_proba(X["val"])[:, 1]
            ap = average_precision_score(y["val"], val_p) if len(set(y["val"].tolist())) == 2 else 0
            if best is None or ap > best[0]:
                best = (ap, c, model, val_p)
        _ap, c, model, val_p = best
        out[kind] = {"model": model, "c": c, "threshold": best_threshold(y["val"], val_p), "val_p": val_p,
                     "test_p": model.predict_proba(X["test"])[:, 1]}
    for kind, entry in out.items():
        entry["val"] = metrics(y["val"], entry["val_p"], entry["threshold"])
        entry["test"] = metrics(y["test"], entry["test_p"], entry["threshold"])
    return out


def top_features(model, n: int = 14) -> dict:
    """The words a TF-IDF model leans on most, each way."""
    union, classifier = model.steps[0][1], model.steps[-1][1]
    words = union.transformer_list[0][1]
    names = words.get_feature_names_out()
    coef = classifier.coef_[0][:len(names)]
    order = np.argsort(coef)
    return {"towards_yes": [str(names[i]) for i in order[::-1][:n]], "towards_no": [str(names[i]) for i in order[:n]]}


def slices(rows: list[dict], p: np.ndarray, threshold: float, key: str) -> list[dict]:
    out = []
    for value in sorted({r.get(key) or "unknown" for r in rows}):
        idx = [i for i, r in enumerate(rows) if (r.get(key) or "unknown") == value]
        y = np.array([rows[i]["y"] for i in idx])
        if len(idx) < 20:
            out.append({"slice": value, "n": len(idx), "note": "fewer than 20 items: not scored"})
        else:
            m = metrics(y, p[idx], threshold)
            out.append({"slice": value, "n": len(idx), "positives": int(y.sum()), "pr_auc": m["pr_auc"],
                        "f1": m["f1"], "roc_auc": m["roc_auc"]})
    return out


# ── Runs ──────────────────────────────────────────────────────────────

def model_dir(account: str | None) -> str:
    return os.path.join(store.MODEL_DIR, account or "shared")


def next_version(conn: sqlite3.Connection, account: str, kind: str) -> int:
    return conn.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM runs WHERE account = ? AND kind = ? AND "
                        "status = 'done'", (account, kind)).fetchone()[0]


def frozen_eval(account: str, test_rows: list[dict]) -> set[str]:
    """The test items of this persona's first run, written once and reused by every later version."""
    path = os.path.join(model_dir(account), "eval-set.json")
    if os.path.exists(path):
        with open(path) as fh:
            return set(json.load(fh)["items"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ids = sorted(r["id"] for r in test_rows)
    with open(path, "w") as fh:
        json.dump({"created_at": store.now(), "items": ids}, fh)
    return set(ids)


def train(conn: sqlite3.Connection, account: str, *, embed_fn=None, progress=print, kind: str = "preference") -> dict:
    """Train one version of a persona's preference model ("preference") or the shared whose-feed model ("feed")."""
    import joblib

    started = datetime.datetime.now(datetime.timezone.utc)
    if kind == "feed":
        rows, bias, owner = feed_dataset(conn), None, "shared"
    else:
        rows, bias, owner = dataset(conn, account), biases.load(account), account
    parts = splits(rows)
    if min(len(parts["train"]), len(parts["val"]), len(parts["test"])) < 30:
        raise ValueError(f"Not enough labelled items to train ({len(rows)}); label the corpus first.")
    version = next_version(conn, owner, kind)
    run_id = f"{owner}-{kind}-v{version}-{started:%Y%m%dT%H%M%S}"
    progress(f"{run_id}: {len(parts['train'])} train / {len(parts['val'])} val / {len(parts['test'])} test")
    fitted = fit_candidates(parts, (lambda r: rule_scores(r, bias)) if bias else None, embed_fn, progress)
    chosen = max(("tfidf", "embedding"), key=lambda k: fitted[k]["val"]["pr_auc"] or 0)
    best = fitted[chosen]
    test = parts["test"]
    report = {
        "run": run_id, "account": owner, "kind": kind, "version": version, "task": (
            "Will this persona watch it to the end and engage? (Jev's label)" if kind == "preference"
            else f"Whose feed is it from? (1 = @{store.ACCOUNTS[0]})"),
        "persona": {"name": bias.name, "fingerprint": bias.fingerprint} if bias else None,
        "data": {"items": len(rows), **{f"{k}_n": len(v) for k, v in parts.items()},
                 **{f"{k}_positives": int(sum(r["y"] for r in v)) for k, v in parts.items()},
                 "origins": count(rows, "origin"), "languages": count(rows, "language"),
                 "batches": sorted({r["batch"] for r in rows})},
        "candidates": {k: {"name": MODEL_NAMES[k], "c": v.get("c"), "threshold": v["threshold"],
                           "val": v["val"], "test": v["test"]} for k, v in fitted.items()},
        "chosen": chosen, "chosen_because": "highest validation PR-AUC",
        "slices": {"origin": slices(test, best["test_p"], best["threshold"], "origin"),
                   "language": slices(test, best["test_p"], best["threshold"], "language")},
        "top_features": top_features(fitted["tfidf"]["model"]),
        "errors": errors(test, best["test_p"], best["threshold"]),
        "features": "the reel's text only (caption, on-screen text, speech), handles and persona-city words scrubbed",
        "embedding_model": EMBEDDING_MODEL,
    }
    frozen = frozen_eval(f"{owner}-{kind}" if kind == "feed" else owner, test)
    idx = [i for i, r in enumerate(test) if r["id"] in frozen]
    if idx:
        y = np.array([test[i]["y"] for i in idx])
        report["frozen_eval"] = {"n": len(idx), "of": len(frozen),
                                 **metrics(y, best["test_p"][idx], best["threshold"])}
    path = os.path.join(model_dir(owner), f"{kind}-v{version}.joblib")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"kind": chosen, "model": best["model"], "threshold": best["threshold"], "task": kind,
                 "account": owner, "fingerprint": bias.fingerprint if bias else None, "version": version,
                 "run": run_id, "embedding_model": EMBEDDING_MODEL, "created": started.isoformat()}, path)
    conn.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, 'done', ?, ?, ?, ?, NULL)",
                 (run_id, owner, kind, version, store.now(), len(rows), json.dumps(report["data"]["batches"]),
                  path, json.dumps(report)))
    conn.commit()
    m = report["candidates"][chosen]["test"]
    progress(f"{run_id}: chose {MODEL_NAMES[chosen]}; test PR-AUC {m['pr_auc']}, F1 {m['f1']} "
             f"(majority F1 {report['candidates']['majority']['test']['f1']})")
    return report


def count(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        out[r.get(key) or "unknown"] = out.get(r.get(key) or "unknown", 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def errors(rows: list[dict], p: np.ndarray, threshold: float, n: int = 4) -> list[dict]:
    y = np.array([r["y"] for r in rows])
    out = []
    for wrong, label in ((np.where((p >= threshold) & (y == 0))[0], "predicted yes, teacher said no"),
                         (np.where((p < threshold) & (y == 1))[0], "predicted no, teacher said yes")):
        for i in sorted(wrong, key=lambda i: -abs(p[i] - threshold))[:n]:
            out.append({"kind": label, "p": round(float(p[i]), 3), "teacher": rows[i].get("teacher_score"),
                        "origin": rows[i]["origin"], "id": rows[i]["id"], "text": rows[i]["text"][:220]})
    return out


# ── Using a trained model ─────────────────────────────────────────────

_loaded: dict[str, dict] = {}


def latest_path(owner: str, kind: str = "preference") -> str | None:
    folder = model_dir(owner)
    if not os.path.isdir(folder):
        return None
    found = [(int(m.group(1)), f) for f in os.listdir(folder) if (m := re.fullmatch(rf"{kind}-v(\d+)\.joblib", f))]
    return os.path.join(folder, max(found)[1]) if found else None


def load(owner: str, kind: str = "preference") -> dict | None:
    import joblib

    path = latest_path(owner, kind)
    if path is None:
        return None
    if _loaded.get(f"{owner}|{kind}", {}).get("path") != path:
        _loaded[f"{owner}|{kind}"] = {**joblib.load(path), "path": path}
    return _loaded[f"{owner}|{kind}"]


def predict(saved: dict, texts: list[str], embed_fn=None) -> np.ndarray:
    """Probabilities from a saved model, for texts not in the corpus."""
    rows = [{"id": f"q:{store.text_hash(t)}", "text": t} for t in texts]
    if saved["kind"] == "tfidf":
        X = np.array([scrub(t) for t in texts], dtype=object)
    else:
        X = (embed_fn or vectors.embed_texts)([scrub(r["text"]) for r in rows])
    return saved["model"].predict_proba(X)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the persona preference models (and the whose-feed model)")
    parser.add_argument("--account", choices=store.ACCOUNTS)
    parser.add_argument("--feed", action="store_true", help="also train the whose-feed model")
    args = parser.parse_args()
    conn = store.connect()
    for account in [args.account] if args.account else store.ACCOUNTS:
        train(conn, account, progress=lambda m: print(m, flush=True))
    if args.feed or not args.account:
        train(conn, store.ACCOUNTS[0], kind="feed", progress=lambda m: print(m, flush=True))
    conn.close()


if __name__ == "__main__":
    main()
