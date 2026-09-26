"""Tests for ml/: the labelled dataset and the preference model — synthetic data, no embedding model."""

import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import numpy as np

import ml.train as train
from database.db import init_db, log_impression
from ml.dataset import Example, load_examples, reel_text


def fake_embed(texts):
    return np.array([[1.0, 0.1] if "news" in t else [0.1, 1.0] for t in texts])


def synthetic(n_each: int = 10) -> list[Example]:
    news = [Example(f"n{i}", f"news_page_{i}", "acct", f"breaking news update {i}", 0.8) for i in range(n_each)]
    junk = [Example(f"j{i}", f"fun_page_{i}", "acct", f"dance song mood {i}", 0.0) for i in range(n_each)]
    return news + junk


class DatasetTests(unittest.TestCase):
    def test_reel_text(self):
        self.assertEqual(reel_text("Caption ", None, " speech"), "Caption | speech")
        self.assertEqual(reel_text("", "", ""), "")
        self.assertEqual(len(reel_text("x" * 5000, None, None)), 1500)

    def test_load_examples_one_per_reel_from_one_scorer(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        init_db(conn)
        review = conn.execute("INSERT INTO sessions (account, scorer) VALUES ('acct', 'review')").lastrowid
        llm = conn.execute("INSERT INTO sessions (account, scorer) VALUES ('acct', 'llm-v1')").lastrowid
        rows = [
            (review, "A", 0.2, 0.1, 0), (review, "A", 0.9, None, 0),   # labelled twice → keep 0.9
            (review, "B", 0.0, 0.1, 0), (review, "AD", 0.0, None, 1),  # ads are not examples
            (llm, "C", 0.9, None, 0),                                  # other scorer
        ]
        for seq, (sid, code, value, rule, is_ad) in enumerate(rows, start=1):
            log_impression(dict(session_id=sid, seq=seq, code=code, creator=f"c_{code}", caption=f"cap {code}",
                                score_basis="score", value_score=value, rule_score=rule, is_ad=is_ad), conn)
        examples = {e.code: e for e in load_examples(conn)}
        self.assertEqual(set(examples), {"A", "B"})
        self.assertEqual((examples["A"].value_score, examples["A"].rule_score), (0.9, 0.1))
        self.assertTrue(examples["A"].informative)
        self.assertFalse(examples["B"].informative)


class TrainTests(unittest.TestCase):
    def test_metrics(self):
        m = train.metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.6, 0.1]))
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (0.5, 0.5, 0.5))
        self.assertEqual(m["roc_auc"], 0.75)

    def test_cross_validation_separates_an_easy_signal(self):
        examples = synthetic()
        results = train.evaluate(examples, folds=5, embed_fn=fake_embed)
        self.assertEqual(results["embedding"]["all"]["roc_auc"], 1.0)
        self.assertNotIn("rules", results)  # no rule scores logged on these examples

    def test_train_and_save(self):
        import joblib

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(train, "MODEL_DIR", tmp):
            path = train.train_and_save("embedding", synthetic(), {"embedding": {}}, embed_fn=fake_embed)
            saved = joblib.load(path)
        self.assertEqual(os.path.basename(path), "informative-embedding.joblib")
        self.assertEqual((saved["kind"], saved["trained_on"], saved["positives"]), ("embedding", 20, 10))
        self.assertTrue(saved["model"].predict(fake_embed(["news again"]))[0])


if __name__ == "__main__":
    unittest.main()
