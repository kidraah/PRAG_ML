"""Tests for ml/: the labelled dataset and the preference model — synthetic data, no embedding model."""

import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import numpy as np

import ml.train as train
from automation.feed_capture import ReelMeta
from database.db import init_db, log_impression
from ml.dataset import Example, load_examples, reel_text
from agent import bias
from ml.scorer import FOLLOWED_FLOOR, ModelScorer, lead_line
from perception.local_evaluator import scorer_version


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


class ScorerTests(unittest.TestCase):
    def test_lead_line(self):
        caption = "Kunal Ghosh's car attacked in Kolkata\nTMC alleges BJP hand. Police probe on.\n#news @ndtv"
        self.assertEqual(lead_line(caption), "Kunal Ghosh's car attacked in Kolkata")
        self.assertEqual(lead_line("GST cut on 12 items. Details inside #news"), "GST cut on 12 items.")
        self.assertEqual(lead_line("#viral #reels", cover_text="संसद में आज नया बिल पेश"), "संसद में आज नया बिल पेश")
        self.assertEqual(lead_line("#India vs #Pakistan at #UNGA: Diplomats #trade charges https://t.co/x #news"),
                         "India vs Pakistan at UNGA: Diplomats trade charges")
        self.assertEqual(lead_line("On the anniversary of Dr. Manmohan Singh, we remember him. He led reforms."),
                         "On the anniversary of Dr. Manmohan Singh, we remember him.")
        self.assertEqual(lead_line("Big news! GST cut on 12 items. Details inside"), "Big news! GST cut on 12 items.")
        self.assertEqual(lead_line("Mail tips to desk@news.com"), "Mail tips to desk@news.com")
        self.assertEqual(lead_line("", None, "short"), "")
        self.assertTrue(lead_line("word " * 100, limit=40).endswith(" …"))

    def test_scores_with_the_saved_model(self):
        for kind in ("embedding", "tfidf"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(train, "MODEL_DIR", tmp), mock.patch("stories.grouping.embed", fake_embed):
                path = train.train_and_save(kind, synthetic(), {}, embed_fn=fake_embed)
                self.assertTrue(ModelScorer.available(path))
                scorer = ModelScorer(path)
                news, reason = scorer.score(ReelMeta(code="N", caption="breaking news update"), "", "")
                junk, _ = scorer.score(ReelMeta(code="J", caption="dance song mood"), "", "")
                followed, why = scorer.score(ReelMeta(code="F", caption="dance song mood", creator_following=True), "", "")
                self.assertTrue(scorer.version.startswith(f"model-{kind}-"))
                self.assertTrue(ModelScorer.informative(news))
                self.assertFalse(ModelScorer.informative(junk))
                self.assertTrue(reason.startswith("model p="))
                self.assertEqual((followed, why.endswith("followed account")), (FOLLOWED_FLOOR, True))
        self.assertFalse(ModelScorer.available(os.path.join(tmp, "missing.joblib")))


class ModelBiasTests(unittest.TestCase):
    def test_a_model_only_scores_for_the_bias_it_was_trained_for(self):
        import joblib

        default, tech = bias.DEFAULT_BIAS, bias.BUILT_IN["tech"]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(train, "MODEL_DIR", tmp):
            path = train.train_and_save("embedding", synthetic(), {}, embed_fn=fake_embed)
            self.assertTrue(ModelScorer.available(path, bias=default))
            self.assertFalse(ModelScorer.available(path, bias=tech))
            tech_path = train.train_and_save("tfidf", synthetic(), {}, bias=tech.fingerprint)
            self.assertTrue(ModelScorer.available(tech_path, bias=tech))
            older = os.path.join(tmp, "older.joblib")  # saved before biases existed: the default bias's
            joblib.dump({"kind": "tfidf", "model": None, "created": "2026-09-26"}, older)
            self.assertTrue(ModelScorer.available(older, bias=default))
            self.assertFalse(ModelScorer.available(older, bias=tech))

    def test_which_bias_a_scorers_labels_came_from(self):
        tech = bias.BUILT_IN["tech"]
        with mock.patch.object(train, "list_accounts", return_value=[]):
            self.assertEqual(train.bias_of_scorer("review"), bias.DEFAULT_FINGERPRINT)
            self.assertEqual(train.bias_of_scorer(scorer_version(tech.interests, tech.persona)), tech.fingerprint)
            self.assertEqual(train.bias_of_scorer(scorer_version()), bias.DEFAULT_FINGERPRINT)
            self.assertIsNone(train.bias_of_scorer("0000ffff"))


if __name__ == "__main__":
    unittest.main()
