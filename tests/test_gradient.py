"""Tests for the Gradient (gradient/) and Jev's cover images — no network, no LLM, no MLX."""

import asyncio
import datetime
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from database.db import init_db
from gradient import answers, chat, generate, ingest, label, lora, prefs, retrieval, simulate, store, vectors
from perception import jev, jev_image


def fake_embed(texts):
    """Deterministic bag-of-words vectors: similar words, similar vectors."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.lower().split():
            out[row, int(hashlib.md5(word.strip(".,!?#").encode()).hexdigest(), 16) % 64] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def persona(name="P", fingerprint="fp1", profile=None):
    return SimpleNamespace(name=name, interests="Likes trains.", fingerprint=fingerprint, profile=profile or {
        "summary": "Trains first.", "topic_weights": {"infrastructure": 1.0, "cricket": 0.0},
        "stance": {"supportive": 1.0, "critical": 0.1, "neutral": 0.5}, "languages": {"en": 1.0}},
        watch_words=["train"], skip_words=["makeup"])


class Temp(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.tmp = tmp
        self.enterContext(mock.patch.object(store, "DB_PATH", os.path.join(tmp, "corpus.db")))
        self.enterContext(mock.patch.object(store, "MODEL_DIR", os.path.join(tmp, "models")))
        self.enterContext(mock.patch.object(vectors, "VECTORS_PATH", os.path.join(tmp, "vectors.npz")))
        self.enterContext(mock.patch.object(vectors, "_cache", None))
        self.enterContext(mock.patch.object(vectors, "embed_texts", fake_embed))
        self.enterContext(mock.patch.object(lora, "DATA_DIR", os.path.join(tmp, "lora")))
        self.conn = store.connect()
        self.addCleanup(self.conn.close)


class StoreTests(Temp):
    def test_same_text_is_the_same_item(self):
        b = store.new_batch(self.conn, "t")
        first, new = store.add_item(self.conn, {"origin": "generated", "caption": "A new  Road", "group_id": "g"}, b)
        again, new2 = store.add_item(self.conn, {"origin": "rss", "caption": "a new road", "group_id": "h"}, b)
        self.assertEqual((first, new, new2), (again, True, False))

    def test_splits_are_fixed_per_group(self):
        self.assertEqual(store.split_for("brief-1"), store.split_for("brief-1"))
        shares = [store.split_for(f"g{i}") for i in range(3000)]
        self.assertAlmostEqual(shares.count("train") / 3000, 0.7, delta=0.04)
        b = store.new_batch(self.conn, "t")
        for n in range(5):
            store.add_item(self.conn, {"origin": "generated", "caption": f"reel {n}", "group_id": "one-brief"}, b)
        self.assertEqual(len({r[0] for r in self.conn.execute("SELECT split FROM items")}), 1)


class GenerateTests(Temp):
    def test_briefs_are_seeded_and_follow_the_profile(self):
        p = persona()
        a, b = generate.brief("acct", 3, p), generate.brief("acct", 3, p)
        self.assertEqual(a, b)
        topics = [generate.brief("acct", n, p)["topic"] for n in range(300)]
        self.assertGreater(topics.count("infrastructure"), 200)  # weight 1 vs 0, plus a little exploration
        self.assertGreater(len(set(topics)), 3)  # exploration still reaches other topics
        infra = next(generate.brief("acct", n, p) for n in range(300) if topics[n] == "infrastructure")
        self.assertTrue(all(s["stance"] in ("supportive", "critical", "neutral") for s in infra["slots"]))
        self.assertIn("JSON", generate.user_message(infra))

    def test_writer_field_names_are_normalised(self):
        reel = generate.normalised({"creator_handle": "@x", "burned_text": "Big", "caption": " c ", "transcript": ["a", "b"],
                                    "viewer_action": "LIKE", "take": "nice"})
        self.assertEqual((reel["creator"], reel["on_screen"], reel["caption"], reel["transcript"], reel["viewer_action"],
                          reel["viewer_take"]), ("@x", "Big", "c", "a b", "like", "nice"))
        self.assertEqual(generate.normalised({"viewer_action": "dance"})["viewer_action"], "watch")

    def test_a_saved_brief_fills_the_feed_takes_and_intent_labels(self):
        p, b = persona(), generate.brief("acct", 0, persona())
        reels = [{"creator": "@a b", "caption": "cap one", "on_screen": "s", "transcript": "words " * 10,
                  "viewer_action": "skip", "viewer_take": "not for me", "stance": "neutral", "format": "vlog",
                  "language": "en"}]
        self.assertEqual(generate.save_brief(self.conn, b, reels, p, "b1"), 1)
        item = self.conn.execute("SELECT * FROM items").fetchone()
        self.assertEqual((item["origin"], item["creator"], item["group_id"]), ("generated", "@a", b["id"]))
        self.assertEqual(self.conn.execute("SELECT via FROM feed").fetchone()[0], "generator")
        self.assertEqual(tuple(self.conn.execute("SELECT action, take FROM takes").fetchone()), ("skip", "not for me"))
        self.assertEqual(self.conn.execute("SELECT label FROM labels WHERE teacher = 'generator'").fetchone()[0], 0)

    def test_takes_are_written_for_feed_reels_without_one(self):
        b = store.new_batch(self.conn, "t")
        for n in range(3):
            item, _ = store.add_item(self.conn, {"origin": "instagram", "caption": f"session reel {n}"}, b)
            store.add_to_feed(self.conn, "mlproject01", item, b, "session")
        self.conn.execute("INSERT INTO takes VALUES (?, 'mlproject01', 'like', 'had one')", (item,))
        self.conn.commit()
        seen = []

        async def ask(_session, url, headers, payload, timeout, spend):
            seen.append(payload["messages"][1]["content"])
            return {"takes": [{"n": 1, "viewer_action": "save", "viewer_take": "useful"},
                              {"n": 2, "viewer_action": "skip", "viewer_take": "meh"}, {"n": 9, "viewer_take": "x"}]}

        with mock.patch.object(generate.biases, "load", return_value=persona()), \
                mock.patch.object(generate, "_ask", ask), mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}):
            stats = asyncio.run(generate.write_takes("mlproject01", progress=lambda m: None))
        self.assertEqual(stats["done"], 2)  # the reel that had a take wasn't asked about; n=9 isn't a reel
        self.assertNotIn("session reel 2", seen[0])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM takes").fetchone()[0], 3)

    def test_the_request_names_the_backend(self):
        b = generate.brief("acct", 0, persona())
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}):
            url, headers, body = generate.request(b, "viewer", "openrouter")
        self.assertIn("openrouter.ai", url)
        self.assertEqual((headers["Authorization"], body["reasoning"]), ("Bearer k", {"enabled": False}))
        url, headers, body = generate.request(b, "viewer", "local")
        self.assertEqual((headers, body["reasoning_effort"]), ({}, "none"))


class IngestTests(Temp):
    def reels_db(self):
        path = os.path.join(self.tmp, "reels.db")
        conn = sqlite3.connect(path)
        init_db(conn)
        live = conn.execute("INSERT INTO sessions (account, dry_run) VALUES ('mlproject01', 0)").lastrowid
        dry = conn.execute("INSERT INTO sessions (account, dry_run) VALUES ('mlproject01', 1)").lastrowid
        for sid, code, caption in ((live, "A1", "Vande Bharat to Indore"), (dry, "B2", "Kohli century"), (dry, "B2", "x"),
                                   (live, "AD", "buy now")):
            conn.execute("INSERT INTO impressions (session_id, seq, code, caption, creator, is_ad, shown_at) "
                         "VALUES (?, 1, ?, ?, 'c', ?, '2026-10-01 10:00:00')", (sid, code, caption, int(code == "AD")))
        conn.commit()
        conn.close()
        return path

    def test_session_reels_join_the_feed_once_dry_runs_included(self):
        path = self.reels_db()
        self.assertEqual(ingest.ingest_sessions(self.conn, ["mlproject01"], path), {"mlproject01": 2})
        self.assertEqual(ingest.ingest_sessions(self.conn, ["mlproject01"], path), {"mlproject01": 0})
        rows = self.conn.execute("SELECT i.origin, i.source_url, f.via, f.shown_at FROM items i JOIN feed f "
                                 "ON f.item_id = i.id ORDER BY i.id").fetchall()
        self.assertEqual([tuple(r) for r in rows][0], ("instagram", "https://www.instagram.com/reel/A1/", "session",
                                                       "2026-10-01 10:00:00"))

    def test_the_recommender_draws_headlines_by_score(self):
        b = store.new_batch(self.conn, "t")
        for n in range(40):
            item, _ = store.add_item(self.conn, {"origin": "rss", "title": f"h{n}", "text": f"headline {n}"}, b)
            store.add_label(self.conn, item, "acct", "fp", "jev", "v", 0.95 if n < 10 else 0.05, int(n < 10))
        wiki, _ = store.add_item(self.conn, {"origin": "wikipedia", "text": "an article"}, b)
        store.add_label(self.conn, wiki, "acct", "fp", "jev", "v", 1.0, 1)
        self.assertEqual(ingest.recommend(self.conn, "acct", n=10), 10)
        picked = [r[0] for r in self.conn.execute("SELECT i.title FROM feed f JOIN items i ON i.id = f.item_id")]
        self.assertGreaterEqual(sum(int(t[1:]) < 10 for t in picked), 7)  # mostly what it scores highly
        self.assertNotIn(None, picked)  # headlines only: the article isn't served
        self.assertEqual(ingest.recommend(self.conn, "acct", n=10), 0)  # topped up already

    def test_language_guess(self):
        self.assertEqual(ingest.language_of("भारत की नई ट्रेन"), "hi")
        self.assertEqual(ingest.language_of("bhai ye train kya mast hai"), "hinglish")
        self.assertEqual(ingest.language_of("A new train", "hi"), "hi")


class LabelTests(Temp):
    def test_jev_labels_what_is_missing_and_stops_at_the_cap(self):
        b = store.new_batch(self.conn, "t")
        for n in range(6):
            store.add_item(self.conn, {"origin": "generated", "caption": f"reel {n}", "creator": "@c"}, b)
        p = persona()
        calls = []

        async def judge(_session, text, *_a, **_k):
            calls.append(text)
            return jev.JevScore(value_score=0.75 if "reel 1" in text else 0.25, confidence=1, seconds=0.1, cost=0.4)

        with mock.patch.object(label.biases, "load", return_value=p), mock.patch.object(label.jev, "judge_async", judge):
            stats = asyncio.run(label.label_account(self.conn, "acct", parallel=1, max_spend=1.0, progress=lambda m: None))
            self.assertEqual((stats["done"], stats["left"]), (3, 3))  # 3 × $0.40 passes the $1 cap
            stats = asyncio.run(label.label_account(self.conn, "acct", parallel=1, max_spend=10, progress=lambda m: None))
        self.assertEqual(stats["done"], 3)
        self.assertEqual(len(calls), 6)  # nothing judged twice
        self.assertTrue(calls[0].startswith("Creator: @c\nCaption: reel"))
        labels = dict(self.conn.execute("SELECT i.caption, l.label FROM labels l JOIN items i ON i.id = l.item_id"))
        self.assertEqual((labels["reel 1"], labels["reel 2"]), (1, 0))


class PrefsTests(Temp):
    def corpus(self):
        b = store.new_batch(self.conn, "t")
        rng = np.random.default_rng(0)
        likes = {"mlproject01": "railway expressway metro train", "mlproject02": "union strike wages workers"}
        for n in range(600):
            topic = n % 3
            words = ["railway expressway metro train", "union strike wages workers", "makeup skincare fashion"][topic]
            text = f"{' '.join(rng.permutation(words.split()))} reel {n} @creator{n} Pune"
            item, _ = store.add_item(self.conn, {"origin": "generated", "text": text, "group_id": f"g{n // 4}"}, b)
            for account, liked in likes.items():
                y = int(words == liked) if rng.random() > 0.1 else int(words != liked)
                store.add_label(self.conn, item, account, f"fp-{account}", "jev", "v", float(y), y)
            store.add_to_feed(self.conn, "mlproject01" if topic == 0 else "mlproject02", item, b, "generator")
        self.conn.commit()

    def load(self, account):
        return persona(fingerprint=f"fp-{account}")

    def test_each_persona_gets_its_own_model_and_a_frozen_eval_set(self):
        self.corpus()
        with mock.patch.object(prefs.biases, "load", side_effect=self.load):
            one = prefs.train(self.conn, "mlproject01", embed_fn=fake_embed, progress=lambda m: None)
            two = prefs.train(self.conn, "mlproject02", embed_fn=fake_embed, progress=lambda m: None)
            again = prefs.train(self.conn, "mlproject01", embed_fn=fake_embed, progress=lambda m: None)
        self.assertEqual((one["version"], two["version"], again["version"]), (1, 1, 2))
        for report in (one, two):
            best = report["candidates"][report["chosen"]]["test"]
            self.assertGreater(best["f1"], report["candidates"]["majority"]["test"]["f1"])
            self.assertGreater(best["pr_auc"], 0.7)  # 10% of labels are flipped on purpose
        self.assertEqual(one["frozen_eval"]["n"], again["frozen_eval"]["n"])
        self.assertTrue(prefs.latest_path("mlproject01").endswith("preference-v2.joblib"))
        self.assertTrue(prefs.latest_path("mlproject02").endswith("preference-v1.joblib"))
        self.assertEqual(prefs.load("mlproject02")["fingerprint"], "fp-mlproject02")  # never the other persona's
        self.assertIn("railway", " ".join(one["top_features"]["towards_yes"]))
        self.assertIn("union", " ".join(two["top_features"]["towards_yes"]))

    def test_features_never_see_handles_or_the_persona_city(self):
        self.assertEqual(prefs.scrub("@desi_infra Pune ki metro #Pune puneri").split(), ["ki", "metro"])
        self.assertNotIn("indore", prefs.scrub("Indore's road").lower())

    def test_the_whose_feed_model(self):
        self.corpus()
        report = prefs.train(self.conn, "mlproject01", kind="feed", embed_fn=fake_embed, progress=lambda m: None)
        self.assertEqual(report["account"], "shared")
        self.assertGreater(report["candidates"][report["chosen"]]["test"]["pr_auc"], 0.9)
        saved = prefs.load("shared", "feed")
        self.assertGreater(prefs.predict(saved, ["railway metro train"], embed_fn=fake_embed)[0], 0.5)

    def test_too_little_data_says_so(self):
        with mock.patch.object(prefs.biases, "load", side_effect=self.load), self.assertRaises(ValueError):
            prefs.train(self.conn, "mlproject01", embed_fn=fake_embed, progress=lambda m: None)


class RetrievalTests(Temp):
    def test_a_persona_only_finds_its_own_feed(self):
        b = store.new_batch(self.conn, "t")
        mine, _ = store.add_item(self.conn, {"origin": "generated", "text": "fuel prices rise again petrol"}, b)
        theirs, _ = store.add_item(self.conn, {"origin": "generated", "text": "fuel prices petrol rise"}, b)
        store.add_to_feed(self.conn, "mlproject01", mine, b, "generator")
        store.add_to_feed(self.conn, "mlproject02", theirs, b, "generator")
        with mock.patch.object(retrieval.prefs, "load", return_value=None):
            found = retrieval.search(self.conn, "mlproject01", "petrol fuel prices")
            self.assertEqual([i["id"] for i in found["items"]], [mine])
            self.assertFalse(retrieval.search(self.conn, "mlproject01", "cricket world cup")["enough"])


class ChatTests(unittest.TestCase):
    def test_citations_are_checked_against_what_was_retrieved(self):
        text, cited, invalid = chat.checked_citations("Roads are better [1][3]. Also [2] and [1].", 2)
        self.assertEqual((text, cited, invalid), ("Roads are better [1]. Also [2] and [1].", [1, 2], [3]))

    def test_an_answer_cites_only_its_evidence(self):
        evidence = {"items": [{"id": "x", "text": "petrol up", "creator": "@a"}], "model": "m", "enough": True}
        with mock.patch.object(chat.retrieval, "search", return_value=evidence), \
                mock.patch.object(chat, "generate", return_value=("Prices are up [1] [4].", "lora-v1")) as gen, \
                mock.patch.object(chat.lora, "system_prompt", return_value="SYS"):
            a = chat.answer(None, "mlproject01", "fuel?")
        self.assertEqual((a["text"], a["cited"], a["invalid_citations"], a["adapter"], a["typing"]),
                         ("Prices are up [1].", [1], [4], "lora-v1", False))
        self.assertIn("[1] @a: petrol up", gen.call_args.args[1])  # the evidence is in the prompt

    def test_a_cut_off_reply_ends_at_its_last_sentence(self):
        self.assertEqual(chat.complete("Roads are **better** [1]. Fares went up and"), "Roads are better [1].")
        self.assertEqual(chat.complete("Done."), "Done.")
        self.assertEqual(chat.complete("no stop at all here"), "no stop at all here…")

    def test_reasoning_tags_and_stray_tokens_are_cleaned(self):
        self.assertEqual(chat.clean("<think>\n\n퍙\n\nGood breakdown [3]."), "Good breakdown [3].")
        self.assertEqual(chat.clean("<think>plan</think>Answer."), "Answer.")
        self.assertEqual(chat.clean("</tool_call>\n\n\n\nSeeing this makes me angry. [5]"), "Seeing this makes me angry. [5]")

    def test_a_debate_streams_turn_by_turn_and_stops_when_asked(self):
        seen = []

        def fake_generate(account, system, user, max_tokens=0, *, history=None, on_text=None, should_stop=lambda: False):
            on_text("Half a")
            seen.append([e.get("text") for e in job["events"]])  # the panel sees the reply being written
            return (f"{account} says it.", "lora-v2")

        job = {"params": {"topic": "metro fares", "rounds": 2}, "events": [], "stop": False}
        with mock.patch.object(chat.retrieval, "search", return_value={"items": [], "model": None, "enough": False}), \
                mock.patch.object(chat, "generate", fake_generate), mock.patch.object(chat.lora, "system_prompt",
                                                                                      return_value="SYS"):
            chat.debate_job(job, None)
        kinds = [(e["type"], e.get("account"), e.get("round")) for e in job["events"]]
        self.assertEqual(kinds, [("turn", "mlproject01", 1), ("turn", "mlproject02", 1), ("turn", "mlproject01", 2),
                                 ("turn", "mlproject02", 2), ("moderator", None, None)])
        self.assertEqual(seen[0], ["Half a"])
        self.assertFalse(any(e["typing"] for e in job["events"]))
        job = {"params": {"topic": "metro fares", "rounds": 3}, "events": [], "stop": True}
        with mock.patch.object(chat, "generate", fake_generate):
            chat.debate_job(job, None)
        self.assertEqual(job["events"], [])

    def test_chat_passes_the_conversation_so_far(self):
        job = {"params": {"account": "mlproject01", "question": "and fares?",
                          "history": [{"role": "user", "text": "metro?"}, {"role": "assistant", "text": "Late [1]."}]},
               "events": [], "stop": False}
        with mock.patch.object(chat.retrieval, "search", return_value={"items": [], "model": None, "enough": False}), \
                mock.patch.object(chat, "generate", return_value=("Up.", "lora-v2")) as gen, \
                mock.patch.object(chat.lora, "system_prompt", return_value="SYS"):
            chat.chat_job(job, None)
        self.assertEqual(gen.call_args.kwargs["history"][1]["text"], "Late [1].")
        self.assertEqual(job["events"][0]["text"], "Up.")

    def test_no_evidence_means_saying_so(self):
        with mock.patch.object(chat.retrieval, "search", return_value={"items": [], "model": None, "enough": False}), \
                mock.patch.object(chat, "generate", return_value=("My feed hasn't shown me that.", None)) as gen, \
                mock.patch.object(chat.lora, "system_prompt", return_value="SYS"):
            a = chat.answer(None, "mlproject01", "quantum?")
        self.assertIn("No reels in your feed", gen.call_args.args[1])
        self.assertFalse(a["grounded"])

    def test_gap_between_two_answers(self):
        a = {"account": "mlproject01", "text": "roads good", "evidence": [{"stance": "supportive", "text": "x"}]}
        b = {"account": "mlproject02", "text": "roads bad", "evidence": [{"stance": "critical", "text": "y"}]}
        with mock.patch.object(chat.vectors, "embed_texts", fake_embed), mock.patch.object(chat.prefs, "load", return_value=None):
            g = chat.gap([a, b])
        self.assertAlmostEqual(g["evidence_divergence"], 1.0, places=2)

    def test_jobs_need_the_models(self):
        with mock.patch.object(chat, "availability", return_value={"ok": False, "why": "no model"}), \
                self.assertRaises(RuntimeError):
            chat.submit("chat", {"account": "mlproject01", "question": "hi"})


class LoraTests(Temp):
    def feed(self, accounts=("mlproject01",), n=200):
        b = store.new_batch(self.conn, "t")
        for i in range(n):
            for account in accounts:
                item, _ = store.add_item(self.conn, {"origin": "generated", "text": f"{account} reel {i} about trains",
                                                     "caption": f"cap {i}", "topic": "infrastructure",
                                                     "group_id": f"g{i // 4}"}, b)
                store.add_to_feed(self.conn, account, item, b, "generator")
                self.conn.execute("INSERT INTO takes VALUES (?, ?, 'like', ?)", (item, account, f"{account} take {i}"))
        store.add_item(self.conn, {"origin": "wikipedia", "title": "Saturn", "text": "a planet"}, b)
        self.conn.executescript(answers.TABLE)

    def test_written_answers_train_and_follow_the_splits(self):
        self.feed()
        with mock.patch.object(answers.biases, "load", return_value=persona()):
            todo = answers.tasks(self.conn, "mlproject01", 40)
        self.assertEqual(len(todo), 40)
        for t in todo:  # a task's reels share its split
            self.assertEqual({it["split"] for it in t["items"]}, {t["split"]})
        for t in todo:
            self.conn.execute("INSERT INTO answers VALUES (?, 'mlproject01', 'answer', ?, 'Q?', ?, ?, 'w', 'now')",
                              (t["id"], json.dumps([it["id"] for it in t["items"]]), f"Written answer {t['id']} [1].", t["split"]))
        with mock.patch.object(lora.biases, "load", return_value=persona()):
            examples = lora.build_examples(self.conn, "mlproject01")
        texts = {s: " ".join(json.dumps(e) for e in examples[s]) for s in examples}
        for t in todo:
            name = {"val": "valid"}.get(t["split"], t["split"])
            self.assertIn(f"Written answer {t['id']}", texts[name])
            for other in set(texts) - {name}:
                self.assertNotIn(f"Written answer {t['id']}", texts[other])
        kinds = " ".join(texts.values())
        self.assertIn("Reels from your feed", kinds)
        self.assertIn("Thoughts?", kinds)
        self.assertIn("hasn't shown me anything about Saturn", kinds)
        self.assertNotIn("saw it differently", kinds)  # no template answers any more
        train = examples["train"]
        written = sum("Written answer" in e["messages"][2]["content"] for e in train)
        self.assertGreater(written, (len(train) - written) // 2)  # short takes stay a minority

    def test_debate_tasks_quote_the_other_persona(self):
        self.feed(("mlproject01", "mlproject02"), n=120)
        todo = answers.tasks(self.conn, "mlproject01", 100)
        debates = [t for t in todo if t["kind"] == "debate"]
        self.assertGreater(len(debates), 5)
        self.assertIn("mlproject02 take", debates[0]["question"])
        self.assertTrue(debates[0]["question"].startswith("Debate on infrastructure."))
        self.assertIn("Task 1 (debate; the question is fixed", answers.prompt(debates[:1]))

    def test_training_log_is_read(self):
        old = "Iter 1: Val loss 3.100, Val took 1s\nIter 20: Train loss 2.500, It/sec 0.80\nIter 40: Train loss 2.000"
        parsed = lora.parse_log(old)
        self.assertEqual((parsed["val_loss"][0]["loss"], len(parsed["train_loss"])), (3.1, 2))
        new = ("  \x1b[1miter   train_loss     tok/s     tokens\x1b[0m\n  \x1b[38;5;244m   1\x1b[0m    \x1b[1;35mval\x1b[0m "
               "\x1b[1m4.162\x1b[0m    5.21s\n     2    \x1b[1;32m4.143 ▼\x1b[0m    \x1b[1m   27\x1b[0m      0.4k\n"
               "    10    2.498 ▲       49      2.0k\nTrainable parameters: 0.182%")
        parsed = lora.parse_log(new)
        self.assertEqual((parsed["val_loss"], parsed["train_loss"][-1], parsed["tokens_per_sec"]),
                         ([{"iter": 1, "loss": 4.162}], {"iter": 10, "loss": 2.498}, 38.0))


class SimulateTests(Temp):
    def test_two_weeks_of_simulated_sessions_kept_apart_and_labelled(self):
        from database import db

        demo_dir = os.path.join(self.tmp, "demo")
        for name, value in (("DEMO_DIR", demo_dir), ("DEMO_DIGEST_DIR", os.path.join(demo_dir, "digests")),
                            ("DEMO_FEED_MODEL", os.path.join(demo_dir, "m.joblib")),
                            ("DEMO_FEED_VECTORS", os.path.join(demo_dir, "v.npz"))):
            self.enterContext(mock.patch.object(simulate, name, value))
        self.enterContext(mock.patch.object(db, "DEMO_DB_PATH", os.path.join(demo_dir, "reels.db")))
        live = os.path.join(self.tmp, "live.db")
        self.enterContext(mock.patch.object(db, "DB_PATH", live))
        b = store.new_batch(self.conn, "t")
        rng = np.random.default_rng(1)
        topics = list(taxonomy_topics := ["cricket", "labour", "astrology", "infrastructure", "indie_music"])
        for n in range(900):
            topic = topics[n % len(topics)]
            news = {"news_title": f"Real story {n}", "news_source": "The Hindu",
                    "news_published_at": "2026-10-13T08:00:00+00:00"} if n % 7 == 0 else {}
            item, _ = store.add_item(self.conn, {"origin": "generated", "caption": f"{topic} reel {n}", "topic": topic,
                                                 "creator": f"@c{n % 40}", "stance": "neutral", "meta": news,
                                                 "source_url": "https://www.thehindu.com/x" if news else None}, b)
            for account in store.ACCOUNTS:
                likes = {"mlproject01": ("cricket", "infrastructure"), "mlproject02": ("labour", "indie_music")}[account]
                store.add_label(self.conn, item, account, f"fp-{account}", "jev", "v",
                                float(0.85 if topic in likes else rng.uniform(0, 0.45)), int(topic in likes))
            store.add_to_feed(self.conn, store.ACCOUNTS[n % 2], item, b, "generator")
        self.conn.commit()
        fake = lambda account: SimpleNamespace(name=account, interests="x", fingerprint=f"fp-{account}", persona=True,  # noqa: E731
                                               profile={"exploration": {"start_temperature": 0.8, "end_temperature": 0.12}})
        with mock.patch.object(simulate.biases, "load", side_effect=fake), \
                mock.patch("stories.digest.load_bias", side_effect=fake), \
                mock.patch("ml.feed_model.train", return_value=None):
            out = simulate.build(days=4, sessions_per_day=2, today=datetime.date(2026, 10, 15), progress=lambda m: None,
                                 embed_fn=fake_embed)
        self.assertFalse(os.path.exists(live))  # the real database is never touched
        demo = sqlite3.connect(os.path.join(demo_dir, "reels.db"))
        sessions = demo.execute("SELECT simulated_date, simulated_start, generated_at FROM demo_sessions").fetchall()
        self.assertEqual(len(sessions), 2 * 4 * 2)
        self.assertEqual(sorted({s[0] for s in sessions}), ["2026-10-11", "2026-10-12", "2026-10-13", "2026-10-14"])
        self.assertTrue(all(s[2] > "2026-10-20" for s in sessions))  # the real generation time, kept apart
        news_days = demo.execute("SELECT MIN(i.shown_at) FROM impressions i JOIN demo_items d ON d.code = i.code "
                                 "WHERE d.news_title IS NOT NULL").fetchone()[0]
        self.assertGreaterEqual(news_days[:10], "2026-10-12")  # a story never shows before its publication day
        actions = dict(demo.execute("SELECT decision, COUNT(*) FROM impressions GROUP BY decision").fetchall())
        self.assertTrue({"SKIP", "FULL_WATCH"} <= set(actions))
        first = demo.execute("SELECT line FROM demo_log ORDER BY session_id, n LIMIT 1").fetchone()[0]
        self.assertIn("DEMO — simulated session", first)
        reel_lines = [r[0] for r in demo.execute("SELECT line FROM demo_log WHERE line LIKE '#%'")]
        self.assertRegex(reel_lines[0], r"^#\d+\s+\S+\s+.*→\s+(SKIP|WATCH|FULL_WATCH|REWATCH)\s+[\d.]+s")
        digests = os.listdir(os.path.join(demo_dir, "digests", "mlproject01"))
        self.assertIn("2026-10-14.json", digests)
        with open(os.path.join(demo_dir, "digests", "mlproject01", "2026-10-14.html")) as fh:
            page = fh.read()
        self.assertIn("DEMO — simulated sessions", page)
        self.assertNotIn("instagram.com/reels/demo", page)
        self.assertIn("mlproject01", out["corpus_feed"])
        self.assertTrue(self.conn.execute("SELECT 1 FROM feed WHERE via = 'demo_session'").fetchone())


class JevImageTests(unittest.TestCase):
    def image(self, size, fmt="JPEG", exif_rotate=False):
        from PIL import Image

        img = Image.new("RGB", size, (200, 30, 30))
        out = io.BytesIO()
        if exif_rotate:
            exif = Image.Exif()
            exif[0x0112] = 6  # rotate 90°
            img.save(out, fmt, exif=exif)
        else:
            img.save(out, fmt)
        return out.getvalue()

    def size(self, data):
        from PIL import Image

        return Image.open(io.BytesIO(data)).size

    def test_covers_fit_jevs_limits(self):
        self.assertEqual(max(self.size(jev_image.prepared(self.image((1080, 1920))))), 768)
        small = self.size(jev_image.prepared(self.image((200, 300), "PNG")))
        self.assertGreaterEqual(min(small), 280)
        self.assertEqual(self.size(jev_image.prepared(self.image((1920, 1080), exif_rotate=True))), (432, 768))
        for bad in (self.image((60, 60)), b"not an image", self.image((300, 5000))):
            with self.assertRaises(jev_image.ImageRejected):
                jev_image.prepared(bad)
        self.assertTrue(jev_image.image_part(b"x")["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_the_cover_cache_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(jev_image, "JEV_IMAGE_CACHE_MB", 0.001):
            jev_image.save_cover("A", b"x" * 600, tmp)
            jev_image.save_cover("B", b"y" * 600, tmp)
            self.assertEqual(jev_image.load_cover("B", tmp), b"y" * 600)
            self.assertIsNone(jev_image.load_cover("A", tmp))  # the oldest went

    def test_images_always_go_to_the_decisions_api(self):
        url, body = jev.build_request("s", {"q": {}}, b"\xff\xd8")
        self.assertTrue(url.endswith("/api/alpha/decisions"))
        self.assertEqual((body["state"][0], body["state"][1]["type"]), ("s", "image_url"))
        url, body = jev.build_request("s", {"q": {}}, endpoint="systemone")
        self.assertEqual((url.endswith("/api/v1/systemone"), body["state"]), (True, "s"))

    def test_a_rejected_image_falls_back_to_text(self):
        calls = []

        async def ask(_session, state, asked, timeout, image=None):
            calls.append(image is not None)
            if image is not None:
                raise jev.JevRejected("Jev HTTP 413: too big")
            return {"answers": {"value": {"score": 4.0}}, "usage": {"cost": 1e-5}}, 0.2

        cover = self.image((1080, 1920))
        with mock.patch.object(jev, "_ask", ask):
            score = asyncio.run(jev.judge_async(None, "Caption: x", "who", labels=False, cover=cover, vision=True))
        self.assertEqual((calls, score.image_used, score.value_score), ([True, False], False, 1.0))
        self.assertIn("image rejected", score.fallback)
        with mock.patch.object(jev, "_ask", ask):
            score = asyncio.run(jev.judge_async(None, "Caption: x", "who", labels=False, cover=None, vision=True))
        self.assertEqual(score.fallback, "no cover image")

    def test_vision_changes_the_fingerprint_text_does_not(self):
        self.assertEqual(jev.jev_version("x", True, vision=False), jev.jev_version("x", True, vision=False))
        self.assertNotEqual(jev.jev_version("x", True, vision=False), jev.jev_version("x", True, vision=True))


if __name__ == "__main__":
    unittest.main()
