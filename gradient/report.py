"""
PRAG — the Gradient's results, as Markdown, from the corpus and the saved run reports.

Every number is read from datasets/gradient/corpus.db and the runs it
records; nothing is typed in. Re-run after every retraining.

    python -m gradient.report [--out docs/results/gradient.md]
"""

import argparse
import json
import os

from agent import bias as biases
from gradient import store
from gradient.api import persona_card
from gradient.prefs import MODEL_NAMES

LANGS = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}


def fmt(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def runs(conn, account, kind):
    return [dict(r) | {"report": json.loads(r["report"])} for r in
            conn.execute("SELECT * FROM runs WHERE account = ? AND kind = ? AND status = 'done' ORDER BY version",
                         (account, kind))]


def pref_section(title: str, history: list[dict]) -> list[str]:
    if not history:
        return [f"### {title}", "", "Not trained yet.", ""]
    r = history[-1]["report"]
    d = r["data"]
    out = [f"### {title}", "", f"Task: {r['task']}  ",
           f"Latest: **v{r['version']}**, {d['items']:,} items: train {d['train_n']:,} ({d['train_positives']:,} yes), "
           f"validation {d['val_n']:,} ({d['val_positives']:,}), test {d['test_n']:,} ({d['test_positives']:,}). "
           f"Chosen: **{MODEL_NAMES[r['chosen']]}** ({r['chosen_because']}).", "",
           "| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Precision | Recall | F1 | Prec@50 | Brier |",
           "|---|---|---|---|---|---|---|---|---|"]
    for kind, c in r["candidates"].items():
        t = c["test"]
        name = c["name"] + (" **(chosen)**" if kind == r["chosen"] else "")
        out.append(f"| {name} | {fmt(c['val']['pr_auc'])} | {fmt(t['pr_auc'])} | {fmt(t['roc_auc'])} | "
                   f"{fmt(t['precision'])} | {fmt(t['recall'])} | {fmt(t['f1'])} | {fmt(t['precision_at_50'])} | "
                   f"{fmt(t['brier'])} |")
    conf = r["candidates"][r["chosen"]]["test"]["confusion"]
    out += ["", f"Confusion on test (chosen model, threshold {r['candidates'][r['chosen']]['threshold']} picked on "
                f"validation): TN {conf['tn']:,}, FP {conf['fp']:,}, FN {conf['fn']:,}, TP {conf['tp']:,}.", "",
            "By slice (test):", "", "| Slice | Items | PR-AUC | F1 |", "|---|---|---|---|"]
    for s in r["slices"]["origin"] + r["slices"]["language"]:
        out.append(f"| {s['slice']} | {s['n']:,} | {'–' if s.get('note') else fmt(s['pr_auc'])} | "
                   f"{'–' if s.get('note') else fmt(s['f1'])} |")
    out += ["", f"Words that push towards yes: {', '.join(r['top_features']['towards_yes'])}.  ",
            f"Towards no: {', '.join(r['top_features']['towards_no'])}.", ""]
    if len(history) > 1 or r.get("frozen_eval"):
        out += ["Versions, on the same frozen evaluation set:", "", "| Version | Trained | Items | PR-AUC | F1 |",
                "|---|---|---|---|---|"]
        for h in history:
            fe = h["report"].get("frozen_eval") or {}
            out.append(f"| v{h['version']} | {h['created_at'][:16]} | {h['items']:,} | {fmt(fe.get('pr_auc'))} | "
                       f"{fmt(fe.get('f1'))} |")
        out.append("")
    return out


def lora_section(account: str, history: list[dict]) -> list[str]:
    if not history:
        return [f"### @{account}: persona language model", "", "No adapter trained yet.", ""]
    out = [f"### @{account}: persona language model", "",
           "| Version | Examples | Steps | Minutes | Scored on | Held-out ppl, base | With adapter | Adapter on the other persona |",
           "|---|---|---|---|---|---|---|---|"]
    for h in history:
        r = h["report"]
        e = r["eval"]
        out.append(f"| lora-v{r['version']}{' (continued)' if r.get('resumed_from') else ''} | {r['examples']['train']:,} | "
                   f"{r['iters']} | {r['minutes']} | {r.get('eval_set') or 'its own held-out examples'} | "
                   f"{fmt(e['base']['ppl'], 2)} | {fmt(e['adapter']['ppl'], 2)} | "
                   f"{fmt((e.get('other') or {}).get('ppl'), 2)} |")
    return out + [""]


def demo_section() -> list[str]:
    """The simulated sessions (gradient/simulate.py), labelled as such."""
    import sqlite3

    from database.db import DEMO_DB_PATH

    if not os.path.exists(DEMO_DB_PATH):
        return []
    demo = sqlite3.connect(DEMO_DB_PATH)
    out = ["## Simulated sessions", "",
           "Not Instagram observations: 14 days of sessions simulated by `python -m gradient.simulate` from the "
           "corpus's written reels, Jev's scores and the session engagement policy, in their own database. Dates are "
           "simulated; each session also records when it was generated.", "",
           "| Account | Simulated days | Sessions | Reels | Watched | Liked | Fit the persona, first day → last day |",
           "|---|---|---|---|---|---|---|"]
    for account in store.ACCOUNTS:
        days = demo.execute("SELECT ds.simulated_date, AVG(i.value_score >= 0.4) FROM impressions i JOIN demo_sessions ds "
                            "ON ds.session_id = i.session_id WHERE ds.account = ? AND i.is_ad = 0 GROUP BY 1 ORDER BY 1",
                            (account,)).fetchall()
        n = demo.execute("SELECT COUNT(DISTINCT ds.session_id), COUNT(*), SUM(i.decision <> 'SKIP'), SUM(i.liked) "
                         "FROM impressions i JOIN demo_sessions ds ON ds.session_id = i.session_id WHERE ds.account = ? "
                         "AND i.is_ad = 0", (account,)).fetchone()
        if days:
            out.append(f"| @{account} | {days[0][0]} to {days[-1][0]} | {n[0]} | {n[1]:,} | {n[2]:,} | {n[3]:,} | "
                       f"{days[0][1]:.0%} → {days[-1][1]:.0%} |")
    batches = demo.execute("SELECT COALESCE(batch, 'base'), MIN(generated_at), COUNT(*) FROM demo_sessions GROUP BY 1 "
                           "ORDER BY 2").fetchall()
    demo.close()
    return out + ["", "Generated: " + "; ".join(f"{name}, {n} sessions, {at} UTC" for name, at, n in batches) + ".", ""]


def build(conn) -> str:
    total = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    origins = conn.execute("SELECT origin, COUNT(*) FROM items GROUP BY origin ORDER BY 2 DESC").fetchall()
    langs = conn.execute("SELECT language, COUNT(*) FROM items GROUP BY language ORDER BY 2 DESC").fetchall()
    usd = {k: 0.0 for k in ("writer", "jev")}
    for row in conn.execute("SELECT counts FROM batches"):
        counts = json.loads(row[0] or "{}")
        usd["writer"] += float((counts.get("generated") or {}).get("usd") or 0)
    out = ["# PRAG Gradient: results", "",
           "Generated by `python -m gradient.report` from `datasets/gradient/corpus.db` and the saved runs. "
           "Every number is measured.", "", "## The corpus", "",
           f"{total:,} unique items in {conn.execute('SELECT COUNT(DISTINCT group_id) FROM items').fetchone()[0]:,} "
           "split groups.", "", "| Where it came from | Items |", "|---|---|"]
    out += [f"| {store.ORIGINS.get(o, o)} | {n:,} |" for o, n in origins]
    out += ["", "Languages: " + ", ".join(f"{LANGS.get(lang, lang)} {n:,}" for lang, n in langs) + ".", ""]
    for account in store.ACCOUNTS:
        p = persona_card(conn, account)
        out += [f"### @{account}: {p['name']}", "", p["summary"] or "", "",
                "Feed: " + ", ".join(f"{n:,} {via}" for via, n in p["feed"].items()) +
                f"; {p['takes']:,} reactions in the persona's own words. Jev says it would watch "
                f"{p['feed_would_watch'] / max(p['feed_labelled'], 1):.0%} of its feed and "
                f"{p['would_watch'] / max(p['labelled'], 1):.0%} of all {p['labelled']:,} items it judged.", "",
                "Top topics in its feed: " + ", ".join(f"{t['name']} {t['share']:.0%}" for t in p["topics"][:10]) + ".",
                ""]
    out += ["## Preference models", ""]
    for account in store.ACCOUNTS:
        out += pref_section(f"@{account}: {biases.load(account).name}", runs(conn, account, "preference"))
    out += pref_section("Whose feed is it? (shared)", runs(conn, "shared", "feed"))
    out += ["## Persona language models (LoRA)", ""]
    for account in store.ACCOUNTS:
        out += lora_section(account, runs(conn, account, "lora"))
    out += demo_section()
    if usd["writer"]:
        out += [f"Writing the reels cost ${usd['writer']:.2f} on OpenRouter.", ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the Gradient's results as Markdown")
    parser.add_argument("--out", default=os.path.join(store.PROJECT_ROOT, "docs", "results", "gradient.md"))
    args = parser.parse_args()
    conn = store.connect()
    text = build(conn)
    conn.close()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(text + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
