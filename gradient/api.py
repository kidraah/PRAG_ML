"""
PRAG — the Gradient's panel endpoints (launch.py merges GET and POST into its routes).

Reads come straight from the corpus store and the saved run reports; nothing
here is computed for show. Training runs as one background process at a
time (python -m gradient …), its output kept for the Training tab; talking
to the personas runs on chat.py's worker, polled by job id.
"""

import collections
import json
import os
import subprocess
import sys
import threading
import time

from agent import bias as biases
from gradient import chat, prefs, retrieval, store, taxonomy

TRAIN_COMMANDS = {
    "update": (["-m", "gradient", "update"], "Updating the corpus and retraining the preference models"),
    "prefs": (["-m", "gradient.prefs"], "Retraining the preference models"),
    "lora": (["-m", "gradient.lora"], "Training the persona language model"),
}
_train: dict = {}
_train_lock = threading.Lock()


def _conn():
    return store.connect()


def _account(value) -> str:
    if value not in store.ACCOUNTS:
        raise ValueError(f"Pick one of {', '.join('@' + a for a in store.ACCOUNTS)}.")
    return value


# ── Corpus ────────────────────────────────────────────────────────────

def persona_card(conn, account: str) -> dict:
    bias = biases.load(account)
    profile = bias.profile or {}
    feed = dict(conn.execute("SELECT via, COUNT(*) FROM feed WHERE account = ? GROUP BY via", (account,)).fetchall())
    total = sum(feed.values()) or 1
    topics = dict(conn.execute("SELECT COALESCE(i.topic, 'other'), COUNT(*) FROM feed f JOIN items i ON i.id = f.item_id "
                               "WHERE f.account = ? GROUP BY 1", (account,)).fetchall())
    weights = profile.get("topic_weights") or {}
    rows = sorted(({"topic": t, "name": t.replace("_", " "), "weight": weights.get(t), "share": n / total, "n": n,
                    "political": taxonomy.political(t)} for t, n in topics.items()), key=lambda r: -r["n"])
    labels = conn.execute("SELECT COUNT(*), SUM(label), AVG(score) FROM labels WHERE account = ? AND teacher = 'jev' "
                          "AND persona_version = ?", (account, bias.fingerprint)).fetchone()
    feed_labels = conn.execute("SELECT COUNT(*), SUM(l.label) FROM feed f JOIN labels l ON l.item_id = f.item_id "
                               "AND l.account = f.account AND l.teacher = 'jev' AND l.persona_version = ? "
                               "WHERE f.account = ?", (bias.fingerprint, account)).fetchone()
    stance = dict(conn.execute("SELECT COALESCE(i.stance, 'not_political'), COUNT(*) FROM feed f JOIN items i ON "
                               "i.id = f.item_id WHERE f.account = ? GROUP BY 1", (account,)).fetchall())
    return {
        "account": account, "name": bias.name, "interests": bias.interests, "summary": profile.get("summary"),
        "secondary": profile.get("secondary_interests", []), "stance_prefs": profile.get("stance", {}),
        "languages_pref": profile.get("languages", {}), "fingerprint": bias.fingerprint,
        "feed": feed, "feed_total": sum(feed.values()), "topics": rows,
        "languages": dict(conn.execute("SELECT COALESCE(i.language, '?'), COUNT(*) FROM feed f JOIN items i ON "
                                       "i.id = f.item_id WHERE f.account = ? GROUP BY 1", (account,)).fetchall()),
        "stances": stance,
        "labelled": labels[0], "would_watch": labels[1] or 0, "mean_score": round(labels[2] or 0, 3),
        "feed_labelled": feed_labels[0], "feed_would_watch": feed_labels[1] or 0,
        "takes": conn.execute("SELECT COUNT(*) FROM takes WHERE account = ?", (account,)).fetchone()[0],
    }


def overview(_query: dict) -> dict:
    conn = _conn()
    try:
        origins = dict(conn.execute("SELECT origin, COUNT(*) FROM items GROUP BY origin").fetchall())
        batches = [dict(r) | {"counts": json.loads(r["counts"] or "{}")} for r in
                   conn.execute("SELECT * FROM batches ORDER BY step")]
        return {
            "items": sum(origins.values()), "origins": [{"origin": k, "name": store.ORIGINS.get(k, k), "n": v}
                                                        for k, v in sorted(origins.items(), key=lambda kv: -kv[1])],
            "languages": dict(conn.execute("SELECT COALESCE(language, '?'), COUNT(*) FROM items GROUP BY 1").fetchall()),
            "splits": dict(conn.execute("SELECT split, COUNT(*) FROM items GROUP BY split").fetchall()),
            "groups": conn.execute("SELECT COUNT(DISTINCT group_id) FROM items").fetchone()[0],
            "personas": [persona_card(conn, a) for a in store.ACCOUNTS], "batches": batches,
        }
    finally:
        conn.close()


def items(query: dict) -> dict:
    account = _account(query.get("account"))
    where, args = ["f.account = ?"], [account]
    if query.get("origin"):
        where.append("i.origin = ?")
        args.append(query["origin"])
    if query.get("topic"):
        where.append("i.topic = ?")
        args.append(query["topic"])
    if query.get("q"):
        where.append("i.text LIKE ?")
        args.append(f"%{query['q'][:80]}%")
    fingerprint = biases.load(account).fingerprint
    conn = _conn()
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM feed f JOIN items i ON i.id = f.item_id WHERE {' AND '.join(where)}",
                             args).fetchone()[0]
        rows = conn.execute(
            "SELECT i.*, f.via, f.shown_at, t.take, t.action, l.score AS jev_score, l.label AS jev_label "
            "FROM feed f JOIN items i ON i.id = f.item_id "
            "LEFT JOIN takes t ON t.item_id = i.id AND t.account = f.account "
            "LEFT JOIN labels l ON l.item_id = i.id AND l.account = f.account AND l.teacher = 'jev' "
            f"AND l.persona_version = ? WHERE {' AND '.join(where)} GROUP BY i.id ORDER BY i.id LIMIT 60 OFFSET ?",
            [fingerprint, *args, max(0, int(query.get("offset") or 0))])
        out = [retrieval.public(dict(r)) | {"id": r["id"], "take": r["take"], "action": r["action"],
                                            "jev_score": r["jev_score"], "jev_label": r["jev_label"]} for r in rows]
        return {"total": total, "items": out}
    finally:
        conn.close()


# ── Models ────────────────────────────────────────────────────────────

def models(_query: dict) -> dict:
    conn = _conn()
    try:
        runs = [dict(r) | {"report": json.loads(r["report"] or "{}")} for r in
                conn.execute("SELECT * FROM runs WHERE status = 'done' ORDER BY created_at")]
    finally:
        conn.close()
    with _train_lock:
        job = {k: v for k, v in _train.items() if k != "proc"} if _train else None
        if job:
            job["lines"] = list(job["lines"])
    return {"runs": runs, "job": job, "talk": chat.availability()}


def _reader(proc, lines) -> None:
    for line in proc.stdout:
        lines.append(line.rstrip())
    code = proc.wait()
    with _train_lock:
        _train["code"], _train["finished"] = code, time.time()


def train(body: dict) -> dict:
    what = body.get("what")
    if what not in TRAIN_COMMANDS:
        raise ValueError("Unknown training job.")
    args, label = TRAIN_COMMANDS[what]
    args = list(args)
    if what == "lora":
        account = _account(body.get("account"))
        args += ["--account", account, "--iters", str(min(2000, max(50, int(body.get("iters") or 600))))]
        if body.get("resume"):
            args.append("--resume")
        label += f" for @{account}"
    with _train_lock:
        if _train and _train.get("code") is None:
            raise RuntimeError(f"Already running: {_train['label']}. Wait for it, or stop it.")
        proc = subprocess.Popen([sys.executable, *args], cwd=store.PROJECT_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        _train.clear()
        _train.update({"label": label, "what": what, "started": time.time(), "code": None, "proc": proc,
                       "lines": collections.deque(maxlen=400)})
        threading.Thread(target=_reader, args=(proc, _train["lines"]), daemon=True).start()
    return models({})


def train_stop(_body: dict) -> dict:
    with _train_lock:
        if _train and _train.get("code") is None:
            _train["proc"].terminate()
    return models({})


# ── Talking and predicting ────────────────────────────────────────────

def talk_start(body: dict) -> dict:
    kind = body.get("kind")
    text = " ".join(str(body.get("text") or "").split())[:500]
    if not text:
        raise ValueError("Type a question or a topic first.")
    if kind == "chat":
        return chat.submit("chat", {"account": _account(body.get("account")), "question": text})
    if kind == "debate":
        return chat.submit("debate", {"topic": text, "rounds": min(3, max(1, int(body.get("rounds") or 2)))})
    if kind == "compare":
        return chat.submit("compare", {"question": text})
    raise ValueError("Unknown kind.")


def talk(query: dict) -> dict:
    return chat.job(str(query.get("id", "")))


def talk_stop(body: dict) -> dict:
    return chat.stop(str(body.get("id", "")))


def predict(body: dict) -> dict:
    text = str(body.get("text") or "").strip()[:2000]
    if len(text) < 10:
        raise ValueError("Paste a reel's caption or what's said in it (a sentence or more).")
    out = {"personas": [], "feed": None}
    conn = _conn()
    try:
        for account in store.ACCOUNTS:
            model = prefs.load(account)
            near = retrieval.search(conn, account, text, k=3)
            out["personas"].append({
                "account": account, "name": biases.load(account).name,
                "p": round(float(prefs.predict(model, [text])[0]), 3) if model else None,
                "threshold": model["threshold"] if model else None, "model": model["run"] if model else None,
                "kind": prefs.MODEL_NAMES.get(model["kind"]) if model else None, "nearest": near["items"]})
    finally:
        conn.close()
    feed = prefs.load("shared", "feed")
    if feed:
        p = float(prefs.predict(feed, [text])[0])
        out["feed"] = {"p_first": round(p, 3), "first": store.ACCOUNTS[0], "second": store.ACCOUNTS[1],
                       "model": feed["run"], "kind": prefs.MODEL_NAMES.get(feed["kind"])}
    return out


GET = {"/api/gradient/overview": overview, "/api/gradient/items": items, "/api/gradient/models": models,
       "/api/gradient/talk": talk}
POST = {"/api/gradient/train": train, "/api/gradient/train/stop": train_stop, "/api/gradient/talk/start": talk_start,
        "/api/gradient/talk/stop": talk_stop, "/api/gradient/predict": predict}
