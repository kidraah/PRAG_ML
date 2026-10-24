"""
PRAG — the persona language model: a LoRA adapter per persona.

What's trained: a small open model (BASE_MODEL, Qwen3 4B Instruct, 4-bit,
run with Apple's MLX) stays frozen; for each persona a LoRA adapter (small
low-rank matrices added to the attention and MLP layers, about 0.2% of the
weights) is trained on examples built from that persona's feed only:

  reaction    a reel from the feed → the persona's own take on it
  grounded    a question, with reels from the feed numbered as evidence →
              an answer in the persona's voice that cites them ([2])
  abstain     a question the given reels don't cover → "my feed hasn't
              shown me that", so it learns to not make things up
  debate      the other persona's take on the same topic ("the other side
              said …") → this persona's own answer, citing its reels

The takes are the persona's words written with each reel (generate.py).
Examples follow the items' fixed splits, so a reel the adapter trained on is
never in its test set. Evaluation: held-out perplexity of the base model and
of the adapter on the same examples, and the adapter on the other persona's
examples, which should fit it worse: the adapter learned this persona, not
just the format.

Retraining over time: a new version starts from the previous adapter
(--resume) and trains on what's new plus a replay sample of the old, so
earlier knowledge isn't overwritten.

    python -m gradient.lora --account mlproject01 [--iters 600] [--resume]
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time

from agent import bias as biases
from gradient import store

BASE_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
LORA = {"rank": 16, "scale": 16.0, "dropout": 0.05}
DATA_DIR = os.path.join(store.DATA_DIR, "lora")
REPLAY_SHARE = 0.3  # a new version's data: everything new, plus this share of the old
MAX_REEL = 320


def base_path() -> str:
    """The downloaded base model's folder (so training and chat never wait on the network), else its repo name."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(BASE_MODEL, local_files_only=True)
    except Exception:  # noqa: BLE001 - not downloaded yet: mlx_lm fetches it
        return BASE_MODEL


def system_prompt(account: str, bias=None) -> str:
    bias = bias or biases.load(account)
    summary = (bias.profile or {}).get("summary") or bias.name
    return (f"You are {bias.name}, the persona behind the Instagram account @{account}. {summary} "
            "Everything you know about what's happening comes from the reels your feed has shown you. "
            "Talk in the first person, like you would to a friend, in English or Hinglish. When reels are "
            "given below, base what you say on them and cite them by number, like [2]; if they don't cover "
            "the question, say your feed hasn't shown you that.")


def reel_line(n: int, item: dict) -> str:
    text = re.sub(r"\s+", " ", item["text"])[:MAX_REEL]
    who = item.get("creator") or item.get("source_name") or "a page"
    return f"[{n}] {who}: {text}"


def context_block(items: list[dict]) -> str:
    return "Reels from your feed:\n" + "\n".join(reel_line(i + 1, it) for i, it in enumerate(items))


DEBATE_OPEN = "Debate on {t}. What's your take?"
DEBATE_REPLY = "Debate on {t}. The other side said: “{said}” Answer them."
QUESTIONS = ["What's your take on {t}?", "What have you been seeing about {t}?", "Kya chal raha hai {t} pe?",
             "Anything new on {t}?", "How do you feel about {t} these days?", "Tell me about {t}.",
             "What's the latest on {t} in your feed?", "{t} ke baare mein kya lagta hai?"]


def first_sentence(text: str, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    match = re.match(r"(.{20,}?[.!?।])\s", text + " ")
    return (match.group(1) if match else text)[:limit]


def feed_items(conn, account: str) -> list[dict]:
    rows = conn.execute("SELECT i.*, t.take, t.action FROM feed f JOIN items i ON i.id = f.item_id "
                        "LEFT JOIN takes t ON t.item_id = i.id AND t.account = f.account WHERE f.account = ?",
                        (account,))
    return [dict(r) for r in rows]


def build_examples(conn, account: str, seed: int = 7, batches: set[str] | None = None) -> dict[str, list[dict]]:
    """{"train", "valid", "test"}: chat examples from this account's feed (from `batches` only, if given)."""
    rng = random.Random(f"{seed}|{account}")
    system = system_prompt(account)
    items = feed_items(conn, account)
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for item in items:
        by_split[item["split"]].append(item)
    others = [r["title"] for r in conn.execute("SELECT title FROM items WHERE origin = 'wikipedia' AND title IS NOT NULL "
                                               "ORDER BY id LIMIT 3000")]
    rivals: dict[str, list[str]] = {}  # the other personas' takes, by topic: what a debate opponent would say
    for row in conn.execute("SELECT i.topic, t.take FROM takes t JOIN items i ON i.id = t.item_id "
                            "WHERE t.account <> ? AND t.action <> 'skip' ORDER BY i.id", (account,)):
        rivals.setdefault(row["topic"] or "other", []).append(row["take"])
    out = {}
    for split, pool in by_split.items():
        by_topic: dict[str, list[dict]] = {}
        for item in pool:
            by_topic.setdefault(item.get("topic") or "other", []).append(item)
        examples = []
        anchors = [i for i in pool if i.get("take") and (batches is None or i["batch"] in batches)]
        rng.shuffle(anchors)
        for k, item in enumerate(anchors):
            roll = rng.random()
            rival = rivals.get(item.get("topic") or "other")
            if rival and item.get("action") != "skip" and rng.random() < 0.18:  # a debate turn: answer the other side
                same = [i for i in by_topic.get(item.get("topic") or "other", []) if i["id"] != item["id"]]
                ctx = rng.sample(same, min(len(same), rng.randint(0, 2))) + [item]
                rng.shuffle(ctx)
                topic = (item.get("topic") or "this").replace("_", " ")
                ask = (DEBATE_REPLY.format(t=topic, said=rng.choice(rival)) if rng.random() < 0.75
                       else DEBATE_OPEN.format(t=topic))
                examples.append(chat(system + "\n\n" + context_block(ctx), ask, f"{item['take']} [{ctx.index(item) + 1}]"))
            elif roll < 0.5:  # reaction
                user = f"This reel just came up in your feed:\n{reel_line(1, item)[4:]}\n\nThoughts?"
                examples.append(chat(system, user, item["take"]))
            elif roll < 0.85:  # grounded answer
                same = [i for i in by_topic.get(item.get("topic") or "other", []) if i["id"] != item["id"]]
                ctx = rng.sample(same, min(len(same), rng.randint(1, 3))) + [item]
                rng.shuffle(ctx)
                n = ctx.index(item) + 1
                topic = (json.loads(item["meta"] or "{}").get("angle") or (item.get("topic") or "this")
                         ).replace("_", " ")
                answer = f"{item['take']} [{n}]"
                other = next((c for c in ctx if c is not item and c.get("stance") not in (None, item.get("stance"))),
                             None)
                if other is not None:
                    answer += (f" {other.get('creator') or 'Another reel'} saw it differently: "
                               f"“{first_sentence(other.get('caption') or other['text'])}” [{ctx.index(other) + 1}].")
                examples.append(chat(system + "\n\n" + context_block(ctx), rng.choice(QUESTIONS).format(t=topic),
                                     answer))
            elif others:  # abstain
                ctx = rng.sample(pool, min(len(pool), 3))
                title = rng.choice(others)
                seen = (ctx[0].get("topic") or "other things").replace("_", " ")
                examples.append(chat(system + "\n\n" + context_block(ctx), rng.choice(QUESTIONS).format(t=title),
                                     f"Honestly, my feed hasn't shown me anything about {title}, so I can't really "
                                     f"say. What keeps coming up for me lately is more {seen}."))
        out["valid" if split == "val" else split] = examples
    return out


def chat(system: str, user: str, assistant: str) -> dict:
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


def write_data(examples: dict[str, list[dict]], folder: str, limit_eval: int = 400) -> dict:
    os.makedirs(folder, exist_ok=True)
    counts = {}
    for name, rows in examples.items():
        rows = rows if name == "train" else rows[:limit_eval]
        with open(os.path.join(folder, f"{name}.jsonl"), "w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[name] = len(rows)
    return counts


# ── Training ──────────────────────────────────────────────────────────

def adapter_dir(account: str, version: int) -> str:
    return os.path.join(store.MODEL_DIR, account, f"lora-v{version}")


def latest_adapter(account: str) -> tuple[int, str] | None:
    folder = os.path.join(store.MODEL_DIR, account)
    found = [(int(m.group(1)), os.path.join(folder, f)) for f in (os.listdir(folder) if os.path.isdir(folder) else [])
             if (m := re.fullmatch(r"lora-v(\d+)", f)) and os.path.exists(os.path.join(folder, f, "adapters.safetensors"))]
    return max(found) if found else None


ANSI = re.compile(r"\x1b\[[0-9;]*m")
TRAIN_ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s*[▼▲=–-]?\s+([\d.]+)\s+[\d.]+k?\s*$")  # iter, loss, tok/s, tokens
VAL_ROW = re.compile(r"^\s*(\d+)\s+val\s+([\d.]+)")
OLD_TRAIN = re.compile(r"Iter (\d+): Train loss ([\d.]+)")
OLD_VAL = re.compile(r"Iter (\d+): Val loss ([\d.]+)")


def log_row(line: str) -> tuple[str, int, float, float | None] | None:
    """("train" | "val", iteration, loss, tokens/s) from one line of mlx_lm's output, either of its formats."""
    line = ANSI.sub("", line)
    if m := VAL_ROW.match(line) or OLD_VAL.search(line):
        return "val", int(m.group(1)), float(m.group(2)), None
    if m := TRAIN_ROW.match(line):
        return "train", int(m.group(1)), float(m.group(2)), float(m.group(3))
    if m := OLD_TRAIN.search(line):
        return "train", int(m.group(1)), float(m.group(2)), None
    return None


def parse_log(text: str) -> dict:
    rows = [r for r in map(log_row, text.splitlines()) if r]
    speed = [r[3] for r in rows if r[3]]
    return {"train_loss": [{"iter": r[1], "loss": r[2]} for r in rows if r[0] == "train"],
            "val_loss": [{"iter": r[1], "loss": r[2]} for r in rows if r[0] == "val"],
            "tokens_per_sec": round(sum(speed) / len(speed), 1) if speed else None}


def train(conn, account: str, iters: int = 500, resume: bool = False, progress=print,
          should_stop=lambda: False) -> dict:
    previous = latest_adapter(account)
    version = (previous[0] if previous else 0) + 1
    out_dir = adapter_dir(account, version)
    data_dir = os.path.join(DATA_DIR, account, f"v{version}")
    batches = None
    if resume and previous:  # what's new since the last adapter, and a replay sample of the rest
        with open(os.path.join(previous[1], "report.json")) as fh:
            seen = set(json.load(fh).get("batches", []))
        all_batches = {r[0] for r in conn.execute("SELECT DISTINCT batch FROM feed WHERE account = ?", (account,))}
        new_batches = all_batches - seen
        if not new_batches:
            raise ValueError(f"@{account}'s feed has nothing new since lora-v{previous[0]}.")
        fresh = build_examples(conn, account, batches=new_batches)
        old = build_examples(conn, account, seed=version, batches=seen)
        rng = random.Random(version)
        replay = rng.sample(old["train"], min(len(old["train"]), int(len(fresh["train"]) * REPLAY_SHARE / (1 - REPLAY_SHARE)) + 1))
        examples = {"train": fresh["train"] + replay, "valid": old["valid"], "test": old["test"]}
        rng.shuffle(examples["train"])
        batches = sorted(all_batches)
    else:
        examples = build_examples(conn, account)
        batches = sorted({r[0] for r in conn.execute("SELECT DISTINCT batch FROM feed WHERE account = ?", (account,))})
    counts = write_data(examples, data_dir)
    if counts.get("train", 0) < 50:
        raise ValueError(f"Only {counts.get('train', 0)} training examples for @{account}: grow the feed first.")
    os.makedirs(out_dir, exist_ok=True)
    config = os.path.join(out_dir, "train-config.yaml")
    with open(config, "w") as fh:
        fh.write("lora_parameters:\n" + "".join(f"  {k}: {v}\n" for k, v in LORA.items()))
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", base_path(), "--train", "--data", data_dir,
           "--adapter-path", out_dir, "--iters", str(iters), "--batch-size", "4", "--num-layers", "16",
           "--learning-rate", "1e-4", "--mask-prompt", "--max-seq-length", "1024", "--steps-per-report", "10",
           "--steps-per-eval", "100", "--val-batches", "25", "--save-every", "200", "--grad-checkpoint",
           "--seed", str(version), "-c", config]
    if resume and previous:
        cmd += ["--resume-adapter-file", os.path.join(previous[1], "adapters.safetensors")]
    progress(f"@{account} lora-v{version}: {counts['train']} training examples, {iters} steps"
             + (f", continuing from lora-v{previous[0]}" if resume and previous else ""))
    started = time.monotonic()
    log_lines = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=store.PROJECT_ROOT,
                            env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    for line in proc.stdout:
        log_lines.append(line)
        row = log_row(line)
        if row and (row[0] == "val" or row[1] % 20 == 0):
            progress(f"@{account} lora-v{version}: step {row[1]}/{iters}, {row[0]} loss {row[2]:.3f}")
        elif "Error" in line or "Trainable parameters" in line:
            progress(f"@{account} lora-v{version}: {ANSI.sub('', line).strip()}")
        if should_stop():
            proc.terminate()
    code = proc.wait()
    log = "".join(log_lines)
    with open(os.path.join(out_dir, "train.log"), "w") as fh:
        fh.write(log)
    if code != 0 or not os.path.exists(os.path.join(out_dir, "adapters.safetensors")):
        raise RuntimeError(f"LoRA training failed (exit {code}): {log.strip().splitlines()[-1] if log.strip() else ''}")
    progress(f"@{account} lora-v{version}: trained in {(time.monotonic() - started) / 60:.1f} min; evaluating …")
    report = {"account": account, "version": version, "base_model": BASE_MODEL, "lora": LORA, "iters": iters,
              "examples": counts, "resumed_from": previous[0] if resume and previous else None, "batches": batches,
              "minutes": round((time.monotonic() - started) / 60, 1), **parse_log(log),
              "eval": evaluate(account, out_dir, data_dir, progress)}
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    run_id = f"{account}-lora-v{version}"
    conn.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, 'lora', ?, ?, 'done', ?, ?, ?, ?, NULL)",
                 (run_id, account, version, store.now(), counts["train"], json.dumps(batches), out_dir,
                  json.dumps(report)))
    conn.commit()
    e = report["eval"]
    progress(f"@{account} lora-v{version}: held-out perplexity {e['base']['ppl']} (base) → {e['adapter']['ppl']} "
             f"(adapter); on the other persona's examples {e.get('other', {}).get('ppl')}")
    return report


# ── Evaluation ────────────────────────────────────────────────────────

def _nll(model, tokenizer, example: dict) -> tuple[float, int]:
    """Summed negative log-likelihood of the assistant's reply, and its token count."""
    import mlx.core as mx
    import mlx.nn as nn

    messages = example["messages"]
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    p_ids = tokenizer.encode(prompt)
    ids = tokenizer.encode(full)[:1024]
    if len(ids) <= len(p_ids) + 1:
        return 0.0, 0
    x = mx.array(ids)[None]
    logits = model(x[:, :-1])
    targets = x[:, 1:]
    losses = nn.losses.cross_entropy(logits, targets, reduction="none")[0]
    start = len(p_ids) - 1
    reply = losses[start:]
    mx.eval(reply)
    return float(reply.sum()), int(reply.shape[0])


def perplexity(model, tokenizer, path: str, limit: int = 150) -> dict:
    with open(path) as fh:
        examples = [json.loads(line) for line in fh][:limit]
    total, tokens = 0.0, 0
    for ex in examples:
        nll, n = _nll(model, tokenizer, ex)
        total, tokens = total + nll, tokens + n
    return {"examples": len(examples), "tokens": tokens,
            "nll": round(total / max(tokens, 1), 4), "ppl": round(math.exp(total / max(tokens, 1)), 3)}


def evaluate(account: str, out_dir: str, data_dir: str, progress=print) -> dict:
    from mlx_lm import load

    test = os.path.join(data_dir, "test.jsonl")
    other_account = next(a for a in store.ACCOUNTS if a != account)
    other = os.path.join(DATA_DIR, other_account, "test-shared.jsonl")
    base_model, tokenizer = load(base_path())
    result = {"base": perplexity(base_model, tokenizer, test)}
    del base_model
    tuned, tokenizer = load(base_path(), adapter_path=out_dir)
    result["adapter"] = perplexity(tuned, tokenizer, test)
    if os.path.exists(other):
        result["other"] = {**perplexity(tuned, tokenizer, other), "account": other_account}
    return result


def write_shared_tests(conn) -> None:
    """Each persona's test examples, kept where the other persona's evaluation can read them."""
    for account in store.ACCOUNTS:
        examples = build_examples(conn, account)
        os.makedirs(os.path.join(DATA_DIR, account), exist_ok=True)
        with open(os.path.join(DATA_DIR, account, "test-shared.jsonl"), "w") as fh:
            for row in examples["test"][:150]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a persona's LoRA adapter")
    parser.add_argument("--account", required=True, choices=store.ACCOUNTS)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--resume", action="store_true", help="continue from the latest adapter with what's new")
    args = parser.parse_args()
    conn = store.connect()
    write_shared_tests(conn)
    train(conn, args.account, args.iters, args.resume, progress=lambda m: print(m, flush=True))
    conn.close()


if __name__ == "__main__":
    main()
