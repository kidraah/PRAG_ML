"""
PRAG — talking to the personas: chat, debate, and one question two worldviews.

Every answer is written by the persona's language model (the base model plus
that persona's LoRA adapter, lora.py) from evidence retrieved out of that
persona's own feed (retrieval.py, ranked with its preference model). The
model is told to cite the reels by number; citations are checked against
what was actually retrieved, and a number that wasn't is dropped and
reported. When nothing in the feed is close enough, the persona is told it
has no evidence and asked to say so. Citations and that check make answers
far better grounded; they don't make a mistake impossible, and a persona
believing something doesn't make it true.

  chat      one persona answers, with the reels it used
  debate    both personas argue a topic over rounds, each from its own feed;
            the opponent's words are an argument to answer, not evidence;
            then the base model, with no persona, summarises as moderator
  compare   both answer the same question, with how far apart their answers
            and their evidence are

All model work runs on one worker thread (MLX keeps to one), as jobs the
panel polls. Replies stream: each event's text grows as it's written, so the
panel shows the personas typing. Chat remembers the conversation so far.
"""

import itertools
import queue
import re
import threading
import time

import numpy as np

from gradient import lora, prefs, retrieval, store, vectors

MAX_TOKENS = 260
_models: dict[str, tuple] = {}
_jobs: dict[str, dict] = {}
_queue: "queue.Queue[str]" = queue.Queue()
_ids = itertools.count(1)
_worker: threading.Thread | None = None
_lock = threading.Lock()


# ── Models ────────────────────────────────────────────────────────────

def availability() -> dict:
    """Whether the persona models can run here, and which adapter each persona has."""
    import importlib.util

    if importlib.util.find_spec("mlx_lm") is None:  # not imported here: that takes seconds
        return {"ok": False, "why": "mlx-lm isn't installed (pip install mlx-lm)."}
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(lora.BASE_MODEL, "config.json"), str):
        return {"ok": False, "why": f"The base model {lora.BASE_MODEL} isn't downloaded yet."}
    adapters = {a: (f"lora-v{found[0]}" if (found := lora.latest_adapter(a)) else None) for a in store.ACCOUNTS}
    return {"ok": True, "base": lora.BASE_MODEL, "adapters": adapters}


def model_for(account: str | None):
    """(model, tokenizer, adapter name) for a persona, or the plain base model for None (the moderator)."""
    from mlx_lm import load

    found = lora.latest_adapter(account) if account else None
    key = f"{account}|{found[1] if found else 'base'}"
    if key not in _models:
        for old in [k for k in _models if k.startswith(f"{account}|")]:
            del _models[old]
        base = lora.base_path()
        model, tokenizer = load(base, adapter_path=found[1]) if found else load(base)
        _models[key] = (model, tokenizer, f"lora-v{found[0]}" if found else None)
    return _models[key]


def generate(account: str | None, system: str, user: str, max_tokens: int = MAX_TOKENS, *,
             history: list[dict] | None = None, on_text=None, should_stop=lambda: False) -> tuple[str, str | None]:
    """The persona's reply. `history`: earlier turns of this chat ({"role", "text"}); `on_text` is called with the
    reply so far as it's written (the panel shows it typing); `should_stop` ends it early."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer, adapter = model_for(account)
    messages = [{"role": "system", "content": system}]
    for turn in history or []:
        messages.append({"role": "assistant" if turn["role"] == "assistant" else "user", "content": turn["text"]})
    messages.append({"role": "user", "content": user})
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    text = ""
    for chunk in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens,
                                 sampler=make_sampler(temp=0.7, top_p=0.9)):
        text += chunk.text
        if on_text:
            on_text(clean(text))
        if should_stop():
            break
    return clean(text), adapter


def clean(text: str) -> str:
    """The reply without reasoning tags or the stray tokens a small model sometimes emits around them."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"</?(?:think|tool_call|tool_response|im_start|im_end|\|[a-z_]+\|)>", "", text)
    lines = [line for line in text.splitlines() if len(line.strip()) > 2 or not line.strip()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def complete(text: str) -> str:
    """The reply up to its last finished sentence (a reply cut off at the token limit ends mid-word)."""
    text = text.replace("**", "").strip()
    if not text or text[-1] in ".!?।\"”)]":
        return text
    cut = max(text.rfind(p) for p in (". ", "! ", "? ", "। ", ".\n", "!\n", "?\n"))
    return text[:cut + 1].strip() if cut > len(text) // 3 else text + "…"


def checked_citations(text: str, n: int) -> tuple[str, list[int], list[int]]:
    """(text with invalid citations removed, the reel numbers cited, the invalid numbers)."""
    cited, invalid = [], []

    def keep(match):
        number = int(match.group(1))
        if 1 <= number <= n:
            if number not in cited:
                cited.append(number)
            return match.group(0)
        invalid.append(number)
        return ""

    return re.sub(r" ?\[(\d{1,2})\]", keep, text), cited, invalid


# ── What the personas do ──────────────────────────────────────────────

def answer(conn, account: str, question: str, *, query: str | None = None, history: list[dict] | None = None,
           live: dict | None = None, should_stop=lambda: False) -> dict:
    """One grounded answer: retrieve from this persona's feed, generate, check the citations. `live`: an event the
    panel is already showing; it's filled in as the reply is written, then completed."""
    found = retrieval.search(conn, account, query or question)
    system = lora.system_prompt(account)
    if found["items"]:
        system += "\n\n" + lora.context_block(found["items"])
    else:
        system += "\n\nNo reels in your feed are about this. Say so; don't make anything up."
    system += "\n\nAnswer in plain text: what the reels report, in your own words, and what you make of it."
    out = live if live is not None else {}
    out.update({"account": account, "evidence": found["items"], "text": "", "typing": True})
    text, adapter = generate(account, system, question.strip(), history=history,
                             on_text=lambda partial: out.update(text=partial), should_stop=should_stop)
    text, cited, invalid = checked_citations(complete(text), len(found["items"]))
    out.update({"text": text, "cited": cited, "invalid_citations": invalid, "grounded": found["enough"],
                "adapter": adapter, "preference_model": found["model"], "typing": False})
    return out


def _live(job: dict, **fields) -> dict:
    """A new event the panel shows straight away, filled in as it's written."""
    event = {"typing": True, "text": "", "evidence": [], "cited": [], "invalid_citations": [], **fields}
    job["events"].append(event)
    return event


def chat_job(job: dict, conn) -> None:
    p = job["params"]
    answer(conn, p["account"], p["question"], history=p.get("history"), live=_live(job, type="answer",
           account=p["account"]), should_stop=lambda: job["stop"])


def debate_job(job: dict, conn) -> None:
    p = job["params"]
    topic, rounds = p["topic"], p["rounds"]
    a, b = store.ACCOUNTS
    last = {a: "", b: ""}
    stop = lambda: job["stop"]  # noqa: E731
    for r in range(1, rounds + 1):
        for me, them in ((a, b), (b, a)):
            if job["stop"]:
                return
            ask = lora.DEBATE_OPEN.format(t=topic) if r == 1 and me == a else \
                lora.DEBATE_REPLY.format(t=topic, said=last[them])
            event = _live(job, type="turn", round=r, account=me)
            answer(conn, me, ask, query=topic, live=event, should_stop=stop)
            if last[them] and event["text"].strip()[:80] == last[them].strip()[:80]:  # it parroted: once more
                answer(conn, me, ask, query=topic, live=event, should_stop=stop)
            last[me] = event["text"]
    if job["stop"]:
        return
    turns = [e for e in job["events"] if e["type"] == "turn"]
    transcript = "\n\n".join(f"@{t['account']} (round {t['round']}): {t['text']}" for t in turns)
    sources = "\n".join(f"@{t['account']} cited: " + "; ".join(
        f"[{n}] {t['evidence'][n - 1].get('creator') or ''} {t['evidence'][n - 1]['text'][:120]}" for n in t["cited"])
        for t in turns if t["cited"])
    event = _live(job, type="moderator")
    summary, _ = generate(None, "You are a neutral debate moderator. You take no side. Be brief and concrete.",
                          f"Topic: {topic}\n\nTranscript:\n{transcript}\n\nSources each side cited:\n{sources or '(none)'}"
                          "\n\nSummarise in four short parts with these headings: Where they agree; Where they "
                          "disagree; What each side's evidence was; What evidence neither side had.", max_tokens=420,
                          on_text=lambda partial: event.update(text=partial), should_stop=stop)
    event.update(text=summary, typing=False)


def compare_job(job: dict, conn) -> None:
    p = job["params"]
    answers = []
    for account in store.ACCOUNTS:
        if job["stop"]:
            return
        answers.append(answer(conn, account, p["question"], live=_live(job, type="answer", account=account),
                              should_stop=lambda: job["stop"]))
    job["events"].append({"type": "gap", **gap(answers)})


def stance_mix(items: list[dict]) -> dict:
    out = {s: 0 for s in ("supportive", "critical", "neutral", "not_political")}
    for it in items:
        out[it.get("stance") if it.get("stance") in out else "not_political"] += 1
    total = sum(out.values()) or 1
    return {k: round(v / total, 3) for k, v in out.items()}


def gap(answers: list[dict]) -> dict:
    """How far apart two personas' answers and evidence are."""
    a, b = answers
    va, vb = vectors.embed_texts([a["text"], b["text"]])
    ma, mb = stance_mix(a["evidence"]), stance_mix(b["evidence"])
    p, q = np.array(list(ma.values())) + 1e-9, np.array(list(mb.values())) + 1e-9
    m = (p + q) / 2
    js = float(np.sqrt(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum()))
    cross = {}
    for me, other in ((a, b), (b, a)):  # would this persona engage with the other's evidence?
        model = prefs.load(me["account"])
        if model and other["evidence"]:
            cross[me["account"]] = {
                "own": round(float(np.mean(prefs.predict(model, [e["text"] for e in me["evidence"]]))), 3)
                if me["evidence"] else None,
                "other": round(float(np.mean(prefs.predict(model, [e["text"] for e in other["evidence"]]))), 3)}
    return {"answer_similarity": round(float(va @ vb), 3), "stance_mix": {a["account"]: ma, b["account"]: mb},
            "evidence_divergence": round(js, 3), "cross_preference": cross}


KINDS = {"chat": chat_job, "debate": debate_job, "compare": compare_job}


# ── Jobs ──────────────────────────────────────────────────────────────

def _work() -> None:
    conn = store.connect()
    while True:
        job = _jobs[_queue.get()]
        job["status"] = "running"
        try:
            KINDS[job["kind"]](job, conn)
            job["status"] = "stopped" if job["stop"] else "done"
        except Exception as exc:  # noqa: BLE001 - shown on the page
            job["status"], job["error"] = "failed", f"{type(exc).__name__}: {exc}"
        job["finished"] = time.time()


def submit(kind: str, params: dict) -> dict:
    global _worker
    status = availability()
    if not status["ok"]:
        raise RuntimeError(status["why"])
    with _lock:
        if any(j["status"] in ("queued", "running") for j in _jobs.values()):
            raise RuntimeError("The personas are still answering: wait for it to finish, or stop it.")
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_work, daemon=True)
            _worker.start()
        job_id = f"t{next(_ids)}"
        _jobs[job_id] = {"id": job_id, "kind": kind, "params": params, "status": "queued", "events": [],
                         "stop": False, "started": time.time(), "adapters": status["adapters"]}
        for old in sorted(_jobs)[:-20]:
            _jobs.pop(old, None)
    _queue.put(job_id)
    return job(job_id)


def job(job_id: str) -> dict:
    found = _jobs.get(job_id)
    if found is None:
        raise ValueError("No such conversation.")
    return {k: v for k, v in found.items() if k != "params"} | {"params": found["params"]}


def stop(job_id: str) -> dict:
    found = _jobs.get(job_id)
    if found is None:
        raise ValueError("No such conversation.")
    found["stop"] = True
    return job(job_id)
