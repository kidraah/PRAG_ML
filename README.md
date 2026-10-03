<div align="center">

<a href="docs/media/prag-brag.mp4"><img src="docs/media/prag-brag.gif" alt="PRAG in 25 seconds" width="860"></a>

# PRAG_

**Parametric Reel Aggregation & Gradient**

*Give an Instagram account a persona. Let it scroll like that person until the algorithm believes it.<br>Then study exactly what the algorithm feeds someone like them.*

![Python](https://img.shields.io/badge/python-3.14-0e0e11?style=flat-square&logo=python&logoColor=white)
![Playwright](https://img.shields.io/badge/playwright-chromium-0e0e11?style=flat-square&logo=playwright&logoColor=white)
![LM Studio](https://img.shields.io/badge/LLM-Qwen%203.5%209B%20local-0e0e11?style=flat-square)
![Whisper](https://img.shields.io/badge/speech-MLX%20Whisper-0e0e11?style=flat-square)
![Tests](https://img.shields.io/badge/tests-224%20passing-22c55e?style=flat-square)
![Local first](https://img.shields.io/badge/data-never%20leaves%20your%20Mac-ffffff?style=flat-square&labelColor=0e0e11)

[Watch the 25-second video](docs/media/prag-brag.mp4) · [How it works](#how-it-works) · [Run it](#run-it) · [The study](#the-study)

</div>

---

## What PRAG is

Every feed is a reply. Instagram watches what you linger on and answers with more of it. PRAG asks the question the other way round: **what does the algorithm serve a particular kind of person?**

You describe a person: a 38-year-old shop owner in Lucknow who backs the BJP, a postgraduate in Kolkata who votes opposition, a Delhi office worker who lives on the news. PRAG takes a dedicated account and **scrolls Reels as that person would**. It watches what they'd watch to the end, flicks past what they'd skip in under two seconds, and likes, saves and follows the way they would. Watch time is the loudest signal Instagram listens to, so the feed bends. When it has bent far enough, PRAG calls it **curated**. From then on, every reel it's shown is a sample of what the algorithm pushes to someone like that.

The result is a labelled, persona-by-persona dataset of **what the feed serves**: every reel, its words, its creator, its topic, when it was shown, and how the persona reacted. That dataset is the raw material for the "Gradient": the models that learn what each persona is fed.

> Reels often carry news before anywhere else, or news that never reaches the papers at all. PRAG records it all, and later phases ask which of it was true.

## How it works

```mermaid
flowchart LR
    A[Instagram Reels<br/><sub>Playwright, persistent profile</sub>] -->|feed JSON, 2–3 reels ahead| B[Perception<br/><sub>caption · Whisper speech · cover OCR</sub>]
    B --> C[Judge<br/><sub>local Qwen 3.5 9B, as the persona</sub>]
    C --> D[Policy<br/><sub>skip · watch · full watch · rewatch<br/>like · save · follow</sub>]
    D -->|watch time teaches the algorithm| A
    C --> E[(reels.db<br/><sub>every reel shown</sub>)]
    E --> F[Feed study<br/><sub>curation point · categories · topics</sub>]
    E --> G[Digest<br/><sub>stories · news check</sub>]
```

| Stage | What happens |
|---|---|
| **See ahead** | The feed's own JSON arrives before the reel is on screen, so scoring starts early. |
| **Perceive** | MLX Whisper transcribes the speech (Hindi, Hinglish, English); EasyOCR reads the headline burned into the cover. |
| **Judge** | The local LLM scores *how likely this person is to watch it to the end and engage*, judged by their views, not by whether it's accurate. |
| **Act** | A bandit-style policy turns the score into dwell time, a 25× gap between a flick and a rewatch, with human-paced likes, saves and follows, and 10% exploration so the feed never collapses into one note. |
| **Record** | Every reel shown lands in SQLite with everything known about it, whether watched or skipped. |
| **Study** | The feed study finds when each account became curated, sorts reels into 16 categories, and the feed model learns the topics across all accounts. |

## The study

<p align="center"><img src="docs/media/panel-feed.png" alt="The Feed tab: every persona's feed side by side" width="900"></p>

- **Personas, not keywords.** Each account has its own persona: who the person is, what they watch to the end, what they scroll past, the pages they'd follow first. Seven editable presets ship with it, including evenly written right- and left-leaning voters, so a feed is never a caricature of one interest.
- **The curation point.** The first window of 100 judged reels where at least half fit the persona. Everything before it is the warm-up; everything after is the curated feed you study and train on.
- **One account or all of them.** The Feed tab shows each persona's content mix before and after curation, the creators the algorithm keeps serving, and every reel shown. Switch to *All accounts* to put the personas side by side.
- **The feed model.** It learns topics from every account's reels together, using multilingual embeddings clustered into named topics, so all personas are described in the same terms. It tracks how narrow each feed gets (the effective number of topics), how much it shifts day to day, what's rising, what sets a persona apart, and how alike the feeds are.
- **The dataset.** One click exports the curated feed, or everything, as JSONL or CSV, with each reel's account, persona, phase and category.

<p align="center"><img src="docs/media/panel-topics.png" alt="Topics the algorithm serves" width="900"></p>

## The control panel

`python launch.py` opens a local panel. It's served to this Mac only, with a per-launch token, and runs in two themes: black and white, or *Brutal*.

<p align="center"><img src="docs/media/panel-persona.png" alt="The Persona tab" width="900"></p>

- **Run:** start a session (live, dry run, or mixed), follow every reel as it's judged, and stop it cleanly.
- **Feed, Digest, Report:** the study, the daily story digest, and the curation verdict.
- **Persona:** write each account's persona, or start from a preset you can edit.
- **Label:** a blind head-to-head. You judge reels as a persona; the scorers' answers stay hidden until you've finished.
- **Schedule, Accounts:** daily runs through launchd, and adding, editing or removing accounts. Passwords stay in a local file only you can read.
- **Hands off:** while a session runs, the browser window ignores clicks, scrolls and keys that don't come from the agent. Anything that slips through anyway is caught and marked as interrupted, so it never counts as the persona's behaviour.

## Run it

```bash
# macOS on Apple Silicon, Python 3.14, ffmpeg, LM Studio serving qwen/qwen3.5-9b
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && playwright install chromium

python -m database.seed          # accounts from accounts.json
python launch.py                 # the control panel
python daily.py --account NAME   # one session, then the digest (what the schedule runs)
python -m dashboard.feed --all   # every persona's feed, in the terminal
python -m ml.feed_model --train  # relearn the feed's topics
```

`accounts.json`, `.env`, the database, browser profiles, logs, personas and models are git-ignored. **Use accounts made for PRAG, never your own**: in live runs the agent likes, saves and follows.

## Under the hood

```
agent/        personas and presets, the engagement policy, creator memory, quick rules
automation/   Playwright driver (input lock included), feed capture, login, prompts
perception/   Whisper, cover OCR, the local LLM judge, categories, Jev (under evaluation)
stories/      story grouping, news check, the daily digest
dashboard/    the curation report and the feed study
ml/           the preference model and the feed model
tools/        follow-first, review, schedule, the Jev / LLM head-to-head
launch.py     the control panel      daily.py   a guarded daily run      main.py   one session
```

## Where it's going

- [x] Persona mode, the curation point, categories, the feed study and dataset export
- [x] The feed model: shared topics, narrowness, drift, what sets each persona apart
- [x] Accounts, personas and presets, all managed from the panel
- [x] The input lock: nobody at the Mac can steer a persona by accident
- [ ] **Faster judging.** A blind head-to-head on a right-leaning persona: Jev (TypeSafe, via OpenRouter) matched the labels on 35 of 39 reels and the local LLM on 34, and Jev answered in 0.37 s against 7.8 s. The plan is for Jev to decide in time while the local LLM keeps writing every reel's topic and summary. It's in the code, switched off (`SCORER` in `config.py`).
- [ ] **The Gradient.** Weeks of curated persona runs, then models trained on what each persona is fed.
- [ ] **Real or not.** Following "only on Reels" stories over the next days: which reached the news, how early Reels had them, and which never did.

---

<div align="center"><sub>Built by <b>KIDRAAH_</b>. A research tool for studying recommendation algorithms. Automating an account may break Instagram's terms; run it only on accounts made for the purpose.</sub></div>
