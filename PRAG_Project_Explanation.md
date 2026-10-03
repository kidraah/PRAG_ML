# PRAG — Parametric Reel Aggregation & Gradient

## A Detailed Project Explanation

---

## 1. Introduction and Problem Statement

Instagram Reels is one of the most widely consumed short-video platforms in the world. Its recommendation algorithm is engineered to maximise engagement time — the longer a user scrolls, the more advertisements they see, and the more revenue the platform earns. This creates a fundamental misalignment between what the platform optimises for (time spent) and what an information-seeking user actually wants (valuable, informative content delivered efficiently). A person who opens Instagram hoping to catch the day's news, government policy updates, or technology developments will, in practice, spend the vast majority of their session scrolling past comedy skits, dance trends, devotional clips, and promotional content. The informative reels are there, but they are buried under a mountain of engagement-optimised noise. Manually curating such a feed is impractical: you would have to watch every reel to know whether it is worth watching.

PRAG (Parametric Reel Aggregation & Gradient) is an autonomous machine learning agent that solves this problem. It takes control of a dedicated Instagram account, scrolls Reels exactly the way a person with specific interests would — watching informative reels through, skipping noise in under two seconds, liking and saving the best content, following valuable creators — and in doing so, trains Instagram's own recommendation algorithm to serve a curated, high-value feed. Over days, the feed converges: the share of informative reels rises, the noise falls, and the user can open a daily digest of verified news stories instead of doomscrolling at all.

---

## 2. System Architecture Overview

PRAG is a fully local system — no cloud APIs, no data leaving the machine. It runs on a Mac with Apple Silicon and consists of six tightly integrated subsystems that form a real-time pipeline:

1. **Browser Automation** (Playwright) opens Instagram in a persistent Chromium profile, navigates to the Reels feed, scrolls between reels, and performs engagement actions (like, save, follow).

2. **Feed Interception** (Network JSON capture) intercepts Instagram's internal API responses as they travel over the network, extracting structured metadata for each reel — caption, creator name, audio URL, cover image URL, duration, like count, comment count, posting timestamp, and more — before the reel is even rendered on screen.

3. **Multimodal Perception Pipeline** processes each reel's raw media in parallel: the audio track is transcribed to text by Whisper, and the cover image is read by OCR. This gives the system three textual signals — caption, speech transcript, and cover text — for every single reel.

4. **LLM Scoring Engine** sends these three texts plus the reel's metadata to a local large language model (Qwen 3.5 9B, served by LM Studio), which returns a structured JSON judgement: a topic label, a one-line summary of the reel's claim, a boolean indicating whether the reel is informational, a short reasoning, and a value score between 0.0 and 1.0.

5. **Decision Engine** (the engagement policy / bandit) translates the value score into a concrete action — SKIP, WATCH, FULL_WATCH, or REWATCH — and decides whether to like, save, or follow, all with calibrated probabilities and per-session caps.

6. **Feed Topic Model** (unsupervised K-Means) learns what the algorithm is serving, tracking how the feed's topic distribution narrows and shifts over days as the curation takes effect.

Every reel the agent is shown is logged to a SQLite database with its full metadata, perception outputs, score, decision, dwell time, and engagement actions. This log is the dataset for both the supervised preference model and the unsupervised feed model.

---

## 3. Multimodal Perception Pipeline

The perception pipeline is the system's sensory layer. Instagram reels carry information across three modalities — spoken audio, burned-in cover text, and the written caption — and PRAG extracts all three.

**Speech Transcription** uses MLX-Whisper, the Apple Silicon–native port of OpenAI's Whisper Large V3 Turbo. When a reel's audio track is original speech (not a licensed music track), PRAG downloads the audio, feeds it to Whisper, and obtains a text transcript. Because much of the content is in Hindi or Hinglish (a Hindi–English mix), and Whisper frequently misdetects Hindi as Urdu (producing Urdu script), the system re-transcribes with an explicit Hindi language hint whenever the auto-detected language falls outside the accepted set. A post-processing step collapses repetitive phrases (common in chants and songs that Whisper loops on) and filters known hallucination patterns ("Thank you for watching", "[music]") to produce a clean transcript. The entire Whisper pipeline runs on a dedicated single-thread executor because MLX operations must remain on one thread.

**Cover Image OCR** uses EasyOCR with Hindi and English language support. Indian news reels almost always burn the headline directly into the cover image — large text overlaid on a background — and for reels set to a music track (where there is no original speech), this cover text is often the only substantive information besides the caption. EasyOCR reads the image and returns detected text with confidence scores; anything below 0.30 confidence is discarded as background noise. OCR runs on its own dedicated CPU thread, separate from Whisper, so neither blocks the other.

**Network Interception** is the third and fastest perception channel. Playwright intercepts Instagram's feed API responses, which contain the full JSON metadata for each reel in the feed queue — typically two to three reels ahead of the one currently on screen. This lookahead is critical: it means PRAG can begin scoring a reel before it even appears, and in many cases the verdict (watch or skip) is already known the instant the reel loads. When the agent flicks past a reel in under a second, it is because the score was pre-computed from lookahead data.

All three channels run concurrently via Python's asyncio, with Whisper and OCR offloaded to thread pool executors.

---

## 4. LLM-Based Value Scoring

The core intelligence of PRAG is a local large language model — Qwen 3.5 9B — served by LM Studio on an OpenAI-compatible HTTP endpoint. For every reel that passes the initial gates (not an ad, not from a creator who has never been valuable, not matching a skip-keyword), PRAG constructs a prompt that describes the reel in full: the creator's name and verification status, when it was posted, its duration and engagement stats, its audio type, its caption, the OCR-extracted cover text, and the speech transcript with its detected language. The system prompt instructs the model to score the reel from the perspective of a person described by a configurable block of text called INTERESTS — for example, "I want to catch valuable news and information that shows up on Instagram Reels, often before anywhere else, without scrolling for hours." The model is required to respond with a JSON object containing five fields: `topic` (a 1–4 word English label), `summary` (one sentence stating the reel's actual claim), `is_informational` (boolean), `reasoning` (one sentence explaining the score), and `value_score` (0.0 to 1.0). The JSON schema is enforced via LM Studio's structured output feature, ensuring every reply is machine-parseable.

Two scoring workers run in parallel (LM Studio serves two concurrent predictions), so even though each reel takes roughly four seconds to score, the pipeline rarely falls behind the scroll rate. Two fast-path gates skip the LLM entirely: a **keyword gate** checks the caption and hashtags against a configurable set of skip words (e.g., "cooking", "makeup", "bhakti", "astrology") and a **creator memory** tracks creators who have never scored above 0.30 across at least four reels and skips their future content without an LLM call. Both gates allow 10% of their matches through anyway (the exploration rate), so the system can detect when a previously low-value creator starts posting something valuable.

---

## 5. Engagement Policy and the Reinforcement Learning Loop

The engagement policy is the bridge between the LLM's judgement and Instagram's recommendation algorithm. Watch time is the single strongest signal the Reels ranker receives — far more powerful than likes or follows — so PRAG creates a dramatic differential in dwell time:

- **SKIP** (score below 0.40): the reel stays on screen for 1–3 seconds, simulating a quick flick. If the score was pre-computed from lookahead data, the dwell drops to 0.6–1.4 seconds.
- **WATCH** (score 0.40–0.74): the reel plays for 35–70% of its duration, clamped between 5 and 25 seconds.
- **FULL_WATCH** (score 0.75–0.84): the reel plays for 90–100% of its duration (8–75 seconds), and there is a 35–85% chance of a like (rising linearly with the score).
- **REWATCH** (score 0.85–1.00): the reel loops and plays through again (up to 120 seconds total), with a higher chance of a like and a 30–70% chance of a save. These reels' creators also become candidates for a follow after three high-scoring reels.

This 25× difference in dwell time between a skip and a rewatch is the gradient that reshapes the feed. Instagram's algorithm interprets short dwells as disinterest and long dwells as engagement, gradually adjusting the content mix.

An **exploration mechanism** (ε = 0.10) watches 10% of borderline reels (score 0.25–0.39) that would normally be skipped. This prevents the feed from collapsing into a narrow filter bubble and ensures the agent keeps discovering new creators and topics. This is a classic ε-greedy strategy from multi-armed bandit theory.

Engagement actions are capped per session — at most 20 likes, 5 saves, and 3 follows — and are probabilistic, not deterministic. A real user does not like every reel they enjoy; they like some fraction of them. PRAG mimics this with probability ramps: the chance of a like starts at 35% at the like threshold and rises to 85% at a perfect score. The daily runner supports a "mixed" engagement mode: the first session of the day uses live engagement (likes, saves, follows), while subsequent sessions use watch time only, capping the total daily engagement to human-plausible levels.

The **convergence metric** is the curation target: when at least 50% of the last 100 organic reels score above 0.40 (the watch threshold), the feed is considered curated — the algorithm has learned what this persona values.

---

## 6. Supervised Preference Model (ml/train.py)

Over time, the LLM's labels accumulate into a dataset of scored reels. PRAG trains a fast supervised model on these labels so that it can score reels in milliseconds when the LLM is unavailable (LM Studio is not running, the Mac is on battery, or a quick daily run is preferred). Two feature representations are compared: **TF-IDF** (character 2–5 grams of caption, cover text, and transcript — effective across Hindi, English, and Hinglish without any language-specific tokenisation) and **sentence embeddings** (the same multilingual sentence-transformer used for story grouping, `paraphrase-multilingual-MiniLM-L12-v2`). Both are fed into a class-balanced logistic regression. Evaluation uses 5-fold stratified group K-fold cross-validation, where the groups are creators: every test fold contains creators the model has never seen during training, so the metrics measure generalisation to new accounts, not memorised caption styles. The better-performing model is saved and used as the fallback scorer.

---

## 7. Unsupervised Feed Topic Model (ml/feed_model.py)

The feed topic model answers the question: "What is the algorithm serving this persona, and how is it changing?" Every scored reel's text — topic label, summary, caption, cover text, and transcript — is embedded with the multilingual sentence model and clustered with K-Means (k = √(reels/2), capped at 24). Topics are named automatically using class-based TF-IDF: each topic's most distinctive words (weighted by how common the word is within the topic versus how many topics it appears in) become its label. The model computes the topic distribution (mix) for each persona's feed, the effective number of topics (e^entropy of the distribution, which falls as the feed narrows), the Jensen-Shannon distance between consecutive days (measuring how much the feed moved), rising topics (topics the latest days show more of), and topics a persona is served more than the overall average. When multiple personas are running, the model also computes a similarity matrix showing how alike their feeds are. The model is retrained automatically after every daily run.

---

## 8. The Persona System

PRAG supports multiple Instagram accounts, each scrolling as a different persona — a complete description of a fictional person with specific interests, political leanings, profession, and regional context. The LLM scores every reel from that persona's point of view. Six built-in presets are provided: a News Follower (Delhi office worker), a Right-Leaning Voter (Lucknow shop owner), a Left-Leaning Voter (Kolkata student), a Tech Enthusiast (Bengaluru engineer), an Investor (Mumbai chartered accountant), and a Science & Health Reader (Pune doctor). Each persona carries not only the interests prompt but also watch words (for the quick rules), skip words (for the keyword gate), creator-name hints (for identifying relevant pages), and seed accounts to follow before the first session. Custom personas can be created and edited through a web panel.

---

## 9. Story Grouping, Verification, and the Daily Digest

Phase 2 of the pipeline turns raw scored reels into a curated news briefing. **Story grouping** embeds each informative reel's topic and summary with the multilingual sentence model and merges reels whose cosine distance is below 0.45 into a single story — one event reported by multiple creators. **News verification** takes each story's summary, embeds it, and compares it against Google News headlines from the past 3–30 days. A cosine similarity of 0.60 or above means a real news source covered the same event, and the story is labelled "verified"; lower similarities yield "plausible" or "unverified" labels. The **daily digest** is an HTML page generated after every daily run, listing stories ranked by score with their verification status. A macOS notification is posted when the digest is ready, with the top story's headline in the alert. The result is that the user never needs to scroll at all — they open a single page and read the day's curated, verified news.

---

## 10. Conclusion

PRAG is a complete machine learning system that touches supervised learning (the preference model), unsupervised learning (the feed topic model), reinforcement learning (the ε-greedy bandit engagement policy that trains Instagram's recommender through differential watch time), natural language processing (LLM scoring, speech transcription, OCR, sentence embeddings), and multimodal perception (audio, image, and structured metadata fusion). It runs entirely locally on Apple Silicon, uses no cloud APIs, and autonomously curates a social media feed from noise into signal — then verifies and delivers the signal as a daily briefing.
